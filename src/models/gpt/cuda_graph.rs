//! CUDA capture/replay lifecycle and guarded KV fallback.

use super::GPTModel;
use crate::Result;
use candle_core::Tensor;

#[cfg(feature = "cuda")]
use super::conditioning::check_kv_capacity;
#[cfg(feature = "cuda")]
use super::sampling::{DecoderSampler, MIN_TOKENS_BEFORE_EOS};
#[cfg(feature = "cuda")]
use crate::utils::{KvCacheManager, StaticKvManager};
#[cfg(feature = "cuda")]
use candle_core::DType;

impl GPTModel {
    /// CUDA graph-accelerated decode. Requires the `cuda` feature.
    ///
    /// Flow:
    ///  1. Prefill with dynamic KV — captured graphs don't help here (variable shapes).
    ///  2. Convert KV to pre-allocated static buffers (fixed `[max_kv_len]` shapes).
    ///  3. Pre-allocate stable input/mask buffers for the graph boundary.
    ///  4. Warm the allocator with eager steps through the first EOS-eligible sample.
    ///  5. Capture one full decode step into a CUDA graph.
    ///  6. For each remaining step:
    ///     a. Add the sampled token embedding and its position on the CPU.
    ///     b. H2D-copy the result into the pre-allocated input buffer.
    ///     c. H2D-copy the current position once for all attention layers.
    ///     d. Unmask the new position across all attention heads on the GPU.
    ///     e. Launch graph (replays ~160 kernels as a single submission).
    ///     f. Sample from the (still-live) logits tensor written by the graph.
    ///
    /// Uses dynamic KV on non-CUDA devices and eager static KV for non-F32 CUDA models.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_cuda_graph(
        &self,
        phoneme_ids: &[usize],
        prompt_tokens: &[usize],
        pre_aligned_bert: Option<&Tensor>,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_new_tokens: usize,
        max_kv_len: usize,
    ) -> Result<Vec<usize>> {
        if phoneme_ids.is_empty() || max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        #[cfg(feature = "cuda")]
        {
            use std::sync::Arc;

            let candle_core::Device::Cuda(cuda_dev) = &self.device else {
                return self.generate_with_prompts_aligned_bert_kv_cache(
                    phoneme_ids,
                    prompt_tokens,
                    pre_aligned_bert,
                    top_k,
                    top_p,
                    temperature,
                    repetition_penalty,
                    max_new_tokens,
                );
            };
            // Graph replay uploads F32 embeddings. Other CUDA dtypes can finish
            // the request with the same static KV operations used in the eager prefix.
            if self.dtype != DType::F32 {
                return self.generate_with_static_kv(
                    phoneme_ids,
                    prompt_tokens,
                    pre_aligned_bert,
                    top_k,
                    top_p,
                    temperature,
                    repetition_penalty,
                    max_new_tokens,
                    max_kv_len,
                );
            }

            let text_seq = phoneme_ids.len();
            let prompt_seq = prompt_tokens.len();
            let total_prefill = text_seq + prompt_seq;
            let hidden_size = self.transformer.config.hidden_size;
            check_kv_capacity(total_prefill, max_new_tokens, max_kv_len)?;

            // Set memory pool release threshold to never-release BEFORE any allocations.
            // When graph-managed cuMemAllocAsync allocations are freed (cuMemFreeAsync) inside
            // the graph, the pool retains the memory instead of returning it to the OS.
            // On each graph replay the pool hands out the SAME virtual address for each
            // allocation node — identical to PyTorch's CUDA-graph stability trick.
            // Without this, CUDA 13.3/WSL2 may hand out different addresses on replay and
            // then fail to update the captured kernel parameters (stale pointer = ILLEGAL_ADDRESS).
            unsafe {
                use candle_core::cuda_backend::cudarc::driver;
                let cu_device = driver::result::device::get(0)
                    .map_err(|e| candle_core::Error::Msg(format!("cu_device get: {e:?}")))?;
                let pool = driver::result::device::get_default_mem_pool(cu_device)
                    .map_err(|e| candle_core::Error::Msg(format!("get_default_mem_pool: {e:?}")))?;
                let threshold: u64 = u64::MAX;
                driver::result::mem_pool::set_attribute(
                    pool,
                    driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const u64 as *mut std::ffi::c_void,
                )
                .map_err(|e| candle_core::Error::Msg(format!("set pool threshold: {e:?}")))?;
            }

            // ── Prefill (same as static_kv path) ─────────────────────────────────
            let x_emb = self.embed_phonemes(phoneme_ids)?;
            let x_emb = if let Some(bert) = pre_aligned_bert {
                let bert_f32 = bert.to_dtype(DType::F32)?;
                let x_f32 = x_emb.to_dtype(DType::F32)?;
                (x_f32 + bert_f32)?.to_dtype(self.dtype)?
            } else {
                x_emb
            };
            let prefill_input = self.prepare_prefill(&x_emb, prompt_tokens)?;
            let hybrid_mask = self.create_hybrid_mask(text_seq, total_prefill)?;
            let mut dyn_kv = KvCacheManager::new(self.num_layers);
            let prefill_out = self.transformer.forward_from_embedding_kv(
                &prefill_input,
                Some(&hybrid_mask),
                &mut dyn_kv,
            )?;
            let mut static_kv = StaticKvManager::from_dynamic(dyn_kv, max_kv_len)?;
            let mut sampler =
                DecoderSampler::new(self, top_k, top_p, temperature, repetition_penalty);

            // ── First token ───────────────────────────────────────────────────────
            let last_hidden = prefill_out.narrow(1, total_prefill - 1, 1)?.squeeze(0)?;
            let first_logits = last_hidden.matmul(&self.ar_predict_layer_t)?.squeeze(0)?;
            let Some(next_token) = sampler.sample_next(&first_logits, prompt_tokens, &[])? else {
                return Ok(vec![]);
            };
            let mut generated = vec![next_token];

            // Acquire the CUDA stream early — needed for raw memcpy helpers below.
            let stream: Arc<candle_core::cuda_backend::cudarc::driver::CudaStream> =
                cuda_dev.cuda_stream();

            // ── Pre-allocate stable graph-boundary buffers ────────────────────────
            // These buffers must be allocated BEFORE capture so their CUDA memory addresses
            // remain stable across graph launches (graph-managed cuMemAllocAsync allocations
            // may change address between launches).
            let n_heads = self.transformer.config.num_attention_heads;

            // input_buf: [1, 1, hidden_size] — updated via raw H2D copy before each graph replay
            let input_buf = Tensor::zeros((1usize, 1usize, hidden_size), self.dtype, &self.device)?;

            // attn_mask_buf: [1, n_heads, 1, max_kv_len] — PRE-EXPANDED (no expand() inside graph).
            // Storing as full [n_heads, max_kv_len] avoids a non-contiguous broadcast view inside
            // forward_kv_graphable, which would allocate a strides-buffer via cuMemAllocAsync.
            // A stale strides pointer on second graph replay causes CUDA_ERROR_ILLEGAL_ADDRESS.
            let attn_mask_buf = {
                let mut v = vec![-1e9f32; n_heads * max_kv_len];
                for h in 0..n_heads {
                    for i in 0..total_prefill {
                        v[h * max_kv_len + i] = 0.0f32;
                    }
                }
                Tensor::from_slice(&v, (1usize, n_heads, 1usize, max_kv_len), &self.device)?
                    .to_dtype(self.dtype)?
            };

            // logits_out: [n_vocab] F32 — graph output is copied into this stable buffer.
            // Allocated PRE-capture so the address doesn't change between graph launches.
            let n_vocab = self.ar_predict_layer.dim(0)?;
            let logits_out = Tensor::zeros((n_vocab,), DType::F32, &self.device)?;

            // ── Raw CUDA update helpers (NO pool allocation between graph launches) ──────
            // scatter_set uses cuMemAllocAsync for index/strides tensors. Any pool allocation
            // between graph launches disturbs the pool's free-list order. On the next launch,
            // the pool returns different addresses for the graph's own cuMemAllocAsync nodes,
            // causing stale captured kernel parameters → CUDA_ERROR_ILLEGAL_ADDRESS.
            //
            // cuMemcpyHtoDAsync / cuMemcpyDtoDAsync do NOT allocate from the memory pool,
            // so they leave the pool state undisturbed between launches.
            use candle_core::cuda_backend::{
                cudarc::driver::{result as cudarc_result, DevicePtr},
                CudaStorageSlice,
            };

            // Helper: get raw CUDA device pointer for a contiguous tensor (at start_offset).
            // SAFETY: tensor must stay alive for the duration of any async copy using the pointer.
            let get_cuda_ptr = |tensor: &Tensor| -> candle_core::Result<u64> {
                let (storage, layout) = tensor.storage_and_layout();
                let esize = tensor.dtype().size_in_bytes();
                let start = layout.start_offset();
                match &*storage {
                    candle_core::Storage::Cuda(cs) => {
                        let cstream = cs.device.cuda_stream();
                        let base = match &cs.slice {
                            CudaStorageSlice::F32(s) => {
                                let (p, _g) = s.device_ptr(&cstream);
                                p
                            }
                            CudaStorageSlice::F16(s) => {
                                let (p, _g) = s.device_ptr(&cstream);
                                p
                            }
                            CudaStorageSlice::BF16(s) => {
                                let (p, _g) = s.device_ptr(&cstream);
                                p
                            }
                            CudaStorageSlice::I64(s) => {
                                let (p, _g) = s.device_ptr(&cstream);
                                p
                            }
                            _ => {
                                return Err(candle_core::Error::Msg(
                                    "unsupported dtype in get_cuda_ptr".into(),
                                ))
                            }
                        };
                        Ok(base + (start * esize) as u64)
                    }
                    _ => Err(candle_core::Error::Msg(
                        "tensor is not on CUDA device".into(),
                    )),
                }
            };

            let raw_stream = stream.cu_stream();

            // Precompute audio embedding weights on CPU for pool-free input updates.
            // D2H copy is done once here; each decode step uses a CPU-side emb+PE computation.
            let audio_emb_cpu: Vec<f32> = self
                .audio_embedding
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let audio_pos_cpu: Vec<f32> = self
                .audio_positional
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;

            // Cache the raw pointer for input_buf (stable for entire function lifetime)
            let input_buf_ptr = get_cuda_ptr(&input_buf)?;
            // Cache the raw pointer for attn_mask_buf
            let mask_buf_ptr = get_cuda_ptr(&attn_mask_buf)?;
            let mask_esize = attn_mask_buf.dtype().size_in_bytes();
            let logits_out_ptr = get_cuda_ptr(&logits_out)?;
            let mut input_staging = vec![0.0f32; hidden_size];

            // Update input_buf from the exact precomputed embedding and positional values.
            let mut update_input_raw = |token: usize, pe_pos: usize| -> candle_core::Result<()> {
                let off = token * hidden_size;
                input_staging.copy_from_slice(&audio_emb_cpu[off..off + hidden_size]);
                let pos_off = pe_pos * hidden_size;
                for (value, position) in input_staging
                    .iter_mut()
                    .zip(&audio_pos_cpu[pos_off..pos_off + hidden_size])
                {
                    *value += *position;
                }
                unsafe {
                    cudarc_result::memcpy_htod_async(
                        input_buf_ptr,
                        input_staging.as_slice(),
                        raw_stream,
                    )
                    .map_err(|e| candle_core::Error::Msg(format!("htod input_buf: {e:?}")))?;
                }
                Ok(())
            };

            // One column of the contiguous [heads, max_kv_len] mask: width and pitch are bytes.
            let update_mask_raw = |cache_len: usize| -> candle_core::Result<()> {
                unsafe {
                    candle_core::cuda_backend::cudarc::driver::sys::cuMemsetD2D8Async(
                        mask_buf_ptr + (cache_len * mask_esize) as u64,
                        max_kv_len * mask_esize,
                        0,
                        mask_esize,
                        n_heads,
                        raw_stream,
                    )
                    .result()
                }
                .map_err(|e| candle_core::Error::Msg(format!("unmask position {cache_len}: {e:?}")))
            };

            // ── Eager decode step for warmup (uses static KV's append(), not graphable) ──
            let eager_step =
                |gen: &Vec<usize>, static_kv: &mut StaticKvManager| -> Result<Tensor> {
                    let prev = *gen.last().unwrap();
                    let emb = self.lookup_audio_token(prev)?;
                    let pe_pos = prompt_seq + gen.len() - 1;
                    let pos_emb = self.add_sine_positional_at(&emb, pe_pos)?;
                    let hidden = self
                        .transformer
                        .forward_from_embedding_static(&pos_emb, static_kv)?;
                    Ok(hidden
                        .squeeze(0)?
                        .matmul(&self.ar_predict_layer_t)?
                        .squeeze(0)?)
                };

            // Prefill already sampled one token. Give the decoder one chance to stop
            // at EOS before paying for capture; these steps also warm the allocator.
            let mut reached_eos = false;
            for _ in 0..MIN_TOKENS_BEFORE_EOS {
                if generated.len() >= max_new_tokens {
                    break;
                }
                let pos_before = static_kv.len(); // position this step will write to
                let logits = eager_step(&generated, &mut static_kv)?;
                // Unmask the position that eager_step just appended on the same stream.
                update_mask_raw(pos_before)?;
                let Some(tok) = sampler.sample_next(&logits, prompt_tokens, &generated)? else {
                    reached_eos = true;
                    break;
                };
                generated.push(tok);
            }
            if reached_eos || generated.len() >= max_new_tokens {
                return Ok(generated);
            }

            self.device.synchronize()?;
            let mut guard_kv = static_kv.fork_valid()?;
            let expected_capture_logits = eager_step(&generated, &mut guard_kv)?;
            let expected_capture_values = expected_capture_logits
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;

            // ── Prepare pre-allocated buffers for the capture step ────────────────
            {
                let prev = *generated.last().unwrap();
                let pe_pos = prompt_seq + generated.len() - 1;
                update_input_raw(prev, pe_pos)?;
            }
            let cache_len_before_capture = static_kv.len();
            // Every layer appends at the same position. Bind their captured scatter
            // operations to one index buffer; the K/V buffers stay per-layer.
            let (first, rest) = static_kv.layers.split_first_mut().ok_or_else(|| {
                candle_core::Error::Msg("CUDA graph requires at least one transformer layer".into())
            })?;
            for layer in rest {
                layer.pos_idx = first.pos_idx.clone();
            }
            let pos_ptr = get_cuda_ptr(&first.pos_idx)?;
            let mut pos_staging = vec![0i64; first.pos_idx.elem_count()];
            let mut update_position = |pos: usize| {
                pos_staging.fill(pos as i64);
                unsafe {
                    cudarc_result::memcpy_htod_async(pos_ptr, pos_staging.as_slice(), raw_stream)
                }
                .map_err(|e| candle_core::Error::Msg(format!("htod pos_idx: {e:?}")))
            };
            update_position(cache_len_before_capture)?;
            update_mask_raw(cache_len_before_capture)?;
            // Finish graph input updates before capture begins.
            stream
                .synchronize()
                .map_err(|e| candle_core::Error::Msg(format!("sync before capture: {e:?}")))?;

            // Release free-listed prefill blocks, including BERT intermediates, before
            // capture. The graph then owns a stable allocator state across replays.
            unsafe {
                use candle_core::cuda_backend::cudarc::driver;
                let cu_device = driver::result::device::get(0)
                    .map_err(|e| candle_core::Error::Msg(format!("cu_device get (trim): {e:?}")))?;
                let pool =
                    driver::result::device::get_default_mem_pool(cu_device).map_err(|e| {
                        candle_core::Error::Msg(format!("get_default_mem_pool (trim): {e:?}"))
                    })?;
                driver::result::mem_pool::trim_to(pool, 0)
                    .map_err(|e| candle_core::Error::Msg(format!("cuMemPoolTrimTo: {e:?}")))?;
                // Re-set threshold to never-release AFTER trim so graph-managed allocs
                // are pinned from this point forward.
                let threshold: u64 = u64::MAX;
                driver::result::mem_pool::set_attribute(
                    pool,
                    driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const u64 as *mut std::ffi::c_void,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!("set pool threshold (post-trim): {e:?}"))
                })?;
            }

            // ── CUDA graph capture ────────────────────────────────────────────────
            use candle_core::cuda_backend::cudarc::driver::sys::{
                CUgraphInstantiate_flags, CUstreamCaptureMode,
            };
            stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| candle_core::Error::Msg(format!("begin_capture: {e:?}")))?;

            // The graph captures: scatter-based decode + logits projection + a raw D2D
            // copy into logits_out (stable pre-allocated buffer).
            // cuMemAllocAsync allocations inside capture hold placeholder pointers that only
            // become valid after graph.launch(). logits_tmp MUST be dropped before end_capture
            // so its free_async is also captured (graph owns the full alloc/free lifecycle).
            {
                let logits_tmp = self
                    .transformer
                    .forward_from_embedding_graphable(&input_buf, &static_kv, &attn_mask_buf)?
                    .squeeze(0)?
                    .matmul(&self.ar_predict_layer_t)?
                    .squeeze(0)?;
                let logits_tmp_ptr = get_cuda_ptr(&logits_tmp)?;
                unsafe {
                    cudarc_result::memcpy_dtod_async(
                        logits_out_ptr,
                        logits_tmp_ptr,
                        n_vocab * DType::F32.size_in_bytes(),
                        raw_stream,
                    )
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("capture logits D2D copy: {e:?}"))
                    })?;
                }
                // logits_tmp drops here (still inside capture) → free_async captured ✓
            }

            let graph_opt = stream
                .end_capture(
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .map_err(|e| candle_core::Error::Msg(format!("end_capture: {e:?}")))?;

            let Some(graph) = graph_opt else {
                tracing::warn!("[GPT cuda-graph] empty graph; falling back to dynamic KV");
                return self.generate_with_prompts_aligned_bert_kv_cache(
                    phoneme_ids,
                    prompt_tokens,
                    pre_aligned_bert,
                    top_k,
                    top_p,
                    temperature,
                    repetition_penalty,
                    max_new_tokens,
                );
            };

            // Graph capture only RECORDS operations — they were NOT executed.
            // Launch now to execute the captured decode step and populate logits_out.
            graph
                .launch()
                .map_err(|e| candle_core::Error::Msg(format!("graph launch (first): {e:?}")))?;
            stream
                .synchronize()
                .map_err(|e| candle_core::Error::Msg(format!("stream sync (first): {e:?}")))?;
            static_kv.step();
            let actual_capture_values = logits_out.to_vec1::<f32>()?;
            let max_error = expected_capture_values
                .iter()
                .zip(actual_capture_values.iter())
                .map(|(&left, &right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            if !max_error.is_finite() || max_error > 1e-3 {
                tracing::warn!(
                    "[GPT cuda-graph] first graph logits diverged (max_error={max_error:.6e}); continuing with guarded KV"
                );
                let Some(token) =
                    sampler.sample_next(&expected_capture_logits, prompt_tokens, &generated)?
                else {
                    return Ok(generated);
                };
                generated.push(token);

                for _ in generated.len()..max_new_tokens {
                    let logits = eager_step(&generated, &mut guard_kv)?;
                    let Some(token) = sampler.sample_next(&logits, prompt_tokens, &generated)?
                    else {
                        break;
                    };
                    generated.push(token);
                }
                tracing::info!(
                    "[GPT cuda-graph] guarded KV generated {} tokens total",
                    generated.len()
                );
                return Ok(generated);
            }
            drop(expected_capture_logits);
            drop(guard_kv);
            {
                let Some(tok) = sampler.sample_next(&logits_out, prompt_tokens, &generated)? else {
                    tracing::info!("[GPT cuda-graph] EOS at capture step");
                    return Ok(generated);
                };
                generated.push(tok);
            }

            tracing::info!(
                "[GPT cuda-graph] Graph captured ({} eager + 1 capture steps); replaying",
                MIN_TOKENS_BEFORE_EOS
            );

            // ── Graph replay loop ─────────────────────────────────────────────────
            for step in generated.len()..max_new_tokens {
                // 1. Compute emb+PE on CPU, H2D copy to input_buf — zero pool allocations
                let prev = *generated.last().unwrap();
                let pe_pos = prompt_seq + generated.len() - 1;
                update_input_raw(prev, pe_pos)?;

                // 2. Update pos_idx (H2D copy, zero pool allocations)
                let cur_len = static_kv.len();
                tracing::debug!(
                    "[GPT cuda-graph] step {step}: cur_len={cur_len} max_kv_len={max_kv_len}"
                );
                if cur_len >= max_kv_len {
                    tracing::warn!("[GPT cuda-graph] KV cache overflow: cur_len={cur_len} >= max_kv_len={max_kv_len}, stopping");
                    break;
                }
                update_position(cur_len)?;

                // 3. Unmask the new position for all heads.
                update_mask_raw(cur_len)?;

                // 5. Launch graph (replays all 16 layers of attention + FFN + logits proj)
                // No explicit sync needed: input updates and graph launch are on the same stream,
                // so CUDA guarantees the copies complete before the graph kernel reads.
                graph.launch().map_err(|e| {
                    candle_core::Error::Msg(format!("graph launch step {step}: {e:?}"))
                })?;
                // Sync stream before D2H copy so logits_out is fully written.
                stream.synchronize().map_err(|e| {
                    candle_core::Error::Msg(format!("stream sync step {step}: {e:?}"))
                })?;

                static_kv.step();

                // 6. Sample (D2H; logits_out is stable pre-allocated buffer written by graph)
                let Some(tok) = sampler.sample_next(&logits_out, prompt_tokens, &generated)? else {
                    break;
                };
                generated.push(tok);
            }

            tracing::info!(
                "[GPT cuda-graph] Generated {} tokens total",
                generated.len()
            );
            Ok(generated)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = max_kv_len;
            self.generate_with_prompts_aligned_bert_kv_cache(
                phoneme_ids,
                prompt_tokens,
                pre_aligned_bert,
                top_k,
                top_p,
                temperature,
                repetition_penalty,
                max_new_tokens,
            )
        }
    }
}
