//! Reference-conditioned plain, dynamic KV and static KV decoding.

use super::GPTModel;
use super::conditioning::check_kv_capacity;
use super::sampling::DecoderSampler;
use crate::Result;
use crate::utils::{KvCacheManager, StaticKvManager};
use candle_core::{DType, Tensor};

impl GPTModel {
    /// Generate semantic tokens matching Python's infer_panel behavior.
    ///
    /// This method uses prompt audio tokens as the audio context, matching
    /// Python's Text2SemanticLightningModule.model.infer_panel().
    ///
    /// # Arguments
    /// * `phoneme_ids` - Input phoneme sequence
    /// * `prompt_tokens` - Prompt audio token IDs (from reference audio)
    /// * `bert_features` - Optional BERT features [batch, 1024, seq_len] or [batch, seq_len, 1024]
    /// * `top_k` - Top-k sampling parameter
    /// * `top_p` - Top-p (nucleus) sampling parameter
    /// * `temperature` - Sampling temperature
    ///
    /// # Returns
    /// Vector of generated semantic token IDs
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_prompts(
        &self,
        phoneme_ids: &[usize],
        prompt_tokens: &[usize],
        bert_features: Option<&Tensor>,
        word2ph: &[usize],
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_tokens: usize,
    ) -> Result<Vec<usize>> {
        self.generate_with_prompts_inner(
            phoneme_ids,
            prompt_tokens,
            bert_features,
            None,
            word2ph,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            max_tokens,
        )
    }

    /// Like `generate_with_prompts` but accepts pre-aligned 512-dim BERT features
    /// [1, all_phones, 512] ready to add directly to text embeddings.
    /// Use this when ref+target BERT features are pre-concatenated externally.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_prompts_aligned_bert(
        &self,
        phoneme_ids: &[usize],
        prompt_tokens: &[usize],
        pre_aligned_bert: Option<&Tensor>,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_tokens: usize,
    ) -> Result<Vec<usize>> {
        self.generate_with_prompts_inner(
            phoneme_ids,
            prompt_tokens,
            None,
            pre_aligned_bert,
            &[],
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            max_tokens,
        )
    }

    /// KV-cache version of `generate_with_prompts_aligned_bert`.
    ///
    /// Prefills the cache with the full text+prompt sequence in one forward pass,
    /// then generates each audio token with a single-token forward pass (O(1) per step
    /// instead of O(n) for the non-cached version).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_prompts_aligned_bert_kv_cache(
        &self,
        phoneme_ids: &[usize],
        prompt_tokens: &[usize],
        pre_aligned_bert: Option<&Tensor>,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_tokens: usize,
    ) -> Result<Vec<usize>> {
        if phoneme_ids.is_empty() || max_tokens == 0 {
            return Ok(Vec::new());
        }

        let text_seq = phoneme_ids.len();
        let prompt_seq = prompt_tokens.len();
        let max_new_tokens = max_tokens;

        // Build text embeddings + BERT fusion (same as non-cached path)
        let x_emb = self.embed_phonemes(phoneme_ids)?;

        let x_emb = if let Some(aligned) = pre_aligned_bert {
            x_emb.broadcast_add(aligned)?
        } else {
            x_emb
        };
        let prefill_input = self.prepare_prefill(&x_emb, prompt_tokens)?;
        let total_prefill = text_seq + prompt_seq;
        let hybrid_mask = self.create_hybrid_mask(text_seq, total_prefill)?;

        let mut kv_cache = KvCacheManager::new(self.num_layers);
        let prefill_out = self.transformer.forward_from_embedding_kv(
            &prefill_input,
            Some(&hybrid_mask),
            &mut kv_cache,
        )?;

        // Get logits from the last prefill position
        let last_hidden = prefill_out.narrow(1, total_prefill - 1, 1)?.squeeze(0)?;
        let logits = last_hidden.matmul(&self.ar_predict_layer_t)?.squeeze(0)?;

        let mut generated_tokens: Vec<usize> = Vec::new();
        let mut sampler = DecoderSampler::new(self, top_k, top_p, temperature, repetition_penalty);

        // First token sampling from prefill logits (step 0 — always mask EOS)
        let Some(next_token) = sampler.sample_next(&logits, prompt_tokens, &generated_tokens)?
        else {
            return Ok(generated_tokens);
        };
        generated_tokens.push(next_token);

        // Autoregressive generation: single token per forward pass
        for _ in 1..max_new_tokens {
            let prev_token = *generated_tokens.last().unwrap();
            let new_emb = self.lookup_audio_token(prev_token)?;
            let audio_pe_pos = prompt_seq + generated_tokens.len() - 1;
            let new_pos = self.add_sine_positional_at(&new_emb, audio_pe_pos)?;

            // Single-token forward: new token attends to ALL cached K/V (no mask needed)
            let hidden =
                self.transformer
                    .forward_from_embedding_kv(&new_pos, None, &mut kv_cache)?;

            let logits = hidden
                .squeeze(0)?
                .matmul(&self.ar_predict_layer_t)?
                .squeeze(0)?;

            let Some(next_token) =
                sampler.sample_next(&logits, prompt_tokens, &generated_tokens)?
            else {
                break;
            };

            generated_tokens.push(next_token);
        }

        tracing::info!(
            "[GPT kv] Total generated tokens: {}",
            generated_tokens.len()
        );
        Ok(generated_tokens)
    }

    /// Like `generate_with_prompts_aligned_bert_kv_cache`, but after prefill the KV cache is
    /// converted to pre-allocated fixed-size buffers (`StaticKvManager`).
    /// This eliminates the O(n) `Tensor::cat` allocations in the dynamic KV approach, replacing
    /// them with an in-place `scatter_set` that writes only the new K/V token at each step.
    ///
    /// The storage is fixed, while eager attention only reads the valid prefix. CUDA Graph uses
    /// the separate fixed-shape graphable path below.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_static_kv(
        &self,
        phoneme_ids: &[usize],
        prompt_tokens: &[usize],
        pre_aligned_bert: Option<&Tensor>,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_new_tokens: usize,
        max_kv_len: usize, // max total KV sequence length (prefill + generated)
    ) -> Result<Vec<usize>> {
        if phoneme_ids.is_empty() || max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        let text_seq = phoneme_ids.len();
        let prompt_seq = prompt_tokens.len();
        let total_prefill = text_seq + prompt_seq;

        check_kv_capacity(total_prefill, max_new_tokens, max_kv_len)?;

        // ── Build text embeddings ────────────────────────────────────────────────
        let x_emb = self.embed_phonemes(phoneme_ids)?;

        // Inject BERT features if available
        let x_emb = if let Some(bert) = pre_aligned_bert {
            let bert_f32 = bert.to_dtype(DType::F32)?;
            let x_f32 = x_emb.to_dtype(DType::F32)?;
            // Debug: dump BERT and x_emb stats
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

        // ── Convert dynamic KV → static pre-allocated KV ────────────────────────
        let mut static_kv = StaticKvManager::from_dynamic(dyn_kv, max_kv_len)?;
        let mut sampler = DecoderSampler::new(self, top_k, top_p, temperature, repetition_penalty);

        // ── First token from prefill logits ─────────────────────────────────────
        let last_hidden = prefill_out.narrow(1, total_prefill - 1, 1)?.squeeze(0)?;
        let logits = last_hidden.matmul(&self.ar_predict_layer_t)?.squeeze(0)?;
        let Some(next_token) = sampler.sample_next(&logits, prompt_tokens, &[])? else {
            return Ok(vec![]);
        };
        let mut generated_tokens = vec![next_token];

        // ── Autoregressive decode with static KV ─────────────────────────────────
        for _ in 1..max_new_tokens {
            let prev_token = *generated_tokens.last().unwrap();
            let new_emb = self.lookup_audio_token(prev_token)?;
            let pe_pos = prompt_seq + generated_tokens.len() - 1;
            let new_pos = self.add_sine_positional_at(&new_emb, pe_pos)?;

            let hidden = self
                .transformer
                .forward_from_embedding_static(&new_pos, &mut static_kv)?;

            let logits = hidden
                .squeeze(0)?
                .matmul(&self.ar_predict_layer_t)?
                .squeeze(0)?;

            let Some(next_token) =
                sampler.sample_next(&logits, prompt_tokens, &generated_tokens)?
            else {
                break;
            };

            generated_tokens.push(next_token);
        }

        tracing::info!(
            "[GPT static-kv] Generated {} tokens",
            generated_tokens.len()
        );
        Ok(generated_tokens)
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_with_prompts_inner(
        &self,
        phoneme_ids: &[usize],
        prompt_tokens: &[usize],
        bert_features: Option<&Tensor>,
        pre_aligned_bert: Option<&Tensor>, // [1, all_phones, 512] — bypasses projection+alignment
        word2ph: &[usize],
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_tokens: usize,
    ) -> Result<Vec<usize>> {
        if phoneme_ids.is_empty() || max_tokens == 0 {
            return Ok(Vec::new());
        }

        let text_seq = phoneme_ids.len();
        let prompt_seq = prompt_tokens.len();
        let x_emb = self.embed_phonemes(phoneme_ids)?;

        // Step 2: Add BERT features (either pre-aligned 512-dim, or raw 1024-dim with projection)
        let x_emb = if let Some(aligned) = pre_aligned_bert {
            // Pre-aligned 512-dim features: add directly without any processing
            x_emb.broadcast_add(aligned)?
        } else if let (Some(bert), Some((proj_w, proj_b))) = (bert_features, &self.bert_proj) {
            let bert_dims = bert.dims();
            let bert_reshaped = if bert_dims.len() == 3 && bert_dims[1] == 1024 {
                bert.transpose(1, 2)? // [batch, 1024, seq] -> [batch, seq, 1024]
            } else {
                bert.clone()
            };
            let last_dim = bert_reshaped.dims().last().copied();
            if last_dim == Some(1024) {
                let proj_w_3d = proj_w.t()?.unsqueeze(0)?;
                let projected = bert_reshaped.matmul(&proj_w_3d)?;
                let projected =
                    projected.broadcast_add(&proj_b.reshape((1, 1, proj_b.dims()[0]))?)?;
                let target_seq = x_emb.dims()[1];
                let aligned = self.align_bert_to_phonemes(&projected, target_seq, word2ph)?;
                x_emb.broadcast_add(&aligned)?
            } else {
                x_emb
            }
        } else {
            x_emb
        };

        let mut xy_pos = self.prepare_prefill(&x_emb, prompt_tokens)?;

        let mut generated_tokens = Vec::new();
        let max_new_tokens = max_tokens;
        let mut sampler = DecoderSampler::new(self, top_k, top_p, temperature, repetition_penalty);

        for _ in 0..max_new_tokens {
            let total_seq = xy_pos.dims()[1];

            // Use hybrid mask: text bidirectional + text blocked from audio + audio causal
            let mask = self.create_hybrid_mask(text_seq, total_seq)?;
            let hidden = self
                .transformer
                .forward_all_layers_with_mask(&xy_pos, &mask)?;

            // Project last position to vocab
            let last_hidden = hidden.narrow(1, total_seq - 1, 1)?.squeeze(0)?;
            let logits = last_hidden.matmul(&self.ar_predict_layer_t)?.squeeze(0)?;

            let Some(next_token) =
                sampler.sample_next(&logits, prompt_tokens, &generated_tokens)?
            else {
                break;
            };

            generated_tokens.push(next_token);

            let new_emb = self.lookup_audio_token(next_token)?;
            // Audio PE position = prompt_seq + generated_so_far (not text_seq + total_audio)
            let audio_pe_pos = prompt_seq + generated_tokens.len() - 1;
            let new_pos = self.add_sine_positional_at(&new_emb, audio_pe_pos)?;
            xy_pos = Tensor::cat(&[&xy_pos, &new_pos], 1)?;
        }

        tracing::info!("[GPT] Total generated tokens: {}", generated_tokens.len());
        Ok(generated_tokens)
    }
}
