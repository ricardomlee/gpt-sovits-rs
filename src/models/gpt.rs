//! GPT Model for semantic token prediction
//!
//! This module implements the GPT model from GPT-SoVITS, which uses:
//! - Fused QKV projection (in_proj_weight combines Q, K, V weights)
//! - RoPE (Rotary Position Embedding) instead of learned positions
//! - Separate text and audio embeddings
//! - BERT feature projection
//! - Hubert feature projection for prosody guidance
//! - MRTE (Multi-Reference Timbre Encoder) for advanced fusion

use candle_core::{Device, DType, Tensor};
use crate::{Result, Error};
use crate::utils::{StateDict, load_safetensors, KvCacheManager, StaticKvManager};
use super::transformer::{TransformerGPTSoVITS, TransformerConfig};
use super::mrte::MRTE;

/// GPT Model for semantic token prediction
pub struct GPTModel {
    text_embedding: Tensor,      // model.ar_text_embedding.word_embeddings.weight [vocab_size, hidden_size]
    audio_embedding: Tensor,     // model.ar_audio_embedding.word_embeddings.weight [1025, hidden_size]
    bert_proj: Option<(Tensor, Tensor)>, // (weight, bias) for BERT features [512, 1024], [512]
    hubert_proj: Option<(Tensor, Tensor)>, // (weight, bias) for Hubert features [512, 768], [512]
    mrte: Option<MRTE>,          // MRTE module for advanced cross-attention fusion
    transformer: TransformerGPTSoVITS,
    ar_predict_layer: Tensor,    // output projection [vocab_size, hidden_size]
    text_pos_alpha: f32,         // Learned alpha for text positional encoding
    audio_pos_alpha: f32,        // Learned alpha for audio positional encoding
    device: Device,
    dtype: DType,
    vocab_size: usize,
    num_layers: usize,           // Number of transformer layers for KV cache
}

/// Lookup embeddings handling both text and audio tokens
fn mixed_embedding_lookup(
    text_emb: &Tensor,
    audio_emb: &Tensor,
    indices: &Tensor,
    text_vocab_size: usize,
) -> Result<Tensor> {
    let dims = indices.dims();
    if dims.len() != 2 {
        return Err(candle_core::Error::UnexpectedShape {
            msg: "Expected 2D input for embedding".to_string(),
            expected: candle_core::Shape::from(&[1usize, 1]),
            got: candle_core::Shape::from(dims),
        }.into());
    }

    let (batch, seq_len) = (dims[0], dims[1]);
    let hidden_size = text_emb.dims()[1];
    let device = text_emb.device();

    // Flatten indices to 1D for processing
    let indices_flat: Vec<i64> = indices.flatten_all()?.to_vec1()?;

    // Lookup each index - text tokens use text_emb, audio tokens use audio_emb
    let mut embeddings = Vec::with_capacity(indices_flat.len());
    for &idx in &indices_flat {
        let emb = if (idx as usize) < text_vocab_size {
            // Text token - use text embedding
            text_emb.get(idx as usize)?
        } else {
            // Audio/semantic token - use audio embedding
            // Audio tokens are in range [0, audio_vocab), so subtract text_vocab_size
            let audio_idx = (idx as usize).saturating_sub(text_vocab_size);
            if audio_idx >= audio_emb.dims()[0] {
                tracing::warn!("audio_idx {} out of range for audio_emb {:?}", audio_idx, audio_emb.dims());
                return Err(candle_core::Error::UnexpectedShape {
                    msg: format!("Audio token index {} out of range [0, {})", audio_idx, audio_emb.dims()[0]),
                    expected: candle_core::Shape::from(&[1usize, 1]),
                    got: candle_core::Shape::from(&[audio_idx, 1]),
                }.into());
            }
            audio_emb.get(audio_idx)?
        };
        embeddings.push(emb);
    }

    // Stack: [batch*seq_len, hidden]
    let stacked = Tensor::stack(&embeddings, 0)?.to_device(device)?;

    // Reshape to [batch, seq_len, hidden]
    stacked.reshape((batch, seq_len, hidden_size))
        .map_err(|e| e.into())
}

impl GPTModel {
    /// Load model from safetensors file
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu, DType::F32)
    }

    /// Load model with specific device
    pub fn load_with_device(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        // Load weights from safetensors
        let weights_map = load_safetensors(path)?;
        let state_dict = StateDict::new(weights_map);

        // Load text embedding: [vocab_size, hidden_size]
        let text_emb_key = "model.ar_text_embedding.word_embeddings.weight";
        let text_embedding = state_dict.get(text_emb_key)?
            .to_device(device)?
            .to_dtype(dtype)?;

        let vocab_size = text_embedding.dims()[0];
        let hidden_size = text_embedding.dims()[1];

        // Load audio embedding: [1025, hidden_size]
        let audio_emb_key = "model.ar_audio_embedding.word_embeddings.weight";
        let audio_embedding = state_dict.get(audio_emb_key)?
            .to_device(device)?
            .to_dtype(dtype)?;

        // Load BERT projection (optional): weight [512, 1024], bias [512]
        let bert_proj = if state_dict.contains("model.bert_proj.weight") {
            let bert_weight = state_dict.get("model.bert_proj.weight")?.to_device(device)?.to_dtype(dtype)?;
            let bert_bias = state_dict.get("model.bert_proj.bias")?.to_device(device)?.to_dtype(dtype)?;
            Some((bert_weight, bert_bias))
        } else {
            None
        };

        // Load Hubert projection (optional): weight [512, 768], bias [512]
        let hubert_proj = if state_dict.contains("model.hubert_proj.weight") {
            let hubert_weight = state_dict.get("model.hubert_proj.weight")?.to_device(device)?.to_dtype(dtype)?;
            let hubert_bias = state_dict.get("model.hubert_proj.bias")?.to_device(device)?.to_dtype(dtype)?;
            Some((hubert_weight, hubert_bias))
        } else {
            None
        };

        // Load MRTE module (optional) - for advanced cross-attention fusion
        // MRTE is used when both BERT and Hubert features are available
        let mrte = if state_dict.contains("model.mrte.cross_attention.conv_q.weight") {
            // Convert StateDict to HashMap for VarBuilder
            let mrte_vb = candle_nn::VarBuilder::from_tensors(
                state_dict.as_hash_map().clone(),
                dtype,
                device,
            );
            // Check if we can access MRTE weights
            match MRTE::new(768, 512, 512, 8, mrte_vb.pp("model.mrte")) {
                Ok(mrte) => Some(mrte),
                Err(_) => None,
            }
        } else {
            None
        };

        // Count number of transformer layers
        let mut num_hidden_layers = 0;
        for key in state_dict.keys() {
            if key.starts_with("model.h.layers.") && key.contains(".self_attn.in_proj_weight") {
                num_hidden_layers += 1;
            }
        }

        // GPT-SoVITS v2 uses 16 attention heads for 512 hidden (head_dim=32)
        let num_attention_heads = 16;

        // Get intermediate size from FFN
        let intermediate_size = state_dict.get("model.h.layers.0.linear1.weight")?.dims()[0];

        // Create config for transformer
        let config = TransformerConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            max_seq_len: 2048,
        };

        // Create transformer with GPT-SoVITS style weights
        let transformer = TransformerGPTSoVITS::new(config, &state_dict, device, dtype)?;

        // Load output projection: [vocab_size, hidden_size]
        // Load positional encoding alpha parameters
        let text_pos_alpha = if let Ok(t) = state_dict.get("model.ar_text_position.alpha") {
            let v: Vec<f32> = t.to_dtype(DType::F32)?.to_vec1()?;
            v[0]
        } else {
            1.0
        };
        let audio_pos_alpha = if let Ok(t) = state_dict.get("model.ar_audio_position.alpha") {
            let v: Vec<f32> = t.to_dtype(DType::F32)?.to_vec1()?;
            v[0]
        } else {
            1.0
        };

        let ar_predict_layer = state_dict.get("model.ar_predict_layer.weight")?
            .to_device(device)?
            .to_dtype(dtype)?;

        Ok(Self {
            text_embedding,
            audio_embedding,
            bert_proj,
            hubert_proj,
            mrte,
            transformer,
            ar_predict_layer,
            text_pos_alpha,
            audio_pos_alpha,
            device: device.clone(),
            dtype,
            vocab_size,
            num_layers: num_hidden_layers,
        })
    }

    /// Sample next token from logits using top-k and top-p filtering
    /// Sample a token from logits, matching Python's `logits_to_probs` order exactly:
    ///   1. Repetition penalty (in logit domain)
    ///   2. top_k filter: set non-top-k logits to -inf
    ///   3. top_p filter: using softmax(logits) WITHOUT temperature, set overflow to -inf
    ///   4. Temperature + softmax → final probs
    ///   5. Multinomial sample
    ///
    /// Returns `(sampled_token, argmax_token)` — both computed from ONE D2H transfer.
    /// `prompt_tokens` = prompt audio tokens (for rep penalty matching Python's y which includes prompt)
    fn sample_token(
        logits: &Tensor,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        prompt_tokens: &[usize],
        generated_tokens: &[usize],
    ) -> Result<(usize, usize)> {
        let logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
        let n = logits_vec.len();

        // Compute argmax of raw logits (before any filtering) — Python: torch.argmax(logits)
        let argmax = logits_vec.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut logits_vec = logits_vec;

        // 1. Repetition penalty — Python uses set(y.flatten()) so each unique token is penalized once
        if repetition_penalty != 1.0 {
            let mut seen = std::collections::HashSet::new();
            for &token in prompt_tokens.iter().chain(generated_tokens.iter()) {
                if token < n && seen.insert(token) {
                    if logits_vec[token] > 0.0 {
                        logits_vec[token] /= repetition_penalty;
                    } else {
                        logits_vec[token] *= repetition_penalty;
                    }
                }
            }
        }

        // 2. top_k: find the k-th largest logit value, set all below to -inf
        if top_k > 0 && top_k < n {
            let mut sorted = logits_vec.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let pivot = sorted[top_k - 1];
            for v in logits_vec.iter_mut() {
                if *v < pivot {
                    *v = f32::NEG_INFINITY;
                }
            }
        }

        // 3. top_p: cumulative prob filter on UNTEMPERATURE-SCALED softmax (matches Python)
        if top_p < 1.0 {
            let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = logits_vec.iter().map(|&x| (x - max_l).exp()).collect();
            let exp_sum: f32 = exp_vals.iter().sum();
            let probs_unit: Vec<f32> = exp_vals.iter().map(|&e| e / exp_sum).collect();

            let mut idx_probs: Vec<(usize, f32)> = probs_unit.iter()
                .copied().enumerate().collect();
            idx_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumsum = 0.0f32;
            for &(idx, prob) in &idx_probs {
                if cumsum > top_p {
                    logits_vec[idx] = f32::NEG_INFINITY;
                }
                cumsum += prob;
            }
        }

        // 4. Temperature + softmax → final probs
        let t = if temperature > 0.0 { temperature } else { 1.0 };
        let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_vec.iter().map(|&x| ((x - max_l) / t).exp()).collect();
        let exp_sum: f32 = exp_vals.iter().sum();
        let final_probs: Vec<f32> = exp_vals.iter().map(|&e| e / exp_sum).collect();

        // 5. Multinomial sample
        let rand_val = rand::random::<f32>();
        let mut cumsum = 0.0f32;
        for (i, &prob) in final_probs.iter().enumerate() {
            cumsum += prob;
            if rand_val <= cumsum {
                return Ok((i, argmax));
            }
        }

        // Fallback: argmax of final probs
        let best = final_probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok((best, argmax))
    }

    /// Generate semantic tokens from phoneme IDs (without BERT/Hubert features)
    ///
    /// # Arguments
    /// * `phoneme_ids` - Input phoneme sequence
    /// * `top_k` - Top-k sampling parameter
    /// * `top_p` - Top-p (nucleus) sampling parameter
    /// * `temperature` - Sampling temperature
    ///
    /// # Returns
    /// Vector of semantic token IDs
    pub fn generate(
        &self,
        phoneme_ids: &[usize],
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_tokens: usize,
    ) -> Result<Vec<usize>> {
        self.generate_with_features(phoneme_ids, None, None, top_k, top_p, temperature, repetition_penalty, max_tokens)
    }

    /// Generate semantic tokens with BERT and Hubert features
    ///
    /// # Arguments
    /// * `phoneme_ids` - Input phoneme sequence
    /// * `bert_features` - Optional BERT features [batch, seq_len, 768]
    /// * `hubert_features` - Optional Hubert features [batch, frames, 768]
    /// * `top_k` - Top-k sampling parameter
    /// * `top_p` - Top-p (nucleus) sampling parameter
    /// * `temperature` - Sampling temperature
    ///
    /// # Returns
    /// Vector of semantic token IDs
    pub fn generate_with_features(
        &self,
        phoneme_ids: &[usize],
        bert_features: Option<&Tensor>,
        hubert_features: Option<&Tensor>,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_tokens: usize,
    ) -> Result<Vec<usize>> {
        if phoneme_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Convert phoneme IDs to tensor [1, seq_len]
        let input_ids: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
        let mut current_ids = Tensor::new(input_ids.as_slice(), &self.device)?
            .unsqueeze(0)?;

        let mut generated_tokens = Vec::new();
        let max_new_tokens = max_tokens;

        // Prepare BERT projection if available
        let bert_proj_result = if let Some(bert) = bert_features {
            if let Some((proj_w, proj_b)) = &self.bert_proj {
                // Project BERT features: [batch, seq, 1024] @ [1024, 512] + bias -> [batch, seq, 512]
                // BERT output is [1, seq_len, 1024]
                let bert_dims = bert.dims();
                let bert_reshaped = if bert_dims.len() == 3 && bert_dims[1] == 1024 {
                    // Shape: [batch, 1024, seq] -> transpose to [batch, seq, 1024]
                    bert.transpose(1, 2)?
                } else {
                    bert.clone()
                };

                // Ensure last dim is 1024 for projection
                if bert_reshaped.dims().last().copied() == Some(1024) {
                    // Candle requires same dims for batched matmul: [1, seq, 1024] @ [1, 1024, 512]
                    let proj_w_3d = proj_w.t()?.unsqueeze(0)?;
                    let projected = bert_reshaped.matmul(&proj_w_3d)?;
                    let projected = projected.broadcast_add(&proj_b.reshape((1, 1, proj_b.dims()[0]))?)?;
                    Some(projected)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Prepare Hubert projection
        let hubert_proj_result = if let Some(hubert) = hubert_features {
            if let Some((proj_w, proj_b)) = &self.hubert_proj {
                // Project Hubert features: [batch, frames, 768] @ [768, 512] + bias -> [batch, frames, 512]
                let hubert_dims = hubert.dims();
                if hubert_dims.len() >= 2 && hubert_dims.last().copied() == Some(768) {
                    // Transpose if needed to get [batch, frames, 768]
                    let hubert_reshaped = if hubert_dims.len() == 3 && hubert_dims[1] == 768 {
                        hubert.transpose(1, 2)?
                    } else {
                        hubert.clone()
                    };

                    // Project to hidden size (Candle requires same dims for batched matmul)
                    let proj_w_3d = proj_w.t()?.unsqueeze(0)?;
                    let projected = hubert_reshaped.matmul(&proj_w_3d)?;
                    let projected = projected.broadcast_add(&proj_b.reshape((1, 1, proj_b.dims()[0]))?)?;
                    Some(projected)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Autoregressive generation
        for _step in 0..max_new_tokens {
            let seq_len = current_ids.dims()[1];

            // Get token embeddings - use mixed lookup for text + audio tokens
            let token_emb = mixed_embedding_lookup(&self.text_embedding, &self.audio_embedding, &current_ids, self.vocab_size)?;

            // Fuse features using MRTE if available, otherwise fall back to simple addition
            let fused_emb = if let Some(ref mrte) = self.mrte {
                if let (Some(bert), Some(hubert)) = (bert_proj_result.as_ref(), hubert_proj_result.as_ref()) {
                    let _token_emb_t = token_emb.transpose(1, 2)?;
                    let bert_t = bert.transpose(1, 2)?;
                    let hubert_t = hubert.transpose(1, 2)?;
                    let ones_mask = Tensor::ones((1, 1, seq_len), DType::F32, &self.device)?;
                    match mrte.forward(&hubert_t, &ones_mask, &bert_t, &ones_mask, None) {
                        Ok(mrte_out) => {
                            let mrte_frames = mrte_out.dims()[2];
                            let mrte_aligned = if mrte_frames >= seq_len {
                                mrte_out.narrow(2, 0, seq_len)?
                            } else {
                                let last_frame = mrte_out.narrow(2, mrte_frames - 1, 1)?;
                                let mut frames = vec![mrte_out.clone()];
                                for _ in 0..(seq_len - mrte_frames) {
                                    frames.push(last_frame.clone());
                                }
                                Tensor::cat(&frames, 2).unwrap_or_else(|_| mrte_out.clone())
                            };
                            mrte_aligned.transpose(1, 2).unwrap_or_else(|_| token_emb.clone())
                        }
                        Err(_) => token_emb.clone(),
                    }
                } else {
                    token_emb.clone()
                }
            } else {
                let mut fused_emb = token_emb.clone();
                if let Some(ref bert_proj) = bert_proj_result {
                    if bert_proj.dims().len() >= 2 {
                        let bert_seq_len = bert_proj.dims()[1];
                        if bert_seq_len >= seq_len {
                            let bert_narrowed = if bert_seq_len > seq_len {
                                bert_proj.narrow(1, 0, seq_len)?
                            } else {
                                bert_proj.clone()
                            };
                            if bert_narrowed.dims() == fused_emb.dims() {
                                let scale = 0.5f32;
                                let scaled_bert = bert_narrowed.broadcast_mul(&Tensor::full(scale, bert_narrowed.dims(), &self.device)?)?;
                                fused_emb = fused_emb.broadcast_add(&scaled_bert)?;
                            }
                        }
                    }
                }
                if let Some(ref hubert_proj) = hubert_proj_result {
                    let hubert_frames = hubert_proj.dims()[1];
                    if hubert_frames > 0 {
                        let hubert_aligned = if hubert_frames >= seq_len {
                            hubert_proj.narrow(1, 0, seq_len)?
                        } else {
                            let last_frame = hubert_proj.narrow(1, hubert_frames - 1, 1)?;
                            let mut frames = vec![hubert_proj.clone()];
                            for _ in 0..(seq_len - hubert_frames) {
                                frames.push(last_frame.clone());
                            }
                            Tensor::cat(&frames, 1).unwrap_or_else(|_| hubert_proj.clone())
                        };
                        if hubert_aligned.dims() == fused_emb.dims() {
                            let scale = 0.3f32;
                            let scaled_hubert = hubert_aligned.broadcast_mul(&Tensor::full(scale, hubert_aligned.dims(), &self.device)?)?;
                            fused_emb = fused_emb.broadcast_add(&scaled_hubert)?;
                        }
                    }
                }
                fused_emb
            };

            // Forward pass through transformer
            let hidden = self.transformer.forward_from_embedding(&fused_emb)?;

            // Project to vocab
            let last_hidden = hidden.narrow(1, seq_len - 1, 1)?.squeeze(0)?;
            let logits = last_hidden.matmul(&self.ar_predict_layer.t()?)?.squeeze(0)?;

            // Sample next token (no prompt tokens for generate_with_features path)
            let (next_token, _argmax) = Self::sample_token(&logits, top_k, top_p, temperature, repetition_penalty, &[], &generated_tokens)?;

            // Check for end-of-sequence token
            let audio_vocab_size = self.ar_predict_layer.dims()[0];
            if next_token >= audio_vocab_size - 1 {
                break;
            }

            generated_tokens.push(next_token);

            // Append token to input for next iteration
            let next_tensor = Tensor::new(&[next_token as i64], &self.device)?;
            current_ids = Tensor::cat(&[current_ids, next_tensor.unsqueeze(0)?], 1)?;
        }

        Ok(generated_tokens)
    }

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
            phoneme_ids, prompt_tokens, bert_features, None, word2ph,
            top_k, top_p, temperature, repetition_penalty, max_tokens,
        )
    }

    /// Like `generate_with_prompts` but accepts pre-aligned 512-dim BERT features
    /// [1, all_phones, 512] ready to add directly to text embeddings.
    /// Use this when ref+target BERT features are pre-concatenated externally.
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
            phoneme_ids, prompt_tokens, None, pre_aligned_bert, &[],
            top_k, top_p, temperature, repetition_penalty, max_tokens,
        )
    }

    /// KV-cache version of `generate_with_prompts_aligned_bert`.
    ///
    /// Prefills the cache with the full text+prompt sequence in one forward pass,
    /// then generates each audio token with a single-token forward pass (O(1) per step
    /// instead of O(n) for the non-cached version).
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
        if phoneme_ids.is_empty() {
            return Ok(Vec::new());
        }

        let text_seq = phoneme_ids.len();
        let prompt_seq = prompt_tokens.len();
        let audio_vocab_size = self.ar_predict_layer.dims()[0];
        let max_new_tokens = max_tokens;

        // Build text embeddings + BERT fusion (same as non-cached path)
        let text_ids: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
        let text_tensor = Tensor::new(text_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let x_emb = self.lookup_tokens(&self.text_embedding, &text_tensor, text_seq)?;

        let x_emb = if let Some(aligned) = pre_aligned_bert {
            x_emb.broadcast_add(aligned)?
        } else {
            x_emb
        };
        let x_emb = self.add_sine_positional(&x_emb, "text")?;

        // Build audio prompt embeddings
        let prompt_ids: Vec<i64> = prompt_tokens.iter().map(|&x| x as i64).collect();
        let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let y_emb = self.lookup_tokens(&self.audio_embedding, &prompt_tensor, prompt_seq)?;
        let y_pos = self.add_sine_positional(&y_emb, "audio")?;

        // Prefill: run the full text+prompt sequence through the transformer, filling KV cache
        let prefill_input = Tensor::cat(&[&x_emb, &y_pos], 1)?;
        let total_prefill = text_seq + prompt_seq;
        let hybrid_mask = self.create_hybrid_mask(text_seq, total_prefill)?;

        let mut kv_cache = KvCacheManager::new(self.num_layers);
        let prefill_out = self.transformer.forward_from_embedding_kv(
            &prefill_input, Some(&hybrid_mask), &mut kv_cache
        )?;

        // Get logits from the last prefill position
        let last_hidden = prefill_out.narrow(1, total_prefill - 1, 1)?.squeeze(0)?;
        let logits = last_hidden.matmul(&self.ar_predict_layer.t()?)?.squeeze(0)?;

        let mut generated_tokens: Vec<usize> = Vec::new();

        // First token sampling from prefill logits (step 0 — always mask EOS)
        let logits_for_sampling = logits.narrow(0, 0, audio_vocab_size - 1)?;
        let (next_token, _argmax) = Self::sample_token(
            &logits_for_sampling, top_k, top_p, temperature, repetition_penalty,
            prompt_tokens, &generated_tokens,
        )?;
        if next_token >= audio_vocab_size - 1 {
            return Ok(generated_tokens);
        }
        generated_tokens.push(next_token);

        // Autoregressive generation: single token per forward pass
        for step in 1..max_new_tokens {
            let prev_token = *generated_tokens.last().unwrap();
            let new_ids = Tensor::new(&[prev_token as i64], &self.device)?.unsqueeze(0)?;
            let new_emb = self.lookup_tokens(&self.audio_embedding, &new_ids, 1)?;
            let audio_pe_pos = prompt_seq + generated_tokens.len() - 1;
            let new_pos = self.add_sine_positional_at(&new_emb, audio_pe_pos)?;

            // Single-token forward: new token attends to ALL cached K/V (no mask needed)
            let hidden = self.transformer.forward_from_embedding_kv(
                &new_pos, None, &mut kv_cache
            )?;

            let logits = hidden.squeeze(0)?.matmul(&self.ar_predict_layer.t()?)?.squeeze(0)?;

            let is_eos_masked = step < 11;
            let effective_vocab = if is_eos_masked { audio_vocab_size - 1 } else { audio_vocab_size };
            let logits_for_sampling = logits.narrow(0, 0, effective_vocab)?;

            // sample_token returns (sampled, argmax) from a single D2H transfer
            let (next_token, argmax) = Self::sample_token(
                &logits_for_sampling, top_k, top_p, temperature, repetition_penalty,
                prompt_tokens, &generated_tokens,
            )?;

            let argmax_eos = !is_eos_masked && argmax == audio_vocab_size - 1;

            if next_token >= audio_vocab_size - 1 || argmax_eos {
                break;
            }
            generated_tokens.push(next_token);
        }

        tracing::info!("[GPT kv] Total generated tokens: {}", generated_tokens.len());
        Ok(generated_tokens)
    }

    /// Like `generate_with_prompts_aligned_bert_kv_cache`, but after prefill the KV cache is
    /// converted to pre-allocated fixed-size buffers (`StaticKvManager`).
    /// This eliminates the O(n) `Tensor::cat` allocations in the dynamic KV approach, replacing
    /// them with an in-place `scatter_set` that writes only the new K/V token at each step.
    ///
    /// All attention shapes during decode are constant → compatible with CUDA graph capture.
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
        max_kv_len: usize,  // max total KV sequence length (prefill + generated)
    ) -> Result<Vec<usize>> {
        let audio_vocab_size = self.ar_predict_layer.dims()[0];
        let text_seq = phoneme_ids.len();
        let prompt_seq = prompt_tokens.len();
        let total_prefill = text_seq + prompt_seq;

        assert!(
            total_prefill + max_new_tokens <= max_kv_len,
            "max_kv_len ({max_kv_len}) must be >= total_prefill ({total_prefill}) + max_new_tokens ({max_new_tokens})"
        );

        // ── Build text embeddings ────────────────────────────────────────────────
        let phone_ids: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
        let phone_tensor = Tensor::new(phone_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let x_emb = self.lookup_tokens(&self.text_embedding, &phone_tensor, text_seq)?;

        // Inject BERT features if available
        let x_emb = if let Some(bert) = pre_aligned_bert {
            let bert_f32 = bert.to_dtype(DType::F32)?;
            let x_f32 = x_emb.to_dtype(DType::F32)?;
            // Debug: dump BERT and x_emb stats
            (x_f32 + bert_f32)?.to_dtype(self.dtype)?
        } else {
            x_emb
        };
        let x_emb = self.add_sine_positional(&x_emb, "text")?;

        // ── Build audio prompt embeddings ────────────────────────────────────────
        let prompt_ids: Vec<i64> = prompt_tokens.iter().map(|&x| x as i64).collect();
        let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let y_emb = self.lookup_tokens(&self.audio_embedding, &prompt_tensor, prompt_seq)?;
        let y_pos = self.add_sine_positional(&y_emb, "audio")?;

        // ── Prefill (dynamic KV) ─────────────────────────────────────────────────
        let prefill_input = Tensor::cat(&[&x_emb, &y_pos], 1)?;
        let hybrid_mask = self.create_hybrid_mask(text_seq, total_prefill)?;
        let mut dyn_kv = KvCacheManager::new(self.num_layers);
        let prefill_out = self.transformer.forward_from_embedding_kv(
            &prefill_input, Some(&hybrid_mask), &mut dyn_kv,
        )?;

        // ── Convert dynamic KV → static pre-allocated KV ────────────────────────
        let mut static_kv = StaticKvManager::from_dynamic(dyn_kv, max_kv_len)?;

        // ── First token from prefill logits ─────────────────────────────────────
        let last_hidden = prefill_out.narrow(1, total_prefill - 1, 1)?.squeeze(0)?;
        let logits = last_hidden.matmul(&self.ar_predict_layer.t()?)?.squeeze(0)?;
        let logits_for_sampling = logits.narrow(0, 0, audio_vocab_size - 1)?;
        let (next_token, _) = Self::sample_token(
            &logits_for_sampling, top_k, top_p, temperature, repetition_penalty,
            prompt_tokens, &[],
        )?;
        if next_token >= audio_vocab_size - 1 {
            return Ok(vec![]);
        }
        let mut generated_tokens = vec![next_token];

        // ── Autoregressive decode with static KV ─────────────────────────────────
        for step in 1..max_new_tokens {
            let prev_token = *generated_tokens.last().unwrap();
            let new_ids = Tensor::new(&[prev_token as i64], &self.device)?.unsqueeze(0)?;
            let new_emb = self.lookup_tokens(&self.audio_embedding, &new_ids, 1)?;
            let pe_pos = prompt_seq + generated_tokens.len() - 1;
            let new_pos = self.add_sine_positional_at(&new_emb, pe_pos)?;

            let hidden = self.transformer.forward_from_embedding_static(
                &new_pos, &mut static_kv,
            )?;

            let logits = hidden.squeeze(0)?.matmul(&self.ar_predict_layer.t()?)?.squeeze(0)?;

            let is_eos_masked = step < 11;
            let effective_vocab = if is_eos_masked { audio_vocab_size - 1 } else { audio_vocab_size };
            let logits_for_sampling = logits.narrow(0, 0, effective_vocab)?;

            let (next_token, argmax) = Self::sample_token(
                &logits_for_sampling, top_k, top_p, temperature, repetition_penalty,
                prompt_tokens, &generated_tokens,
            )?;

            let argmax_eos = !is_eos_masked && argmax == audio_vocab_size - 1;
            if next_token >= audio_vocab_size - 1 || argmax_eos {
                break;
            }
            generated_tokens.push(next_token);
        }

        tracing::info!("[GPT static-kv] Generated {} tokens", generated_tokens.len());
        Ok(generated_tokens)
    }

    /// CUDA graph-accelerated decode. Requires the `cuda` feature.
    ///
    /// Flow:
    ///  1. Prefill with dynamic KV — captured graphs don't help here (variable shapes).
    ///  2. Convert KV to pre-allocated static buffers (fixed [max_kv_len] shapes).
    ///  3. Pre-allocate stable input/mask buffers for the graph boundary.
    ///  4. Warmup 3 decode steps to prime CUDA's stream-ordered allocator pool.
    ///  5. Capture one full decode step into a CUDA graph.
    ///  6. For each remaining step:
    ///     a. Compute new token embedding (eager, ~3 fast kernels).
    ///     b. D2D-copy embedding into the pre-allocated input buffer.
    ///     c. H2D-copy new position value into each layer's `pos_idx` buffer.
    ///     d. H2D-copy updated mask into the mask buffer.
    ///     e. Launch graph (replays ~160 kernels as a single submission).
    ///     f. Sample from the (still-live) logits tensor written by the graph.
    ///
    /// Falls back to `generate_with_static_kv` on non-CUDA devices.
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
        #[cfg(feature = "cuda")]
        {
            use std::sync::Arc;

            let candle_core::Device::Cuda(cuda_dev) = &self.device else {
                return self.generate_with_static_kv(
                    phoneme_ids, prompt_tokens, pre_aligned_bert,
                    top_k, top_p, temperature, repetition_penalty,
                    max_new_tokens, max_kv_len,
                );
            };

            // CUDA graph capture is unsafe when BERT features are present: intermediate
            // BERT tensors freed to the default async-malloc pool before capture disturb
            // the pool's free-list, causing cuMemAllocAsync inside the graph to get
            // different virtual addresses on the second replay → CUDA_ERROR_ILLEGAL_ADDRESS.
            // Fall back to the static-kv (eager) path which still uses pre-allocated KV
            // cache for GPU efficiency without the graph launch optimisation.
            if pre_aligned_bert.is_some() {
                return self.generate_with_static_kv(
                    phoneme_ids, prompt_tokens, pre_aligned_bert,
                    top_k, top_p, temperature, repetition_penalty,
                    max_new_tokens, max_kv_len,
                );
            }

            let audio_vocab_size = self.ar_predict_layer.dims()[0];
            let text_seq = phoneme_ids.len();
            let prompt_seq = prompt_tokens.len();
            let total_prefill = text_seq + prompt_seq;
            let hidden_size = self.transformer.config.hidden_size;
            assert!(total_prefill + max_new_tokens <= max_kv_len);

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
                ).map_err(|e| candle_core::Error::Msg(format!("set pool threshold: {e:?}")))?;
            }

            // ── Prefill (same as static_kv path) ─────────────────────────────────
            let phone_ids: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
            let phone_tensor = Tensor::new(phone_ids.as_slice(), &self.device)?.unsqueeze(0)?;
            let x_emb = self.lookup_tokens(&self.text_embedding, &phone_tensor, text_seq)?;
            let x_emb = if let Some(bert) = pre_aligned_bert {
                let bert_f32 = bert.to_dtype(DType::F32)?;
                let x_f32 = x_emb.to_dtype(DType::F32)?;
                (x_f32 + bert_f32)?.to_dtype(self.dtype)?
            } else { x_emb };
            let x_emb = self.add_sine_positional(&x_emb, "text")?;

            let prompt_ids: Vec<i64> = prompt_tokens.iter().map(|&x| x as i64).collect();
            let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;
            let y_emb = self.lookup_tokens(&self.audio_embedding, &prompt_tensor, prompt_seq)?;
            let y_pos = self.add_sine_positional(&y_emb, "audio")?;

            let prefill_input = Tensor::cat(&[&x_emb, &y_pos], 1)?;
            let hybrid_mask = self.create_hybrid_mask(text_seq, total_prefill)?;
            let mut dyn_kv = KvCacheManager::new(self.num_layers);
            let prefill_out = self.transformer.forward_from_embedding_kv(
                &prefill_input, Some(&hybrid_mask), &mut dyn_kv,
            )?;
            let mut static_kv = StaticKvManager::from_dynamic(dyn_kv, max_kv_len)?;

            // ── First token ───────────────────────────────────────────────────────
            let last_hidden = prefill_out.narrow(1, total_prefill - 1, 1)?.squeeze(0)?;
            let first_logits = last_hidden.matmul(&self.ar_predict_layer.t()?)?.squeeze(0)?;
            let (next_token, _) = Self::sample_token(
                &first_logits.narrow(0, 0, audio_vocab_size - 1)?,
                top_k, top_p, temperature, repetition_penalty, prompt_tokens, &[],
            )?;
            if next_token >= audio_vocab_size - 1 { return Ok(vec![]); }
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

            // logits_out: [n_vocab] F32 — the graph scatter_sets computed logits into this stable buf.
            // Allocated PRE-capture so the address doesn't change between graph launches.
            let n_vocab = self.ar_predict_layer.dim(0)?;
            let logits_out = Tensor::zeros((n_vocab,), DType::F32, &self.device)?;
            // full_idx: [n_vocab] — used for scatter_set to overwrite all logit slots each step
            let full_idx = Tensor::arange(0i64, n_vocab as i64, &self.device)?;

            // ── Raw CUDA update helpers (NO pool allocation between graph launches) ──────
            // scatter_set uses cuMemAllocAsync for index/strides tensors. Any pool allocation
            // between graph launches disturbs the pool's free-list order. On the next launch,
            // the pool returns different addresses for the graph's own cuMemAllocAsync nodes,
            // causing stale captured kernel parameters → CUDA_ERROR_ILLEGAL_ADDRESS.
            //
            // cuMemcpyHtoDAsync / cuMemcpyDtoDAsync do NOT allocate from the memory pool,
            // so they leave the pool state undisturbed between launches.
            use candle_core::cuda_backend::{
                CudaStorageSlice,
                cudarc::driver::{DevicePtr, result as cudarc_result},
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
                            CudaStorageSlice::F32(s)  => { let (p, _g) = s.device_ptr(&*cstream); p }
                            CudaStorageSlice::F16(s)  => { let (p, _g) = s.device_ptr(&*cstream); p }
                            CudaStorageSlice::BF16(s) => { let (p, _g) = s.device_ptr(&*cstream); p }
                            CudaStorageSlice::I64(s)  => { let (p, _g) = s.device_ptr(&*cstream); p }
                            _ => return Err(candle_core::Error::Msg("unsupported dtype in get_cuda_ptr".into())),
                        };
                        Ok(base + (start * esize) as u64)
                    }
                    _ => Err(candle_core::Error::Msg("tensor is not on CUDA device".into())),
                }
            };

            let raw_stream = stream.cu_stream();

            // Precompute audio embedding weights on CPU for pool-free input updates.
            // D2H copy is done once here; each decode step uses a CPU-side emb+PE computation.
            let audio_emb_cpu: Vec<f32> = self.audio_embedding
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let half_dim = hidden_size / 2;
            // div_term[i] = (1/10000)^(2i/hidden_size)  — same as add_sine_positional_with_alpha
            let div_term: Vec<f64> = (0..half_dim)
                .map(|i| (-(i as f64 * 2.0) * (10000.0f64.ln()) / (hidden_size as f64)).exp())
                .collect();
            let audio_alpha = self.audio_pos_alpha;

            // Cache the raw pointer for input_buf (stable for entire function lifetime)
            let input_buf_ptr = get_cuda_ptr(&input_buf)?;
            // Cache the raw pointer for attn_mask_buf
            let mask_buf_ptr = get_cuda_ptr(&attn_mask_buf)?;
            let mask_esize = attn_mask_buf.dtype().size_in_bytes() as u64;

            // Update input_buf: compute emb + alpha*PE on CPU, H2D copy — zero pool allocations.
            let update_input_raw = |token: usize, pe_pos: usize| -> candle_core::Result<()> {
                let off = token * hidden_size;
                let mut combined = audio_emb_cpu[off..off + hidden_size].to_vec();
                let pos = pe_pos as f64;
                for i in 0..half_dim {
                    let v = pos * div_term[i];
                    combined[2 * i]     += audio_alpha * v.sin() as f32;
                    combined[2 * i + 1] += audio_alpha * v.cos() as f32;
                }
                if input_buf.dtype() != DType::F32 {
                    return Err(candle_core::Error::Msg(
                        "CUDA graph raw update only supports F32 models".into()
                    ));
                }
                unsafe {
                    cudarc_result::memcpy_htod_async(input_buf_ptr, combined.as_slice(), raw_stream)
                        .map_err(|e| candle_core::Error::Msg(format!("htod input_buf: {e:?}")))?;
                }
                Ok(())
            };

            // Update pos_idx: H2D copy of [new_pos; n_elements] — zero pool allocations.
            let update_pos_idx_raw = |pos_idx: &Tensor, new_pos: usize| -> candle_core::Result<()> {
                let n = pos_idx.elem_count();
                let vals: Vec<i64> = vec![new_pos as i64; n];
                let ptr = get_cuda_ptr(pos_idx)?;
                unsafe {
                    cudarc_result::memcpy_htod_async(ptr, vals.as_slice(), raw_stream)
                        .map_err(|e| candle_core::Error::Msg(format!("htod pos_idx: {e:?}")))?;
                }
                Ok(())
            };

            // Update mask: H2D write zero-bytes at strided positions for `cache_len` in all heads.
            // mask_buf is [1, n_heads, 1, max_kv_len] C-contiguous.
            // Element [0, h, 0, p] is at flat index h*max_kv_len + p.
            let update_mask_raw = |cache_len: usize| -> candle_core::Result<()> {
                // Zeroing a float (F32/F16/BF16) is always [0u8; esize]
                let zero = [0u8; 4];
                let nbytes = mask_esize as usize;
                for h in 0..n_heads {
                    let byte_off = (h * max_kv_len + cache_len) as u64 * mask_esize;
                    unsafe {
                        cudarc_result::memcpy_htod_async(
                            mask_buf_ptr + byte_off,
                            &zero[..nbytes],
                            raw_stream,
                        ).map_err(|e| candle_core::Error::Msg(format!("htod mask h={h}: {e:?}")))?;
                    }
                }
                Ok(())
            };


            // ── Eager decode step for warmup (uses static KV's append(), not graphable) ──
            let eager_step = |gen: &Vec<usize>, static_kv: &mut StaticKvManager| -> Result<Tensor> {
                let prev = *gen.last().unwrap();
                let ids = Tensor::new(&[prev as i64], &self.device)?.unsqueeze(0)?;
                let emb = self.lookup_tokens(&self.audio_embedding, &ids, 1)?;
                let pe_pos = prompt_seq + gen.len() - 1;
                let pos_emb = self.add_sine_positional_at(&emb, pe_pos)?;
                let hidden = self.transformer.forward_from_embedding_static(&pos_emb, static_kv)?;
                Ok(hidden.squeeze(0)?.matmul(&self.ar_predict_layer.t()?)?.squeeze(0)?)
            };

            // ── Warmup: 3 eager steps to prime the allocator pool ─────────────────
            // Also unmask each new KV position in attn_mask_buf: warmup steps write to
            // positions total_prefill..total_prefill+WARMUP in the KV cache, and the
            // graph's attention must see those positions as valid (0.0) on replay.
            const WARMUP: usize = 3;
            for step in 1..=WARMUP {
                if generated.len() >= max_new_tokens { break; }
                let pos_before = static_kv.len(); // position this step will write to
                let logits = eager_step(&generated, &mut static_kv)?;
                // Unmask the position that eager_step just appended (uses H2D async on same stream)
                update_mask_raw(pos_before)?;
                let is_eos = step < 11;
                let ev = if is_eos { audio_vocab_size - 1 } else { audio_vocab_size };
                let (tok, ag) = Self::sample_token(&logits.narrow(0, 0, ev)?, top_k, top_p, temperature, repetition_penalty, prompt_tokens, &generated)?;
                if tok >= audio_vocab_size - 1 || (!is_eos && ag == audio_vocab_size - 1) { break; }
                generated.push(tok);
            }
            if generated.len() >= max_new_tokens { return Ok(generated); }

            self.device.synchronize()?;

            // ── Prepare pre-allocated buffers for the capture step ────────────────
            {
                let prev = *generated.last().unwrap();
                let pe_pos = prompt_seq + generated.len() - 1;
                update_input_raw(prev, pe_pos)?;
            }
            let cache_len_before_capture = static_kv.len();
            for layer in &static_kv.layers {
                update_pos_idx_raw(&layer.pos_idx, cache_len_before_capture)?;
            }
            update_mask_raw(cache_len_before_capture)?;
            // Flush all H2D copies before capture begins
            stream.synchronize()
                .map_err(|e| candle_core::Error::Msg(format!("sync before capture: {e:?}")))?;

            // Trim the memory pool: release all free-listed blocks (e.g. from BERT prefill
            // intermediate tensors) back to the OS before capture.  After trim the pool's
            // free-list is empty, so every cuMemAllocAsync during graph capture is the
            // first (and only) tenant of its address — guaranteeing the same pointer is
            // returned on every subsequent replay.
            unsafe {
                use candle_core::cuda_backend::cudarc::driver;
                let cu_device = driver::result::device::get(0)
                    .map_err(|e| candle_core::Error::Msg(format!("cu_device get (trim): {e:?}")))?;
                let pool = driver::result::device::get_default_mem_pool(cu_device)
                    .map_err(|e| candle_core::Error::Msg(format!("get_default_mem_pool (trim): {e:?}")))?;
                driver::result::mem_pool::trim_to(pool, 0)
                    .map_err(|e| candle_core::Error::Msg(format!("cuMemPoolTrimTo: {e:?}")))?;
                // Re-set threshold to never-release AFTER trim so graph-managed allocs
                // are pinned from this point forward.
                let threshold: u64 = u64::MAX;
                driver::result::mem_pool::set_attribute(
                    pool,
                    driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const u64 as *mut std::ffi::c_void,
                ).map_err(|e| candle_core::Error::Msg(format!("set pool threshold (post-trim): {e:?}")))?;
            }

            // ── CUDA graph capture ────────────────────────────────────────────────
            use candle_core::cuda_backend::cudarc::driver::sys::{
                CUstreamCaptureMode, CUgraphInstantiate_flags,
            };
            stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| candle_core::Error::Msg(format!("begin_capture: {e:?}")))?;

            // The graph captures: scatter-based decode + logits projection +
            // scatter_set into logits_out (stable pre-allocated buffer).
            // cuMemAllocAsync allocations inside capture hold placeholder pointers that only
            // become valid after graph.launch(). logits_tmp MUST be dropped before end_capture
            // so its free_async is also captured (graph owns the full alloc/free lifecycle).
            {
                let logits_tmp = self.transformer
                    .forward_from_embedding_graphable(&input_buf, &static_kv, &attn_mask_buf)?
                    .squeeze(0)?
                    .matmul(&self.ar_predict_layer.t()?)?
                    .squeeze(0)?;
                // Scatter all logits into stable pre-allocated buffer — captured in graph
                logits_out.scatter_set(&full_idx, &logits_tmp, 0)?;
                // logits_tmp drops here (still inside capture) → free_async captured ✓
            }

            let graph_opt = stream
                .end_capture(unsafe { std::mem::transmute::<u32, CUgraphInstantiate_flags>(0u32) })
                .map_err(|e| candle_core::Error::Msg(format!("end_capture: {e:?}")))?;

            let Some(graph) = graph_opt else {
                tracing::warn!("[GPT cuda-graph] empty graph; falling back to static-kv");
                static_kv.step();  // account for the decode_step we ran above
                for step in generated.len()..max_new_tokens {
                    let logits = eager_step(&generated, &mut static_kv)?;
                    let is_eos = step < 11;
                    let ev = if is_eos { audio_vocab_size - 1 } else { audio_vocab_size };
                    let (tok, ag) = Self::sample_token(&logits.narrow(0, 0, ev)?, top_k, top_p, temperature, repetition_penalty, prompt_tokens, &generated)?;
                    if tok >= audio_vocab_size - 1 || (!is_eos && ag == audio_vocab_size - 1) { break; }
                    generated.push(tok);
                }
                return Ok(generated);
            };

            // Graph capture only RECORDS operations — they were NOT executed.
            // Launch now to execute the captured decode step and populate logits_out.
            graph.launch()
                .map_err(|e| candle_core::Error::Msg(format!("graph launch (first): {e:?}")))?;
            stream.synchronize()
                .map_err(|e| candle_core::Error::Msg(format!("stream sync (first): {e:?}")))?;
            static_kv.step();
            {
                let step = generated.len();
                let is_eos = step < 11;
                let ev = if is_eos { audio_vocab_size - 1 } else { audio_vocab_size };
                let (tok, ag) = Self::sample_token(
                    &logits_out.narrow(0, 0, ev)?,
                    top_k, top_p, temperature, repetition_penalty, prompt_tokens, &generated,
                )?;
                if tok >= audio_vocab_size - 1 || (!is_eos && ag == audio_vocab_size - 1) {
                    tracing::info!("[GPT cuda-graph] EOS at capture step");
                    return Ok(generated);
                }
                generated.push(tok);
            }

            tracing::info!("[GPT cuda-graph] Graph captured ({} warmup + 1 capture steps); replaying", WARMUP);

            // ── Graph replay loop ─────────────────────────────────────────────────
            for step in generated.len()..max_new_tokens {
                // 1. Compute emb+PE on CPU, H2D copy to input_buf — zero pool allocations
                let prev = *generated.last().unwrap();
                let pe_pos = prompt_seq + generated.len() - 1;
                update_input_raw(prev, pe_pos)?;

                // 2. Update pos_idx (H2D copy, zero pool allocations)
                let cur_len = static_kv.len();
                tracing::debug!("[GPT cuda-graph] step {step}: cur_len={cur_len} max_kv_len={max_kv_len}");
                if cur_len >= max_kv_len {
                    tracing::warn!("[GPT cuda-graph] KV cache overflow: cur_len={cur_len} >= max_kv_len={max_kv_len}, stopping");
                    break;
                }
                for layer in &static_kv.layers {
                    update_pos_idx_raw(&layer.pos_idx, cur_len)?;
                }

                // 3. Update mask: unmask the new position (H2D copy, zero pool allocations)
                update_mask_raw(cur_len)?;

                // 5. Launch graph (replays all 16 layers of attention + FFN + logits proj)
                // No explicit sync needed: H2D copies and graph launch are on the same stream,
                // so CUDA guarantees the copies complete before the graph kernel reads.
                graph.launch()
                    .map_err(|e| candle_core::Error::Msg(format!("graph launch step {step}: {e:?}")))?;
                // Sync stream before D2H copy so logits_out is fully written.
                stream.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("stream sync step {step}: {e:?}")))?;

                static_kv.step();

                // 6. Sample (D2H; logits_out is stable pre-allocated buffer written by graph)
                let is_eos = step < 11;
                let ev = if is_eos { audio_vocab_size - 1 } else { audio_vocab_size };
                let (tok, ag) = Self::sample_token(
                    &logits_out.narrow(0, 0, ev)?,
                    top_k, top_p, temperature, repetition_penalty, prompt_tokens, &generated,
                )?;
                if tok >= audio_vocab_size - 1 || (!is_eos && ag == audio_vocab_size - 1) { break; }
                generated.push(tok);
            }

            tracing::info!("[GPT cuda-graph] Generated {} tokens total", generated.len());
            Ok(generated)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.generate_with_static_kv(
                phoneme_ids, prompt_tokens, pre_aligned_bert,
                top_k, top_p, temperature, repetition_penalty,
                max_new_tokens, max_kv_len,
            )
        }
    }

    fn generate_with_prompts_inner(
        &self,
        phoneme_ids: &[usize],
        prompt_tokens: &[usize],
        bert_features: Option<&Tensor>,
        pre_aligned_bert: Option<&Tensor>,  // [1, all_phones, 512] — bypasses projection+alignment
        word2ph: &[usize],
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_tokens: usize,
    ) -> Result<Vec<usize>> {
        if phoneme_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Convert phoneme IDs to tensor [1, text_seq]
        let text_ids: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
        let text_tensor = Tensor::new(text_ids.as_slice(), &self.device)?.unsqueeze(0)?;

        // Convert prompt tokens to tensor [1, prompt_seq]
        let prompt_ids: Vec<i64> = prompt_tokens.iter().map(|&x| x as i64).collect();
        let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;

        let text_seq = phoneme_ids.len();
        let prompt_seq = prompt_tokens.len();

        // Step 1: Embed text tokens [1, text_seq, 512]
        let x_emb = self.lookup_tokens(&self.text_embedding, &text_tensor, text_seq)?;

        // Step 2: Add BERT features (either pre-aligned 512-dim, or raw 1024-dim with projection)
        let x_emb = if let Some(aligned) = pre_aligned_bert {
            // Pre-aligned 512-dim features: add directly without any processing
            x_emb.broadcast_add(aligned)?
        } else if let (Some(bert), Some((proj_w, proj_b))) = (bert_features, &self.bert_proj) {
            let bert_dims = bert.dims();
            let bert_reshaped = if bert_dims.len() == 3 && bert_dims[1] == 1024 {
                bert.transpose(1, 2)?  // [batch, 1024, seq] -> [batch, seq, 1024]
            } else {
                bert.clone()
            };
            let last_dim = bert_reshaped.dims().last().copied();
            if last_dim == Some(1024) {
                let proj_w_3d = proj_w.t()?.unsqueeze(0)?;
                let projected = bert_reshaped.matmul(&proj_w_3d)?;
                let projected = projected.broadcast_add(&proj_b.reshape((1, 1, proj_b.dims()[0]))?)?;
                let target_seq = x_emb.dims()[1];
                let aligned = self.align_bert_to_phonemes(&projected, target_seq, word2ph)?;
                x_emb.broadcast_add(&aligned)?
            } else {
                x_emb
            }
        } else {
            x_emb
        };

        // Step 3: Add text positional encoding
        let x_emb = self.add_sine_positional(&x_emb, "text")?;

        // Step 4: Embed prompt audio tokens [1, prompt_seq, 512]
        let y_emb = self.lookup_tokens(&self.audio_embedding, &prompt_tensor, prompt_seq)?;

        // Step 5: Add audio positional encoding
        let y_pos = self.add_sine_positional(&y_emb, "audio")?;

        // Step 6: Concatenate text + audio along sequence dimension
        let mut xy_pos = Tensor::cat(&[&x_emb, &y_pos], 1)?;
        let mut generated_tokens = Vec::new();
        let max_new_tokens = max_tokens;
        let audio_vocab_size = self.ar_predict_layer.dims()[0];

        for step in 0..max_new_tokens {
            let total_seq = xy_pos.dims()[1];

            // Use hybrid mask: text bidirectional + text blocked from audio + audio causal
            let mask = self.create_hybrid_mask(text_seq, total_seq)?;
            let hidden = self.transformer.forward_all_layers_with_mask(&xy_pos, &mask)?;

            // Project last position to vocab
            let last_hidden = hidden.narrow(1, total_seq - 1, 1)?.squeeze(0)?;
            let logits = last_hidden.matmul(&self.ar_predict_layer.t()?)?.squeeze(0)?;

            // For first 11 steps, mask out EOS logit
            let is_eos_masked = step < 11;
            let effective_vocab_size = if is_eos_masked { audio_vocab_size - 1 } else { audio_vocab_size };
            let logits_for_sampling = if is_eos_masked {
                logits.narrow(0, 0, effective_vocab_size)?
            } else {
                logits.clone()
            };

            // sample_token returns (sampled, argmax) from a single D2H transfer
            let (next_token, argmax) = Self::sample_token(&logits_for_sampling, top_k, top_p, temperature, repetition_penalty, prompt_tokens, &generated_tokens)?;

            // Python stop condition: argmax(logits_for_sampling) == EOS OR sampled == EOS
            let argmax_eos = !is_eos_masked && argmax == audio_vocab_size - 1;
            if next_token >= audio_vocab_size - 1 || argmax_eos {
                break;
            }

            generated_tokens.push(next_token);

            let new_ids = Tensor::new(&[next_token as i64], &self.device)?.unsqueeze(0)?;
            let new_emb = self.lookup_tokens(&self.audio_embedding, &new_ids, 1)?;
            // Audio PE position = prompt_seq + generated_so_far (not text_seq + total_audio)
            let audio_pe_pos = prompt_seq + generated_tokens.len() - 1;
            let new_pos = self.add_sine_positional_at(&new_emb, audio_pe_pos)?;
            xy_pos = Tensor::cat(&[&xy_pos, &new_pos], 1)?;
        }

        tracing::info!("[GPT] Total generated tokens: {}", generated_tokens.len());
        Ok(generated_tokens)
    }

    /// Project and align raw BERT features to phone level.
    ///
    /// Takes raw ONNX BERT output [1, seq+2, 1024] (includes CLS/SEP), applies bert_proj,
    /// strips CLS/SEP, and expands to phone level via word2ph.
    /// Returns [1, n_phones, 512].
    ///
    /// Used by the inference pipeline to pre-align ref and target BERT features separately
    /// before concatenating them for joint GPT conditioning.
    pub fn project_and_align_bert(
        &self,
        bert_raw: &Tensor,  // [1, seq_with_cls_sep, 1024]
        word2ph: &[usize],
        n_phones: usize,
    ) -> Result<Tensor> {
        let (proj_w, proj_b) = self.bert_proj.as_ref()
            .ok_or_else(|| Error::ModelLoadError("bert_proj not loaded".to_string()))?;

        let bert = if bert_raw.dims().len() == 3 && bert_raw.dims()[1] == 1024 {
            bert_raw.transpose(1, 2)?
        } else {
            bert_raw.clone()
        };
        let proj_w_3d = proj_w.t()?.unsqueeze(0)?;
        let projected = bert.matmul(&proj_w_3d)?;
        let projected = projected.broadcast_add(&proj_b.reshape((1, 1, proj_b.dims()[0]))?)?;
        self.align_bert_to_phonemes(&projected, n_phones, word2ph)
    }

    /// Align BERT features (from ONNX model including CLS/SEP) to phoneme sequence.
    ///
    /// Python's pipeline:
    ///   1. Extract BERT hidden states (char-level, includes CLS/SEP tokens)
    ///   2. Remove CLS (first) and SEP (last) tokens → bert_content[i] per char
    ///   3. Expand: repeat bert_content[i] word2ph[i] times (skip if 0 for punctuation)
    ///
    /// When word2ph is provided: uses exact word2ph expansion (matches Python exactly).
    /// When word2ph is empty: falls back to nearest-neighbor interpolation.
    ///
    /// # Arguments
    /// * `projected` - BERT projected features [1, bert_seq, hidden] (including CLS/SEP)
    /// * `target_seq` - Target phoneme sequence length
    /// * `word2ph` - Per-character phoneme counts from G2P (empty → use nearest-neighbor)
    fn align_bert_to_phonemes(&self, projected: &Tensor, target_seq: usize, word2ph: &[usize]) -> Result<Tensor> {
        let bert_full_len = projected.dims()[1];

        // Strip CLS (index 0) and SEP (last index)
        let bert_content = if bert_full_len >= 2 {
            projected.narrow(1, 1, bert_full_len - 2)?
        } else {
            projected.clone()
        };
        let bert_len = bert_content.dims()[1];

        if bert_len == 0 {
            return Ok(projected.clone());
        }

        // Use word2ph expansion if provided and consistent with bert_content
        if !word2ph.is_empty() && word2ph.len() == bert_len {
            let total_phonemes: usize = word2ph.iter().sum();
            if total_phonemes == target_seq {
                let mut frames = Vec::with_capacity(target_seq);
                for (i, &count) in word2ph.iter().enumerate() {
                    for _ in 0..count {
                        frames.push(bert_content.narrow(1, i, 1)?);
                    }
                }
                return Tensor::cat(&frames, 1).map_err(|e| e.into());
            } else {
                tracing::debug!("word2ph sum={} != target_seq={}, falling back to nearest-neighbor",
                    total_phonemes, target_seq);
            }
        }

        // Fallback: nearest-neighbor interpolation from bert_len to target_seq
        if bert_len == target_seq {
            Ok(bert_content)
        } else if bert_len > target_seq {
            bert_content.narrow(1, 0, target_seq).map_err(|e| e.into())
        } else {
            let mut frames = Vec::with_capacity(target_seq);
            for i in 0..target_seq {
                let bert_idx = (i * bert_len / target_seq).min(bert_len - 1);
                frames.push(bert_content.narrow(1, bert_idx, 1)?);
            }
            Tensor::cat(&frames, 1).map_err(|e| e.into())
        }
    }

    /// Create hybrid attention mask matching Python's infer_panel:
    /// Hybrid attention mask matching Python's infer_panel:
    /// - Text→Text: bidirectional (no masking)
    /// - Text→Audio: BLOCKED (True in Python x_attn_mask_pad)
    /// - Audio→Text: allowed (no masking)
    /// - Audio→Audio: causal (block future j > i)
    fn create_hybrid_mask(&self, text_seq: usize, total_seq: usize) -> Result<Tensor> {
        let mut mask = vec![0.0f32; total_seq * total_seq]; // 0 = allow, 1 = block

        for i in 0..text_seq {
            // Text positions: block attending to audio positions
            for j in text_seq..total_seq {
                mask[i * total_seq + j] = 1.0;
            }
        }
        for i in text_seq..total_seq {
            // Audio positions: block future audio
            for j in (i + 1)..total_seq {
                mask[i * total_seq + j] = 1.0;
            }
        }

        Ok(Tensor::from_vec(mask, (total_seq, total_seq), &self.device)?)
    }

    /// Lookup token embeddings for a tensor of token IDs.
    /// Stays entirely on GPU via Tensor::embedding (no D2H transfer).
    /// ids: [batch, seq] (i64) → output: [batch, seq, hidden]
    fn lookup_tokens(&self, embedding: &Tensor, ids: &Tensor, _seq: usize) -> Result<Tensor> {
        let orig_dims = ids.dims().to_vec();
        let hidden = embedding.dims()[1];
        // index_select (used internally by Tensor::embedding) requires 1D ids
        let ids_flat = ids.flatten_all()
            .map_err(|e| crate::Error::InferenceError(e.to_string()))?
            .to_dtype(candle_core::DType::U32)
            .map_err(|e| crate::Error::InferenceError(e.to_string()))?;
        let flat_emb = embedding.embedding(&ids_flat)
            .map_err(|e| crate::Error::InferenceError(e.to_string()))?;
        // Restore original batch/seq dims: [batch*seq, hidden] → [batch, seq, hidden]
        let mut out_dims = orig_dims;
        out_dims.push(hidden);
        flat_emb.reshape(out_dims)
            .map_err(|e| crate::Error::InferenceError(e.to_string()))
    }

    /// Add sinusoidal positional encoding with learned alpha scaling
    fn add_sine_positional(&self, x: &Tensor, kind: &str) -> Result<Tensor> {
        self.add_sine_positional_with_alpha(x, 0, kind)
    }

    /// Add sinusoidal positional encoding starting at a specific position offset
    fn add_sine_positional_at(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        // New tokens during generation are audio tokens
        self.add_sine_positional_with_alpha(x, start_pos, "audio")
    }

    /// Add sinusoidal positional encoding with explicit alpha
    fn add_sine_positional_with_alpha(&self, x: &Tensor, start_pos: usize, kind: &str) -> Result<Tensor> {
        let dims = x.dims();
        let seq = dims[1];
        let hidden = dims[2];

        let alpha = match kind {
            "text" => self.text_pos_alpha,
            _ => self.audio_pos_alpha,
        };

        // Generate sinusoidal positional encoding
        let half_dim = hidden / 2;
        let div_term: Vec<f64> = (0..half_dim)
            .map(|i| (-((i as f64) * 2.0) * (10000.0f64.ln()) / (hidden as f64)).exp())
            .collect();

        let mut pe = vec![0.0f32; seq * hidden];
        for t in 0..seq {
            let pos = (start_pos + t) as f64;
            for i in 0..half_dim {
                let val = pos * div_term[i];
                pe[t * hidden + 2 * i] = val.sin() as f32;
                pe[t * hidden + 2 * i + 1] = val.cos() as f32;
            }
        }

        // output = x + alpha * pe  (cast pe to match x dtype for FP16 compatibility)
        let pe_tensor = Tensor::from_vec(pe, (1, seq, hidden), &self.device)?.to_dtype(x.dtype())?;
        let scaled_pe = pe_tensor.broadcast_mul(&Tensor::full(alpha, x.dims(), &self.device)?.to_dtype(x.dtype())?)?;
        Ok(x.broadcast_add(&scaled_pe)?)
    }

    // ========================
    // Public helper methods for debugging
    // ========================

    /// Lookup text token embeddings for a tensor of token IDs.
    /// Public wrapper around the private lookup_tokens for text embeddings.
    pub fn lookup_text_tokens(&self, ids: &Tensor, seq: usize) -> Result<Tensor> {
        self.lookup_tokens(&self.text_embedding, ids, seq)
    }

    /// Lookup audio token embeddings for a tensor of token IDs.
    /// Public wrapper around the private lookup_tokens for audio embeddings.
    pub fn lookup_audio_tokens(&self, ids: &Tensor, seq: usize) -> Result<Tensor> {
        self.lookup_tokens(&self.audio_embedding, ids, seq)
    }

    /// Add sinusoidal positional encoding with learned alpha scaling.
    /// Public wrapper around the private add_sine_positional.
    pub fn add_sine_positional_pub(&self, x: &Tensor, kind: &str) -> Result<Tensor> {
        self.add_sine_positional(x, kind)
    }

    /// Get a reference to the BERT projection (weight, bias) if available.
    pub fn bert_proj_ref(&self) -> Option<&(Tensor, Tensor)> {
        self.bert_proj.as_ref()
    }

    /// Compute the output of the first transformer layer for debugging.
    ///
    /// This method reproduces the exact same computation as the beginning of
    /// `generate_with_prompts` up through the first transformer layer, allowing
    /// comparison with Python reference implementations.
    ///
    /// # Arguments
    /// * `phoneme_ids` - Input phoneme sequence
    /// * `bert_features` - Optional BERT features [batch, 1024, seq_len] or [batch, seq_len, 1024]
    /// * `prompt_tokens` - Prompt audio token IDs (from reference audio)
    ///
    /// # Returns
    /// Flattened hidden state after the first transformer layer as Vec<f32>
    pub fn debug_layer0_output(
        &self,
        phoneme_ids: &[usize],
        bert_features: Option<&Tensor>,
        prompt_tokens: &[usize],
    ) -> Result<Vec<f32>> {
        if phoneme_ids.is_empty() {
            return Err(crate::Error::InferenceError(
                "phoneme_ids cannot be empty".to_string(),
            ));
        }

        // Convert phoneme IDs to tensor [1, text_seq]
        let text_ids: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
        let text_tensor = Tensor::new(text_ids.as_slice(), &self.device)?.unsqueeze(0)?;

        // Convert prompt tokens to tensor [1, prompt_seq]
        let prompt_ids: Vec<i64> = prompt_tokens.iter().map(|&x| x as i64).collect();
        let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;

        let text_seq = phoneme_ids.len();
        let prompt_seq = prompt_tokens.len();

        // Step 1: Embed text tokens [1, text_seq, 512]
        let x_emb = self.lookup_tokens(&self.text_embedding, &text_tensor, text_seq)?;

        // Step 2: Project and add BERT features
        let x_emb = if let (Some(bert), Some((proj_w, proj_b))) = (bert_features, &self.bert_proj) {
            let bert_dims = bert.dims();
            let bert_reshaped = if bert_dims.len() == 3 && bert_dims[1] == 1024 {
                bert.transpose(1, 2)?  // [batch, 1024, seq] -> [batch, seq, 1024]
            } else {
                bert.clone()
            };
            if bert_reshaped.dims().last().copied() == Some(1024) {
                let proj_w_3d = proj_w.t()?.unsqueeze(0)?;
                let projected = bert_reshaped.matmul(&proj_w_3d)?;
                let projected = projected.broadcast_add(&proj_b.reshape((1, 1, proj_b.dims()[0]))?)?;
                // Align BERT features: strip CLS/SEP, expand via word2ph to phoneme seq length
                let text_seq = x_emb.dims()[1];
                let aligned = self.align_bert_to_phonemes(&projected, text_seq, &[])?;
                // Python: x = x + self.bert_proj(...) — no scaling
                x_emb.broadcast_add(&aligned)?
            } else {
                x_emb
            }
        } else {
            x_emb
        };

        // Step 3: Add text positional encoding
        let x_emb = self.add_sine_positional(&x_emb, "text")?;

        // Step 4: Embed prompt audio tokens [1, prompt_seq, 512]
        let y_emb = self.lookup_tokens(&self.audio_embedding, &prompt_tensor, prompt_seq)?;

        // Step 5: Add audio positional encoding
        let y_pos = self.add_sine_positional(&y_emb, "audio")?;

        // Step 6: Concatenate text + audio along sequence dimension
        let xy_pos = Tensor::cat(&[&x_emb, &y_pos], 1)?;

        // Step 7: Create hybrid attention mask
        let total_seq = text_seq + prompt_seq;
        let mask = self.create_hybrid_mask(text_seq, total_seq)?;

        // Step 8: Run through the first transformer layer only
        let hidden = self.transformer.forward_first_layer(&xy_pos, &mask)?;

        // Flatten and return as Vec<f32>
        let flat = hidden.flatten_all()?;
        flat.to_vec1().map_err(|e| crate::Error::InferenceError(e.to_string()))
    }

    /// Get a reference to the ar_predict_layer for debugging
    pub fn ar_predict_layer_ref(&self) -> Result<&Tensor> {
        Ok(&self.ar_predict_layer)
    }

    /// Run the full transformer (all 24 layers) on the input embeddings
    /// and return the hidden state. This is used for debugging layer-by-layer
    /// numerical accuracy. Uses causal mask internally.
    pub fn run_full_transformer(&self, xy_pos: &Tensor) -> Result<Tensor> {
        self.transformer.forward_all_layers(xy_pos)
    }

    /// Run all transformer layers with a provided attention mask.
    pub fn run_transformer_with_mask(&self, xy_pos: &Tensor, mask: &Tensor) -> Result<Tensor> {
        self.transformer.forward_all_layers_with_mask(xy_pos, mask)
    }

    /// Run first transformer layer and return debug intermediates:
    /// (attn_out, norm1_out, linear1_out, relu_out, final_out)
    pub fn debug_layer0_intermediates(&self, xy_pos: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        self.transformer.forward_first_layer_debug(xy_pos, mask)
    }

    /// Get model device
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

impl crate::models::Model for GPTModel {
    fn load(path: &str) -> Result<Self> {
        Self::load(path)
    }

    fn device(&self) -> &str {
        match self.device {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Metal(_) => "mps",
        }
    }

    fn to_device(&mut self, device: &str) -> Result<()> {
        let new_device = match device {
            "cuda" => Device::new_cuda_with_stream(0),
            "mps" => Device::new_metal(0),
            _ => Ok(Device::Cpu),
        }
        .map_err(|e| Error::ModelLoadError(e.to_string()))?;

        self.device = new_device;
        Ok(())
    }
}

