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
use crate::utils::{StateDict, load_safetensors, KvCacheManager};
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
        Self::load_with_device(path, &Device::Cpu)
    }

    /// Load model with specific device
    pub fn load_with_device(path: &str, device: &Device) -> Result<Self> {
        // Load weights from safetensors
        let weights_map = load_safetensors(path)?;
        let state_dict = StateDict::new(weights_map);

        // Load text embedding: [vocab_size, hidden_size]
        let text_emb_key = "model.ar_text_embedding.word_embeddings.weight";
        let text_embedding = state_dict.get(text_emb_key)?
            .to_device(device)?
            .to_dtype(DType::F32)?;

        let vocab_size = text_embedding.dims()[0];
        let hidden_size = text_embedding.dims()[1];

        // Load audio embedding: [1025, hidden_size]
        let audio_emb_key = "model.ar_audio_embedding.word_embeddings.weight";
        let audio_embedding = state_dict.get(audio_emb_key)?
            .to_device(device)?
            .to_dtype(DType::F32)?;

        // Load BERT projection (optional): weight [512, 1024], bias [512]
        let bert_proj = if state_dict.contains("model.bert_proj.weight") {
            let bert_weight = state_dict.get("model.bert_proj.weight")?.to_device(device)?.to_dtype(DType::F32)?;
            let bert_bias = state_dict.get("model.bert_proj.bias")?.to_device(device)?.to_dtype(DType::F32)?;
            Some((bert_weight, bert_bias))
        } else {
            None
        };

        // Load Hubert projection (optional): weight [512, 768], bias [512]
        let hubert_proj = if state_dict.contains("model.hubert_proj.weight") {
            let hubert_weight = state_dict.get("model.hubert_proj.weight")?.to_device(device)?.to_dtype(DType::F32)?;
            let hubert_bias = state_dict.get("model.hubert_proj.bias")?.to_device(device)?.to_dtype(DType::F32)?;
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
                DType::F32,
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
        let transformer = TransformerGPTSoVITS::new(config, &state_dict, device)?;

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
            .to_dtype(DType::F32)?;

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
            dtype: DType::F32,
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

        // output = x + alpha * pe
        let pe_tensor = Tensor::from_vec(pe, (1, seq, hidden), &self.device)?;
        let scaled_pe = pe_tensor.broadcast_mul(&Tensor::full(alpha, x.dims(), &self.device)?)?;
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
            "cuda" => Device::new_cuda(0),
            "mps" => Device::new_metal(0),
            _ => Ok(Device::Cpu),
        }
        .map_err(|e| Error::ModelLoadError(e.to_string()))?;

        self.device = new_device;
        Ok(())
    }
}
