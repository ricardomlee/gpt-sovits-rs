//! GPT Model for semantic token prediction
//!
//! This module implements the GPT model from GPT-SoVITS, which uses:
//! - Fused QKV projection (in_proj_weight combines Q, K, V weights)
//! - RoPE (Rotary Position Embedding) instead of learned positions
//! - Separate text and audio embeddings
//! - BERT feature projection
//! - Hubert feature projection for prosody guidance
//! - MRTE (Multi-Reference Timbre Encoder) for advanced fusion

use candle_core::{Device, DType, Tensor, D};
use candle_nn::ops::softmax;
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
                eprintln!("WARN: audio_idx {} out of range for audio_emb {:?}", audio_idx, audio_emb.dims());
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

        // Infer number of attention heads (GPT-SoVITS uses 8 heads for 512 hidden)
        let num_attention_heads = 8;

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
    fn sample_token(
        logits: &Tensor,
        top_k: usize,
        top_p: f32,
        temperature: f32,
    ) -> Result<usize> {
        let mut logits = logits.to_dtype(DType::F32)?;

        // Apply temperature
        if temperature != 1.0 && temperature > 0.0 {
            let t = Tensor::full(temperature, logits.dims(), &logits.device())?;
            logits = logits.broadcast_div(&t)?;
        }

        // Get sorted indices and values for top-p filtering
        let probs = softmax(&logits, D::Minus1)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // Create (prob, index) pairs and sort by probability descending
        let mut indexed_probs: Vec<(f32, usize)> = probs_vec.iter()
            .copied()
            .enumerate()
            .map(|(i, p)| (p, i))
            .collect();
        indexed_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-k
        if top_k < indexed_probs.len() {
            indexed_probs.truncate(top_k);
        }

        // Apply top-p (nucleus) sampling
        let mut cumsum = 0.0f32;
        let mut cutoff_index = indexed_probs.len();
        for (i, (prob, _)) in indexed_probs.iter().enumerate() {
            cumsum += prob;
            if cumsum >= top_p {
                cutoff_index = i + 1;
                break;
            }
        }
        indexed_probs.truncate(cutoff_index);

        // Renormalize probabilities
        let total: f32 = indexed_probs.iter().map(|(p, _)| p).sum();
        let normalized: Vec<f32> = indexed_probs.iter().map(|(p, _)| p / total).collect();

        // Sample from distribution
        let rand_val = rand::random::<f32>();
        let mut cumsum = 0.0f32;
        for (prob, &index) in normalized.iter().zip(indexed_probs.iter().map(|(_, i)| i)) {
            cumsum += prob;
            if rand_val <= cumsum {
                return Ok(index);
            }
        }

        // Fallback: return the most likely token
        Ok(indexed_probs.first().map(|(_, i)| *i).unwrap_or(0))
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
    ) -> Result<Vec<usize>> {
        self.generate_with_features(phoneme_ids, None, None, top_k, top_p, temperature)
    }

    /// Generate semantic tokens with BERT and Hubert features (with KV cache optimization)
    ///
    /// This method uses KV cache to speed up autoregressive generation.
    /// KV cache avoids recomputing K and V for previous tokens.
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
    pub fn generate_with_features_kv_cache(
        &self,
        phoneme_ids: &[usize],
        bert_features: Option<&Tensor>,
        hubert_features: Option<&Tensor>,
        top_k: usize,
        top_p: f32,
        temperature: f32,
    ) -> Result<Vec<usize>> {
        if phoneme_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Initialize KV cache for all transformer layers
        let mut kv_cache_manager = KvCacheManager::new(self.num_layers);

        // Convert phoneme IDs to tensor [1, seq_len]
        let input_ids: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
        let mut current_ids = Tensor::new(input_ids.as_slice(), &self.device)?
            .unsqueeze(0)?;

        let mut generated_tokens = Vec::new();
        let max_new_tokens = 500;

        // Prepare BERT projection if available
        let bert_proj_result = if let Some(bert) = bert_features {
            if let Some((proj_w, proj_b)) = &self.bert_proj {
                let bert_dims = bert.dims();
                // BERT output is [1, seq_len, 1024]
                let bert_reshaped = if bert_dims.len() == 3 && bert_dims[1] == 1024 {
                    bert.transpose(1, 2)?
                } else {
                    bert.clone()
                };

                // Project from 1024 to 512 (Candle requires same dims for batched matmul)
                if bert_reshaped.dims().last().copied() == Some(1024) {
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
                let hubert_dims = hubert.dims();
                if hubert_dims.len() >= 2 && hubert_dims.last().copied() == Some(768) {
                    let hubert_reshaped = if hubert_dims.len() == 3 && hubert_dims[1] == 768 {
                        hubert.transpose(1, 2)?
                    } else {
                        hubert.clone()
                    };

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

        // Autoregressive generation with KV cache
        for _step in 0..max_new_tokens {
            let seq_len = current_ids.dims()[1];

            // Get token embeddings
            let token_emb = mixed_embedding_lookup(&self.text_embedding, &self.audio_embedding, &current_ids, self.vocab_size)?;

            // Fuse with BERT/Hubert features (only for the first position in generation)
            let mut fused_emb = token_emb.clone();
            if _step == 0 {
                // Only fuse features at the beginning of generation
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
            }

            // For KV cache, we only pass the last token embedding
            // The cache contains all previous tokens
            let last_token_emb = fused_emb.narrow(1, seq_len - 1, 1)?;

            // Create causal mask for current sequence length (including cached tokens)
            let total_seq_len = kv_cache_manager.len() + 1;
            let causal_mask = TransformerGPTSoVITS::create_causal_mask(total_seq_len, &self.device)?;

            // Forward through transformer with KV cache
            // last_token_emb: [1, 1, hidden], hidden output: [1, 1, hidden]
            let hidden = self.transformer.forward_from_embedding_kv(&last_token_emb, Some(&causal_mask), &mut kv_cache_manager)?;

            // Project to vocab: [1, 1, hidden] @ [vocab, hidden]^T -> [1, 1, vocab]
            let last_hidden = hidden.squeeze(0)?; // [1, hidden]
            let logits = last_hidden.matmul(&self.ar_predict_layer.t()?)?; // [1, vocab]
            let logits = logits.squeeze(0)?; // [vocab]

            // Sample next token
            let next_token = Self::sample_token(&logits, top_k, top_p, temperature)?;

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
    ) -> Result<Vec<usize>> {
        if phoneme_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Convert phoneme IDs to tensor [1, seq_len]
        let input_ids: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
        let mut current_ids = Tensor::new(input_ids.as_slice(), &self.device)?
            .unsqueeze(0)?;

        let mut generated_tokens = Vec::new();
        let max_new_tokens = 500;

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
                // Use MRTE for advanced cross-attention fusion
                // MRTE expects: [batch, channels, seq_len] format
                if let (Some(bert), Some(hubert)) = (bert_proj_result.as_ref(), hubert_proj_result.as_ref()) {
                    // Transpose embeddings: [1, seq, 512] -> [1, 512, seq]
                    let _token_emb_t = token_emb.transpose(1, 2)?;

                    // Prepare BERT features as text encoding [1, 512, bert_seq]
                    let bert_t = bert.transpose(1, 2)?;

                    // Prepare Hubert features as content encoding [1, 512, hubert_frames]
                    let hubert_t = hubert.transpose(1, 2)?;

                    // Create masks
                    let ones_mask = Tensor::ones((1, 1, seq_len), DType::F32, &self.device)?;

                    // MRTE forward: content (Hubert) attends to text (BERT)
                    match mrte.forward(&hubert_t, &ones_mask, &bert_t, &ones_mask, None) {
                        Ok(mrte_out) => {
                            // MRTE output: [1, 512, hubert_frames]
                            // Need to align to token_emb seq_len
                            let mrte_frames = mrte_out.dims()[2];
                            let mrte_aligned = if mrte_frames >= seq_len {
                                mrte_out.narrow(2, 0, seq_len)?
                            } else {
                                // Repeat last frame
                                let last_frame = mrte_out.narrow(2, mrte_frames - 1, 1)?;
                                let mut frames = vec![mrte_out.clone()];
                                for _ in 0..(seq_len - mrte_frames) {
                                    frames.push(last_frame.clone());
                                }
                                Tensor::cat(&frames, 2).unwrap_or_else(|_| mrte_out.clone())
                            };
                            // Transpose back: [1, 512, seq] -> [1, seq, 512]
                            mrte_aligned.transpose(1, 2).unwrap_or_else(|_| token_emb.clone())
                        }
                        Err(_) => token_emb.clone(),
                    }
                } else {
                    token_emb.clone()
                }
            } else {
                // Fallback: simple residual fusion with BERT and Hubert
                let mut fused_emb = token_emb.clone();

                // Fuse with BERT features if available
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

                // Fuse with Hubert features if available
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

            // Project to vocab: [1, seq_len, hidden] @ [vocab, hidden]^T -> [1, seq_len, vocab]
            let last_hidden = hidden.narrow(1, seq_len - 1, 1)?; // [1, 1, hidden]
            let last_hidden = last_hidden.squeeze(0)?; // [1, hidden]
            let logits = last_hidden.matmul(&self.ar_predict_layer.t()?)?; // [1, vocab]
            let logits = logits.squeeze(0)?; // [vocab]

            // Sample next token
            let next_token = Self::sample_token(&logits, top_k, top_p, temperature)?;

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
        top_k: usize,
        top_p: f32,
        temperature: f32,
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
                x_emb.broadcast_add(&projected)?
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
        let max_new_tokens = 500;
        let audio_vocab_size = self.ar_predict_layer.dims()[0];

        // Step 7: Autoregressive generation
        for _step in 0..max_new_tokens {
            let total_seq = xy_pos.dims()[1];

            // Create hybrid attention mask matching Python's behavior:
            // - Text positions (0..text_seq): full bidirectional attention
            // - Audio positions (text_seq..total_seq): attend to all text + causal audio
            // True = masked (blocked), False = visible
            let mask = self.create_hybrid_mask(text_seq, total_seq)?;

            // Forward pass through transformer with custom mask
            let hidden = self.transformer.forward_from_embedding_kv(
                &xy_pos,
                Some(&mask),
                &mut crate::utils::KvCacheManager::new(self.num_layers),
            )?;

            // Project last position to vocab
            let last_hidden = hidden.narrow(1, total_seq - 1, 1)?.squeeze(0)?; // [1, hidden]
            let logits = last_hidden.matmul(&self.ar_predict_layer.t()?)?.squeeze(0)?; // [vocab]

            // Sample next token
            let next_token = Self::sample_token(&logits, top_k, top_p, temperature)?;

            // Check for EOS (but allow at least 10 tokens like Python)
            if _step >= 10 && next_token >= audio_vocab_size - 1 {
                break;
            }

            generated_tokens.push(next_token);

            // Embed new token and append
            let new_ids = Tensor::new(&[next_token as i64], &self.device)?.unsqueeze(0)?;
            let new_emb = self.lookup_tokens(&self.audio_embedding, &new_ids, 1)?;
            // Add positional encoding for the new position
            let new_pos = self.add_sine_positional_at(&new_emb, total_seq)?;
            xy_pos = Tensor::cat(&[&xy_pos, &new_pos], 1)?;
        }

        Ok(generated_tokens)
    }

    /// Create hybrid attention mask matching Python's infer_panel:
    /// - Text positions: full bidirectional attention (no masking)
    /// - Audio positions: attend to all text + causal among audio
    fn create_hybrid_mask(&self, text_seq: usize, total_seq: usize) -> Result<Tensor> {
        let mut mask = vec![0.0f32; total_seq * total_seq]; // 0 = visible, 1 = masked

        for i in text_seq..total_seq {  // Only for audio positions
            for j in text_seq..total_seq {
                if j > i {
                    // Audio position i cannot attend to future audio position j
                    mask[i * total_seq + j] = 1.0;
                }
            }
        }
        // Text positions (0..text_seq) have no masking - full bidirectional
        // Audio positions can see all text positions (0..text_seq) - no masking

        Ok(Tensor::from_vec(mask, (total_seq, total_seq), &self.device)?)
    }

    /// Lookup token embeddings for a tensor of token IDs
    fn lookup_tokens(&self, embedding: &Tensor, ids: &Tensor, _seq: usize) -> Result<Tensor> {
        // embedding: [vocab, hidden], ids: [batch, seq]
        // Use gather or index_select
        let flat_ids = ids.flatten_all()?;
        let mut embeddings = Vec::new();
        let id_vec: Vec<i64> = flat_ids.to_vec1()?;
        for &id in &id_vec {
            embeddings.push(embedding.get(id as usize)?);
        }
        let batch = ids.dims()[0];
        let seq = ids.dims()[1];
        let hidden = embedding.dims()[1];
        let stacked = Tensor::stack(&embeddings, 0)?;
        stacked.reshape((batch, seq, hidden)).map_err(|e| crate::Error::InferenceError(e.to_string()))
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
                x_emb.broadcast_add(&projected)?
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
