//! GPT Model for semantic token prediction
//!
//! This module implements the GPT model from GPT-SoVITS, which uses:
//! - Fused QKV projection (in_proj_weight combines Q, K, V weights)
//! - RoPE (Rotary Position Embedding) instead of learned positions
//! - Separate text and audio embeddings
//! - BERT feature projection
//! - Hubert feature projection for prosody guidance

use candle_core::{Device, DType, Tensor, D};
use candle_nn::ops::softmax;
use crate::{Result, Error};
use crate::utils::{StateDict, load_safetensors};
use super::transformer::{TransformerGPTSoVITS, TransformerConfig};

/// GPT Model for semantic token prediction
pub struct GPTModel {
    text_embedding: Tensor,      // model.ar_text_embedding.word_embeddings.weight [vocab_size, hidden_size]
    #[allow(dead_code)]
    audio_embedding: Tensor,     // model.ar_audio_embedding.word_embeddings.weight [1025, hidden_size]
    bert_proj: Option<(Tensor, Tensor)>, // (weight, bias) for BERT features [512, 1024], [512]
    hubert_proj: Option<(Tensor, Tensor)>, // (weight, bias) for Hubert features [512, 768], [512]
    transformer: TransformerGPTSoVITS,
    ar_predict_layer: Tensor,    // output projection [vocab_size, hidden_size]
    device: Device,
    dtype: DType,
    vocab_size: usize,
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
        let ar_predict_layer = state_dict.get("model.ar_predict_layer.weight")?
            .to_device(device)?
            .to_dtype(DType::F32)?;

        Ok(Self {
            text_embedding,
            audio_embedding,
            bert_proj,
            hubert_proj,
            transformer,
            ar_predict_layer,
            device: device.clone(),
            dtype: DType::F32,
            vocab_size,
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
                // Project BERT features: [batch, seq, 768] @ [768, 512] + bias -> [batch, seq, 512]
                // BERT output is typically [1, 768, seq] or [1, seq, 768]
                let bert_dims = bert.dims();
                let bert_reshaped = if bert_dims.len() == 3 && bert_dims[1] == 768 {
                    // Shape: [batch, 768, seq] -> transpose to [batch, seq, 768]
                    bert.transpose(1, 2)?
                } else {
                    bert.clone()
                };

                // Ensure last dim is 768 for projection
                if bert_reshaped.dims().last().copied() == Some(768) {
                    let projected = bert_reshaped.matmul(&proj_w.t()?)?;
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

                    // Project to hidden size
                    let projected = hubert_reshaped.matmul(&proj_w.t()?)?;
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

            // Fuse with BERT features if available
            let mut fused_emb = token_emb;
            if let Some(ref bert_proj) = bert_proj_result {
                // Ensure shapes match for fusion
                if bert_proj.dims().len() >= 2 {
                    let bert_seq_len = bert_proj.dims()[1];
                    if bert_seq_len >= seq_len {
                        // Narrow BERT features to match current sequence length
                        let bert_narrowed = if bert_seq_len > seq_len {
                            bert_proj.narrow(1, 0, seq_len)?
                        } else {
                            bert_proj.clone()
                        };
                        // Add BERT features to embeddings (residual connection)
                        // Scale factor to balance embeddings and BERT features
                        if bert_narrowed.dims() == fused_emb.dims() {
                            // Apply layer norm-like scaling
                            let scale = 0.5f32;
                            let scaled_bert = bert_narrowed.broadcast_mul(&Tensor::full(scale, bert_narrowed.dims(), &self.device)?)?;
                            fused_emb = fused_emb.broadcast_add(&scaled_bert)?;
                        }
                    }
                }
            }

            // Fuse with Hubert features if available
            if let Some(ref hubert_proj) = hubert_proj_result {
                // Hubert features are in frame space (e.g., 49 frames for 1s audio)
                // We need to interpolate/align to match the phoneme sequence length
                let hubert_frames = hubert_proj.dims()[1];

                if hubert_frames > 0 {
                    // Simple approach: repeat Hubert features to match sequence length
                    // or use the last available frame if sequence is longer
                    let hubert_aligned = if hubert_frames >= seq_len {
                        // Narrow Hubert features to match sequence
                        hubert_proj.narrow(1, 0, seq_len)?
                    } else {
                        // Repeat the last Hubert frame to fill the sequence
                        // This is a simplification; full MRTE uses cross-attention
                        let last_frame = hubert_proj.narrow(1, hubert_frames - 1, 1)?;
                        let mut frames = vec![hubert_proj.clone()];
                        for _ in 0..(seq_len - hubert_frames) {
                            frames.push(last_frame.clone());
                        }
                        Tensor::cat(&frames, 1).unwrap_or_else(|_| hubert_proj.clone())
                    };

                    // Ensure shapes match and fuse
                    if hubert_aligned.dims() == fused_emb.dims() {
                        // Apply scaling for Hubert features (prosody guidance)
                        let scale = 0.3f32;
                        let scaled_hubert = hubert_aligned.broadcast_mul(&Tensor::full(scale, hubert_aligned.dims(), &self.device)?)?;
                        fused_emb = fused_emb.broadcast_add(&scaled_hubert)?;
                    }
                }
            }

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

    /// Get model device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get model dtype
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
