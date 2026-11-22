//! Compatibility generator used without reference semantic tokens.
//!
//! This preserves the historical feature-fusion and stopping behavior. Reference-
//! conditioned decoding is implemented separately in `decode` and `cuda_graph`.

use super::GPTModel;
use crate::Result;
use candle_core::{DType, Tensor};

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
        }
        .into());
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
                tracing::warn!(
                    "audio_idx {} out of range for audio_emb {:?}",
                    audio_idx,
                    audio_emb.dims()
                );
                return Err(candle_core::Error::UnexpectedShape {
                    msg: format!(
                        "Audio token index {} out of range [0, {})",
                        audio_idx,
                        audio_emb.dims()[0]
                    ),
                    expected: candle_core::Shape::from(&[1usize, 1]),
                    got: candle_core::Shape::from(&[audio_idx, 1]),
                }
                .into());
            }
            audio_emb.get(audio_idx)?
        };
        embeddings.push(emb);
    }

    // Stack: [batch*seq_len, hidden]
    let stacked = Tensor::stack(&embeddings, 0)?.to_device(device)?;

    // Reshape to [batch, seq_len, hidden]
    stacked
        .reshape((batch, seq_len, hidden_size))
        .map_err(|e| e.into())
}

impl GPTModel {
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
        self.generate_with_features(
            phoneme_ids,
            None,
            None,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            max_tokens,
        )
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
    #[allow(clippy::too_many_arguments)]
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
        let mut current_ids = Tensor::new(input_ids.as_slice(), &self.device)?.unsqueeze(0)?;

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
                    let projected =
                        projected.broadcast_add(&proj_b.reshape((1, 1, proj_b.dims()[0]))?)?;
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
                    let projected =
                        projected.broadcast_add(&proj_b.reshape((1, 1, proj_b.dims()[0]))?)?;
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
            let token_emb = mixed_embedding_lookup(
                &self.text_embedding,
                &self.audio_embedding,
                &current_ids,
                self.vocab_size,
            )?;

            // Fuse features using MRTE if available, otherwise fall back to simple addition
            let fused_emb = if let Some(ref mrte) = self.mrte {
                if let (Some(bert), Some(hubert)) =
                    (bert_proj_result.as_ref(), hubert_proj_result.as_ref())
                {
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
                            mrte_aligned
                                .transpose(1, 2)
                                .unwrap_or_else(|_| token_emb.clone())
                        }
                        Err(_) => token_emb.clone(),
                    }
                } else {
                    token_emb.clone()
                }
            } else {
                let mut fused_emb = token_emb.clone();
                if let Some(ref bert_proj) = bert_proj_result
                    && bert_proj.dims().len() >= 2
                {
                    let bert_seq_len = bert_proj.dims()[1];
                    if bert_seq_len >= seq_len {
                        let bert_narrowed = if bert_seq_len > seq_len {
                            bert_proj.narrow(1, 0, seq_len)?
                        } else {
                            bert_proj.clone()
                        };
                        if bert_narrowed.dims() == fused_emb.dims() {
                            let scale = 0.5f32;
                            let scaled_bert = bert_narrowed.broadcast_mul(&Tensor::full(
                                scale,
                                bert_narrowed.dims(),
                                &self.device,
                            )?)?;
                            fused_emb = fused_emb.broadcast_add(&scaled_bert)?;
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
                            let scaled_hubert = hubert_aligned.broadcast_mul(&Tensor::full(
                                scale,
                                hubert_aligned.dims(),
                                &self.device,
                            )?)?;
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
            let logits = last_hidden.matmul(&self.ar_predict_layer_t)?.squeeze(0)?;

            // Sample next token (no prompt tokens for generate_with_features path)
            let (next_token, _argmax) = self.sample_token(
                &logits,
                top_k,
                top_p,
                temperature,
                repetition_penalty,
                &[],
                &generated_tokens,
            )?;

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
}
