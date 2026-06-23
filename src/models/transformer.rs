//! Transformer Module for GPT Model
//!
//! Implementation of transformer encoder-decoder architecture

use candle_core::{Tensor, Device, D, DType};
use crate::Result;
use crate::utils::{StateDict, Embedding, Linear, LayerNorm, KvCacheManager};

/// Multi-head attention mechanism
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    /// Fused QKV projection weight [3*hidden, hidden] + bias [3*hidden].
    /// When set, replaces 3 separate wq/wk/wv matmuls with a single matmul.
    w_qkv: Option<Linear>,
    pub n_heads: usize,
    pub head_dim: usize,
    pub scale: f64,
}

impl MultiHeadAttention {
    pub fn new(q: Linear, k: Linear, v: Linear, o: Linear, n_heads: usize) -> Self {
        let out_features = q.out_features();
        let head_dim = out_features / n_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        Self { wq: q, wk: k, wv: v, wo: o, w_qkv: None, n_heads, head_dim, scale }
    }

    /// Constructor that keeps the fused QKV weight for a single matmul at inference time.
    /// `w_qkv` weight is [3*hidden, hidden], bias is [3*hidden].
    pub fn new_fused(q: Linear, k: Linear, v: Linear, o: Linear, n_heads: usize, w_qkv: Linear) -> Self {
        let out_features = q.out_features();
        let head_dim = out_features / n_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        Self { wq: q, wk: k, wv: v, wo: o, w_qkv: Some(w_qkv), n_heads, head_dim, scale }
    }

    /// Compute Q, K, V — uses fused matmul when available, else 3 separate projections.
    fn project_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let hidden = self.n_heads * self.head_dim;
        if let Some(w) = &self.w_qkv {
            let qkv = w.forward(x)?; // [batch, seq, 3*hidden]
            let q = qkv.narrow(D::Minus1, 0, hidden)?.contiguous()?;
            let k = qkv.narrow(D::Minus1, hidden, hidden)?.contiguous()?;
            let v = qkv.narrow(D::Minus1, hidden * 2, hidden)?.contiguous()?;
            Ok((q, k, v))
        } else {
            Ok((self.wq.forward(x)?, self.wk.forward(x)?, self.wv.forward(x)?))
        }
    }

    /// Split tensor into multiple heads
    fn split_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        // x: [batch, seq_len, hidden_size] -> [batch, n_heads, seq_len, head_dim]
        x.reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)
            .map_err(|e| e.into())
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let dims = x.dims();
        let (batch_size, seq_len, _) = (dims[0], dims[1], dims[2]);

        let (q, k, v) = self.project_qkv(x)?;

        // Split into heads
        let q = self.split_heads(&q, batch_size, seq_len)?;
        let k = self.split_heads(&k, batch_size, seq_len)?;
        let v = self.split_heads(&v, batch_size, seq_len)?;

        // Scaled dot-product attention: attn = softmax(Q @ K^T / sqrt(d_k))
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let q_contiguous = q.contiguous()?;
        let attn_weights = q_contiguous.matmul(&k_t)?;

        // Apply scale - convert to match attn_weights dtype
        let scale_val = self.scale as f32;
        let scale_tensor = Tensor::full(scale_val, attn_weights.dims(), &attn_weights.device())?;
        let scale_tensor = scale_tensor.to_dtype(attn_weights.dtype())?;
        let attn_weights = attn_weights.broadcast_mul(&scale_tensor)?;

        // Apply causal mask if provided
        let attn_weights = if let Some(m) = mask {
            // Expand mask to match attention weights shape
            let mask_expanded = m.broadcast_left((batch_size, self.n_heads))?;
            // Convert mask to match attn_weights dtype
            let mask_expanded = mask_expanded.to_dtype(attn_weights.dtype())?;
            // Mask with a large negative value for positions we want to ignore
            let neg_inf_val = -1e9f32;
            let neg_inf = Tensor::full(neg_inf_val, attn_weights.dims(), &attn_weights.device())?;
            let neg_inf = neg_inf.to_dtype(attn_weights.dtype())?;
            let mask_weighted = mask_expanded.broadcast_mul(&neg_inf)?;
            attn_weights.add(&mask_weighted)?
        } else {
            attn_weights
        };

        // Softmax over last dimension
        let attn_probs = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        // Apply attention to values: [batch, n_heads, seq_len, head_dim]
        let attn_probs = attn_probs.contiguous()?;
        let v_contiguous = v.contiguous()?;
        let attn_output = attn_probs.matmul(&v_contiguous)?;

        // Concatenate heads: [batch, seq_len, hidden_size]
        // Must call contiguous() after transpose since reshape requires contiguous layout
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.n_heads * self.head_dim))?;

        // Output projection
        self.wo.forward(&attn_output)
    }

    /// Forward with KV cache for inference
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional KV cache for this layer
    /// * `use_cache` - Whether to update cache (for inference) or just read (for prefill)
    ///
    /// # Returns
    /// Output tensor and optionally updated KV cache
    pub fn forward_kv(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&mut crate::utils::KvCache>,
        use_cache: bool,
    ) -> Result<Tensor> {
        let dims = x.dims();
        let (batch_size, seq_len, _) = (dims[0], dims[1], dims[2]);

        let (q, k, v) = self.project_qkv(x)?;

        // Split into heads
        let q = self.split_heads(&q, batch_size, seq_len)?;
        let k = self.split_heads(&k, batch_size, seq_len)?;
        let v = self.split_heads(&v, batch_size, seq_len)?;

        // Update KV cache if enabled
        let (k, v) = if use_cache {
            if let Some(cache) = cache {
                cache.update(k, v)?
            } else {
                (k, v)
            }
        } else {
            (k, v)
        };

        // Scaled dot-product attention with cached K, V
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let q_contiguous = q.contiguous()?;
        let attn_weights = q_contiguous.matmul(&k_t)?;

        // Apply scale
        let scale_val = self.scale as f32;
        let scale_tensor = Tensor::full(scale_val, attn_weights.dims(), &attn_weights.device())?;
        let scale_tensor = scale_tensor.to_dtype(attn_weights.dtype())?;
        let attn_weights = attn_weights.broadcast_mul(&scale_tensor)?;

        // Apply causal mask if provided
        // For KV cache, attn_weights shape is [batch, heads, 1, total_seq_len]
        // We need the last row of the mask (for the new token)
        let attn_weights = if let Some(m) = mask {
            let total_seq_len = k.dims()[2];
            let mask_for_q = if seq_len == 1 && total_seq_len > 1 {
                // Only have Q for the last token, get last row of mask
                m.narrow(0, total_seq_len - 1, 1)?
            } else {
                m.clone()
            };
            let mask_expanded = mask_for_q.broadcast_left((batch_size, self.n_heads))?;
            let mask_expanded = mask_expanded.to_dtype(attn_weights.dtype())?;
            let neg_inf_val = -1e9f32;
            let neg_inf = Tensor::full(neg_inf_val, attn_weights.dims(), &attn_weights.device())?;
            let neg_inf = neg_inf.to_dtype(attn_weights.dtype())?;
            let mask_weighted = mask_expanded.broadcast_mul(&neg_inf)?;
            attn_weights.add(&mask_weighted)?
        } else {
            attn_weights
        };

        // Softmax
        let attn_probs = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        // Apply attention to values
        let attn_probs = attn_probs.contiguous()?;
        let v_contiguous = v.contiguous()?;
        let attn_output = attn_probs.matmul(&v_contiguous)?;

        // Concatenate heads (contiguous needed before reshape after transpose)
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.n_heads * self.head_dim))?;

        // Output projection
        self.wo.forward(&attn_output)
    }
}

/// Feed-forward network with SwiGLU activation
#[derive(Debug, Clone)]
pub struct FeedForward {
    linear1: Linear,  // up projection [hidden, intermediate]
    linear2: Linear,  // down projection [intermediate, hidden]
}

impl FeedForward {
    pub fn new(linear1: Linear, linear2: Linear) -> Self {
        Self { linear1, linear2 }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // ReLU FFN: linear2(relu(linear1(x))) - matching Python
        let hidden = self.linear1.forward(x)?;
        let activated = hidden.clamp(0.0, f32::MAX)?;
        Ok(self.linear2.forward(&activated)?)
    }

    pub fn linear1_forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear1.forward(x)
    }

    pub fn linear2_forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear2.forward(x)
    }
}

/// Transformer block
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub feed_forward: FeedForward,
    pub attn_norm: LayerNorm,
    pub ffn_norm: LayerNorm,
}

impl TransformerBlock {
    pub fn new(attn: MultiHeadAttention, ff: FeedForward, attn_norm: LayerNorm, ffn_norm: LayerNorm) -> Self {
        Self { attention: attn, feed_forward: ff, attn_norm, ffn_norm }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Post-LN: x = norm1(x + attn(x)), x = norm2(x + ff(x)) — matches Python T2SBlock
        let attn_output = self.attention.forward(x, mask)?;
        let x = self.attn_norm.forward(&x.add(&attn_output)?)?;
        let ff_output = self.feed_forward.forward(&x)?;
        Ok(self.ffn_norm.forward(&x.add(&ff_output)?)?)
    }

    /// Forward with KV cache for inference
    pub fn forward_kv(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&mut crate::utils::KvCache>,
        use_cache: bool,
    ) -> Result<Tensor> {
        // Post-LN: x = norm1(x + attn(x)), x = norm2(x + ff(x))
        let attn_output = self.attention.forward_kv(x, mask, cache, use_cache)?;
        let x = self.attn_norm.forward(&x.add(&attn_output)?)?;
        let ff_output = self.feed_forward.forward(&x)?;
        Ok(self.ffn_norm.forward(&x.add(&ff_output)?)?)
    }

    /// Forward with debug intermediates: returns (attn_out, norm1_out, linear1_out, relu_out, final_out)
    pub fn forward_debug(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        // Post-LN: attn on raw x, norm1 after residual
        let attn_output = self.attention.forward(x, mask)?;
        let norm1_out = self.attn_norm.forward(&x.add(&attn_output)?)?;
        let linear1_out = self.feed_forward.linear1_forward(&norm1_out)?;
        let relu_out = linear1_out.relu()?;
        let ffn_out = self.feed_forward.linear2_forward(&relu_out)?;
        let final_out = self.ffn_norm.forward(&norm1_out.add(&ffn_out)?)?;
        Ok((attn_output, norm1_out, linear1_out, relu_out, final_out))
    }

    /// Forward debug with Q,K,V projections and attention output exposed
    pub fn forward_debug_qkv(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // Post-LN: QKV on raw x
        let q_raw = self.attention.wq.forward(x)?;
        let k_raw = self.attention.wk.forward(x)?;
        let v_raw = self.attention.wv.forward(x)?;
        let attn_output = self.attention.forward(x, mask)?;
        let x_after_attn = self.attn_norm.forward(&x.add(&attn_output)?)?;
        let ffn_out = self.feed_forward.forward(&x_after_attn)?;
        let final_out = self.ffn_norm.forward(&x_after_attn.add(&ffn_out)?)?;
        Ok((q_raw, k_raw, v_raw, final_out))
    }
}

/// Transformer model configuration
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_seq_len: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 512,
            hidden_size: 512,
            intermediate_size: 2048,
            num_hidden_layers: 8,
            num_attention_heads: 8,
            max_seq_len: 1024,
        }
    }
}

/// Transformer model for GPT-based semantic prediction
#[derive(Debug, Clone)]
pub struct Transformer {
    pub config: TransformerConfig,
    token_embedding: Embedding,
    position_embedding: Tensor,
    layers: Vec<TransformerBlock>,
    norm: LayerNorm,
    output_projection: Linear,
    device: Device,
}

impl Transformer {
    pub fn new(config: TransformerConfig, weights: &StateDict, device: &Device) -> Result<Self> {
        // Create embeddings
        let token_embedding = weights.get_embedding("tok_emb.weight")?;

        // Create position embeddings
        let pos_emb = weights.get("pos_emb.weight")?
            .narrow(0, 0, config.max_seq_len)?
            .to_device(device)?
            .clone();

        // Create transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("layers.{}", i);

            // Attention projections
            let wq = weights.get_linear(&format!("{}.attention.wq", prefix))?;
            let wk = weights.get_linear(&format!("{}.attention.wk", prefix))?;
            let wv = weights.get_linear(&format!("{}.attention.wv", prefix))?;
            let wo = weights.get_linear(&format!("{}.attention.wo", prefix))?;
            let attn = MultiHeadAttention::new(wq, wk, wv, wo, config.num_attention_heads);

            // Feed-forward network (ReLU)
            let w1 = Linear::new(
                weights.get(&format!("{}.ffn.w1.weight", prefix))?.to_device(device)?.to_dtype(DType::F32)?,
                weights.get(&format!("{}.ffn.w1.bias", prefix)).ok().and_then(|t| t.to_device(device).ok()).and_then(|t| t.to_dtype(DType::F32).ok())
            );
            let w2 = Linear::new(
                weights.get(&format!("{}.ffn.w2.weight", prefix))?.to_device(device)?.to_dtype(DType::F32)?,
                weights.get(&format!("{}.ffn.w2.bias", prefix)).ok().and_then(|t| t.to_device(device).ok()).and_then(|t| t.to_dtype(DType::F32).ok())
            );
            let ff = FeedForward::new(w1, w2);

            // Layer norms with F32 conversion
            let attn_norm = LayerNorm::new(
                weights.get(&format!("{}.attn_norm.weight", prefix))?.to_device(device)?.to_dtype(DType::F32)?,
                weights.get(&format!("{}.attn_norm.bias", prefix))?.to_device(device)?.to_dtype(DType::F32)?
            );
            let ffn_norm = LayerNorm::new(
                weights.get(&format!("{}.ffn_norm.weight", prefix))?.to_device(device)?.to_dtype(DType::F32)?,
                weights.get(&format!("{}.ffn_norm.bias", prefix))?.to_device(device)?.to_dtype(DType::F32)?
            );

            let block = TransformerBlock::new(attn, ff, attn_norm, ffn_norm);
            layers.push(block);
        }

        // Create final norm with F32 conversion
        let norm = LayerNorm::new(
            weights.get("norm.weight")?.to_device(device)?.to_dtype(DType::F32)?,
            weights.get("norm.bias")?.to_device(device)?.to_dtype(DType::F32)?
        );

        // Create output projection with F32 conversion
        let output_projection = Linear::new(
            weights.get("output.weight")?.to_device(device)?.to_dtype(DType::F32)?,
            weights.get("output.bias").ok().and_then(|t| t.to_device(device).ok()).and_then(|t| t.to_dtype(DType::F32).ok())
        );

        Ok(Self {
            config,
            token_embedding,
            position_embedding: pos_emb,
            layers,
            norm,
            output_projection,
            device: device.clone(),
        })
    }

    /// Create causal attention mask
    pub fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if i >= j { 1.0f32 } else { 0.0f32 }).collect::<Vec<_>>())
            .collect();
        Tensor::from_vec(mask, (seq_len, seq_len), device).map_err(|e| e.into())
    }

    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;

        // Get embeddings
        let token_emb = self.token_embedding.forward(input_ids)?;

        // Get position embeddings
        let pos_emb = self.position_embedding.narrow(0, 0, seq_len)?;

        // Combine embeddings
        let mut x = token_emb.broadcast_add(&pos_emb)?;

        // Create causal mask
        let causal_mask = Self::create_causal_mask(seq_len, &self.device)?;

        // Apply transformer layers
        for layer in &self.layers {
            x = layer.forward(&x, Some(&causal_mask))?;
        }

        // Final normalization
        x = self.norm.forward(&x)?;

        // Output projection
        self.output_projection.forward(&x)
    }

    /// Generate tokens autoregressively
    ///
    /// # Arguments
    /// * `prompt` - Input prompt tensor
    /// * `max_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature
    /// * `top_k` - Top-k filtering
    /// * `top_p` - Top-p (nucleus) filtering
    pub fn generate(
        &self,
        prompt: &Tensor,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<Vec<u32>> {
        use candle_core::D;
        use candle_nn::ops::softmax;

        let mut current = prompt.clone();
        let mut generated = Vec::new();

        for _ in 0..max_tokens {
            let seq_len = current.dims()[1];

            // Forward pass
            let logits = self.forward(&current)?;

            // Get logits for last position: [batch, seq_len, vocab] -> [vocab]
            let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(0)?;

            // Apply temperature
            let mut logits = last_logits.to_dtype(candle_core::DType::F32)?;
            if temperature != 1.0 && temperature > 0.0 {
                let t = candle_core::Tensor::full(temperature, logits.dims(), &logits.device())?;
                logits = logits.broadcast_div(&t)?;
            }

            // Softmax to get probabilities
            let probs = softmax(&logits, D::Minus1)?;
            let probs_vec: Vec<f32> = probs.to_vec1()?;

            // Create (prob, index) pairs and sort descending
            let mut indexed_probs: Vec<(f32, usize)> = probs_vec
                .into_iter()
                .enumerate()
                .map(|(i, p)| (p, i))
                .collect();
            indexed_probs.sort_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Apply top-k
            if top_k < indexed_probs.len() && top_k > 0 {
                indexed_probs.truncate(top_k);
            }

            // Apply top-p (nucleus) sampling
            let mut cumsum = 0.0f32;
            let mut cutoff = indexed_probs.len();
            for (i, (prob, _)) in indexed_probs.iter().enumerate() {
                cumsum += *prob;
                if cumsum >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed_probs.truncate(cutoff);

            // Renormalize
            let total: f32 = indexed_probs.iter().map(|(p, _)| p).sum();
            if total > 0.0 {
                for (prob, _) in indexed_probs.iter_mut() {
                    *prob /= total;
                }
            }

            // Sample
            let rand_val = rand::random::<f32>();
            let mut cumsum = 0.0f32;
            let next_token = indexed_probs
                .iter()
                .find(|&&(prob, _)| {
                    cumsum += prob;
                    rand_val <= cumsum
                })
                .map(|(_, idx)| *idx as u32)
                .unwrap_or(indexed_probs.first().map(|(_, i)| *i as u32).unwrap_or(0));

            // Check for EOS (last token in vocab)
            if next_token >= (self.config.vocab_size - 1) as u32 {
                break;
            }

            generated.push(next_token);

            // Append to input for next iteration
            let next_tensor =
                candle_core::Tensor::new(&[next_token as i64], &self.device)?.unsqueeze(0)?;
            current = candle_core::Tensor::cat(&[current, next_tensor], 1)?;
        }

        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask() {
        let mask = Transformer::create_causal_mask(4, &Device::Cpu).unwrap();
        let mask_vec: Vec<f32> = mask.to_vec2().unwrap().iter().flatten().copied().collect();

        // First row should be [1, 0, 0, 0]
        assert_eq!(mask_vec[0], 1.0);
        assert_eq!(mask_vec[1], 0.0);
    }
}

/// Transformer for GPT-SoVITS with fused QKV and SwiGLU
#[derive(Debug, Clone)]
pub struct TransformerGPTSoVITS {
    #[allow(dead_code)]
    pub config: TransformerConfig,
    token_embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    device: Device,
}

impl TransformerGPTSoVITS {
    pub fn new(config: TransformerConfig, weights: &StateDict, device: &Device, dtype: DType) -> Result<Self> {
        // Load text embedding with dtype conversion
        let token_embedding_weight = weights
            .get("model.ar_text_embedding.word_embeddings.weight")?
            .to_device(device)?
            .to_dtype(dtype)?;
        let token_embedding = Embedding::new(token_embedding_weight);

        // Create transformer layers with GPT-SoVITS format
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.h.layers.{}", i);

            // Load fused QKV projection: weight [3*hidden, hidden], bias [3*hidden]
            // Keep the fused weight for a single matmul at inference time.
            let in_proj_weight = weights.get(&format!("{}.self_attn.in_proj_weight", prefix))?
                .to_device(device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let in_proj_bias = weights.get(&format!("{}.self_attn.in_proj_bias", prefix))?
                .to_device(device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let w_qkv = Linear::new(in_proj_weight.clone(), Some(in_proj_bias.clone()));

            // Also split for the separate wq/wk/wv fields (used by forward_debug_qkv only)
            let hidden = config.hidden_size;
            let q_weight = in_proj_weight.narrow(0, 0, hidden)?.contiguous()?;
            let k_weight = in_proj_weight.narrow(0, hidden, hidden)?.contiguous()?;
            let v_weight = in_proj_weight.narrow(0, hidden * 2, hidden)?.contiguous()?;
            let q_bias = in_proj_bias.narrow(0, 0, hidden)?.contiguous()?;
            let k_bias = in_proj_bias.narrow(0, hidden, hidden)?.contiguous()?;
            let v_bias = in_proj_bias.narrow(0, hidden * 2, hidden)?.contiguous()?;
            let wq = Linear::new(q_weight, Some(q_bias));
            let wk = Linear::new(k_weight, Some(k_bias));
            let wv = Linear::new(v_weight, Some(v_bias));

            // Load output projection with dtype conversion
            let wo_weight = weights.get(&format!("{}.self_attn.out_proj.weight", prefix))?
                .to_device(device)?
                .to_dtype(dtype)?;
            let wo_bias = weights.get(&format!("{}.self_attn.out_proj.bias", prefix))?
                .to_device(device)?
                .to_dtype(dtype)?;
            let wo = Linear::new(wo_weight, Some(wo_bias));

            let attn = MultiHeadAttention::new_fused(wq, wk, wv, wo, config.num_attention_heads, w_qkv);

            // Load FFN with dtype conversion
            let linear1_weight = weights.get(&format!("{}.linear1.weight", prefix))?
                .to_device(device)?
                .to_dtype(dtype)?;
            let linear1_bias = weights.get(&format!("{}.linear1.bias", prefix))?
                .to_device(device)?
                .to_dtype(dtype)?;
            let linear2_weight = weights.get(&format!("{}.linear2.weight", prefix))?
                .to_device(device)?
                .to_dtype(dtype)?;
            let linear2_bias = weights.get(&format!("{}.linear2.bias", prefix))?
                .to_device(device)?
                .to_dtype(dtype)?;

            // Simple ReLU FFN: linear2(relu(linear1(x))) - matching Python
            let w1 = Linear::new(linear1_weight, Some(linear1_bias));
            let w2 = Linear::new(linear2_weight, Some(linear2_bias));

            let ff = FeedForward::new(w1, w2);

            // Load layer norms: keep F32 for numerical stability
            let attn_norm_weight = weights.get(&format!("{}.norm1.weight", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
            let attn_norm_bias = weights.get(&format!("{}.norm1.bias", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
            let attn_norm = LayerNorm::new(attn_norm_weight, attn_norm_bias);

            let ffn_norm_weight = weights.get(&format!("{}.norm2.weight", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
            let ffn_norm_bias = weights.get(&format!("{}.norm2.bias", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
            let ffn_norm = LayerNorm::new(ffn_norm_weight, ffn_norm_bias);

            let block = TransformerBlock::new(attn, ff, attn_norm, ffn_norm);
            layers.push(block);
        }

        // GPT-SoVITS doesn't have a final layer norm

        Ok(Self {
            config,
            token_embedding,
            layers,
            device: device.clone(),
        })
    }

    /// Create causal attention mask
    pub fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
        Transformer::create_causal_mask(seq_len, device)
    }

    /// Forward pass without position embeddings (uses RoPE internally or none)
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;

        // Get token embeddings
        let mut x = self.token_embedding.forward(input_ids)?;

        // Create causal mask
        let causal_mask = Self::create_causal_mask(seq_len, &self.device)?;

        // Apply transformer layers
        for layer in &self.layers {
            x = layer.forward(&x, Some(&causal_mask))?;
        }

        Ok(x)
    }

    /// Forward pass from pre-computed embeddings
    pub fn forward_from_embedding(&self, embeddings: &Tensor) -> Result<Tensor> {
        let (_, seq_len, _) = embeddings.dims3()?;

        let mut x = embeddings.clone();

        // Create causal mask
        let causal_mask = Self::create_causal_mask(seq_len, &self.device)?;

        // Apply transformer layers
        for layer in &self.layers {
            x = layer.forward(&x, Some(&causal_mask))?;
        }

        Ok(x)
    }

    /// Forward from embeddings with KV cache support
    pub fn forward_from_embedding_kv(
        &self,
        embeddings: &Tensor,
        mask: Option<&Tensor>,
        kv_cache_manager: &mut KvCacheManager,
    ) -> Result<Tensor> {
        let mut x = embeddings.clone();

        // Pass mask through as-is:
        // - Some(mask): used for prefill with hybrid/causal mask
        // - None: single-token decode — new token attends to ALL cached K/V (no masking needed)
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let cache = kv_cache_manager.get_or_create(layer_idx);
            x = layer.forward_kv(&x, mask, Some(cache), true)?;
        }

        Ok(x)
    }

    /// Forward through only the first transformer layer
    pub fn forward_first_layer(&self, embeddings: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let mut x = embeddings.clone();
        if self.layers.is_empty() {
            return Err(crate::Error::InferenceError(
                "No transformer layers available".to_string(),
            ));
        }
        x = self.layers[0].forward(&x, Some(mask))?;
        Ok(x)
    }

    /// Forward through all transformer layers with provided mask
    pub fn forward_all_layers_with_mask(&self, embeddings: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let mut x = embeddings.clone();
        for layer in &self.layers {
            x = layer.forward(&x, Some(mask))?;
        }
        Ok(x)
    }

    /// Forward through all transformer layers with causal mask
    pub fn forward_all_layers(&self, embeddings: &Tensor) -> Result<Tensor> {
        let (_, seq_len, _) = embeddings.dims3()?;
        let causal_mask = Self::create_causal_mask(seq_len, &self.device)?;
        self.forward_all_layers_with_mask(embeddings, &causal_mask)
    }

    /// Forward through first layer, returning all intermediates for debugging
    pub fn forward_first_layer_debug(
        &self,
        embeddings: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        // (attn_out, norm1_out, linear1_out, relu_out, final_out)
        if self.layers.is_empty() {
            return Err(crate::Error::InferenceError(
                "No transformer layers available".to_string(),
            ));
        }
        let block = &self.layers[0];
        block.forward_debug(embeddings, Some(mask))
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}
