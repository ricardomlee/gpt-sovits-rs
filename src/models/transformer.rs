//! Transformer Module for GPT Model
//!
//! Implementation of transformer encoder-decoder architecture

use candle_core::{Tensor, Device, D, DType};
use crate::Result;
use crate::utils::{StateDict, Embedding, Linear, LayerNorm};

/// Multi-head attention mechanism
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiHeadAttention {
    pub fn new(q: Linear, k: Linear, v: Linear, o: Linear, n_heads: usize) -> Self {
        let out_features = q.out_features();
        let head_dim = out_features / n_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        Self { wq: q, wk: k, wv: v, wo: o, n_heads, head_dim, scale }
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

        // Compute Q, K, V projections
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

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
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.n_heads * self.head_dim))?;

        // Output projection
        self.wo.forward(&attn_output)
    }
}

/// Feed-forward network with SwiGLU activation
#[derive(Debug, Clone)]
pub struct FeedForward {
    w1: Linear,  // gate projection
    w2: Linear,  // output projection
    w3: Linear,  // up projection
}

impl FeedForward {
    pub fn new(w1: Linear, w2: Linear, w3: Linear) -> Self {
        Self { w1, w2, w3 }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: (swish(w1(x)) * w3(x)) @ w2
        let gate = self.w1.forward(x)?;
        let up = self.w3.forward(x)?;

        // Swish activation: x * sigmoid(x) = x / (1 + exp(-x))
        let gate_exp = gate.neg()?.exp()?;
        let ones = Tensor::ones_like(&gate_exp)?;
        let sigmoid = ones.broadcast_add(&gate_exp)?.recip()?;
        let gate_swish = gate.broadcast_mul(&sigmoid)?;

        // Element-wise multiplication
        let ff_output = gate_swish.broadcast_mul(&up)?;

        Ok(self.w2.forward(&ff_output)?)
    }
}

/// Transformer block
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    attn_norm: LayerNorm,
    ffn_norm: LayerNorm,
}

impl TransformerBlock {
    pub fn new(attn: MultiHeadAttention, ff: FeedForward, attn_norm: LayerNorm, ffn_norm: LayerNorm) -> Self {
        Self { attention: attn, feed_forward: ff, attn_norm, ffn_norm }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-norm architecture
        let normed = self.attn_norm.forward(x)?;
        let attn_output = self.attention.forward(&normed, mask)?;
        // Residual connection
        let x = x.add(&attn_output)?;

        let normed = self.ffn_norm.forward(&x)?;
        let ff_output = self.feed_forward.forward(&normed)?;
        // Residual connection
        Ok(x.add(&ff_output)?)
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

            // Feed-forward network (SwiGLU uses 3 linear layers)
            let w1 = Linear::new(
                weights.get(&format!("{}.ffn.w1.weight", prefix))?.to_device(device)?.to_dtype(DType::F32)?,
                weights.get(&format!("{}.ffn.w1.bias", prefix)).ok().and_then(|t| t.to_device(device).ok()).and_then(|t| t.to_dtype(DType::F32).ok())
            );
            let w2 = Linear::new(
                weights.get(&format!("{}.ffn.w2.weight", prefix))?.to_device(device)?.to_dtype(DType::F32)?,
                weights.get(&format!("{}.ffn.w2.bias", prefix)).ok().and_then(|t| t.to_device(device).ok()).and_then(|t| t.to_dtype(DType::F32).ok())
            );
            let w3 = Linear::new(
                weights.get(&format!("{}.ffn.w3.weight", prefix))?.to_device(device)?.to_dtype(DType::F32)?,
                weights.get(&format!("{}.ffn.w3.bias", prefix)).ok().and_then(|t| t.to_device(device).ok()).and_then(|t| t.to_dtype(DType::F32).ok())
            );
            let ff = FeedForward::new(w1, w2, w3);

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
    pub config: TransformerConfig,
    token_embedding: Embedding,
    layers: Vec<TransformerBlock>,
    device: Device,
}

impl TransformerGPTSoVITS {
    pub fn new(config: TransformerConfig, weights: &StateDict, device: &Device) -> Result<Self> {
        // Load text embedding with F32 conversion
        let token_embedding_weight = weights
            .get("model.ar_text_embedding.word_embeddings.weight")?
            .to_device(device)?
            .to_dtype(DType::F32)?;
        let token_embedding = Embedding::new(token_embedding_weight);

        // Create transformer layers with GPT-SoVITS format
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.h.layers.{}", i);

            // Load fused QKV projection: [hidden * 3, hidden]
            // Split into separate Q, K, V weights with F32 conversion
            let in_proj_weight = weights.get(&format!("{}.self_attn.in_proj_weight", prefix))?
                .to_device(device)?
                .to_dtype(DType::F32)?;
            let in_proj_bias = weights.get(&format!("{}.self_attn.in_proj_bias", prefix))?
                .to_device(device)?
                .to_dtype(DType::F32)?;

            // Split QKV weight into three parts: [hidden, hidden] each
            let hidden = config.hidden_size;
            let q_weight = in_proj_weight.narrow(0, 0, hidden)?;
            let k_weight = in_proj_weight.narrow(0, hidden, hidden)?;
            let v_weight = in_proj_weight.narrow(0, hidden * 2, hidden)?;

            // Split QKV bias into three parts
            let q_bias = in_proj_bias.narrow(0, 0, hidden)?;
            let k_bias = in_proj_bias.narrow(0, hidden, hidden)?;
            let v_bias = in_proj_bias.narrow(0, hidden * 2, hidden)?;

            // Create Linear weights manually
            let wq = Linear::new(q_weight, Some(q_bias));
            let wk = Linear::new(k_weight, Some(k_bias));
            let wv = Linear::new(v_weight, Some(v_bias));

            // Load output projection with F32 conversion
            let wo_weight = weights.get(&format!("{}.self_attn.out_proj.weight", prefix))?
                .to_device(device)?
                .to_dtype(DType::F32)?;
            let wo_bias = weights.get(&format!("{}.self_attn.out_proj.bias", prefix))?
                .to_device(device)?
                .to_dtype(DType::F32)?;
            let wo = Linear::new(wo_weight, Some(wo_bias));

            let attn = MultiHeadAttention::new(wq, wk, wv, wo, config.num_attention_heads);

            // Load FFN with F32 conversion
            let linear1_weight = weights.get(&format!("{}.linear1.weight", prefix))?
                .to_device(device)?
                .to_dtype(DType::F32)?;
            let linear1_bias = weights.get(&format!("{}.linear1.bias", prefix))?
                .to_device(device)?
                .to_dtype(DType::F32)?;
            let linear2_weight = weights.get(&format!("{}.linear2.weight", prefix))?
                .to_device(device)?
                .to_dtype(DType::F32)?;
            let linear2_bias = weights.get(&format!("{}.linear2.bias", prefix))?
                .to_device(device)?
                .to_dtype(DType::F32)?;

            // Use linear1 for both gate (w1) and up (w3), linear2 for output (w2)
            let w1 = Linear::new(linear1_weight.clone(), Some(linear1_bias.clone()));
            let w3 = Linear::new(linear1_weight.clone(), Some(linear1_bias.clone()));
            let w2 = Linear::new(linear2_weight, Some(linear2_bias));

            let ff = FeedForward::new(w1, w2, w3);

            // Load layer norms with F32 conversion
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

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}
