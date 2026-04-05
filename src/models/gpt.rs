//! GPT Model for semantic token prediction

use candle_core::{Device, DType, Tensor, D};
use candle_nn::ops::softmax;
use crate::{Result, Error};
use crate::utils::{StateDict, load_safetensors};
use super::transformer::{Transformer, TransformerConfig};

/// GPT Model for semantic token prediction
pub struct GPTModel {
    transformer: Transformer,
    device: Device,
    dtype: DType,
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

        // Infer config from weights
        let vocab_size = state_dict.get("tok_emb.weight")?.dims()[0];
        let hidden_size = state_dict.get("tok_emb.weight")?.dims()[1];

        // Count number of layers
        let mut num_hidden_layers = 0;
        for key in state_dict.keys() {
            if key.starts_with("layers.") && key.contains(".attention.wq") {
                num_hidden_layers += 1;
            }
        }

        // Infer number of attention heads
        let wq_out_features = state_dict.get("layers.0.attention.wq.weight")?.dims()[0];
        let num_attention_heads = if hidden_size == 512 { 8 } else { wq_out_features / (hidden_size / num_hidden_layers.max(1)) };

        let max_seq_len = state_dict.get("pos_emb.weight")?.dims()[0];

        let config = TransformerConfig {
            vocab_size,
            hidden_size,
            intermediate_size: state_dict.get("layers.0.ffn.w1.weight")?.dims()[0],
            num_hidden_layers,
            num_attention_heads,
            max_seq_len,
        };

        // Create transformer
        let transformer = Transformer::new(config, &state_dict, device)?;

        Ok(Self {
            transformer,
            device: device.clone(),
            dtype: DType::F32,
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

    /// Generate semantic tokens from phoneme IDs
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
        if phoneme_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Convert phoneme IDs to tensor [1, seq_len]
        let input_ids: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
        let mut current_ids = Tensor::new(input_ids.as_slice(), &self.device)?
            .unsqueeze(0)?;

        let mut generated_tokens = Vec::new();
        let max_new_tokens = 200; // Maximum tokens to generate

        // Autoregressive generation
        for _ in 0..max_new_tokens {
            let seq_len = current_ids.dims()[1];

            // Forward pass through transformer
            let logits = self.transformer.forward(&current_ids)?;

            // Get logits for the last position: [1, seq_len, vocab_size] -> [vocab_size]
            let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(0)?;

            // Sample next token
            let next_token = Self::sample_token(&last_logits, top_k, top_p, temperature)?;

            // Check for end-of-sequence token (assuming vocab_size - 1 is EOS)
            if next_token >= self.transformer.config.vocab_size - 1 {
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
