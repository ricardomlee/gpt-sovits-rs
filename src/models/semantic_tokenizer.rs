//! Semantic Token Extractor
//!
//! Extracts discrete semantic tokens from audio using SSL projection + codebook.
//! This matches Python's `vits_model.extract_latent(hubert_feature)` which produces
//! the prompt tokens used by the GPT for speaker conditioning.

use candle_core::{Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module};
use crate::utils::{StateDict, load_safetensors};
use crate::Result;

/// Extracts semantic tokens (codebook indices) from Hubert features
pub struct SemanticTokenizer {
    ssl_conv: Conv1d,
    codebook: Tensor,    // Codebook [1024, 768]
}

impl SemanticTokenizer {
    /// Load from safetensors file
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu)
    }

    pub fn load_with_device(path: &str, device: &Device) -> Result<Self> {
        let weights_map = load_safetensors(path)?;
        let state_dict = StateDict::new(weights_map);

        // SSL projection weights
        let ssl_weight = state_dict.get("ssl_proj.weight")?
            .to_device(device)?
            .to_dtype(candle_core::DType::F32)?;

        let ssl_bias = state_dict.get("ssl_proj.bias")?
            .to_device(device)?
            .to_dtype(candle_core::DType::F32)?;

        // Codebook from quantizer
        let codebook = state_dict.get("quantizer.vq.layers.0._codebook.embed")?
            .to_device(device)?
            .to_dtype(candle_core::DType::F32)?;

        // Determine stride from weight shape (kernel size)
        let weight_dims = ssl_weight.dims();
        let stride = if weight_dims.len() == 3 && weight_dims[2] == 2 {
            2  // 25hz mode
        } else {
            1  // 50hz mode
        };

        let padding = weight_dims.get(2).map(|&k| k / 2).unwrap_or(0);

        eprintln!("[SemanticTokenizer] ssl_weight={:?}, codebook={:?}, stride={}",
            ssl_weight.dims(), codebook.dims(), stride);

        let config = Conv1dConfig {
            stride,
            padding,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let ssl_conv = Conv1d::new(ssl_weight, Some(ssl_bias), config);

        Ok(Self {
            ssl_conv,
            codebook,
        })
    }

    /// Extract semantic tokens from Hubert features
    /// hubert_features: [1, 768, T_hubert]
    /// Returns: Vec<usize> of codebook indices [T_tokens]
    pub fn extract(&self, hubert_features: &Tensor) -> Result<Vec<usize>> {
        // Apply SSL projection: Conv1d with stride
        let ssl_out = self.ssl_conv.forward(hubert_features)?;
        // ssl_out: [1, 768, T_tokens]

        // Reshape to [T_tokens, 768] for codebook matching
        let frames = ssl_out.squeeze(0)?.transpose(0, 1)?;
        // frames: [T_tokens, 768]

        // Compute distances to codebook entries
        // ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        let frames_sq = frames.sqr()?;
        let frames_norm = frames_sq.sum_keepdim(1)?;  // [T, 1]

        let codebook_sq = self.codebook.sqr()?;
        let codebook_norm = codebook_sq.sum_keepdim(1)?;  // [1024, 1]

        let dot = frames.matmul(&self.codebook.t()?)?;  // [T, 1024]

        let dist = frames_norm.broadcast_add(&codebook_norm.t()?)?;  // [T, 1024]
        let twice_dot = dot.broadcast_mul(&Tensor::full(2.0f32, dot.dims(), &self.codebook.device())?)?;
        let dist = dist.broadcast_sub(&twice_dot)?;

        // Argmin along codebook dimension
        let codes = Self::argmin_2d(&dist)?;
        Ok(codes)
    }

    /// Argmin along last dimension of [T, N] tensor
    fn argmin_2d(t: &Tensor) -> Result<Vec<usize>> {
        let dims = t.dims();
        let n = dims[1];
        let t_len = dims[0];
        let data: Vec<f32> = t.flatten_all()?.to_vec1()?;

        let mut indices = Vec::with_capacity(t_len);
        for i in 0..t_len {
            let start = i * n;
            let mut min_idx = 0;
            let mut min_val = data[start];
            for j in 1..n {
                let val = data[start + j];
                if val < min_val {
                    min_val = val;
                    min_idx = j;
                }
            }
            indices.push(min_idx);
        }
        Ok(indices)
    }
}
