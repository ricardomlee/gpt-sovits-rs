//! Semantic Token Extractor
//!
//! Extracts discrete semantic tokens from audio using SSL projection + codebook.
//! This matches Python's `vits_model.extract_latent(hubert_feature)` which produces
//! the prompt tokens used by the GPT for speaker conditioning.

use crate::Result;
use crate::utils::{StateDict, load_safetensors};
use candle_core::{Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module};

/// Extracts semantic tokens (codebook indices) from Hubert features
pub struct SemanticTokenizer {
    ssl_conv: Conv1d,
    codebook_t: Tensor,
    codebook_norm_t: Tensor,
}

impl SemanticTokenizer {
    /// Load from safetensors file
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu)
    }

    pub fn load_with_device(path: &str, device: &Device) -> Result<Self> {
        let weights_map = load_safetensors(path)?;
        let state_dict = StateDict::new(weights_map);
        Self::from_state_dict(&state_dict, device)
    }

    pub(crate) fn from_state_dict(state_dict: &StateDict, device: &Device) -> Result<Self> {
        // SSL projection weights
        let ssl_weight = state_dict
            .get("ssl_proj.weight")?
            .to_device(device)?
            .to_dtype(candle_core::DType::F32)?;

        let ssl_bias = state_dict
            .get("ssl_proj.bias")?
            .to_device(device)?
            .to_dtype(candle_core::DType::F32)?;

        // Codebook from quantizer
        let codebook = state_dict
            .get("quantizer.vq.layers.0._codebook.embed")?
            .to_device(device)?
            .to_dtype(candle_core::DType::F32)?;

        // Determine stride from weight shape (kernel size)
        let weight_dims = ssl_weight.dims();
        let stride = if weight_dims.len() == 3 && weight_dims[2] == 2 {
            2 // 25hz mode
        } else {
            1 // 50hz mode
        };

        // Python: nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2) uses default padding=0
        let padding = 0;

        tracing::debug!(
            "[SemanticTokenizer] ssl_weight={:?}, codebook={:?}, stride={}",
            ssl_weight.dims(),
            codebook.dims(),
            stride
        );

        let config = Conv1dConfig {
            stride,
            padding,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let ssl_conv = Conv1d::new(ssl_weight, Some(ssl_bias), config);

        let codebook_t = codebook.t()?;
        let codebook_norm_t = codebook.sqr()?.sum_keepdim(1)?.t()?;

        Ok(Self {
            ssl_conv,
            codebook_t,
            codebook_norm_t,
        })
    }

    /// Extract semantic tokens from Hubert features
    /// hubert_features: [1, 768, T_hubert]
    /// Returns: `Vec<usize>` of codebook indices `[T_tokens]`
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
        let frames_norm = frames_sq.sum_keepdim(1)?; // [T, 1]

        let dot = frames.matmul(&self.codebook_t)?; // [T, 1024]

        let dist = frames_norm.broadcast_add(&self.codebook_norm_t)?; // [T, 1024]
        let twice_dot = dot.affine(2.0, 0.0)?;
        let dist = dist.broadcast_sub(&twice_dot)?;

        Self::argmin_2d(&dist)
    }

    fn argmin_2d(distances: &Tensor) -> Result<Vec<usize>> {
        Ok(distances
            .argmin(1)?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|index| index as usize)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::collections::HashMap;

    fn tokenizer_weights(kernel: usize, dtype: DType) -> Result<HashMap<String, Tensor>> {
        let device = &Device::Cpu;
        Ok(HashMap::from([
            (
                "ssl_proj.weight".into(),
                Tensor::ones((1, 1, kernel), dtype, device)?.affine(1.0 / kernel as f64, 0.0)?,
            ),
            ("ssl_proj.bias".into(), Tensor::ones(1, dtype, device)?),
            (
                "quantizer.vq.layers.0._codebook.embed".into(),
                Tensor::new(&[[-2f32], [0.], [2.]], device)?.to_dtype(dtype)?,
            ),
        ]))
    }

    #[test]
    fn shared_and_file_weights_extract_expected_tokens() -> Result<()> {
        let device = &Device::Cpu;
        let features = Tensor::new(&[[[-3f32, -3., -1., -1., 1., 1.]]], device)?;
        for dtype in [DType::F32, DType::F16] {
            for (kernel, expected) in [(1, vec![0, 0, 1, 1, 2, 2]), (2, vec![0, 1, 2])] {
                let weights = tokenizer_weights(kernel, dtype)?;
                let file = tempfile::NamedTempFile::new().unwrap();
                candle_core::safetensors::save(&weights, file.path())?;
                let state_dict = StateDict::new(weights);
                let shared = SemanticTokenizer::from_state_dict(&state_dict, device)?;
                drop(state_dict);
                let standalone = SemanticTokenizer::load(file.path().to_str().unwrap())?;
                assert_eq!(shared.extract(&features)?, expected);
                assert_eq!(standalone.extract(&features)?, expected);
            }
        }
        Ok(())
    }

    #[test]
    fn shared_and_file_weights_report_the_same_missing_keys() -> Result<()> {
        for key in [
            "ssl_proj.weight",
            "ssl_proj.bias",
            "quantizer.vq.layers.0._codebook.embed",
        ] {
            let mut weights = tokenizer_weights(1, DType::F32)?;
            weights.remove(key);
            let file = tempfile::NamedTempFile::new().unwrap();
            candle_core::safetensors::save(&weights, file.path())?;
            let state_dict = StateDict::new(weights);
            let shared_error = SemanticTokenizer::from_state_dict(&state_dict, &Device::Cpu)
                .err()
                .expect("missing key should fail shared loading");
            let file_error = SemanticTokenizer::load(file.path().to_str().unwrap())
                .err()
                .expect("missing key should fail file loading");
            assert!(matches!(shared_error, crate::Error::ModelLoadError(_)));
            assert!(shared_error.to_string().contains(key));
            assert_eq!(shared_error.to_string(), file_error.to_string());
        }
        Ok(())
    }

    #[test]
    fn argmin_returns_one_index_per_frame() -> Result<()> {
        let distances = Tensor::new(&[[3f32, 1., 2.], [-1., 4., 0.]], &Device::Cpu)?;
        assert_eq!(SemanticTokenizer::argmin_2d(&distances)?, vec![1, 0]);
        Ok(())
    }
}
