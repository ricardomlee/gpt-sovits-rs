//! GPT semantic-token model.
//!
//! Reference-conditioned backends share conditioning and sampling rules. The
//! historical unprompted generator is isolated in the compatibility module.

mod conditioning;
mod cuda_graph;
mod debug;
mod decode;
mod legacy;
mod sampling;
mod weights;

use super::mrte::MRTE;
use super::transformer::TransformerGPTSoVITS;
use crate::Result;
use crate::utils::sampling::SamplingRng;
use candle_core::{DType, Device, Tensor};

#[cfg(test)]
use crate::Error;
#[cfg(test)]
use weights::build_scaled_sine_positions;

/// GPT Model for semantic token prediction
pub struct GPTModel {
    text_embedding: Tensor, // model.ar_text_embedding.word_embeddings.weight [vocab_size, hidden_size]
    audio_embedding: Tensor, // model.ar_audio_embedding.word_embeddings.weight [1025, hidden_size]
    bert_proj: Option<(Tensor, Tensor)>, // (weight, bias) for BERT features [512, 1024], [512]
    hubert_proj: Option<(Tensor, Tensor)>, // (weight, bias) for Hubert features [512, 768], [512]
    mrte: Option<MRTE>,     // MRTE module for advanced cross-attention fusion
    transformer: TransformerGPTSoVITS,
    ar_predict_layer: Tensor, // output projection [vocab_size, hidden_size]
    ar_predict_layer_t: Tensor, // cached transpose [hidden_size, vocab_size]
    text_positional: Tensor,  // Precomputed scaled sine positions [1, max_seq_len, hidden_size]
    audio_positional: Tensor, // Precomputed scaled sine positions [1, max_seq_len, hidden_size]
    device: Device,
    dtype: DType,
    vocab_size: usize,
    num_layers: usize, // Number of transformer layers for KV cache
    sampling_rng: SamplingRng,
}

impl GPTModel {
    /// Get the model compute dtype
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
}

#[cfg(test)]
#[path = "gpt/contracts.rs"]
mod contract_tests;
#[cfg(test)]
pub(crate) use contract_tests::model as test_model;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Model;

    #[test]
    fn device_requests_leave_loaded_weights_on_their_original_device() {
        // Device-policy checks do not need transformer layers.
        let weights: std::collections::HashMap<_, _> = [
            "model.ar_text_embedding.word_embeddings.weight",
            "model.ar_audio_embedding.word_embeddings.weight",
            "model.h.layers.0.linear1.weight",
            "model.ar_predict_layer.weight",
        ]
        .into_iter()
        .map(|name| {
            (
                name.to_owned(),
                Tensor::zeros((16, 16), DType::F32, &Device::Cpu).unwrap(),
            )
        })
        .collect();
        let file = tempfile::NamedTempFile::new().unwrap();
        candle_core::safetensors::save(&weights, file.path()).unwrap();
        let mut model = GPTModel::load(file.path().to_str().unwrap()).unwrap();
        model.to_device("cpu").unwrap();
        model.to_device("CPU").unwrap();
        for requested in ["invalid", "auto", "cuda", "mps", "cuda:1"] {
            let error = model
                .to_device(requested)
                .expect_err("runtime migration must be rejected");
            assert!(matches!(error, Error::ConfigError(_)), "{error}");
            assert_eq!(model.device(), "cpu");
            assert!(model.text_embedding.device().is_cpu());
            assert!(model.audio_embedding.device().is_cpu());
            assert!(model.ar_predict_layer.device().is_cpu());
        }
    }

    #[test]
    fn precomputes_scaled_sine_positions() {
        let positions = build_scaled_sine_positions(2, 4, 2.0, &Device::Cpu, DType::F32).unwrap();
        let values = positions.to_vec3::<f32>().unwrap();

        assert_eq!(values[0][0], vec![0.0, 2.0, 0.0, 2.0]);
        let expected = [
            2.0 * 1.0f32.sin(),
            2.0 * 1.0f32.cos(),
            2.0 * 0.01f32.sin(),
            2.0 * 0.01f32.cos(),
        ];
        for (actual, expected) in values[0][1].iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
