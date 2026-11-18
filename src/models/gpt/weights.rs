//! Weight loading and precomputed positional tables.

use super::GPTModel;
use crate::models::mrte::MRTE;
use crate::models::transformer::{TransformerConfig, TransformerGPTSoVITS};
use crate::utils::sampling::SamplingRng;
use crate::utils::{load_safetensors, StateDict};
use crate::Result;
use candle_core::{DType, Device, Tensor};

pub(super) fn build_scaled_sine_positions(
    max_seq_len: usize,
    hidden_size: usize,
    alpha: f32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let half_dim = hidden_size / 2;
    let div_term: Vec<f64> = (0..half_dim)
        .map(|i| (-((i as f64) * 2.0) * (10000.0f64.ln()) / hidden_size as f64).exp())
        .collect();
    let mut positions = vec![0.0f32; max_seq_len * hidden_size];

    for pos in 0..max_seq_len {
        for (i, &scale) in div_term.iter().enumerate() {
            let value = pos as f64 * scale;
            positions[pos * hidden_size + 2 * i] = value.sin() as f32;
            positions[pos * hidden_size + 2 * i + 1] = value.cos() as f32;
        }
    }

    let positions =
        Tensor::from_vec(positions, (1usize, max_seq_len, hidden_size), device)?.to_dtype(dtype)?;
    let alpha = Tensor::full(alpha, positions.dims(), device)?.to_dtype(dtype)?;
    positions.broadcast_mul(&alpha).map_err(Into::into)
}

impl GPTModel {
    /// Load model from safetensors file
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu, DType::F32)
    }

    /// Load model with specific device
    pub fn load_with_device(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let weights_map = load_safetensors(path)?;
        let state_dict = StateDict::new(weights_map);
        Self::from_state_dict(&state_dict, device, dtype)
    }

    /// Construct the model from decoded tensors without reading a checkpoint file.
    pub fn from_state_dict(state_dict: &StateDict, device: &Device, dtype: DType) -> Result<Self> {
        // Load text embedding: [vocab_size, hidden_size]
        let text_emb_key = "model.ar_text_embedding.word_embeddings.weight";
        let text_embedding = state_dict
            .get(text_emb_key)?
            .to_device(device)?
            .to_dtype(dtype)?;

        let vocab_size = text_embedding.dims()[0];
        let hidden_size = text_embedding.dims()[1];

        // Load audio embedding: [1025, hidden_size]
        let audio_emb_key = "model.ar_audio_embedding.word_embeddings.weight";
        let audio_embedding = state_dict
            .get(audio_emb_key)?
            .to_device(device)?
            .to_dtype(dtype)?;

        // Load BERT projection (optional): weight [512, 1024], bias [512]
        let bert_proj = if state_dict.contains("model.bert_proj.weight") {
            let bert_weight = state_dict
                .get("model.bert_proj.weight")?
                .to_device(device)?
                .to_dtype(dtype)?;
            let bert_bias = state_dict
                .get("model.bert_proj.bias")?
                .to_device(device)?
                .to_dtype(dtype)?;
            Some((bert_weight, bert_bias))
        } else {
            None
        };

        // Load Hubert projection (optional): weight [512, 768], bias [512]
        let hubert_proj = if state_dict.contains("model.hubert_proj.weight") {
            let hubert_weight = state_dict
                .get("model.hubert_proj.weight")?
                .to_device(device)?
                .to_dtype(dtype)?;
            let hubert_bias = state_dict
                .get("model.hubert_proj.bias")?
                .to_device(device)?
                .to_dtype(dtype)?;
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
                dtype,
                device,
            );
            // Check if we can access MRTE weights
            MRTE::new(768, 512, 512, 8, mrte_vb.pp("model.mrte")).ok()
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
        let max_seq_len = 2048;
        let config = TransformerConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            max_seq_len,
        };

        // Create transformer with GPT-SoVITS style weights
        let transformer = TransformerGPTSoVITS::new(config, state_dict, device, dtype)?;

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

        let ar_predict_layer = state_dict
            .get("model.ar_predict_layer.weight")?
            .to_device(device)?
            .to_dtype(dtype)?;
        let ar_predict_layer_t = ar_predict_layer.t()?;
        let text_positional =
            build_scaled_sine_positions(max_seq_len, hidden_size, text_pos_alpha, device, dtype)?;
        let audio_positional =
            build_scaled_sine_positions(max_seq_len, hidden_size, audio_pos_alpha, device, dtype)?;

        Ok(Self {
            text_embedding,
            audio_embedding,
            bert_proj,
            hubert_proj,
            mrte,
            transformer,
            ar_predict_layer,
            ar_predict_layer_t,
            text_positional,
            audio_positional,
            device: device.clone(),
            dtype,
            vocab_size,
            num_layers: num_hidden_layers,
            sampling_rng: SamplingRng::default(),
        })
    }
}
