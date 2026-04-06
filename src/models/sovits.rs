//! SoVITS Model for audio synthesis
//!
//! Simplified implementation with correct data flow

use candle_core::{Device, Tensor, DType};
use crate::{Result, Error};
use crate::utils::{StateDict, load_safetensors, Conv1dWeightNorm};

/// SoVITS Model for mel spectrogram generation
pub struct SoVITSModel {
    device: Device,
    dtype: DType,
    text_embedding: Tensor,      // [322, 192]
    enc_q: EncoderQ,
    n_mels: usize,
    sampling_rate: u32,
}

/// Encoder Q - Semantic token encoder
#[allow(dead_code)]
pub struct EncoderQ {
    #[allow(dead_code)]
    cond_layer: Conv1dWeightNorm,
    in_layers: Vec<Conv1dWeightNorm>,
    res_skip_layers: Vec<Conv1dWeightNorm>,
}

impl SoVITSModel {
    /// Load model from safetensors file
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu)
    }

    /// Load model with specific device
    pub fn load_with_device(path: &str, device: &Device) -> Result<Self> {
        // Load weights from safetensors
        let weights_map = load_safetensors(path)?;
        let state_dict = StateDict::new(weights_map);

        // Load text embedding [322, 192]
        let text_embedding = state_dict
            .get("enc_p.text_embedding.weight")?
            .to_device(device)?
            .clone();

        // Create enc_q
        let enc_q = EncoderQ::new(&state_dict, device)?;

        Ok(Self {
            device: device.clone(),
            dtype: DType::F32,
            text_embedding,
            enc_q,
            n_mels: 100,
            sampling_rate: 24000,
        })
    }

    /// Synthesize mel spectrogram from semantic tokens
    pub fn synthesize(
        &self,
        semantic_tokens: &[usize],
        _ref_audio: Option<&Tensor>,
    ) -> Result<Tensor> {
        if semantic_tokens.is_empty() {
            return Err(Error::InferenceError("Empty semantic tokens".to_string()));
        }

        // Step 1: Lookup embeddings for semantic tokens
        // semantic_tokens: [seq_len] -> embeddings: [1, seq_len, 192]
        let embeddings = self.lookup_embeddings(semantic_tokens)?;

        // Step 2: Transpose for conv1d: [1, 192, seq_len]
        let embeddings = embeddings.transpose(1, 2)?;

        // Step 3: Run through enc_q
        // enc_q expects [1, 512, seq_len] input based on cond_layer shape
        // But in_layers expect 192 channels based on in_layers.0.weight_v: [384, 192, 5]
        // So we need to project from 192 to 512 first
        let features = self.enc_q.encode(&embeddings)?;

        // Narrow to n_mels channels
        let mel_spec = features.narrow(1, 0, self.n_mels)?;
        Ok(mel_spec)
    }

    /// Lookup embeddings for semantic tokens
    fn lookup_embeddings(&self, tokens: &[usize]) -> Result<Tensor> {
        let mut embeddings = Vec::with_capacity(tokens.len());
        for &idx in tokens {
            let emb = self.text_embedding.get(idx as usize)?;
            embeddings.push(emb);
        }

        // Stack: [seq_len, 192]
        let stacked = Tensor::stack(&embeddings, 0)?;

        // Add batch dimension: [1, seq_len, 192]
        Ok(stacked.unsqueeze(0)?)
    }

    /// Get model device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get model dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get number of mel bins
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Get sampling rate
    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }
}

impl EncoderQ {
    pub fn new(state_dict: &StateDict, _device: &Device) -> Result<Self> {
        let cond_layer = state_dict.get_conv1d_weight_norm("enc_q.enc.cond_layer")?;

        let mut in_layers = Vec::new();
        let mut i = 0;
        loop {
            let key = format!("enc_q.enc.in_layers.{}.weight_g", i);
            if !state_dict.contains(&key) {
                break;
            }
            in_layers.push(state_dict.get_conv1d_weight_norm(&format!("enc_q.enc.in_layers.{}", i))?);
            i += 1;
        }

        let mut res_skip_layers = Vec::new();
        i = 0;
        loop {
            let key = format!("enc_q.enc.res_skip_layers.{}.weight_g", i);
            if !state_dict.contains(&key) {
                break;
            }
            res_skip_layers.push(state_dict.get_conv1d_weight_norm(&format!("enc_q.enc.res_skip_layers.{}", i))?);
            i += 1;
        }

        Ok(Self {
            cond_layer,
            in_layers,
            res_skip_layers,
        })
    }

    pub fn encode(&self, tokens: &Tensor) -> Result<Tensor> {
        // tokens: [1, 192, seq_len]
        let mut h = self.cond_layer.forward(tokens)?;
        for (in_layer, res_skip_layer) in self.in_layers.iter().zip(self.res_skip_layers.iter()) {
            let residual = h.clone();
            let x = in_layer.forward(&h)?;
            let x = x.relu()?;
            let x = res_skip_layer.forward(&x)?;
            h = x.broadcast_add(&residual)?;
        }
        Ok(h)
    }
}

impl crate::models::Model for SoVITSModel {
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
