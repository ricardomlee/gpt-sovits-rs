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
    pre_layer: Conv1dWeightNorm,   // [192, 1025, 1] - projects semantic tokens to 192 dim
    enc_q: EncoderQ,
    n_mels: usize,
    sampling_rate: u32,
}

/// Encoder Q - Semantic token encoder
#[allow(dead_code)]
pub struct EncoderQ {
    proj: Conv1dWeightNorm,
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

        // Load pre layer: [192, 1025, 1] - regular conv, not weight-norm
        let pre_weight = state_dict.get("enc_q.pre.weight")?.to_device(device)?.to_dtype(DType::F32)?;
        let pre_bias = state_dict.get("enc_q.pre.bias")?.to_device(device)?.to_dtype(DType::F32)?;
        // Create Conv1dWeightNorm with dummy g/v for regular conv
        let weight_v = pre_weight;
        let weight_g = Tensor::full(1.0f32, weight_v.dims(), &weight_v.device())?;
        let pre_layer = Conv1dWeightNorm::new(weight_g, weight_v, Some(pre_bias), 1, 0, 1);

        // Create enc_q
        let enc_q = EncoderQ::new(&state_dict, device)?;

        Ok(Self {
            device: device.clone(),
            dtype: DType::F32,
            pre_layer,
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

        // Convert semantic tokens to one-hot like tensor for conv1d
        let seq_len = semantic_tokens.len();
        let mut token_tensor = vec![0.0f32; 1025 * seq_len];
        for (i, &token) in semantic_tokens.iter().enumerate() {
            if token < 1025 {
                token_tensor[token * seq_len + i] = 1.0;
            }
        }
        let token_tensor = Tensor::from_vec(token_tensor, (1, 1025, seq_len), &self.device)?;

        // Step 1: Project through pre layer
        let embeddings = self.pre_layer.forward(&token_tensor)?;

        // Step 2: Run through enc_q
        let features = self.enc_q.encode(&embeddings)?;

        // Narrow to n_mels channels
        let mel_spec = features.narrow(1, 0, self.n_mels)?;

        Ok(mel_spec)
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
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        // Load projection layer: [384, 192, 1] - projects from 192 to 384
        // This is a regular conv, not weight-norm, so we need to create dummy g/v
        let proj_weight = state_dict
            .get("enc_q.proj.weight")?
            .to_device(device)?
            .to_dtype(DType::F32)?;
        let proj_bias = state_dict
            .get("enc_q.proj.bias")?
            .to_device(device)?
            .to_dtype(DType::F32)?;
        let weight_v = proj_weight.clone();
        let weight_g = Tensor::full(1.0f32, weight_v.dims(), &weight_v.device())?;
        let proj = Conv1dWeightNorm::new(weight_g, weight_v, Some(proj_bias), 1, 0, 1);

        // Load all layers with explicit F32 conversion to avoid dtype mismatch
        let cond_layer = Self::load_conv1d_weight_norm(state_dict, "enc_q.enc.cond_layer", device)?;

        let mut in_layers = Vec::new();
        let mut i = 0;
        loop {
            let key = format!("enc_q.enc.in_layers.{}.weight_g", i);
            if !state_dict.contains(&key) {
                break;
            }
            in_layers.push(Self::load_conv1d_weight_norm(
                state_dict,
                &format!("enc_q.enc.in_layers.{}", i),
                device,
            )?);
            i += 1;
        }

        let mut res_skip_layers = Vec::new();
        i = 0;
        loop {
            let key = format!("enc_q.enc.res_skip_layers.{}.weight_g", i);
            if !state_dict.contains(&key) {
                break;
            }
            res_skip_layers.push(Self::load_conv1d_weight_norm(
                state_dict,
                &format!("enc_q.enc.res_skip_layers.{}", i),
                device,
            )?);
            i += 1;
        }

        Ok(Self {
            proj,
            cond_layer,
            in_layers,
            res_skip_layers,
        })
    }

    fn load_conv1d_weight_norm(
        state_dict: &StateDict,
        prefix: &str,
        device: &Device,
    ) -> Result<Conv1dWeightNorm> {
        let weight_g = state_dict
            .get(&format!("{}.weight_g", prefix))?
            .to_device(device)?
            .to_dtype(DType::F32)?;
        let weight_v = state_dict
            .get(&format!("{}.weight_v", prefix))?
            .to_device(device)?
            .to_dtype(DType::F32)?;
        let bias = if state_dict.contains(&format!("{}.bias", prefix)) {
            Some(
                state_dict
                    .get(&format!("{}.bias", prefix))?
                    .to_device(device)?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };

        // Infer kernel size from weight_v shape [out_channels, in_channels, kernel_size]
        let weight_v_shape = weight_v.dims();
        let kernel_size = if weight_v_shape.len() >= 3 {
            weight_v_shape[2]
        } else {
            1
        };
        // Calculate padding to maintain sequence length: padding = (kernel_size - 1) / 2
        let padding = (kernel_size - 1) / 2;

        let stride = 1;
        let dilation = 1;
        Ok(Conv1dWeightNorm::new(weight_g, weight_v, bias, stride, padding, dilation))
    }

    pub fn encode(&self, tokens: &Tensor) -> Result<Tensor> {
        // tokens: [1, 192, seq_len] from pre layer
        // Modified WaveNet: hidden state stays at 192 dims
        // in_layer: 192 -> 384, then take first 192 dims for next layer
        // res_skip: accumulates to output
        let mut h = tokens.clone();
        let mut output: Option<Tensor> = None;

        for (in_layer, res_skip_layer) in self.in_layers.iter().zip(self.res_skip_layers.iter()) {
            let residual = h.clone();

            // in_layer: 192 -> 384
            let x = in_layer.forward(&h)?;
            let x = x.relu()?;

            // Take first 192 channels for next hidden state
            h = x.narrow(1, 0, 192)?;

            // res_skip: 192 -> 384 or 192
            let res_skip = res_skip_layer.forward(&residual)?;

            // Add to output accumulator
            output = Some(match output {
                Some(o) if o.dims()[1] == res_skip.dims()[1] => o.broadcast_add(&res_skip)?,
                Some(o) => o, // Skip if shape mismatch (last layer)
                None => res_skip,
            });
        }

        // Return sum of skip connections
        output.ok_or_else(|| Error::InferenceError("No output".to_string()))
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
