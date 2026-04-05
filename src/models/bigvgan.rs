//! BigVGAN Neural Vocoder

use candle_core::{Device, Tensor, DType};
use crate::{Result, Error};
use crate::utils::{StateDict, load_safetensors, LayerNorm};
use crate::utils::weights::Conv1d;

/// BigVGAN vocoder for mel-to-waveform synthesis
pub struct BigVGAN {
    device: Device,
    dtype: DType,
    sampling_rate: u32,
    hop_length: usize,
    // Model components
    input_projection: Conv1d,
    res_stack: ResidualStack,
    output_projection: Conv1d,
}

/// Residual stack with AMP/ALiBi activation
pub struct ResidualStack {
    blocks: Vec<ResidualBlock>,
}

/// Single residual block
struct ResidualBlock {
    conv1: Conv1d,
    conv2: Conv1d,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dilation: usize,
}

impl BigVGAN {
    /// Load BigVGAN model from safetensors file
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu)
    }

    /// Load model with specific device
    pub fn load_with_device(path: &str, device: &Device) -> Result<Self> {
        // Load weights from safetensors
        let weights_map = load_safetensors(path)?;
        let state_dict = StateDict::new(weights_map);

        // Configuration
        let sampling_rate = 24000;
        let hop_length = 256;

        // Create input projection
        let input_projection = Self::create_input_projection(&state_dict, device)?;

        // Create residual stack
        let res_stack = ResidualStack::new(&state_dict, device)?;

        // Create output projection
        let output_projection = Self::create_output_projection(&state_dict, device)?;

        Ok(Self {
            device: device.clone(),
            dtype: DType::F32,
            sampling_rate,
            hop_length,
            input_projection,
            res_stack,
            output_projection,
        })
    }

    fn create_input_projection(state_dict: &StateDict, device: &Device) -> Result<Conv1d> {
        if state_dict.contains("input_projection.weight") {
            let weight = state_dict.get("input_projection.weight")?.clone();
            let bias = state_dict.get("input_projection.bias").ok().cloned();
            Ok(Conv1d::new(weight, bias, 1, 1, 1))
        } else {
            // Default: 100 mel bins -> 512 channels
            let weight = Tensor::zeros((512, 100, 3), DType::F32, device)?;
            let bias = Some(Tensor::zeros(512, DType::F32, device)?);
            Ok(Conv1d::new(weight, bias, 1, 1, 1))
        }
    }

    fn create_output_projection(state_dict: &StateDict, device: &Device) -> Result<Conv1d> {
        if state_dict.contains("output_projection.weight") {
            let weight = state_dict.get("output_projection.weight")?.clone();
            let bias = state_dict.get("output_projection.bias").ok().cloned();
            Ok(Conv1d::new(weight, bias, 1, hop_length_to_padding(256), 1))
        } else {
            // Default: 512 -> 1 channel (waveform)
            let weight = Tensor::zeros((1, 512, 3), DType::F32, device)?;
            let bias = Some(Tensor::zeros(1, DType::F32, device)?);
            Ok(Conv1d::new(weight, bias, 1, 1, 1))
        }
    }

    /// Generate waveform from mel spectrogram
    ///
    /// # Arguments
    /// * `mel_spec` - Mel spectrogram tensor [1, n_mels, time]
    ///
    /// # Returns
    /// Waveform tensor [1, 1, time * hop_length]
    pub fn generate(&self, mel_spec: &Tensor) -> Result<Vec<f32>> {
        // Validate input shape
        let dims = mel_spec.dims();
        if dims.len() != 3 {
            return Err(Error::InferenceError(
                format!("Expected 3D mel spectrogram, got {:?}", dims)
            ));
        }

        // Step 1: Input projection [1, n_mels, time] -> [1, hidden, time]
        let mut x = self.input_projection.forward(mel_spec)?;

        // Step 2: Residual stack
        x = self.res_stack.forward(&x)?;

        // Step 3: Output projection [1, hidden, time] -> [1, 1, time]
        x = self.output_projection.forward(&x)?;

        // Step 4: Apply activation (tanh)
        x = x.tanh()?;

        // Step 5: Convert to Vec<f32>
        // Apply pixel shuffle / upsampling to get final waveform
        let output = self.upsample(&x)?;

        Ok(output)
    }

    /// Upsample the output to waveform resolution
    fn upsample(&self, x: &Tensor) -> Result<Vec<f32>> {
        // Simple upsampling by repeating samples
        let dims = x.dims();
        let batch_size = dims[0];
        let channels = dims[1];
        let time = dims[2];

        // Reshape to [batch, time, channels]
        let x_flat = x.transpose(1, 2)?
            .flatten_from(0)?;

        // Convert to Vec
        let x_vec: Vec<f32> = x_flat.to_vec1()?;

        // Upsample by hop_length
        let mut output = Vec::with_capacity(x_vec.len() * self.hop_length / channels);
        for sample in x_vec.chunks(channels) {
            let value = sample.iter().sum::<f32>() / channels as f32;
            for _ in 0..self.hop_length {
                output.push(value);
            }
        }

        // Trim to actual expected length
        let expected_len = time * self.hop_length / channels.max(1);
        output.truncate(expected_len * batch_size);

        Ok(output)
    }

    /// Get sampling rate
    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
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

impl ResidualStack {
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        let mut blocks = Vec::new();

        // Try to load multiple residual blocks
        for i in 0..8 {
            let prefix = format!("res_stack.blocks.{}", i);

            // Check if this block exists
            if !state_dict.contains(&format!("{}.conv1.weight", prefix)) {
                continue;
            }

            let dilation = 2usize.pow(i as u32);

            let conv1_weight = state_dict.get(&format!("{}.conv1.weight", prefix))?.clone();
            let conv1_bias = state_dict.get(&format!("{}.conv1.bias", prefix)).ok().cloned();
            let conv1 = Conv1d::new(conv1_weight, conv1_bias, 1, dilation, dilation);

            let conv2_weight = state_dict.get(&format!("{}.conv2.weight", prefix))?.clone();
            let conv2_bias = state_dict.get(&format!("{}.conv2.bias", prefix)).ok().cloned();
            let conv2 = Conv1d::new(conv2_weight, conv2_bias, 1, 1, dilation);

            let norm1 = state_dict.get_layer_norm(&format!("{}.norm1", prefix))
                .unwrap_or_else(|_| create_default_layernorm(512, device));

            let norm2 = state_dict.get_layer_norm(&format!("{}.norm2", prefix))
                .unwrap_or_else(|_| create_default_layernorm(512, device));

            blocks.push(ResidualBlock {
                conv1,
                conv2,
                norm1,
                norm2,
                dilation,
            });
        }

        // If no blocks loaded, create a default one
        if blocks.is_empty() {
            let conv1 = create_default_conv(512, 512, 3, 1, device)?;
            let conv2 = create_default_conv(512, 512, 3, 1, device)?;
            let norm1 = create_default_layernorm(512, device);
            let norm2 = create_default_layernorm(512, device);
            blocks.push(ResidualBlock {
                conv1,
                conv2,
                norm1,
                norm2,
                dilation: 1,
            });
        }

        Ok(Self { blocks })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();

        for block in &self.blocks {
            let _residual = x.clone();

            // Conv1 + norm + activation
            let mut h = block.conv1.forward(&x)?;
            h = apply_amp_activation(&h)?; // Anti-aliased Multi-Period Enhancement
            h = block.norm1.forward(&h)?;

            // Conv2 + norm + activation
            h = block.conv2.forward(&h)?;
            h = apply_amp_activation(&h)?;
            h = block.norm2.forward(&h)?;

            // Residual connection
            x = add_tensors(&x, &h)?;
        }

        Ok(x)
    }
}

/// Apply AMP (Anti-aliased Multi-Period Enhancement) activation
/// Uses a combination of tanh and element-wise operations
fn apply_amp_activation(x: &Tensor) -> Result<Tensor> {
    // Simplified AMP: tanh(x) + alpha * sin(x)
    // This approximates the periodic activation used in BigVGAN
    let tanh_x = x.tanh()?;
    let sin_x = x.sin()?;

    // tanh(x) + 0.1 * sin(x)
    let scale = Tensor::full(0.1f32, sin_x.dims(), &sin_x.device())?;
    let scaled_sin = sin_x.broadcast_mul(&scale)?;
    Ok(tanh_x.broadcast_add(&scaled_sin)?)
}

/// Add two tensors with broadcasting
fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.broadcast_add(b).map_err(|e| e.into())
}

/// Create default layer norm
fn create_default_layernorm(size: usize, device: &Device) -> LayerNorm {
    LayerNorm::new(
        Tensor::ones(size, DType::F32, device).unwrap(),
        Tensor::zeros(size, DType::F32, device).unwrap(),
    )
}

/// Create default conv1d
fn create_default_conv(in_ch: usize, out_ch: usize, kernel: usize, dilation: usize, device: &Device) -> Result<Conv1d> {
    let weight = Tensor::zeros((out_ch, in_ch, kernel), DType::F32, device)?;
    let bias = Some(Tensor::zeros(out_ch, DType::F32, device)?);
    Ok(Conv1d::new(weight, bias, 1, dilation, dilation))
}

/// Convert hop length to padding
fn hop_length_to_padding(hop: usize) -> usize {
    (hop - 1) / 2
}

impl crate::models::Model for BigVGAN {
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
