//! BigVGAN Neural Vocoder

use candle_core::{Device, Tensor, DType};
use crate::{Result, Error};
use crate::utils::{StateDict, load_safetensors};
use crate::utils::weights::Conv1dWeightNorm;

/// BigVGAN vocoder for mel-to-waveform synthesis
pub struct BigVGAN {
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
    sampling_rate: u32,
    #[allow(dead_code)]
    hop_length: usize,
    // Model components
    conv_pre: Conv1dWeightNorm,
    ups: Vec<Upsample>,
    resblocks: Vec<ResidualBlock>,
    conv_post: Conv1dWeightNorm,
}

/// Upsampling layer
struct Upsample {
    conv: Conv1dWeightNorm,
    #[allow(dead_code)]
    upsample_factor: usize,
}

/// Residual stack with AMP/ALiBi activation
struct ResidualBlock {
    convs1: Vec<Conv1dWeightNorm>,
    convs2: Vec<Conv1dWeightNorm>,
    activations: Vec<Activation>,
}

impl ResidualBlock {
    /// Load all 18 residual blocks from state dict
    fn load_all(state_dict: &StateDict, device: &Device) -> Result<Vec<Self>> {
        let mut blocks = Vec::new();

        for i in 0..18 {
            let prefix = format!("resblocks.{}", i);

            // Check if this block exists
            if !state_dict.contains(&format!("{}.convs1.0.weight_v", prefix)) {
                continue;
            }

            // Load multiple conv1 layers (convs1.0, convs1.1, convs1.2)
            let mut convs1 = Vec::new();
            let mut convs2 = Vec::new();
            let mut activations = Vec::new();

            for j in 0..3 {
                let conv1_prefix = format!("{}.convs1.{}", prefix, j);
                let conv2_prefix = format!("{}.convs2.{}", prefix, j);
                let act_prefix = format!("{}.activations.{}", prefix, j);

                if state_dict.contains(&format!("{}.weight_v", conv1_prefix)) {
                    convs1.push(Self::load_conv(state_dict, &conv1_prefix, device)?);
                }
                if state_dict.contains(&format!("{}.weight_v", conv2_prefix)) {
                    convs2.push(Self::load_conv(state_dict, &conv2_prefix, device)?);
                }

                // Check for activation with optional downsampler
                let has_downsample = state_dict.contains(&format!(
                    "{}.downsample.lowpass.filter",
                    format!("{}.activations.{}", prefix, j)
                ));

                activations.push(Activation {
                    downsample: if has_downsample {
                        Some(Downsample::new(state_dict, &act_prefix, device)?)
                    } else {
                        None
                    },
                    act: ActivationFn::LeakyReLU { alpha: 0.1 },
                });
            }

            if !convs1.is_empty() || !convs2.is_empty() {
                blocks.push(ResidualBlock {
                    convs1,
                    convs2,
                    activations,
                });
            }
        }

        Ok(blocks)
    }

    fn load_conv(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Conv1dWeightNorm> {
        let weight_g = state_dict.get(&format!("{}.weight_g", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
        let weight_v = state_dict.get(&format!("{}.weight_v", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
        let bias = state_dict.get(&format!("{}.bias", prefix)).ok().cloned()
            .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
            .transpose()?;

        // Get kernel size from weight_v shape [out, in, kernel]
        let weight_v_shape = weight_v.dims();
        let kernel_size = if weight_v_shape.len() >= 3 {
            weight_v_shape[2]
        } else {
            1
        };

        // Calculate padding
        let padding = (kernel_size - 1) / 2;

        Ok(Conv1dWeightNorm::new(weight_g, weight_v, bias, 1, padding, 1))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        let residual = x.clone();

        // Process through convs1 and activations in sequence
        for (i, (conv1, conv2)) in self.convs1.iter().zip(self.convs2.iter()).enumerate() {
            // Conv1
            let mut h = conv1.forward(&x)?;

            // Activation (LeakyReLU)
            h = Self::leaky_relu(&h, 0.1)?;

            // Optional downsampler
            if let Some(activation) = self.activations.get(i) {
                if let Some(downsample) = &activation.downsample {
                    h = downsample.forward(&h)?;
                }
            }

            // Conv2
            h = conv2.forward(&h)?;

            // Activation (LeakyReLU)
            h = Self::leaky_relu(&h, 0.1)?;

            // Add to input for this sub-layer
            x = x.broadcast_add(&h)?;
        }

        // Final residual connection
        x = x.broadcast_add(&residual)?;

        Ok(x)
    }

    fn leaky_relu(x: &Tensor, alpha: f32) -> Result<Tensor> {
        // LeakyReLU: max(x, alpha * x)
        let zeros = Tensor::zeros_like(x)?;
        let scaled = x.broadcast_mul(&Tensor::full(alpha, x.dims(), &x.device())?)?;
        Ok(x.broadcast_gt(&zeros)?.where_cond(&x, &scaled)?)
    }
}

/// Activation module with downsampler
struct Activation {
    downsample: Option<Downsample>,
    #[allow(dead_code)]
    act: ActivationFn,
}

/// Downsampler for multi-period enhancement
struct Downsample {
    lowpass: LowpassFilter,
}

impl Downsample {
    fn new(state_dict: &StateDict, act_prefix: &str, device: &Device) -> Result<Self> {
        let prefix = format!("{}.downsample", act_prefix);

        // Load lowpass filter and move to device
        // Note: This model variant only has lowpass filter, no convolution
        let filter_data = state_dict
            .get(&format!("{}.lowpass.filter", prefix))?
            .to_device(device)?
            .to_dtype(DType::F32)?
            .clone();

        Ok(Self {
            lowpass: LowpassFilter { filter: filter_data },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply lowpass filter only (no convolution in this model variant)
        self.lowpass.apply(x)
    }
}

/// Activation function types
enum ActivationFn {
    #[allow(dead_code)]
    LeakyReLU { alpha: f32 },
    #[allow(dead_code)]
    Snake { alpha: Tensor },
}

/// Lowpass filter coefficients
struct LowpassFilter {
    filter: Tensor,
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

        // Configuration - BigVGAN base config
        let sampling_rate = 24000;
        let hop_length = 256;
        let _num_mels = 100;

        // Load input projection (conv_pre): num_mels -> 1536 channels
        let conv_pre = Self::load_conv(&state_dict, "conv_pre", device)?;

        // Load upsampling layers with interleaved resblocks
        // Architecture: conv_pre → ups.0 → resblocks 0-2 → ups.1 → resblocks 3-5 → ... → conv_post
        let mut ups = Vec::new();
        for i in 0..6 {
            let prefix = format!("ups.{}.0", i);
            if state_dict.contains(&format!("{}.weight_v", prefix)) {
                let conv = Self::load_conv(&state_dict, &prefix, device)?;
                // Upsample factors from kernel stride
                let upsample_factor = if i < 2 { 8 } else { 4 };
                ups.push(Upsample {
                    conv,
                    upsample_factor,
                });
            }
        }

        // Load residual stack (resblocks.0 to resblocks.17)
        let resblocks = ResidualBlock::load_all(&state_dict, device)?;

        // Load output projection (conv_post): 24 -> 1 channel
        let conv_post = Self::load_conv(&state_dict, "conv_post", device)?;

        Ok(Self {
            device: device.clone(),
            dtype: DType::F32,
            sampling_rate,
            hop_length,
            conv_pre,
            ups,
            resblocks,
            conv_post,
        })
    }

    fn load_conv(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Conv1dWeightNorm> {
        let weight_g = state_dict.get(&format!("{}.weight_g", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
        let weight_v = state_dict.get(&format!("{}.weight_v", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
        let bias = state_dict.get(&format!("{}.bias", prefix)).ok().cloned()
            .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
            .transpose()?;

        // Get kernel size from weight_v shape
        let weight_v_shape = weight_v.dims();
        let kernel_size = if weight_v_shape.len() >= 3 {
            weight_v_shape[2]
        } else {
            1
        };

        // Calculate padding to maintain sequence length
        let padding = (kernel_size - 1) / 2;

        Ok(Conv1dWeightNorm::new(weight_g, weight_v, bias, 1, padding, 1))
    }

    /// Generate waveform from mel spectrogram
    ///
    /// # Arguments
    /// * `mel_spec` - Mel spectrogram tensor [batch, n_mels, time]
    ///
    /// # Returns
    /// Waveform as Vec<f32>
    pub fn generate(&self, mel_spec: &Tensor) -> Result<Vec<f32>> {
        // Validate input shape
        let dims = mel_spec.dims();
        if dims.len() != 3 {
            return Err(Error::InferenceError(
                format!("Expected 3D mel spectrogram [batch, n_mels, time], got {:?}", dims)
            ));
        }

        // Step 1: Input projection [batch, n_mels, time] -> [batch, 1536, time]
        let mut x = self.conv_pre.forward(mel_spec)?;

        // Step 2: Interleaved upsampling and resblocks
        // Architecture: ups.0 → resblocks 0-2 → ups.1 → resblocks 3-5 → ... → ups.5 → resblocks 15-17
        for (i, up) in self.ups.iter().enumerate() {
            // Upsample
            x = self.upsample_forward(&x, &up.conv)?;

            // Run corresponding resblock group (3 blocks per upsampling layer)
            let resblock_start = i * 3;
            let resblock_end = resblock_start + 3;

            for block in &self.resblocks[resblock_start..resblock_end] {
                x = block.forward(&x)?;
            }
        }

        // Step 3: Output projection [batch, 24, time] -> [batch, 1, time]
        x = self.conv_post.forward(&x)?;

        // Step 4: Apply tanh activation
        x = x.tanh()?;

        // Convert to Vec<f32>
        let output: Vec<f32> = x.flatten_all()?.to_vec1()?;

        Ok(output)
    }

    /// Forward pass through upsampling layer with transposed convolution
    fn upsample_forward(&self, x: &Tensor, conv: &Conv1dWeightNorm) -> Result<Tensor> {
        let weight = conv.get_weight()?;
        let weight_dims = weight.dims();

        let in_channels_w = weight_dims[0];
        let out_channels_w = weight_dims[1];
        let kernel_size = weight_dims[2];
        let stride = in_channels_w / out_channels_w;

        // For ConvTranspose1d, candle expects weight [in, out, kernel]
        // Our weight is already in correct format

        // For 'same' padding: output_length = input_length * stride
        let padding = (kernel_size - stride) / 2;

        // conv_transpose1d: input [N, in, L] -> output [N, out, L*stride]
        Ok(x.conv_transpose1d(&weight, padding, 0, stride, 1, 1)?)
    }

    /// Get sampling rate
    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }

    /// Get model device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl LowpassFilter {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        // Apply 1D convolution with the lowpass filter
        // Filter shape: [1, 1, filter_len]
        // Input shape: [batch, channels, time]
        // We need to apply the same 1D filter to each channel independently

        let filter_dims = self.filter.dims();
        let filter_len = filter_dims[2];
        let (channels, _time) = (x.dims()[1], x.dims()[2]);

        // For even-length filters, we need asymmetric padding to preserve sequence length
        // Use pad_left = filter_len / 2, pad_right = filter_len / 2 - 1
        let pad_left = filter_len / 2;
        let pad_right = filter_len - 1 - pad_left;

        // Manually pad the input
        let x_padded = x.pad_with_zeros(2, pad_left, pad_right)?;

        // Reshape filter to [channels, 1, filter_len] for depthwise convolution
        let filter_expanded = if channels > 1 {
            self.filter.broadcast_as((channels, 1, filter_len))?
        } else {
            self.filter.clone()
        };

        // Use grouped convolution with padding=0 since we manually padded
        Ok(x_padded.conv1d(&filter_expanded, 0, 1, 1, channels)?)
    }
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
