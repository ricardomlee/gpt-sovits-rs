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
    res_stack: ResidualStack,
    conv_post: Conv1dWeightNorm,
}

/// Upsampling layer
struct Upsample {
    conv: Conv1dWeightNorm,
    #[allow(dead_code)]
    upsample_factor: usize,
}

/// Residual stack with AMP/ALiBi activation
struct ResidualStack {
    blocks: Vec<ResidualBlock>,
}

/// Single residual block
struct ResidualBlock {
    convs1: Vec<Conv1dWeightNorm>,
    convs2: Vec<Conv1dWeightNorm>,
    activations: Vec<Activation>,
}

/// Activation module with downsampler
struct Activation {
    downsample: Option<Downsample>,
    #[allow(dead_code)]
    act: ActivationFn,
}

/// Downsampler for multi-period enhancement
struct Downsample {
    conv: Conv1dWeightNorm,
    lowpass: LowpassFilter,
}

/// Lowpass filter coefficients
struct LowpassFilter {
    filter: Tensor,
}

/// Activation function types
enum ActivationFn {
    #[allow(dead_code)]
    LeakyReLU { alpha: f32 },
    #[allow(dead_code)]
    Snake { alpha: Tensor },
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

        // Load upsampling layers (ups.0 to ups.5)
        let mut ups = Vec::new();
        for i in 0..6 {
            let prefix = format!("ups.{}", i);
            if state_dict.contains(&format!("{}.weight_v", prefix)) {
                let conv = Self::load_conv(&state_dict, &prefix, device)?;
                // Upsample factors: 8, 8, 4, 4, 4, 4 = 8192 total (but we need 256)
                // Actually the kernel stride handles this
                ups.push(Upsample {
                    conv,
                    upsample_factor: if i < 2 { 8 } else { 4 },
                });
            }
        }

        // Load residual stack (resblocks.0 to resblocks.17)
        let res_stack = ResidualStack::new(&state_dict, device)?;

        // Load output projection (conv_post): 24 -> 1 channel
        let conv_post = Self::load_conv(&state_dict, "conv_post", device)?;

        Ok(Self {
            device: device.clone(),
            dtype: DType::F32,
            sampling_rate,
            hop_length,
            conv_pre,
            ups,
            res_stack,
            conv_post,
        })
    }

    fn load_conv(state_dict: &StateDict, prefix: &str, _device: &Device) -> Result<Conv1dWeightNorm> {
        let weight_g = state_dict.get(&format!("{}.weight_g", prefix))?.clone();
        let weight_v = state_dict.get(&format!("{}.weight_v", prefix))?.clone();
        let bias = state_dict.get(&format!("{}.bias", prefix)).ok().cloned();

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

        // Step 2: Upsampling layers
        // Each upsampling layer increases temporal resolution
        for up in &self.ups {
            x = self.upsample_forward(&x, &up.conv)?;
        }

        // Step 3: Residual stack
        x = self.res_stack.forward(&x)?;

        // Step 4: Output projection [batch, 24, time] -> [batch, 1, time]
        x = self.conv_post.forward(&x)?;

        // Step 5: Apply tanh activation
        x = x.tanh()?;

        // Convert to Vec<f32>
        let output: Vec<f32> = x.flatten_all()?.to_vec1()?;

        Ok(output)
    }

    /// Forward pass through upsampling layer with pixel shuffle
    fn upsample_forward(&self, x: &Tensor, conv: &Conv1dWeightNorm) -> Result<Tensor> {
        // Get weight and apply upsampling via transposed convolution semantics
        // The weight shape tells us the expansion factor
        let weight = conv.get_weight()?;
        let weight_dims = weight.dims();

        // weight: [out_channels, in_channels, kernel]
        // For upsampling: out_channels = in_channels * stride
        let in_channels = weight_dims[1];
        let out_channels = weight_dims[0];
        let stride = out_channels / in_channels;

        // Use pixel shuffle approach:
        // 1. Apply convolution without stride
        // 2. Reshape to interleave channels into time
        let x = conv.forward(x)?;

        // Pixel shuffle: reshape [batch, stride*channels, time] -> [batch, channels, time*stride]
        if stride > 1 {
            let dims = x.dims();
            let batch = dims[0];
            let new_channels = dims[1] / stride;
            let time = dims[2];

            // Reshape to [batch, new_channels, stride, time]
            let x = x.reshape((batch, new_channels, stride, time))?;

            // Transpose to [batch, new_channels, time, stride]
            let x = x.transpose(1, 2)?.transpose(2, 3)?;

            // Reshape to [batch, new_channels, time*stride]
            Ok(x.reshape((batch, new_channels, time * stride))?)
        } else {
            Ok(x)
        }
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

impl ResidualStack {
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        let mut blocks = Vec::new();

        // Load 18 residual blocks (resblocks.0 to resblocks.17)
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
                    convs1.push(Self::load_block_conv(state_dict, &conv1_prefix, device)?);
                }
                if state_dict.contains(&format!("{}.weight_v", conv2_prefix)) {
                    convs2.push(Self::load_block_conv(state_dict, &conv2_prefix, device)?);
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

        Ok(Self { blocks })
    }

    fn load_block_conv(state_dict: &StateDict, prefix: &str, _device: &Device) -> Result<Conv1dWeightNorm> {
        let weight_g = state_dict.get(&format!("{}.weight_g", prefix))?.clone();
        let weight_v = state_dict.get(&format!("{}.weight_v", prefix))?.clone();
        let bias = state_dict.get(&format!("{}.bias", prefix)).ok().cloned();

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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();

        for block in &self.blocks {
            let residual = x.clone();

            // Process through convs1 and activations in sequence
            for (i, (conv1, conv2)) in block.convs1.iter().zip(block.convs2.iter()).enumerate() {
                // Conv1
                let mut h = conv1.forward(&x)?;

                // Activation (LeakyReLU)
                h = self.leaky_relu(&h, 0.1)?;

                // Optional downsampler
                if let Some(activation) = block.activations.get(i) {
                    if let Some(downsample) = &activation.downsample {
                        h = downsample.forward(&h)?;
                    }
                }

                // Conv2
                h = conv2.forward(&h)?;

                // Activation (LeakyReLU)
                h = self.leaky_relu(&h, 0.1)?;

                // Add to input for this sub-layer
                x = x.broadcast_add(&h)?;
            }

            // Final residual connection
            x = x.broadcast_add(&residual)?;
        }

        Ok(x)
    }

    fn leaky_relu(&self, x: &Tensor, alpha: f32) -> Result<Tensor> {
        // LeakyReLU: max(x, alpha * x)
        let zeros = Tensor::zeros_like(x)?;
        let scaled = x.broadcast_mul(&Tensor::full(alpha, x.dims(), &x.device())?)?;
        Ok(x.broadcast_gt(&zeros)?.where_cond(&x, &scaled)?)
    }
}

impl Downsample {
    fn new(state_dict: &StateDict, act_prefix: &str, device: &Device) -> Result<Self> {
        let prefix = format!("{}.downsample", act_prefix);

        // Load convolution
        let conv = ResidualStack::load_block_conv(state_dict, &prefix, device)?;

        // Load lowpass filter
        let filter_data = state_dict.get(&format!("{}.lowpass.filter", prefix))?.clone();

        Ok(Self {
            conv,
            lowpass: LowpassFilter { filter: filter_data },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply lowpass filter then convolution
        let x = self.lowpass.apply(x)?;
        self.conv.forward(&x)
    }
}

impl LowpassFilter {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        // Apply 1D convolution with the lowpass filter
        // Filter shape: [1, 1, filter_len]
        let filter_dims = self.filter.dims();
        let filter_len = filter_dims[2];

        // Padding for causal filtering
        let padding = filter_len / 2;

        // Group convolution with filter
        Ok(x.conv1d(&self.filter, padding, 1, 1, x.dims()[1])?)
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
