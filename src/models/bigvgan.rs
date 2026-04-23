//! BigVGAN Neural Vocoder - Simplified Implementation
//!
//! This implementation focuses on the core components:
//! - Snake activation function with learnable alpha/beta
//! - Proper residual connections
//! - Downsampling filters for multi-period enhancement

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
    activation_post: PostActivation,
}

/// Upsampling layer
struct Upsample {
    conv: Conv1dWeightNorm,
    #[allow(dead_code)]
    upsample_factor: usize,
}

/// Residual stack with Snake activation
struct ResidualBlock {
    convs1: Vec<Conv1dWeightNorm>,
    convs2: Vec<Conv1dWeightNorm>,
    activations: Vec<Activation>,
}

/// Snake activation function: x + alpha * sin^2(beta * x)
struct SnakeParams {
    alpha: Tensor,
    beta: Tensor,
}

impl SnakeParams {
    fn new(alpha: Tensor, beta: Tensor) -> Self {
        Self { alpha, beta }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Snake(x) = x + alpha * sin^2(beta * x)
        let x_dims = x.dims();
        let alpha_dims = self.alpha.dims();
        let _beta_dims = self.beta.dims();

        // Reshape alpha/beta to match input dimensions
        // Input is [batch, channels, time]
        // alpha/beta can be [channels] or scalar-like
        let (alpha_reshaped, beta_reshaped) = if x_dims.len() == 3 {
            let _batch = x_dims[0];
            let channels = x_dims[1];
            let _time = x_dims[2];

            if alpha_dims.len() == 1 && alpha_dims[0] == channels {
                // alpha matches input channels, reshape to [1, channels, 1]
                (self.alpha.reshape((1, channels, 1))?,
                 self.beta.reshape((1, channels, 1))?)
            } else if alpha_dims.len() == 1 && alpha_dims[0] != channels {
                // alpha has different channels than input - use scalar (first element)
                // This handles the case where conv_post output is [1, 1, time] but alpha is [24]
                let alpha_scalar = self.alpha.narrow(0, 0, 1)?;
                let beta_scalar = self.beta.narrow(0, 0, 1)?;
                (alpha_scalar.reshape((1, 1, 1))?,
                 beta_scalar.reshape((1, 1, 1))?)
            } else {
                // alpha is already correct shape
                (self.alpha.clone(), self.beta.clone())
            }
        } else {
            // For other shapes, use scalar
            let alpha_scalar = self.alpha.narrow(0, 0, 1)?;
            let beta_scalar = self.beta.narrow(0, 0, 1)?;
            (alpha_scalar.reshape((1, 1, 1))?,
             beta_scalar.reshape((1, 1, 1))?)
        };

        let beta_x = x.broadcast_mul(&beta_reshaped)?;
        let sin_beta_x = beta_x.sin()?;
        let sin_squared = sin_beta_x.broadcast_mul(&sin_beta_x)?;
        let alpha_sin_squared = sin_squared.broadcast_mul(&alpha_reshaped)?;
        Ok(x.broadcast_add(&alpha_sin_squared)?)
    }
}

/// Activation module with downsampler
struct Activation {
    downsample: Option<Downsample>,
    snake: SnakeParams,
}

/// Downsampler for multi-period enhancement
struct Downsample {
    lowpass: LowpassFilter,
}

impl Downsample {
    fn new(state_dict: &StateDict, act_prefix: &str, device: &Device) -> Result<Self> {
        let prefix = format!("{}.downsample", act_prefix);

        // Load lowpass filter
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
        self.lowpass.apply(x)
    }
}

/// Post-activation module
struct PostActivation {
    downsample: Option<Downsample>,
    upsample: Option<UpsampleFilter>,
    snake: SnakeParams,
}

/// Upsample filter for post-activation
struct UpsampleFilter {
    filter: Tensor,
}

impl UpsampleFilter {
    fn new(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let filter_data = state_dict
            .get(&format!("{}.upsample.filter", prefix))?
            .to_device(device)?
            .to_dtype(DType::F32)?
            .clone();

        Ok(Self { filter: filter_data })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Upsample by 2x using nearest neighbor interpolation, then apply filter
        let x_upsampled = self.nearest_upsample(x, 2)?;
        self.apply_filter(&x_upsampled)
    }

    fn nearest_upsample(&self, x: &Tensor, factor: usize) -> Result<Tensor> {
        let (batch, channels, time) = (x.dims()[0], x.dims()[1], x.dims()[2]);

        let mut samples = Vec::new();
        let x_vec: Vec<f32> = x.flatten_all()?.to_vec1()?;
        for &val in &x_vec {
            for _ in 0..factor {
                samples.push(val);
            }
        }

        Ok(Tensor::from_vec(samples, (batch, channels, time * factor), x.device())?)
    }

    fn apply_filter(&self, x: &Tensor) -> Result<Tensor> {
        let filter_dims = self.filter.dims();
        let filter_len = filter_dims[2];
        let channels = x.dims()[1];

        let pad_left = filter_len / 2;
        let pad_right = filter_len - 1 - pad_left;
        let x_padded = x.pad_with_zeros(2, pad_left, pad_right)?;

        let filter_expanded = if channels > 1 {
            self.filter.broadcast_as((channels, 1, filter_len))?
        } else {
            self.filter.clone()
        };

        Ok(x_padded.conv1d(&filter_expanded, 0, 1, 1, channels)?)
    }
}

impl ResidualBlock {
    /// Load all 18 residual blocks from state dict
    fn load_all(state_dict: &StateDict, device: &Device) -> Result<Vec<Self>> {
        let mut blocks = Vec::new();

        for i in 0..18 {
            let prefix = format!("resblocks.{}", i);

            if !state_dict.contains(&format!("{}.convs1.0.weight_v", prefix)) {
                continue;
            }

            let mut convs1 = Vec::new();
            let mut convs2 = Vec::new();
            let mut activations = Vec::new();

            // Each resblock has 3 conv pairs and 6 activation layers
            // But we simplify: apply convs first, then activations with downsampling only
            for j in 0..3 {
                let conv1_prefix = format!("{}.convs1.{}", prefix, j);
                let conv2_prefix = format!("{}.convs2.{}", prefix, j);

                if state_dict.contains(&format!("{}.weight_v", conv1_prefix)) {
                    convs1.push(Self::load_conv(state_dict, &conv1_prefix, device)?);
                }
                if state_dict.contains(&format!("{}.weight_v", conv2_prefix)) {
                    convs2.push(Self::load_conv(state_dict, &conv2_prefix, device)?);
                }
            }

            // Load activation parameters - use this block's first activation's params
            let alpha_key = format!("resblocks.{}.activations.0.act.alpha", i);
            let beta_key = format!("resblocks.{}.activations.0.act.beta", i);

            let alpha = state_dict.get(&alpha_key)?.to_device(device)?.to_dtype(DType::F32)?;
            let beta = state_dict.get(&beta_key)?.to_device(device)?.to_dtype(DType::F32)?;

            // Create 6 activations but only use downsampling (no upsampling inside resblock)
            for j in 0..6 {
                let act_prefix = format!("{}.activations.{}", prefix, j);
                let has_downsample = state_dict.contains(&format!(
                    "{}.downsample.lowpass.filter", act_prefix
                ));

                activations.push(Activation {
                    downsample: if has_downsample {
                        Some(Downsample::new(state_dict, &act_prefix, device)?)
                    } else {
                        None
                    },
                    snake: SnakeParams::new(alpha.clone(), beta.clone()),
                });
            }

            blocks.push(ResidualBlock {
                convs1,
                convs2,
                activations,
            });
        }

        Ok(blocks)
    }

    fn load_conv(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Conv1dWeightNorm> {
        let weight_g = state_dict.get(&format!("{}.weight_g", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
        let weight_v = state_dict.get(&format!("{}.weight_v", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
        let bias = state_dict.get(&format!("{}.bias", prefix)).ok().cloned()
            .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
            .transpose()?;

        let weight_v_shape = weight_v.dims();
        let kernel_size = if weight_v_shape.len() >= 3 {
            weight_v_shape[2]
        } else {
            1
        };

        let padding = (kernel_size - 1) / 2;

        Ok(Conv1dWeightNorm::new_with_cached(weight_g, weight_v, bias, 1, padding, 1)?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_dims = x.dims().to_vec();
        let mut x_out = x.clone();

        // Simplified forward: apply conv1 -> activation -> conv2 for each of 3 iterations
        for i in 0..3 {
            let act_base = i * 2;

            // Conv1
            x_out = self.convs1[i].forward(&x_out)?;

            // Activation (Snake) with downsampling
            x_out = self.activations[act_base].snake.forward(&x_out)?;
            if let Some(downsample) = &self.activations[act_base].downsample {
                x_out = downsample.forward(&x_out)?;
            }

            // Conv2
            let mut conv2_out = self.convs2[i].forward(&x_out)?;

            // Activation (Snake)
            conv2_out = self.activations[act_base + 1].snake.forward(&conv2_out)?;

            // Add residual
            x_out = x_out.broadcast_add(&conv2_out)?;
        }

        // Final residual connection - only if shapes match
        if x_out.dims().to_vec() == original_dims {
            Ok(x.broadcast_add(&x_out)?)
        } else {
            Ok(x_out)
        }
    }
}

/// Lowpass filter coefficients
struct LowpassFilter {
    filter: Tensor,
}

impl LowpassFilter {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let filter_dims = self.filter.dims();
        let filter_len = filter_dims[2];
        let channels = x.dims()[1];

        let pad_left = filter_len / 2;
        let pad_right = filter_len - 1 - pad_left;
        let x_padded = x.pad_with_zeros(2, pad_left, pad_right)?;

        let filter_expanded = if channels > 1 {
            self.filter.broadcast_as((channels, 1, filter_len))?
        } else {
            self.filter.clone()
        };

        Ok(x_padded.conv1d(&filter_expanded, 0, 1, 1, channels)?)
    }
}

impl BigVGAN {
    /// Load BigVGAN model from safetensors file
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu)
    }

    /// Load model with specific device
    pub fn load_with_device(path: &str, device: &Device) -> Result<Self> {
        let weights_map = load_safetensors(path)?;
        let state_dict = StateDict::new(weights_map);

        let sampling_rate = 24000;
        let hop_length = 256;

        // Load input projection
        let conv_pre = Self::load_conv(&state_dict, "conv_pre", device)?;

        // Load upsampling layers
        let mut ups = Vec::new();
        for i in 0..6 {
            let prefix = format!("ups.{}.0", i);
            if state_dict.contains(&format!("{}.weight_v", prefix)) {
                let conv = Self::load_conv(&state_dict, &prefix, device)?;
                ups.push(Upsample {
                    conv,
                    upsample_factor: 2,
                });
            }
        }

        // Load residual stack
        let resblocks = ResidualBlock::load_all(&state_dict, device)?;

        // Load output projection
        let conv_post = Self::load_conv(&state_dict, "conv_post", device)?;

        // Load post-activation
        let activation_post = Self::load_post_activation(&state_dict, "activation_post", device)?;

        Ok(Self {
            device: device.clone(),
            dtype: DType::F32,
            sampling_rate,
            hop_length,
            conv_pre,
            ups,
            resblocks,
            conv_post,
            activation_post,
        })
    }

    fn load_conv(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Conv1dWeightNorm> {
        let weight_g = state_dict.get(&format!("{}.weight_g", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
        let weight_v = state_dict.get(&format!("{}.weight_v", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
        let bias = state_dict.get(&format!("{}.bias", prefix)).ok().cloned()
            .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
            .transpose()?;

        let weight_v_shape = weight_v.dims();
        let kernel_size = if weight_v_shape.len() >= 3 {
            weight_v_shape[2]
        } else {
            1
        };

        let padding = (kernel_size - 1) / 2;

        Ok(Conv1dWeightNorm::new_with_cached(weight_g, weight_v, bias, 1, padding, 1)?)
    }

    fn load_post_activation(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<PostActivation> {
        let has_downsample = state_dict.contains(&format!("{}.downsample.lowpass.filter", prefix));
        let has_upsample = state_dict.contains(&format!("{}.upsample.filter", prefix));

        let alpha = state_dict.get(&format!("{}.act.alpha", prefix))?.to_device(device)?.to_dtype(DType::F32)?;
        let beta = state_dict.get(&format!("{}.act.beta", prefix))?.to_device(device)?.to_dtype(DType::F32)?;

        Ok(PostActivation {
            downsample: if has_downsample {
                Some(Downsample::new(state_dict, prefix, device)?)
            } else {
                None
            },
            upsample: if has_upsample {
                Some(UpsampleFilter::new(state_dict, prefix, device)?)
            } else {
                None
            },
            snake: SnakeParams::new(alpha, beta),
        })
    }

    /// Generate waveform from mel spectrogram
    pub fn generate(&self, mel_spec: &Tensor) -> Result<Vec<f32>> {
        let dims = mel_spec.dims();
        if dims.len() != 3 {
            return Err(Error::InferenceError(
                format!("Expected 3D mel spectrogram [batch, n_mels, time], got {:?}", dims)
            ));
        }

        // Step 1: Input projection
        let mut x = self.conv_pre.forward(mel_spec)?;

        // Step 2: Interleaved upsampling and resblocks
        for (i, up) in self.ups.iter().enumerate() {
            x = self.upsample_forward(&x, &up.conv, up.upsample_factor)?;

            let resblock_start = i * 3;
            let resblock_end = resblock_start + 3;

            for block in &self.resblocks[resblock_start..resblock_end] {
                x = block.forward(&x)?;
            }
        }

        // Step 3: Output projection
        x = self.conv_post.forward(&x)?;

        // Step 4: Apply post-activation (Snake activation only)
        if let Some(downsample) = &self.activation_post.downsample {
            x = downsample.forward(&x)?;
        }
        x = self.activation_post.snake.forward(&x)?;

        // Step 5: Apply tanh activation
        x = x.tanh()?;

        // Convert to Vec<f32>
        let output: Vec<f32> = x.flatten_all()?.to_vec1()?;

        Ok(output)
    }

    fn upsample_forward(&self, x: &Tensor, conv: &Conv1dWeightNorm, upsample_factor: usize) -> Result<Tensor> {
        let weight = conv.get_weight()?;
        let weight_dims = weight.dims();

        let kernel_size = weight_dims[2];

        // For transposed convolution upsampling:
        // stride = upsample_factor (typically 2)
        // padding = (kernel_size - stride) / 2 to maintain alignment
        let stride = upsample_factor;
        let padding = (kernel_size - stride) / 2;

        Ok(x.conv_transpose1d(&weight, padding, 0, stride, 1, 1)?)
    }

    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }

    pub fn device(&self) -> &Device {
        &self.device
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
