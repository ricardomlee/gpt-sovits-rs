//! SoVITS Decoder Module - HiFi-GAN style vocoder

use candle_core::{Device, DType, Tensor};
use crate::Result;
use crate::utils::{StateDict, Conv1dWeightNorm};

/// Residual Block Type 1 (for HiFi-GAN decoder)
#[derive(Debug, Clone)]
pub struct ResBlock1 {
    convs1: Vec<Conv1dWeightNorm>,
    convs2: Vec<Conv1dWeightNorm>,
}

impl ResBlock1 {
    pub fn load(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();

        // Load convs1 (dilated convolutions)
        for i in 0..3 {
            let key = format!("{}.convs1.{}.weight_v", prefix, i);
            if state_dict.contains(&key) {
                convs1.push(Self::load_conv(state_dict, &format!("{}.convs1.{}", prefix, i), device)?);
            }
        }

        // Load convs2
        for i in 0..3 {
            let key = format!("{}.convs2.{}.weight_v", prefix, i);
            if state_dict.contains(&key) {
                convs2.push(Self::load_conv(state_dict, &format!("{}.convs2.{}", prefix, i), device)?);
            }
        }

        Ok(Self { convs1, convs2 })
    }

    fn load_conv(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Conv1dWeightNorm> {
        let weight_g = state_dict.get(&format!("{}.weight_g", prefix))?
            .to_device(device)?.to_dtype(DType::F32)?;
        let weight_v = state_dict.get(&format!("{}.weight_v", prefix))?
            .to_device(device)?.to_dtype(DType::F32)?;
        let bias = state_dict.get(&format!("{}.bias", prefix))
            .ok()
            .cloned()
            .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
            .transpose()?;

        let weight_v_shape = weight_v.dims();
        let kernel_size = if weight_v_shape.len() >= 3 {
            weight_v_shape[2]
        } else {
            1
        };

        // Dilated convolutions use kernel_size as dilation
        let dilation = kernel_size;
        let padding = (kernel_size * dilation - dilation) / 2;

        Ok(Conv1dWeightNorm::new(weight_g, weight_v, bias, 1, padding, dilation))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x_out = x.clone();

        for i in 0..self.convs1.len() {
            let xt = x.relu()?;
            let mut xt = self.convs1[i].forward(&xt)?;
            xt = xt.relu()?;
            let xt = self.convs2[i].forward(&xt)?;
            x_out = x_out.add(&xt)?;
        }

        Ok(x_out)
    }
}

/// Decoder (HiFi-GAN style vocoder)
#[derive(Debug, Clone)]
pub struct Decoder {
    conv_pre: Conv1dWeightNorm,
    ups: Vec<Upsample>,
    resblocks: Vec<ResBlock1>,
    conv_post: Conv1dWeightNorm,
    cond: Option<Conv1dWeightNorm>,
    gin_channels: usize,
}

#[derive(Debug, Clone)]
struct Upsample {
    conv: Conv1dWeightNorm,
    upsample_factor: usize,
}

impl Decoder {
    /// Load decoder from SoVITS safetensors
    pub fn load(state_dict: &StateDict, device: &Device) -> Result<Self> {
        // Load conv_pre: [512, 192, 7]
        let conv_pre = Self::load_conv(state_dict, "dec.conv_pre", device)?;

        // Load conv_post: [1, 16, 7]
        let conv_post = Self::load_conv_weight(state_dict, "dec.conv_post", device)?;

        // Load condition layer if exists
        let cond = if state_dict.contains("dec.cond.weight_v") {
            Some(Self::load_conv(state_dict, "dec.cond", device)?)
        } else {
            None
        };

        // Load upsampling layers
        // GPT-SoVITS v1/v2 uses upsample_rates = [10, 8, 2, 2, 2] (total 640x)
        // These CANNOT be derived from weight shapes alone - they are training config.
        // The weight shapes are [in_ch, out_ch, kernel] where in_ch/out_ch = 2 for all layers,
        // but the actual stride values are [10, 8, 2, 2, 2] from the training config.
        const UPSAMPLE_RATES: [usize; 5] = [10, 8, 2, 2, 2];
        let mut ups = Vec::new();
        let mut up_idx = 0;
        for i in 0..10 {
            let prefix = format!("dec.ups.{}", i);
            if state_dict.contains(&format!("{}.weight_v", prefix)) {
                let conv = Self::load_conv(state_dict, &prefix, device)?;
                let upsample_factor = if up_idx < UPSAMPLE_RATES.len() {
                    UPSAMPLE_RATES[up_idx]
                } else {
                    2 // fallback
                };
                up_idx += 1;
                ups.push(Upsample { conv, upsample_factor });
            }
        }

        // Load resblock groups (15 resblocks total, using ResBlock1)
        let mut resblocks = Vec::new();
        for i in 0..15 {
            let prefix = format!("dec.resblocks.{}", i);
            if state_dict.contains(&format!("{}.convs1.0.weight_v", prefix)) {
                let block = ResBlock1::load(state_dict, &prefix, device)?;
                resblocks.push(block);
            }
        }

        // Get gin_channels from cond input channels
        let gin_channels = cond.as_ref().map(|c| c.weight_v.dims()[0]).unwrap_or(512);

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            conv_post,
            cond,
            gin_channels,
        })
    }

    fn load_conv(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Conv1dWeightNorm> {
        // Try weight_norm format first
        if state_dict.contains(&format!("{}.weight_g", prefix)) {
            let weight_g = state_dict.get(&format!("{}.weight_g", prefix))?
                .to_device(device)?.to_dtype(DType::F32)?;
            let weight_v = state_dict.get(&format!("{}.weight_v", prefix))?
                .to_device(device)?.to_dtype(DType::F32)?;
            let bias = state_dict.get(&format!("{}.bias", prefix))
                .ok()
                .cloned()
                .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
                .transpose()?;

            let weight_v_shape = weight_v.dims();
            let kernel_size = if weight_v_shape.len() >= 3 {
                weight_v_shape[2]
            } else {
                1
            };

            let padding = (kernel_size - 1) / 2;
            Ok(Conv1dWeightNorm::new(weight_g, weight_v, bias, 1, padding, 1))
        } else {
            // Regular conv format
            let weight = state_dict.get(&format!("{}.weight", prefix))?
                .to_device(device)?.to_dtype(DType::F32)?;
            let bias = state_dict.get(&format!("{}.bias", prefix))
                .ok()
                .cloned()
                .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
                .transpose()?;

            let weight_dims = weight.dims();
            let kernel_size = if weight_dims.len() >= 3 {
                weight_dims[2]
            } else {
                1
            };

            let padding = (kernel_size - 1) / 2;
            let weight_g = Tensor::full(1.0f32, weight.dims(), &weight.device())?;
            Ok(Conv1dWeightNorm::new(weight_g, weight, bias, 1, padding, 1))
        }
    }

    fn load_conv_weight(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Conv1dWeightNorm> {
        // For conv_post, the weight is stored directly (not weight_norm)
        let weight = state_dict.get(&format!("{}.weight", prefix))?
            .to_device(device)?.to_dtype(DType::F32)?;
        let bias = state_dict.get(&format!("{}.bias", prefix))
            .ok()
            .cloned()
            .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
            .transpose()?;

        let weight_dims = weight.dims();
        let kernel_size = if weight_dims.len() >= 3 {
            weight_dims[2]
        } else {
            1
        };

        let padding = (kernel_size - 1) / 2;

        // Create dummy g tensor for weight norm
        let weight_g = Tensor::full(1.0f32, weight.dims(), &weight.device())?;
        Ok(Conv1dWeightNorm::new(weight_g, weight, bias, 1, padding, 1))
    }

    /// Generate waveform from latent features
    pub fn forward(&self, x: &Tensor, g: Option<&Tensor>) -> Result<Vec<f32>> {
        // x: [batch, channels, time]
        let mut x = self.conv_pre.forward(x)?;

        // Add condition if provided
        if let Some(cond) = &self.cond {
            if let Some(g) = g {
                let g_proj = cond.forward(g)?;
                x = x.add(&g_proj)?;
            }
        }

        // Upsampling and resblocks
        for (i, up) in self.ups.iter().enumerate() {
            // LeakyReLU
            x = self.leaky_relu(&x, 0.1)?;

            // Upsample using transposed convolution
            x = self.upsample_forward(&x, &up.conv, up.upsample_factor)?;

            // Apply resblock group (3 resblocks per upsample)
            let resblock_start = i * 3;
            let resblock_end = (resblock_start + 3).min(self.resblocks.len());

            if resblock_start < self.resblocks.len() {
                let mut xs_acc: Option<Tensor> = None;

                for j in resblock_start..resblock_end {
                    let block = &self.resblocks[j];
                    let xs = block.forward(&x)?;
                    xs_acc = Some(match xs_acc {
                        Some(acc) => acc.add(&xs)?,
                        None => xs,
                    });
                }

                if let Some(xs) = xs_acc {
                    let divisor = Tensor::full((resblock_end - resblock_start) as f32, xs.dims(), xs.device())?;
                    x = xs.broadcast_div(&divisor)?;
                }
            }
        }

        // Final activation
        x = self.leaky_relu(&x, 0.1)?;

        // Output projection
        x = self.conv_post.forward(&x)?;

        // Tanh activation
        x = x.tanh()?;

        // Convert to Vec<f32>
        let output: Vec<f32> = x.flatten_all()?.to_vec1()?;
        Ok(output)
    }

    fn leaky_relu(&self, x: &Tensor, slope: f32) -> Result<Tensor> {
        let zeros = Tensor::zeros_like(x)?;
        let positive = x.maximum(&zeros)?;
        let negative = x.minimum(&zeros)?;
        let slope_t = Tensor::full(slope, x.dims(), x.device())?;
        Ok(positive.add(&negative.broadcast_mul(&slope_t)?)?)
    }

    fn upsample_forward(&self, x: &Tensor, conv: &Conv1dWeightNorm, upsample_factor: usize) -> Result<Tensor> {
        let weight = conv.get_weight()?;
        // PyTorch ConvTranspose1d weight format: [in_channels, out_channels, kernel_size]
        // Candle conv_transpose1d uses the same format, no transposition needed
        let weight_dims = weight.dims();
        let kernel_size = weight_dims[2];

        let stride = upsample_factor;
        let padding = (kernel_size - stride) / 2;

        Ok(x.conv_transpose1d(&weight, padding, 0, stride, 1, 1)?)
    }
}
