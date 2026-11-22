//! SoVITS Decoder Module - HiFi-GAN style vocoder

use crate::utils::profiling::{sync_profile_enabled, sync_profile_stage, Instant};
use crate::utils::{Conv1d, Conv1dWeightNorm, StateDict};
use crate::Result;
use candle_core::backend::BackendStorage;
use candle_core::{cpu_backend, CpuStorage, CustomOp1, DType, Device, Layout, Shape, Tensor};
use half::{bf16, f16};
use std::path::Path;
use std::time::Duration;

struct FusedLeakyRelu {
    slope: f32,
}

impl CustomOp1 for FusedLeakyRelu {
    fn name(&self) -> &'static str {
        "fused-leaky-relu"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let output = match storage {
            CpuStorage::BF16(values) => {
                let slope = bf16::from_f32(self.slope);
                CpuStorage::BF16(cpu_backend::unary_map(values, layout, |value| {
                    if value.is_sign_positive() {
                        value
                    } else {
                        value * slope
                    }
                }))
            }
            CpuStorage::F16(values) => {
                let slope = f16::from_f32(self.slope);
                CpuStorage::F16(cpu_backend::unary_map(values, layout, |value| {
                    if value.is_sign_positive() {
                        value
                    } else {
                        value * slope
                    }
                }))
            }
            CpuStorage::F32(values) => {
                CpuStorage::F32(cpu_backend::unary_map(values, layout, |value| {
                    if value.is_sign_positive() {
                        value
                    } else {
                        value * self.slope
                    }
                }))
            }
            CpuStorage::F64(values) => {
                let slope = self.slope as f64;
                CpuStorage::F64(cpu_backend::unary_map(values, layout, |value| {
                    if value.is_sign_positive() {
                        value
                    } else {
                        value * slope
                    }
                }))
            }
            _ => {
                return Err(candle_core::Error::UnsupportedDTypeForOp(
                    storage.dtype(),
                    self.name(),
                ))
            }
        };
        Ok((output, layout.shape().clone()))
    }
}

/// LeakyReLU activation: max(x, 0) + slope * min(x, 0)
pub fn leaky_relu(x: &Tensor, slope: f32) -> Result<Tensor> {
    if x.device().is_cpu() {
        return Ok(x.apply_op1_no_bwd(&FusedLeakyRelu { slope })?);
    }
    let zeros = Tensor::zeros_like(x)?;
    let positive = x.maximum(&zeros)?;
    let negative = x.minimum(&zeros)?;
    let slope = Tensor::full(slope, x.dims(), x.device())?.to_dtype(x.dtype())?;
    Ok(positive.add(&negative.broadcast_mul(&slope)?)?)
}

fn conv_transpose1d(x: &Tensor, weight: &Tensor, padding: usize, stride: usize) -> Result<Tensor> {
    let use_cropped_convolution = (cfg!(feature = "mkl") && x.device().is_cpu())
        || (x.device().is_cuda()
            && x.dtype() == DType::F32
            && x.is_contiguous()
            && weight.is_contiguous());
    if use_cropped_convolution && padding > 0 {
        // Candle's GEMM + col2im fast paths require zero padding. A padded
        // transposed convolution is exactly the zero-padding result cropped at
        // both ends, so keep the fast path and apply the crop explicitly.
        let full = x.conv_transpose1d(weight, 0, 0, stride, 1, 1)?;
        let full_len = full.dim(2)?;
        let output_len = full_len
            .checked_sub(padding.saturating_mul(2))
            .ok_or_else(|| {
                crate::Error::InferenceError("transposed convolution padding exceeds output".into())
            })?;
        return Ok(full.narrow(2, padding, output_len)?);
    }
    Ok(x.conv_transpose1d(weight, padding, 0, stride, 1, 1)?)
}

/// Residual Block Type 1 (for HiFi-GAN decoder)
#[derive(Debug, Clone)]
pub struct ResBlock1 {
    convs1: Vec<Conv1dWeightNorm>,
    convs2: Vec<Conv1dWeightNorm>,
}

impl ResBlock1 {
    pub fn load(
        state_dict: &StateDict,
        prefix: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();

        // Load convs1 (dilated convolutions)
        for i in 0..3 {
            let key = format!("{}.convs1.{}.weight_v", prefix, i);
            if state_dict.contains(&key) {
                convs1.push(Self::load_conv(
                    state_dict,
                    &format!("{}.convs1.{}", prefix, i),
                    device,
                    dtype,
                )?);
            }
        }

        // Load convs2
        for i in 0..3 {
            let key = format!("{}.convs2.{}.weight_v", prefix, i);
            if state_dict.contains(&key) {
                convs2.push(Self::load_conv(
                    state_dict,
                    &format!("{}.convs2.{}", prefix, i),
                    device,
                    dtype,
                )?);
            }
        }

        Ok(Self { convs1, convs2 })
    }

    fn load_conv(
        state_dict: &StateDict,
        prefix: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Conv1dWeightNorm> {
        let weight_g = state_dict
            .get(&format!("{}.weight_g", prefix))?
            .to_device(device)?
            .to_dtype(dtype)?;
        let weight_v = state_dict
            .get(&format!("{}.weight_v", prefix))?
            .to_device(device)?
            .to_dtype(dtype)?;
        let bias = state_dict
            .get(&format!("{}.bias", prefix))
            .ok()
            .cloned()
            .map(|t| t.to_device(device).and_then(|t| t.to_dtype(dtype)))
            .transpose()?;

        // Kernel sizes cycle [3, 7, 11] across resblocks (rb_idx % 3).
        // Dilations are [1, 3, 5] within each resblock.
        // Extract resblock index and conv index from prefix
        let parts: Vec<&str> = prefix.split('.').collect();
        let rb_idx = parts
            .get(2)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let conv_idx = parts
            .last()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        let kernel_sizes = [3, 7, 11];
        let ks = kernel_sizes[rb_idx % 3];
        let dilations = [1, 3, 5];
        let dilation = dilations.get(conv_idx).copied().unwrap_or(1);

        // convs1: padding = dilation * (kernel_size - 1) / 2, uses dilation
        // convs2: padding = (kernel_size - 1) / 2, dilation is ALWAYS 1
        let (padding, actual_dilation) = if prefix.contains(".convs2.") {
            ((ks - 1) / 2, 1)
        } else {
            (dilation * (ks - 1) / 2, dilation)
        };

        Conv1dWeightNorm::new_with_cached(weight_g, weight_v, bias, 1, padding, actual_dilation)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x_out = x.clone();

        for i in 0..self.convs1.len() {
            let xt = leaky_relu(&x_out, 0.1)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = leaky_relu(&xt, 0.1)?;
            let xt = self.convs2[i].forward(&xt)?;
            x_out = x_out.add(&xt)?;
        }

        Ok(x_out)
    }
}

/// Decoder (HiFi-GAN style vocoder)
#[derive(Debug, Clone)]
pub struct Decoder {
    conv_pre: Conv1d,
    ups: Vec<Upsample>,
    resblocks: Vec<ResBlock1>,
    conv_post: Conv1d,
    cond: Option<Conv1d>,
}

#[derive(Debug, Clone)]
struct Upsample {
    conv: Conv1dWeightNorm,
    upsample_factor: usize,
}

impl Decoder {
    /// Load decoder from SoVITS safetensors
    pub fn load(state_dict: &StateDict, device: &Device, dtype: DType) -> Result<Self> {
        // Load conv_pre: [512, 192, 7] - PLAIN conv, NOT weight_norm
        let conv_pre = Self::load_conv_plain(state_dict, "dec.conv_pre", device, dtype)?;

        // Load conv_post: [1, 16, 7] - PLAIN conv, NOT weight_norm
        let conv_post = Self::load_conv_plain(state_dict, "dec.conv_post", device, dtype)?;

        // Load condition layer if exists - PLAIN conv, NOT weight_norm
        let cond = if state_dict.contains("dec.cond.weight") {
            Some(Self::load_conv_plain(
                state_dict, "dec.cond", device, dtype,
            )?)
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
                let conv = Self::load_conv_wn(state_dict, &prefix, device, dtype)?;
                let upsample_factor = if up_idx < UPSAMPLE_RATES.len() {
                    UPSAMPLE_RATES[up_idx]
                } else {
                    2 // fallback
                };
                up_idx += 1;
                ups.push(Upsample {
                    conv,
                    upsample_factor,
                });
            }
        }

        // Load resblock groups (15 resblocks total, using ResBlock1)
        let mut resblocks = Vec::new();
        for i in 0..15 {
            let prefix = format!("dec.resblocks.{}", i);
            if state_dict.contains(&format!("{}.convs1.0.weight_v", prefix)) {
                let block = ResBlock1::load(state_dict, &prefix, device, dtype)?;
                resblocks.push(block);
            }
        }

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            conv_post,
            cond,
        })
    }

    fn load_conv_plain(
        state_dict: &StateDict,
        prefix: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Conv1d> {
        let weight = state_dict
            .get(&format!("{}.weight", prefix))?
            .to_device(device)?
            .to_dtype(dtype)?;
        let bias = state_dict
            .get(&format!("{}.bias", prefix))
            .ok()
            .cloned()
            .map(|t| t.to_device(device).and_then(|t| t.to_dtype(dtype)))
            .transpose()?;

        let weight_dims = weight.dims();
        let kernel_size = if weight_dims.len() >= 3 {
            weight_dims[2]
        } else {
            1
        };

        let padding = (kernel_size - 1) / 2;
        Ok(Conv1d::new(weight, bias, 1, padding, 1))
    }

    /// Load weight-norm conv (used by ups and resblocks which ARE weight_norm)
    fn load_conv_wn(
        state_dict: &StateDict,
        prefix: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Conv1dWeightNorm> {
        let weight_g = state_dict
            .get(&format!("{}.weight_g", prefix))?
            .to_device(device)?
            .to_dtype(dtype)?;
        let weight_v = state_dict
            .get(&format!("{}.weight_v", prefix))?
            .to_device(device)?
            .to_dtype(dtype)?;
        let bias = state_dict
            .get(&format!("{}.bias", prefix))
            .ok()
            .cloned()
            .map(|t| t.to_device(device).and_then(|t| t.to_dtype(dtype)))
            .transpose()?;

        let weight_v_shape = weight_v.dims();
        let kernel_size = if weight_v_shape.len() >= 3 {
            weight_v_shape[2]
        } else {
            1
        };

        let padding = (kernel_size - 1) / 2;
        Conv1dWeightNorm::new_with_cached(weight_g, weight_v, bias, 1, padding, 1)
    }

    /// Generate waveform from latent features
    pub fn forward(&self, x: &Tensor, g: Option<&Tensor>) -> Result<Vec<f32>> {
        self.forward_with_trace(x, g, None)
    }

    fn forward_with_trace(
        &self,
        x: &Tensor,
        g: Option<&Tensor>,
        trace_dir: Option<&Path>,
    ) -> Result<Vec<f32>> {
        let profile_start = Instant::now();
        let device = x.device().clone();
        let trace_tensor = |name: &str, tensor: &Tensor| match trace_dir {
            Some(directory) => self.save_tensor(directory, name, tensor),
            None => Ok(()),
        };
        // x: [batch, channels, time]
        let mut x = self.conv_pre.forward(x)?;
        trace_tensor("debug_conv_pre", &x)?;

        // Add condition if provided
        if let Some(cond) = &self.cond
            && let Some(g) = g
        {
            let g_proj = cond.forward(g)?;
            // g_proj: [batch, channels, 1] → broadcast across time
            x = x.broadcast_add(&g_proj)?;
            trace_tensor("debug_cond", &x)?;
        }
        sync_profile_stage(&device)?;
        let pre_ms = profile_start.elapsed().as_millis();
        let mut upsample_time = Duration::ZERO;
        let mut resblock_time = Duration::ZERO;

        // Upsampling and resblocks
        for (i, up) in self.ups.iter().enumerate() {
            // LeakyReLU
            x = leaky_relu(&x, 0.1)?;

            // Upsample using transposed convolution
            let stage_start = Instant::now();
            x = self.upsample_forward(&x, &up.conv, up.upsample_factor)?;
            if trace_dir.is_some() {
                trace_tensor(&format!("debug_ups{i}"), &x)?;
            }
            sync_profile_stage(&device)?;
            upsample_time += stage_start.elapsed();

            // Apply resblock group (3 resblocks per upsample)
            let stage_start = Instant::now();
            let resblock_start = i * 3;
            let resblock_end = (resblock_start + 3).min(self.resblocks.len());

            if resblock_start < self.resblocks.len() {
                let mut xs_acc: Option<Tensor> = None;

                for block in &self.resblocks[resblock_start..resblock_end] {
                    let xs = block.forward(&x)?;
                    xs_acc = Some(match xs_acc {
                        Some(acc) => acc.add(&xs)?,
                        None => xs,
                    });
                }

                if let Some(xs) = xs_acc {
                    let divisor = Tensor::full(
                        (resblock_end - resblock_start) as f32,
                        xs.dims(),
                        xs.device(),
                    )?
                    .to_dtype(xs.dtype())?;
                    x = xs.broadcast_div(&divisor)?;
                }
                if trace_dir.is_some() {
                    trace_tensor(&format!("debug_resblock{i}"), &x)?;
                }
            }
            sync_profile_stage(&device)?;
            resblock_time += stage_start.elapsed();
        }

        // Final activation (Python uses F.leaky_relu(x) with default slope=0.01)
        let stage_start = Instant::now();
        x = leaky_relu(&x, 0.01)?;

        // Output projection
        x = self.conv_post.forward(&x)?;
        trace_tensor("debug_post_conv", &x)?;

        // Tanh activation
        x = x.tanh()?;
        trace_tensor("debug_audio", &x)?;

        // Convert to Vec<f32> — cast to F32 first since weights may be F16
        let output: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        if device.is_cpu() || sync_profile_enabled() {
            tracing::debug!(
                "profile decoder pre={}ms upsample={}ms resblocks={}ms post={}ms total={}ms",
                pre_ms,
                upsample_time.as_millis(),
                resblock_time.as_millis(),
                stage_start.elapsed().as_millis(),
                profile_start.elapsed().as_millis()
            );
        }
        Ok(output)
    }

    /// Generate waveform and save intermediate outputs for debugging
    pub fn forward_debug(&self, x: &Tensor, g: Option<&Tensor>) -> Result<Vec<f32>> {
        self.forward_with_trace(x, g, Some(Path::new(".")))
    }

    fn save_tensor(&self, directory: &Path, name: &str, t: &Tensor) -> Result<()> {
        let flat: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let dims = t.dims();
        let header = format!(
            "{}\n",
            dims.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let data = flat
            .iter()
            .map(|v| format!("{:.10}", v))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(
            directory.join(format!("{}.txt", name)),
            format!("{}{}\n", header, data),
        )
        .map_err(|e| crate::Error::InferenceError(format!("Failed to save {}: {}", name, e)))
    }

    fn upsample_forward(
        &self,
        x: &Tensor,
        conv: &Conv1dWeightNorm,
        upsample_factor: usize,
    ) -> Result<Tensor> {
        let weight = conv.get_weight()?;
        // PyTorch ConvTranspose1d weight format: [in_channels, out_channels, kernel_size]
        // Candle conv_transpose1d uses the same format, no transposition needed
        let weight_dims = weight.dims();
        let kernel_size = weight_dims[2];

        let stride = upsample_factor;
        let padding = (kernel_size - stride) / 2;

        let out = conv_transpose1d(x, &weight, padding, stride)?;

        // Add bias if present
        if let Some(bias) = &conv.bias {
            let bias_len = bias.dims()[0];
            let bias_reshaped = bias.reshape(&[1, bias_len, 1])?;
            return Ok(out.broadcast_add(&bias_reshaped)?);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_plain_projections_preserve_weights_and_optional_biases() -> Result<()> {
        for bias in [None, Some(0.25f32)] {
            let mut weights = std::collections::HashMap::new();
            for (prefix, weight) in [("dec.conv_pre", 2f32), ("dec.conv_post", 0.5)] {
                weights.insert(
                    format!("{prefix}.weight"),
                    Tensor::new(&[[[weight]]], &Device::Cpu)?,
                );
                if let Some(bias) = bias {
                    weights.insert(
                        format!("{prefix}.bias"),
                        Tensor::new(&[bias], &Device::Cpu)?,
                    );
                }
            }
            let decoder = Decoder::load(&StateDict::new(weights), &Device::Cpu, DType::F32)?;
            let samples = [-2f32, 0., 1., 3.];
            let input = Tensor::new(&[[samples]], &Device::Cpu)?;
            let actual = decoder.forward(&input, None)?;
            let directory = tempfile::tempdir().unwrap();
            let traced = decoder.forward_with_trace(&input, None, Some(directory.path()))?;
            assert_eq!(actual, traced);
            for name in ["debug_conv_pre", "debug_post_conv", "debug_audio"] {
                assert!(directory.path().join(format!("{name}.txt")).is_file());
            }
            assert_eq!(actual.len(), samples.len());
            for (actual, sample) in actual.iter().zip(samples) {
                let x = sample * 2.0 + bias.unwrap_or(0.0);
                let x = if x < 0.0 { x * 0.01 } else { x };
                let expected = (x * 0.5 + bias.unwrap_or(0.0)).tanh();
                assert!((actual - expected).abs() < 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn leaky_relu_uses_the_requested_negative_slope() -> Result<()> {
        let input = Tensor::new(&[-2f32, 0., 3.], &Device::Cpu)?;
        let output = leaky_relu(&input, 0.1)?;
        assert_eq!(output.to_vec1::<f32>()?, [-0.2, 0., 3.]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_transposed_conv_crop_matches_padded_convolution() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;
        assert!(!candle_core::cuda_backend::gemm_reduced_precision_f32());
        for (batch, c_in, c_out, length, kernel, stride, padding) in [
            (1, 2, 3, 5, 4, 2, 1),
            (2, 3, 4, 7, 7, 3, 2),
            (1, 512, 256, 7, 16, 10, 3),
            (1, 256, 128, 11, 16, 8, 4),
            (1, 128, 64, 13, 8, 2, 3),
            (1, 4, 2, 9, 2, 2, 0),
        ] {
            let input = Tensor::arange(0f32, (batch * c_in * length) as f32, &Device::Cpu)?
                .sin()?
                .affine(0.1, 0.0)?
                .reshape((batch, c_in, length))?
                .to_device(&device)?;
            let weight = Tensor::arange(0f32, (c_in * c_out * kernel) as f32, &Device::Cpu)?
                .cos()?
                .affine(0.05, 0.0)?
                .reshape((c_in, c_out, kernel))?
                .to_device(&device)?;
            for (input, weight) in [
                (input.clone(), weight.clone()),
                (
                    Tensor::cat(&[&input, &input], 0)?.narrow(0, batch, batch)?,
                    Tensor::cat(&[&weight, &weight], 0)?.narrow(0, c_in, c_in)?,
                ),
                (input.narrow(2, 1, length - 1)?, weight.clone()),
                (
                    input.clone(),
                    weight.transpose(1, 2)?.contiguous()?.transpose(1, 2)?,
                ),
            ] {
                let expected = input.conv_transpose1d(&weight, padding, 0, stride, 1, 1);
                let actual = conv_transpose1d(&input, &weight, padding, stride);
                if expected.is_err() {
                    assert!(actual.is_err());
                    continue;
                }
                let expected = expected?;
                let actual = actual?;
                assert_eq!(actual.dims(), expected.dims());
                let error = actual
                    .sub(&expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(error < 1e-5, "maximum error was {error}");
            }
        }
        let input = Tensor::ones((1, 2, 4), DType::F32, &device)?;
        let weight = Tensor::ones((2, 3, 4), DType::F32, &device)?;
        assert!(conv_transpose1d(&input, &weight, 6, 2).is_err());
        for dtype in [DType::F16, DType::BF16, DType::F64] {
            let input = input.to_dtype(dtype)?;
            let weight = weight.to_dtype(dtype)?;
            let expected = input.conv_transpose1d(&weight, 1, 0, 2, 1, 1)?;
            let actual = conv_transpose1d(&input, &weight, 1, 2)?;
            assert_eq!(
                actual.to_dtype(DType::F32)?.to_vec3::<f32>()?,
                expected.to_dtype(DType::F32)?.to_vec3::<f32>()?
            );
        }
        Ok(())
    }

    #[cfg(feature = "mkl")]
    #[test]
    fn cpu_transposed_conv_crop_matches_padded_convolution() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::arange(0f32, 8f32, &device)?.reshape((1, 2, 4))?;
        let weight = Tensor::arange(0f32, 24f32, &device)?
            .affine(0.01, -0.1)?
            .reshape((2, 3, 4))?;

        let expected = x.conv_transpose1d(&weight, 1, 0, 2, 1, 1)?;
        let actual = conv_transpose1d(&x, &weight, 1, 2)?;
        assert_eq!(actual.dims(), expected.dims());

        let max_error = actual
            .sub(&expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(max_error < 1e-5, "maximum error was {max_error}");
        Ok(())
    }
}
