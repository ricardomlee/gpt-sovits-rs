//! Encoder Q - Inference-time encoder for reference audio
//!
//! EncQ processes mel spectrograms from reference audio to produce
//! distribution parameters (m, logs) for the flow/decoder.
//!
//! Architecture: WaveNet-style encoder
//! - pre: Conv1d [192, 1025, 1] → projects 1025 mel bins to 192 channels
//! - enc: WaveNet with 16 layers, cond_layer for speaker embedding
//! - proj: Conv1d [384, 192, 1] → outputs m (192) + logs (192)

use crate::utils::{Conv1dWeightNorm, StateDict};
use crate::Result;
use candle_core::{DType, Device, Module, Tensor};

/// WaveNet Encoder for EncQ (same structure as flow WN but with different dimensions)
#[derive(Debug, Clone)]
struct WN {
    in_layers: Vec<Conv1dWeightNorm>,
    res_skip_layers: Vec<Conv1dWeightNorm>,
    cond_layer: Option<Conv1dWeightNorm>,
    n_layers: usize,
}

impl WN {
    pub fn load(
        state_dict: &StateDict,
        prefix: &str,
        device: &Device,
        n_layers: usize,
        dtype: DType,
    ) -> Result<Self> {
        let mut in_layers = Vec::new();
        let mut res_skip_layers = Vec::new();

        // Load in_layers
        for i in 0..n_layers {
            let key = format!("{}.in_layers.{}.weight_v", prefix, i);
            if state_dict.contains(&key) {
                in_layers.push(Self::load_conv(
                    state_dict,
                    &format!("{}.in_layers.{}", prefix, i),
                    device,
                    dtype,
                )?);
            }
        }

        // Load res_skip_layers
        for i in 0..n_layers {
            let key = format!("{}.res_skip_layers.{}.weight_v", prefix, i);
            if state_dict.contains(&key) {
                res_skip_layers.push(Self::load_conv(
                    state_dict,
                    &format!("{}.res_skip_layers.{}", prefix, i),
                    device,
                    dtype,
                )?);
            }
        }

        // Load condition layer (optional)
        let cond_layer = if state_dict.contains(&format!("{}.cond_layer.weight_v", prefix)) {
            Some(Self::load_conv(
                state_dict,
                &format!("{}.cond_layer", prefix),
                device,
                dtype,
            )?)
        } else {
            None
        };

        let n_layers_count = in_layers.len();

        Ok(Self {
            in_layers,
            res_skip_layers,
            cond_layer,
            n_layers: n_layers_count,
        })
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

        let kernel_size = if weight_v.dims().len() >= 3 {
            weight_v.dims()[2]
        } else {
            3
        };
        let padding = (kernel_size - 1) / 2;

        Ok(Conv1dWeightNorm::new_with_cached(
            weight_g, weight_v, bias, 1, padding, 1,
        )?)
    }

    pub fn forward(&self, x: &Tensor, x_mask: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        // Project global conditioning once: [batch, gin_channels, 1] → [batch, 2*hidden*n_layers, 1]
        let g_proj = if let Some(g) = g {
            if let Some(cond) = &self.cond_layer {
                Some(cond.forward(g)?)
            } else {
                None
            }
        } else {
            None
        };

        let hidden_channels = x.dims()[1];
        let mut x = x.clone();
        let mut output = Tensor::zeros_like(&x)?;

        for i in 0..self.n_layers {
            let in_layer = &self.in_layers[i];
            let res_skip_layer = &self.res_skip_layers[i];

            let x_masked = x.broadcast_mul(x_mask)?;
            let x_in = in_layer.forward(&x_masked)?;

            // Slice per-layer conditioning and apply fused_add_tanh_sigmoid_multiply
            let acts = if let Some(ref g) = g_proj {
                let cond_offset = i * 2 * hidden_channels;
                let g_l = g.narrow(1, cond_offset, 2 * hidden_channels)?;
                fused_add_tanh_sigmoid_multiply(&x_in, &g_l, hidden_channels, true)?
            } else {
                // No conditioning: just gated activation on x_in
                let channels = x_in.dims()[1];
                let x_tanh = x_in.narrow(1, 0, channels / 2)?.tanh()?;
                let x_sig = x_in.narrow(1, channels / 2, channels / 2)?;
                let x_sig = candle_nn::ops::sigmoid(&x_sig)?;
                x_tanh.mul(&x_sig)?
            };

            let res_skip_acts = res_skip_layer.forward(&acts)?;

            if i < self.n_layers - 1 {
                // Residual and skip: split res_skip_acts
                let res_acts = res_skip_acts.narrow(1, 0, hidden_channels)?;
                let skip_acts = res_skip_acts.narrow(1, hidden_channels, hidden_channels)?;
                x = x.add(&res_acts)?.broadcast_mul(x_mask)?;
                output = output.add(&skip_acts)?;
            } else {
                // Last layer: only residual
                output = output.add(&res_skip_acts)?;
            }
        }

        Ok(output.broadcast_mul(x_mask)?)
    }
}

/// Fused add tanh-sigmoid multiply (matches Python commons.fused_add_tanh_sigmoid_multiply)
/// Splits both inputs in half along channel dim: tanh(a1)*sigmoid(a2) + tanh(b1)*sigmoid(b2)
/// When broadcast_time is true, b (global conditioning) is broadcast across time dimension of a.
fn fused_add_tanh_sigmoid_multiply(
    a: &Tensor,
    b: &Tensor,
    n_channels: usize,
    broadcast_time: bool,
) -> Result<Tensor> {
    let a_tanh = a.narrow(1, 0, n_channels)?.tanh()?;
    let a_sig = a.narrow(1, n_channels, n_channels)?;
    let a_sig = candle_nn::ops::sigmoid(&a_sig)?;
    let a_out = a_tanh.mul(&a_sig)?;

    let b_tanh = b.narrow(1, 0, n_channels)?.tanh()?;
    let b_sig = b.narrow(1, n_channels, n_channels)?;
    let b_sig = candle_nn::ops::sigmoid(&b_sig)?;
    let b_out = b_tanh.mul(&b_sig)?;

    if broadcast_time {
        // b_out is [batch, n_channels, 1], a_out is [batch, n_channels, time]
        // Broadcast b_out across time dimension
        Ok(a_out.broadcast_add(&b_out)?)
    } else {
        Ok(a_out.add(&b_out)?)
    }
}

/// Encoder Q - processes reference audio mel spectrogram
#[derive(Debug, Clone)]
pub struct EncQ {
    pre: Conv1dWeightNorm,
    enc: WN,
    proj: candle_nn::Conv1d,
    out_channels: usize,
}

impl EncQ {
    /// Load EncQ from state dict
    pub fn load(
        state_dict: &StateDict,
        device: &Device,
        _hidden_channels: usize,
        out_channels: usize,
        dtype: DType,
    ) -> Result<Self> {
        // Load pre projection: [192, 1025, 1]
        let pre = Conv1dWeightNorm::load_regular(state_dict, "enc_q.pre", device, dtype)?;

        // Load WaveNet encoder (16 layers)
        let enc = WN::load(state_dict, "enc_q.enc", device, 16, dtype)?;

        // Load output projection: [384, 192, 1]
        let proj_weight = state_dict
            .get("enc_q.proj.weight")?
            .to_device(device)?
            .to_dtype(dtype)?;
        let proj_bias = state_dict
            .get("enc_q.proj.bias")
            .ok()
            .cloned()
            .map(|t| t.to_device(device).and_then(|t| t.to_dtype(dtype)))
            .transpose()?;

        let proj_config = candle_nn::Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: Default::default(),
        };
        let proj = candle_nn::Conv1d::new(proj_weight, proj_bias, proj_config);

        Ok(Self {
            pre,
            enc,
            proj,
            out_channels,
        })
    }

    /// Forward pass
    /// mel_spec: [batch, 1025, seq_len]
    /// g: optional speaker embedding [batch, 512, 1]
    /// Returns: (m, logs, mask) where m and logs are [batch, out_channels, seq_len]
    pub fn forward(
        &self,
        mel_spec: &Tensor,
        g: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let device = mel_spec.device();
        let seq_len = mel_spec.dims()[2];
        let batch = mel_spec.dims()[0];

        // Create mask
        let lengths: Vec<i64> = vec![seq_len as i64; batch];
        let mask = self.sequence_mask(&lengths, seq_len as i64, device)?;
        let mask_expanded = mask.unsqueeze(1)?;

        // Project mel to hidden dim
        let mut x = self.pre.forward(mel_spec)?;
        x = x.broadcast_mul(&mask_expanded)?;

        // WaveNet encoder
        let h = self.enc.forward(&x, &mask_expanded, g)?;

        // Output projection
        let stats = self.proj.forward(&h)?;
        let stats = stats.broadcast_mul(&mask_expanded)?;

        // Split into m and logs
        let m = stats.narrow(1, 0, self.out_channels)?;
        let logs = stats.narrow(1, self.out_channels, self.out_channels)?;
        let logs = logs.clamp(-5.0, 2.0)?;

        Ok((m, logs, mask))
    }

    fn sequence_mask(&self, lengths: &[i64], max_len: i64, device: &Device) -> Result<Tensor> {
        let batch_size = lengths.len();
        let mut mask = Vec::with_capacity(batch_size * max_len as usize);

        for &len in lengths.iter() {
            for j in 0..max_len {
                if j < len {
                    mask.push(1.0f32);
                } else {
                    mask.push(0.0f32);
                }
            }
        }

        Ok(Tensor::from_vec(
            mask,
            (batch_size, max_len as usize),
            device,
        )?)
    }
}

/// Extension trait for loading regular (non-weight-norm) convolutions from state dict
impl Conv1dWeightNorm {
    /// Load a regular convolution as Conv1dWeightNorm (with dummy weight_g)
    pub fn load_regular(
        state_dict: &StateDict,
        prefix: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
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

        let kernel_size = if weight.dims().len() >= 3 {
            weight.dims()[2]
        } else {
            1
        };
        let padding = (kernel_size - 1) / 2;

        let weight_g = Tensor::full(1.0f32, weight.dims(), &weight.device())?.to_dtype(dtype)?;
        Ok(Conv1dWeightNorm::new_with_cached(
            weight_g, weight, bias, 1, padding, 1,
        )?)
    }
}
