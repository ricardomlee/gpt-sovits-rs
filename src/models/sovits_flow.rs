//! Flow modules for SoVITS
//!
//! Implementation matching the actual model structure

use candle_core::{Device, DType, Tensor};
use crate::Result;
use crate::utils::{StateDict, Conv1dWeightNorm};

/// WaveNet Encoder for Flow-based models
#[derive(Debug, Clone)]
pub struct WN {
    in_layers: Vec<Conv1dWeightNorm>,
    res_skip_layers: Vec<Conv1dWeightNorm>,
    cond_layer: Option<Conv1dWeightNorm>,
    n_layers: usize,
}

impl WN {
    /// Load WN from state dict
    pub fn load(state_dict: &StateDict, prefix: &str, device: &Device, dtype: DType) -> Result<Self> {
        let mut in_layers = Vec::new();
        let mut res_skip_layers = Vec::new();

        // Load in_layers
        let mut i = 0;
        loop {
            let key = format!("{}.in_layers.{}.weight_v", prefix, i);
            if !state_dict.contains(&key) {
                break;
            }
            in_layers.push(Self::load_conv(state_dict, &format!("{}.in_layers.{}", prefix, i), device, dtype)?);
            i += 1;
        }

        // Load res_skip_layers
        i = 0;
        loop {
            let key = format!("{}.res_skip_layers.{}.weight_v", prefix, i);
            if !state_dict.contains(&key) {
                break;
            }
            res_skip_layers.push(Self::load_conv(state_dict, &format!("{}.res_skip_layers.{}", prefix, i), device, dtype)?);
            i += 1;
        }

        // Load condition layer (optional)
        let cond_layer = if state_dict.contains(&format!("{}.cond_layer.weight_v", prefix)) {
            Some(Self::load_conv(state_dict, &format!("{}.cond_layer", prefix), device, dtype)?)
        } else {
            None
        };

        let n_layers = in_layers.len();

        Ok(Self {
            in_layers,
            res_skip_layers,
            cond_layer,
            n_layers,
        })
    }

    fn load_conv(state_dict: &StateDict, prefix: &str, device: &Device, dtype: DType) -> Result<Conv1dWeightNorm> {
        let weight_g = state_dict.get(&format!("{}.weight_g", prefix))?
            .to_device(device)?.to_dtype(dtype)?;
        let weight_v = state_dict.get(&format!("{}.weight_v", prefix))?
            .to_device(device)?.to_dtype(dtype)?;
        let bias = state_dict.get(&format!("{}.bias", prefix))
            .ok()
            .cloned()
            .map(|t| t.to_device(device).and_then(|t| t.to_dtype(dtype)))
            .transpose()?;

        let weight_v_shape = weight_v.dims();
        let kernel_size = if weight_v_shape.len() >= 3 {
            weight_v_shape[2]
        } else {
            3
        };
        let padding = (kernel_size - 1) / 2;

        Ok(Conv1dWeightNorm::new_with_cached(weight_g, weight_v, bias, 1, padding, 1)?)
    }

    /// Forward pass through WaveNet
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
/// Python: in_act = a + b (broadcast), then tanh(in_act[:n]) * sigmoid(in_act[n:])
/// Note: this is NOT the same as tanh(a[:n])*sigmoid(a[n:]) + tanh(b[:n])*sigmoid(b[n:])
fn fused_add_tanh_sigmoid_multiply(a: &Tensor, b: &Tensor, n_channels: usize, _broadcast_time: bool) -> Result<Tensor> {
    // b may be [batch, 2*n_channels, 1] while a is [batch, 2*n_channels, time] — broadcast_add handles this
    let in_act = a.broadcast_add(b)?;
    let t_act = in_act.narrow(1, 0, n_channels)?.tanh()?;
    let s_act = candle_nn::ops::sigmoid(&in_act.narrow(1, n_channels, n_channels)?)?;
    Ok(t_act.mul(&s_act)?)
}

/// Residual Coupling Layer for Normalizing Flow
#[derive(Debug, Clone)]
pub struct ResidualCouplingLayer {
    pre: Conv1dWeightNorm,
    enc: WN,
    post: Conv1dWeightNorm,
    half_channels: usize,
    mean_only: bool,
}

impl ResidualCouplingLayer {
    /// Load from state dict
    pub fn load(state_dict: &StateDict, prefix: &str, device: &Device, mean_only: bool, dtype: DType) -> Result<Self> {
        // Load pre projection
        let pre = Self::load_conv(state_dict, &format!("{}.pre", prefix), device, dtype)?;

        // Load WN encoder
        let enc = WN::load(state_dict, &format!("{}.enc", prefix), device, dtype)?;

        // Load post projection
        let post = Self::load_conv(state_dict, &format!("{}.post", prefix), device, dtype)?;

        // Get half_channels from pre weight shape [out_channels, in_channels, kernel]
        let half_channels = pre.weight_v.dims()[1];

        Ok(Self {
            pre,
            enc,
            post,
            half_channels,
            mean_only,
        })
    }

    fn load_conv(state_dict: &StateDict, prefix: &str, device: &Device, dtype: DType) -> Result<Conv1dWeightNorm> {
        // Try weight_norm format first, fall back to regular weight
        if state_dict.contains(&format!("{}.weight_v", prefix)) {
            let weight_g = state_dict.get(&format!("{}.weight_g", prefix))?
                .to_device(device)?.to_dtype(dtype)?;
            let weight_v = state_dict.get(&format!("{}.weight_v", prefix))?
                .to_device(device)?.to_dtype(dtype)?;
            let bias = state_dict.get(&format!("{}.bias", prefix))
                .ok()
                .cloned()
                .map(|t| t.to_device(device).and_then(|t| t.to_dtype(dtype)))
                .transpose()?;

            let kernel_size = if weight_v.dims().len() >= 3 { weight_v.dims()[2] } else { 1 };
            let padding = (kernel_size - 1) / 2;

            Ok(Conv1dWeightNorm::new_with_cached(weight_g, weight_v, bias, 1, padding, 1)?)
        } else {
            // Regular weight format
            let weight = state_dict.get(&format!("{}.weight", prefix))?
                .to_device(device)?.to_dtype(dtype)?;
            let bias = state_dict.get(&format!("{}.bias", prefix))
                .ok()
                .cloned()
                .map(|t| t.to_device(device).and_then(|t| t.to_dtype(dtype)))
                .transpose()?;

            let kernel_size = if weight.dims().len() >= 3 { weight.dims()[2] } else { 1 };
            let padding = (kernel_size - 1) / 2;

            // Set weight_g = per-channel L2 norm so (w/norm)*norm = w (no spurious normalization)
            let v_sq = weight.sqr()?;
            let v_norm = v_sq.sum(candle_core::D::Minus1)?.sum(candle_core::D::Minus1)?.sqrt()?;
            let out_ch = weight.dims()[0];
            let weight_g = v_norm.reshape((out_ch, 1, 1))?;
            Ok(Conv1dWeightNorm::new_with_cached(weight_g, weight, bias, 1, padding, 1)?)
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor, x_mask: &Tensor, g: Option<&Tensor>, reverse: bool) -> Result<Tensor> {
        // Split input in half
        let x0 = x.narrow(1, 0, self.half_channels)?;
        let x1 = x.narrow(1, self.half_channels, x.dims()[1] - self.half_channels)?;

        // Transform x0 through WN
        let h = self.pre.forward(&x0)?;
        let h = h.broadcast_mul(x_mask)?;
        let h = self.enc.forward(&h, x_mask, g)?;

        let mut stats = self.post.forward(&h)?;
        stats = stats.broadcast_mul(x_mask)?;

        let (m, logs) = if !self.mean_only {
            // Check if stats has enough channels for both m and logs
            let stat_channels = stats.dims()[1];
            if stat_channels >= self.half_channels * 2 {
                let m = stats.narrow(1, 0, self.half_channels)?;
                let logs = stats.narrow(1, self.half_channels, self.half_channels)?;
                (m, logs)
            } else {
                // Post already reduced to single-channel output
                let m = stats;
                let zeros = Tensor::zeros_like(&m)?;
                (m, zeros)
            }
        } else {
            let m = stats;
            let zeros = Tensor::zeros_like(&m)?;
            (m, zeros)
        };

        if !reverse {
            // Forward: x1 = m + x1 * exp(logs)
            let exp_logs = logs.exp()?;
            let x1_new = x1.broadcast_mul(&exp_logs)?;
            let x1_new = x1_new.add(&m)?;
            Ok(Tensor::cat(&[&x0, &x1_new], 1)?)
        } else {
            // Inverse: x1 = (x1 - m) * exp(-logs)
            let neg_logs = logs.neg()?;
            let exp_neg_logs = neg_logs.exp()?;
            let x1_new = x1.sub(&m)?;
            let x1_new = x1_new.broadcast_mul(&exp_neg_logs)?;
            Ok(Tensor::cat(&[&x0, &x1_new], 1)?)
        }
    }
}

/// Residual Coupling Block (stack of coupling layers)
#[derive(Debug, Clone)]
pub struct ResidualCouplingBlock {
    layers: Vec<ResidualCouplingLayer>,
}

impl ResidualCouplingBlock {
    /// Load from state dict
    pub fn load(state_dict: &StateDict, prefix: &str, device: &Device, _n_layers: usize, dtype: DType) -> Result<Self> {
        let mut layers = Vec::new();

        // Flow layers may be at even indices (0, 2, 4, 6) - check up to 8
        // Pre layers use regular "weight" key, not "weight_v"
        for i in 0..8 {
            let layer_prefix = format!("{}.{}", prefix, i);
            if state_dict.contains(&format!("{}.pre.weight", layer_prefix)) {
                let layer = ResidualCouplingLayer::load(state_dict, &layer_prefix, device, false, dtype)?;
                layers.push(layer);
            }
        }

        Ok(Self { layers })
    }

    /// Forward pass through the block
    pub fn forward(&self, x: &Tensor, x_mask: &Tensor, g: Option<&Tensor>, reverse: bool) -> Result<Tensor> {
        let mut x = x.clone();

        if !reverse {
            // Forward direction
            for layer in &self.layers {
                x = layer.forward(&x, x_mask, g, false)?;
                x = self.flip(&x)?;
            }
        } else {
            // Inverse direction (reverse order, reverse operations)
            for layer in self.layers.iter().rev() {
                x = self.flip(&x)?;
                x = layer.forward(&x, x_mask, g, true)?;
            }
        }

        Ok(x)
    }

    fn flip(&self, x: &Tensor) -> Result<Tensor> {
        // Flip along channel dimension
        let channels = x.dims()[1];
        let indices: Vec<i64> = (0..channels).rev().map(|i| i as i64).collect();
        let indices_tensor = Tensor::from_vec(indices, channels, x.device())?;
        Ok(x.index_select(&indices_tensor, 1)?)
    }
}
