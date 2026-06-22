//! Reference Encoder (MelStyleEncoder)
//!
//! Computes speaker embedding (ge) from reference audio spectrogram.
//! This is NOT enc_q - enc_q is a training-time posterior encoder.
//! The ref_enc processes reference audio to produce a 512-d speaker embedding.
//!
//! Architecture (MelStyleEncoder):
//! - spectral: FC(704→128) → Mish → Dropout → FC(128→128) → Mish → Dropout
//! - temporal: Conv1dGLU(128→128, k=5) → Conv1dGLU(128→128, k=5)
//! - self_attn: MultiHeadAttention(2 heads, 128 dim, head=64)
//! - fc: FC(128→512)
//! - temporal_avg_pool: mean over time dimension
//! - output: [batch, 512, 1]

use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax;
use crate::Result;
use crate::utils::StateDict;

/// Linear layer with Mish activation
#[derive(Debug, Clone)]
struct LinearNorm {
    weight: Tensor,
    bias: Tensor,
}

impl LinearNorm {
    fn load(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Self> {
        // Python uses `.fc` suffix: spectral.0.fc.weight, spectral.3.fc.weight
        let weight = state_dict.get(&format!("{}.fc.weight", prefix))?
            .to_device(device)?.to_dtype(DType::F32)?;
        let bias = state_dict.get(&format!("{}.fc.bias", prefix))?
            .to_device(device)?.to_dtype(DType::F32)?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq, in_features], weight: [out, in]
        // Output: [batch, seq, out]
        // Workaround: reshape to 2D, matmul, reshape back
        let dims_x = x.dims();
        let batch = dims_x[0];
        let seq = dims_x[1];
        let x_2d = x.reshape((batch * seq, dims_x[2]))?; // [batch*seq, in]
        let result = x_2d.matmul(&self.weight.transpose(0, 1)?)?; // [batch*seq, out]
        let result = result.reshape((batch, seq, self.weight.dims()[0]))?; // [batch, seq, out]
        let bias_broadcast = self.bias.reshape((1, 1, self.bias.dims()[0]))?;
        result.broadcast_add(&bias_broadcast).map_err(|e| crate::Error::InferenceError(e.to_string()))
    }
}

/// Mish activation: x * tanh(softplus(x))
fn mish(x: &Tensor) -> Result<Tensor> {
    let softplus = x.exp()?.add(&Tensor::full(1.0f32, x.dims(), x.device())?)?.log()?;
    let tanh_soft = softplus.tanh()?;
    Ok(x.broadcast_mul(&tanh_soft)?)
}

/// Conv1d + GLU (Gated Linear Unit)
#[derive(Debug, Clone)]
struct Conv1dGLU {
    conv_weight: Tensor,
    conv_bias: Tensor,
    #[allow(dead_code)]
    kernel_size: usize,
    padding: usize,
}

impl Conv1dGLU {
    fn load(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let conv_prefix = format!("{}.conv1.conv", prefix);
        let weight = state_dict.get(&format!("{}.weight", conv_prefix))?
            .to_device(device)?.to_dtype(DType::F32)?;
        let bias = state_dict.get(&format!("{}.bias", conv_prefix))?
            .to_device(device)?.to_dtype(DType::F32)?;
        let kernel_size = weight.dims()[2];
        let padding = (kernel_size - 1) / 2;
        Ok(Self {
            conv_weight: weight,
            conv_bias: bias,
            kernel_size,
            padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, channels, seq]
        // Conv1d with kernel_size=5, padding=2
        let conv_out = x.conv1d(
            &self.conv_weight,
            self.padding, 1, 1, 1,
        )?;
        // Add bias: [1, out_channels, 1]
        let bias_3d = self.conv_bias.reshape((1, self.conv_bias.dims()[0], 1))?;
        let conv_out = conv_out.broadcast_add(&bias_3d)?;

        // GLU: split in half, apply sigmoid to second half, multiply
        let channels = conv_out.dims()[1];
        let half = channels / 2;
        let a = conv_out.narrow(1, 0, half)?;
        let b = conv_out.narrow(1, half, half)?;
        let b_sig = candle_nn::ops::sigmoid(&b)?;
        Ok(a.broadcast_mul(&b_sig)?)
    }
}

/// Simple Multi-Head Self-Attention
#[derive(Debug, Clone)]
struct MultiHeadSelfAttention {
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    fc: Tensor,
    fc_bias: Tensor,
    n_heads: usize,
    head_dim: usize,
}

impl MultiHeadSelfAttention {
    fn load(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let w_q = state_dict.get(&format!("{}.w_qs.weight", prefix))?
            .to_device(device)?.to_dtype(DType::F32)?;
        let w_k = state_dict.get(&format!("{}.w_ks.weight", prefix))?
            .to_device(device)?.to_dtype(DType::F32)?;
        let w_v = state_dict.get(&format!("{}.w_vs.weight", prefix))?
            .to_device(device)?.to_dtype(DType::F32)?;
        let fc = state_dict.get(&format!("{}.fc.weight", prefix))?
            .to_device(device)?.to_dtype(DType::F32)?;
        let fc_bias = state_dict.get(&format!("{}.fc.bias", prefix))?
            .to_device(device)?.to_dtype(DType::F32)?;

        let hidden_dim = w_q.dims()[0];
        let n_heads = 2;
        let head_dim = hidden_dim / n_heads;

        Ok(Self {
            w_q, w_k, w_v, fc, fc_bias, n_heads, head_dim,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // x: [batch, seq, hidden]
        let dims = x.dims();
        let batch = dims[0];
        let seq = dims[1];
        let hidden = dims[2];

        // Project Q, K, V using 2D reshape (workaround for CUDA matmul issue)
        let q = self.linear_2d(x, &self.w_q)?; // [batch, seq, hidden]
        let k = self.linear_2d(x, &self.w_k)?;
        let v = self.linear_2d(x, &self.w_v)?;

        // Reshape: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q = q.reshape((batch, seq, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((batch, seq, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((batch, seq, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt().recip();
        let k_t = k.transpose(2, 3)?;
        let scores = q.matmul(&k_t)?.broadcast_mul(&Tensor::full(scale as f32, &[batch, self.n_heads, seq, seq], q.device())?)?;

        // Apply mask if provided
        let scores = if let Some(m) = mask {
            // m: [batch, seq, seq]
            let neg_inf = Tensor::full(-1e9f32, scores.dims(), scores.device())?;
            // Broadcast mask: [batch, seq, seq] -> [batch, heads, seq, seq]
            let m_float = m.to_dtype(DType::F32)?;
            let m_expanded = m_float.reshape((batch, 1, seq, seq))?;
            let mask_val = m_expanded.broadcast_mul(&neg_inf)?;
            scores.broadcast_add(&mask_val)?
        } else {
            scores
        };

        let attn = softmax(&scores, 3)?;
        let out = attn.matmul(&v)?; // [batch, heads, seq, head_dim]

        // Reshape: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((batch, seq, hidden))?;

        // Output projection using 2D reshape
        let out = self.linear_2d(&out, &self.fc)?;
        let bias_2d = self.fc_bias.reshape((1, 1, self.fc_bias.dims()[0]))?;
        out.broadcast_add(&bias_2d).map_err(|e| crate::Error::InferenceError(e.to_string()))
    }

    fn linear_2d(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        // x: [batch, seq, in], weight: [out, in]
        // Reshape to 2D, matmul, reshape back
        let d = x.dims();
        let x_2d = x.reshape((d[0] * d[1], d[2]))?;
        let w_t = weight.transpose(0, 1)?;
        let result = x_2d.matmul(&w_t)?;
        result.reshape((d[0], d[1], weight.dims()[0]))
            .map_err(|e| crate::Error::InferenceError(e.to_string()))
    }
}

/// MelStyleEncoder - computes speaker embedding from reference spectrogram
#[derive(Debug, Clone)]
pub struct RefEnc {
    spectral_0: LinearNorm,
    spectral_3: LinearNorm,
    temporal_0: Conv1dGLU,
    temporal_1: Conv1dGLU,
    self_attn: MultiHeadSelfAttention,
    fc: LinearNorm,
}

impl RefEnc {
    pub fn load(state_dict: &StateDict, device: &Device) -> Result<Self> {
        Ok(Self {
            spectral_0: LinearNorm::load(state_dict, "ref_enc.spectral.0", device)?,
            spectral_3: LinearNorm::load(state_dict, "ref_enc.spectral.3", device)?,
            temporal_0: Conv1dGLU::load(state_dict, "ref_enc.temporal.0", device)?,
            temporal_1: Conv1dGLU::load(state_dict, "ref_enc.temporal.1", device)?,
            self_attn: MultiHeadSelfAttention::load(state_dict, "ref_enc.slf_attn", device)?,
            fc: LinearNorm::load(state_dict, "ref_enc.fc", device)?,
        })
    }

    /// Forward pass
    /// x: [batch, 704, time] - reference spectrogram (first 704 bins of 1025-bin STFT, pre-truncated by caller)
    /// mask: [batch, 1, time] - attention mask (0=padded, 1=valid)
    /// Returns: [batch, 512, 1] - speaker embedding
    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Python: x = x.transpose(1, 2)  → [batch, time, n_mel]
        let x = x.transpose(1, 2)?;

        // === spectral ===
        // FC(n_mel → 128)
        let x = self.spectral_0.forward(&x)?;
        // Mish
        let x = mish(&x)?;

        // FC(128 → 128)
        let x = self.spectral_3.forward(&x)?;
        // Mish
        let x = mish(&x)?;

        // === temporal ===
        // x.transpose(1, 2) → [batch, 128, time]
        let x = x.transpose(1, 2)?;
        // Conv1dGLU
        let x = self.temporal_0.forward(&x)?;
        // Conv1dGLU
        let x = self.temporal_1.forward(&x)?;
        // x.transpose(1, 2) → [batch, time, 128]
        let x = x.transpose(1, 2)?;

        // === self-attention ===
        let dims = x.dims();
        let batch = dims[0];
        let time = dims[1];

        // Python: x = x.masked_fill(mask.unsqueeze(-1), 0) → zero padded positions before attention
        // Our mask from caller is [batch, 1, time] with 1=valid, 0=invalid
        // x is [batch, time, 128], need mask as [batch, time, 1] to broadcast
        let x_attn_input = if mask.dims()[1] == 1 {
            let m_2d = mask.reshape((batch, time))?; // [batch, time]
            let m_3d = m_2d.unsqueeze(2)?; // [batch, time, 1]
            x.broadcast_mul(&m_3d)?  // zeros invalid positions
        } else {
            x.clone()
        };

        let slf_attn_mask = if mask.dims()[1] == 1 {
            let m_squeezed = mask.squeeze(1)?;
            let ones = Tensor::full(1.0f32, &[batch, time], x.device())?;
            let inverted = ones.broadcast_sub(&m_squeezed)?;
            let mask_2d = inverted.unsqueeze(1)?;
            Some(mask_2d.broadcast_as((batch, time, time))?)
        } else {
            None
        };

        let x = self.self_attn.forward(&x_attn_input, slf_attn_mask.as_ref())?;

        // === fc ===
        let x = self.fc.forward(&x)?;

        // === temporal average pooling ===
        // Python: if mask exists, mask_fill padded positions to 0, then sum / count
        // refer_mask is [batch, 1, time] with 1=valid, 0=invalid
        let valid_mask = if mask.dims()[1] == 1 {
            Some(mask.squeeze(1)?)  // [batch, time], 1=valid
        } else {
            None
        };

        let w = self.temporal_avg_pool(&x, valid_mask.as_ref())?;

        // Return [batch, 512, 1]
        w.unsqueeze(2).map_err(|e| crate::Error::InferenceError(e.to_string()))
    }

    fn temporal_avg_pool(&self, x: &Tensor, valid_mask: Option<&Tensor>) -> Result<Tensor> {
        // x: [batch, time, 512]
        // valid_mask: [batch, time] - 1=valid, 0=invalid (padded)
        if let Some(m) = valid_mask {
            // m is already 1=valid, 0=invalid
            let m_expanded = m.unsqueeze(2)?; // [batch, time, 1]
            let x_masked = x.broadcast_mul(&m_expanded)?;

            // Count valid positions per sample
            let valid_count = m.sum_keepdim(1)?; // [batch, 1]

            // Sum over time, then divide by count
            let sum = x_masked.sum_keepdim(1)?; // [batch, 1, 512]
            let valid_count_3d = valid_count.broadcast_as((sum.dims()[0], 1, sum.dims()[2]))?; // [batch, 1, 512]
            let result = sum.broadcast_div(&valid_count_3d)?;
            // Squeeze dim 1: [batch, 1, 512] → [batch, 512]
            Ok(result.squeeze(1)?)
        } else {
            Ok(x.mean_keepdim(1)?.squeeze(1)?)
        }
    }
}
