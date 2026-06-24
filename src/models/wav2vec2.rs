//! Wav2Vec2 / HuBERT feature extractor — pure candle implementation.
//!
//! Architecture (chinese-hubert-base, 361 MB):
//!   CNN feature extractor (7 layers, total stride 320, 16 kHz → 50 fps)
//!   Feature projection: LayerNorm(512) → Linear(512→768)
//!   Positional conv: Conv1d(768, 768, 128, groups=16, pad=64) + GELU + residual
//!   Encoder LayerNorm(768)
//!   12 × Transformer layer (pre-LN, 768 hidden, 12 heads, 3072 FFN, GELU)
//!
//! Weights loaded from models/hubert.safetensors (produced by scripts/extract_hubert_weights.py).

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};

// ── CNN feature extractor ─────────────────────────────────────────────────────

struct ConvLayerNorm {
    conv: Conv1d,
    norm: LayerNorm,
}

struct ConvLayerNoNorm {
    conv: Conv1d,
}

enum CnnLayer {
    WithNorm(ConvLayerNorm),
    NoNorm(ConvLayerNoNorm),
}

impl CnnLayer {
    fn load_with_norm(vb: VarBuilder, in_ch: usize, out_ch: usize, kernel: usize, stride: usize) -> Result<Self> {
        let cfg = Conv1dConfig { stride, padding: 0, dilation: 1, groups: 1, cudnn_fwd_algo: None };
        let conv = candle_nn::conv1d_no_bias(in_ch, out_ch, kernel, cfg, vb.pp("conv"))?;
        let norm = candle_nn::layer_norm(out_ch, 1e-5, vb.pp("layer_norm"))?;
        Ok(Self::WithNorm(ConvLayerNorm { conv, norm }))
    }

    fn load_no_norm(vb: VarBuilder, in_ch: usize, out_ch: usize, kernel: usize, stride: usize) -> Result<Self> {
        let cfg = Conv1dConfig { stride, padding: 0, dilation: 1, groups: 1, cudnn_fwd_algo: None };
        let conv = candle_nn::conv1d_no_bias(in_ch, out_ch, kernel, cfg, vb.pp("conv"))?;
        Ok(Self::NoNorm(ConvLayerNoNorm { conv }))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T]
        match self {
            Self::WithNorm(l) => {
                let x = l.conv.forward(x)?;
                // GroupNorm(num_groups=C) = InstanceNorm: normalize per (batch, channel)
                // Implemented as: transpose → LayerNorm over T → transpose back
                // Actually for GroupNorm(groups=C, channels=C) each channel is normalized independently.
                // The layer_norm here has dim=C (one weight per channel) and we apply it
                // after transposing so LN sees [B*C, T] effectively.
                // Simpler: use the fact that LayerNorm over last dim applied on [B, T, C] = per-token norm.
                // For instance norm: normalize each [B, c, T] slice over T.
                let (b, c, t) = x.dims3()?;
                // Reshape [B, C, T] → [B*C, T, 1], apply LN over last dim=1 → then back
                // But our norm has weight/bias of shape [C]. Use manual instance norm instead.
                let x = x.reshape((b * c, t, 1))?;
                // Manual InstanceNorm: normalize each (b,c) slice over T
                let mean = x.mean_keepdim(1)?;                     // [B*C, 1, 1]
                let diff = x.broadcast_sub(&mean)?;                // [B*C, T, 1]
                let var  = (diff.sqr()?.mean_keepdim(1))?;        // [B*C, 1, 1]
                let x_norm = diff.broadcast_div(&(var + 1e-5f64)?.sqrt()?)?;
                let x_norm = x_norm.reshape((b, c, t))?;
                // Apply affine: weight [C] and bias [C] → reshape to [1, C, 1] for broadcast
                let weight = l.norm.weight().reshape((1, c, 1))?;
                let bias   = l.norm.bias()
                    .ok_or_else(|| candle_core::Error::Msg("norm bias missing".into()))?
                    .reshape((1, c, 1))?;
                let x_norm = x_norm.broadcast_mul(&weight)?.broadcast_add(&bias)?;
                gelu(&x_norm)
            }
            Self::NoNorm(l) => gelu(&l.conv.forward(x)?),
        }
    }
}

// ── Attention ─────────────────────────────────────────────────────────────────

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    n_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn load(vb: VarBuilder, hidden: usize, n_heads: usize) -> Result<Self> {
        let head_dim = hidden / n_heads;
        let vb = vb.pp("attention");
        let q   = candle_nn::linear(hidden, hidden, vb.pp("q_proj"))?;
        let k   = candle_nn::linear(hidden, hidden, vb.pp("k_proj"))?;
        let v   = candle_nn::linear(hidden, hidden, vb.pp("v_proj"))?;
        let out = candle_nn::linear(hidden, hidden, vb.pp("out_proj"))?;
        Ok(Self { q, k, v, out, n_heads, head_dim, scale: (head_dim as f64).powf(-0.5) })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = self.q.forward(x)?.reshape((b, t, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = self.k.forward(x)?.reshape((b, t, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = self.v.forward(x)?.reshape((b, t, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let attn = q.matmul(&k.transpose(2, 3)?.contiguous()?)?.affine(self.scale, 0.0)?;
        let attn = candle_nn::ops::softmax(&attn, candle_core::D::Minus1)?;
        let out  = attn.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((b, t, self.n_heads * self.head_dim))?;
        self.out.forward(&out)
    }
}

// ── FFN ───────────────────────────────────────────────────────────────────────

struct Ffn {
    intermediate: Linear,
    output: Linear,
}

impl Ffn {
    fn load(vb: VarBuilder, hidden: usize, intermediate: usize) -> Result<Self> {
        let vb = vb.pp("feed_forward");
        Ok(Self {
            intermediate: candle_nn::linear(hidden, intermediate, vb.pp("intermediate_dense"))?,
            output:       candle_nn::linear(intermediate, hidden, vb.pp("output_dense"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.output.forward(&gelu(&self.intermediate.forward(x)?)?)
    }
}

// ── Transformer layer ─────────────────────────────────────────────────────────

struct TransformerLayer {
    ln1: LayerNorm,
    attn: Attention,
    ln2: LayerNorm,
    ffn: Ffn,
}

impl TransformerLayer {
    fn load(vb: VarBuilder, hidden: usize, n_heads: usize, ffn_dim: usize) -> Result<Self> {
        Ok(Self {
            ln1:  candle_nn::layer_norm(hidden, 1e-5, vb.pp("layer_norm"))?,
            attn: Attention::load(vb.clone(), hidden, n_heads)?,
            ln2:  candle_nn::layer_norm(hidden, 1e-5, vb.pp("final_layer_norm"))?,
            ffn:  Ffn::load(vb, hidden, ffn_dim)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // POST-LN (BERT style): attn → add residual → LN → FFN → add residual → LN
        let x = self.ln1.forward(&(x + self.attn.forward(x)?)?)?;
        let x = self.ln2.forward(&(&x + self.ffn.forward(&x)?)?)?;
        Ok(x)
    }
}

// ── Positional conv embedding ─────────────────────────────────────────────────

struct PosConvEmbed {
    conv: Conv1d,
}

impl PosConvEmbed {
    fn load(vb: VarBuilder, hidden: usize, groups: usize, kernel: usize) -> Result<Self> {
        let cfg = Conv1dConfig { stride: 1, padding: kernel / 2, dilation: 1, groups, cudnn_fwd_algo: None };
        // weight_norm has already been baked into the weight by extract_hubert_weights.py
        let conv = Conv1d::new(
            vb.get((hidden, hidden / groups, kernel), "encoder.pos_conv_embed.conv.weight")?,
            Some(vb.get(hidden, "encoder.pos_conv_embed.conv.bias")?),
            cfg,
        );
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, T, C] → transpose → conv → gelu → transpose → add residual
        let xt = x.transpose(1, 2)?;               // [B, C, T]
        let out = gelu(&self.conv.forward(&xt)?)?;  // [B, C, T]
        let out = out.transpose(1, 2)?;             // [B, T, C]
        // conv output may be T+1 due to padding; trim to original T
        let t = x.dim(1)?;
        let out = out.narrow(1, 0, t)?;
        (x + out)
    }
}

// ── Top-level model ───────────────────────────────────────────────────────────

pub struct Wav2Vec2Model {
    pub(crate) cnn: Vec<CnnLayer>,
    fp_ln: LayerNorm,
    fp_proj: Linear,
    pos_conv: PosConvEmbed,
    enc_ln: LayerNorm,
    layers: Vec<TransformerLayer>,
    dtype: DType,
}

impl CnnLayer {
    pub fn norm_weight(&self) -> Option<&Tensor> {
        match self { Self::WithNorm(l) => Some(l.norm.weight()), Self::NoNorm(_) => None }
    }
}

impl Wav2Vec2Model {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        // CNN: layer 0 has GroupNorm, layers 1-6 have no norm
        let kernels = [10usize, 3, 3, 3, 3, 2, 2];
        let strides = [5usize,  2, 2, 2, 2, 2, 2];
        let mut cnn = Vec::with_capacity(7);
        let vb_cnn = vb.pp("feature_extractor");
        for i in 0..7 {
            let vb_l = vb_cnn.pp(format!("conv_layers.{i}"));
            let in_ch = if i == 0 { 1 } else { 512 };
            if i == 0 {
                cnn.push(CnnLayer::load_with_norm(vb_l, in_ch, 512, kernels[i], strides[i])?);
            } else {
                cnn.push(CnnLayer::load_no_norm(vb_l, in_ch, 512, kernels[i], strides[i])?);
            }
        }

        // Feature projection
        let vb_fp = vb.pp("feature_projection");
        let fp_ln   = candle_nn::layer_norm(512, 1e-5, vb_fp.pp("layer_norm"))?;
        let fp_proj = candle_nn::linear(512, 768, vb_fp.pp("projection"))?;

        // Positional conv (weight_norm baked in during extraction)
        let pos_conv = PosConvEmbed::load(vb.clone(), 768, 16, 128)?;

        // Encoder layer norm
        let enc_ln = candle_nn::layer_norm(768, 1e-5, vb.pp("encoder.layer_norm"))?;

        // 12 transformer layers
        let mut layers = Vec::with_capacity(12);
        for i in 0..12 {
            layers.push(TransformerLayer::load(
                vb.pp(format!("encoder.layers.{i}")), 768, 12, 3072,
            )?);
        }

        Ok(Self { cnn, fp_ln, fp_proj, pos_conv, enc_ln, layers, dtype })
    }

    /// Extract features. Returns `last_hidden_state` [1, T_out, 768].
    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        self.forward_impl(audio, false, None)
    }

    pub fn forward_debug(&self, audio: &Tensor) -> Result<Vec<Tensor>> {
        let mut checkpoints = Vec::new();
        let mut x = audio.unsqueeze(1)?;
        for layer in &self.cnn {
            x = layer.forward(&x)?;
        }
        checkpoints.push(x.clone());           // [0] after all CNN
        let x = x.transpose(1, 2)?.contiguous()?;
        let x_ln = self.fp_ln.forward(&x)?;
        checkpoints.push(x_ln.clone());        // [1] after fp_ln
        let x = self.fp_proj.forward(&x_ln)?;
        checkpoints.push(x.clone());           // [2] after fp_proj
        let x = self.pos_conv.forward(&x)?;
        checkpoints.push(x.clone());           // [3] after pos_conv
        let x = self.enc_ln.forward(&x)?;
        checkpoints.push(x.clone());           // [4] after enc_ln
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < 2 {
                checkpoints.push(x.clone()); // [5] after layer 0, [6] after layer 1
            }
        }
        checkpoints.push(x);                   // final
        Ok(checkpoints)
    }

    fn forward_impl(&self, audio: &Tensor, _debug: bool, _out: Option<&mut Vec<Tensor>>) -> Result<Tensor> {
        // Cast F32 input audio to model dtype (BF16 or F32)
        let audio = if self.dtype != audio.dtype() { audio.to_dtype(self.dtype)? } else { audio.clone() };
        let mut x = audio.unsqueeze(1)?;  // [1, 1, T]
        for layer in &self.cnn {
            x = layer.forward(&x)?;  // [B, 512, T']
        }
        // Transpose + contiguous so candle's CUDA LayerNorm sees a packed tensor
        let x = x.transpose(1, 2)?.contiguous()?;  // [B, T', 512]
        let x = self.fp_proj.forward(&self.fp_ln.forward(&x)?)?;  // [B, T', 768]
        let x = self.pos_conv.forward(&x)?;
        let mut x = self.enc_ln.forward(&x)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        // Always return F32 — downstream (HuBERT semantic tokenizer) expects F32
        if self.dtype != DType::F32 { x.to_dtype(DType::F32) } else { Ok(x) }
    }

    pub fn cnn_norm_weight(&self) -> Result<Tensor> {
        self.cnn[0].norm_weight().ok_or_else(|| candle_core::Error::Msg("no norm".into()))?.clone().flatten_all()
    }

    pub fn load_from_file(path: &std::path::Path, device: &Device) -> crate::Result<Self> {
        Self::load_from_file_with_dtype(path, device, DType::F32)
    }

    pub fn load_from_file_bf16(path: &std::path::Path, device: &Device) -> crate::Result<Self> {
        Self::load_from_file_with_dtype(path, device, DType::BF16)
    }

    pub fn load_from_file_with_dtype(path: &std::path::Path, device: &Device, dtype: DType) -> crate::Result<Self> {
        let weights = candle_core::safetensors::load(path, device)?;
        let vb = VarBuilder::from_tensors(weights, dtype, device);
        Ok(Self::load(vb)?)
    }

    pub fn dtype(&self) -> DType { self.dtype }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn gelu(x: &Tensor) -> Result<Tensor> {
    x.gelu_erf()
}
