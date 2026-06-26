//! SoVITS Encoder P (Text/Semantic Encoder)
//!
//! EncP processes semantic tokens and text features to produce
//! parameters for the flow model.

use crate::utils::{LayerNorm, StateDict};
use crate::Result;
use candle_core::{DType, Device, Module, Tensor};

/// Text Encoder layer with self-attention
#[derive(Debug, Clone)]
pub struct EncoderLayer {
    self_attn: SelfAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl EncoderLayer {
    pub fn load(
        state_dict: &StateDict,
        prefix: &str,
        device: &Device,
        layer_idx: usize,
        n_heads: usize,
        dtype: DType,
    ) -> Result<Self> {
        // Model uses format: enc_p.encoder_ssl.attn_layers.0.conv_q.weight
        let self_attn = SelfAttention::load(state_dict, prefix, device, layer_idx, n_heads, dtype)?;

        // FFN layers
        let ffn = FeedForward::load(state_dict, prefix, device, layer_idx, dtype)?;

        // Layer norms - keep F32 for numerical stability
        let norm1 = LayerNorm::new(
            state_dict
                .get(&format!("{}.norm_layers_1.{}.gamma", prefix, layer_idx))?
                .to_device(device)?
                .to_dtype(DType::F32)?,
            state_dict
                .get(&format!("{}.norm_layers_1.{}.beta", prefix, layer_idx))?
                .to_device(device)?
                .to_dtype(DType::F32)?,
        );
        let norm2 = LayerNorm::new(
            state_dict
                .get(&format!("{}.norm_layers_2.{}.gamma", prefix, layer_idx))?
                .to_device(device)?
                .to_dtype(DType::F32)?,
            state_dict
                .get(&format!("{}.norm_layers_2.{}.beta", prefix, layer_idx))?
                .to_device(device)?
                .to_dtype(DType::F32)?,
        );

        Ok(Self {
            self_attn,
            ffn,
            norm1,
            norm2,
        })
    }

    pub fn forward(&self, x: &Tensor, x_mask: &Tensor) -> Result<Tensor> {
        // Post-norm attention: y = attn(x); x = norm1(x + y)
        // Matches Python: x = self.norm_layers_1[i](x + attn(x))
        let attn_out = self.self_attn.forward(x, x_mask)?;
        let x = self.norm1.forward(&x.add(&attn_out)?)?;

        // Post-norm FFN: y = ffn(x); x = norm2(x + y)
        // Matches Python: x = self.norm_layers_2[i](x + ffn(x))
        let ffn_out = self.ffn.forward(&x)?;
        let x = self.norm2.forward(&x.add(&ffn_out)?)?;

        Ok(x.broadcast_mul(x_mask)?)
    }
}

/// Self-attention mechanism
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SelfAttention {
    conv_q: candle_nn::Conv1d,
    conv_k: candle_nn::Conv1d,
    conv_v: candle_nn::Conv1d,
    conv_o: candle_nn::Conv1d,
    n_heads: usize,
    head_dim: usize,
    emb_rel_k: Option<Tensor>,
    emb_rel_v: Option<Tensor>,
}

impl SelfAttention {
    pub fn load(
        state_dict: &StateDict,
        prefix: &str,
        device: &Device,
        layer_idx: usize,
        _n_heads: usize,
        dtype: DType,
    ) -> Result<Self> {
        let conv_q = load_conv1d(
            state_dict,
            &format!("{}.attn_layers.{}.conv_q", prefix, layer_idx),
            device,
            dtype,
        )?;
        let conv_k = load_conv1d(
            state_dict,
            &format!("{}.attn_layers.{}.conv_k", prefix, layer_idx),
            device,
            dtype,
        )?;
        let conv_v = load_conv1d(
            state_dict,
            &format!("{}.attn_layers.{}.conv_v", prefix, layer_idx),
            device,
            dtype,
        )?;
        let conv_o = load_conv1d(
            state_dict,
            &format!("{}.attn_layers.{}.conv_o", prefix, layer_idx),
            device,
            dtype,
        )?;

        let hidden = conv_q.weight().dims()[0];
        // Derive head_dim from emb_rel_k shape [n_heads_rel, window_size, head_dim]
        let head_dim = if let Ok(emb) =
            state_dict.get(&format!("{}.attn_layers.{}.emb_rel_k", prefix, layer_idx))
        {
            emb.dims()[2]
        } else {
            hidden / 2 // fallback: assume 2 heads (model default)
        };
        let n_heads = hidden / head_dim;

        let emb_rel_k =
            if state_dict.contains(&format!("{}.attn_layers.{}.emb_rel_k", prefix, layer_idx)) {
                Some(
                    state_dict
                        .get(&format!("{}.attn_layers.{}.emb_rel_k", prefix, layer_idx))?
                        .to_device(device)?
                        .to_dtype(dtype)?,
                )
            } else {
                None
            };
        let emb_rel_v =
            if state_dict.contains(&format!("{}.attn_layers.{}.emb_rel_v", prefix, layer_idx)) {
                Some(
                    state_dict
                        .get(&format!("{}.attn_layers.{}.emb_rel_v", prefix, layer_idx))?
                        .to_device(device)?
                        .to_dtype(dtype)?,
                )
            } else {
                None
            };

        Ok(Self {
            conv_q,
            conv_k,
            conv_v,
            conv_o,
            n_heads,
            head_dim,
            emb_rel_k,
            emb_rel_v,
        })
    }

    pub fn forward(&self, x: &Tensor, x_mask: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let (batch, channels, seq_len) = (dims[0], dims[1], dims[2]);

        // Project Q, K, V: [batch, channels, seq_len]
        let q = self.conv_q.forward(x)?;
        let k = self.conv_k.forward(x)?;
        let v = self.conv_v.forward(x)?;

        // Reshape: [batch, n_heads*head_dim, T] → [batch, n_heads, T, head_dim]
        // Python: q.view(b, n_heads, k_ch, T).transpose(2, 3)
        let q = q
            .reshape((batch, self.n_heads, self.head_dim, seq_len))?
            .transpose(2, 3)?;
        let k = k
            .reshape((batch, self.n_heads, self.head_dim, seq_len))?
            .transpose(2, 3)?;
        let v = v
            .reshape((batch, self.n_heads, self.head_dim, seq_len))?
            .transpose(2, 3)?;
        // All: [batch, n_heads, seq_len, head_dim]

        // Scale q: Python scales q BEFORE content and relative key matmuls
        let scale = (self.head_dim as f64).powf(-0.5);
        let q_scaled = (q * scale)?;

        // Content attention scores: [batch, n_heads, T, T]
        let k_t = k.transpose(2, 3)?;
        let mut scores = q_scaled.matmul(&k_t)?;

        // Relative position attention (Shaw et al.) – only for self-attention with window
        if let Some(emb_rel_k) = &self.emb_rel_k {
            let key_rel_emb = Self::get_relative_embeddings(emb_rel_k, seq_len)?;
            // key_rel_emb: [1, 2T-1, d]. Expand across heads: [1, n_heads, d, 2T-1]
            let kre_t = key_rel_emb.transpose(1, 2)?; // [1, d, 2T-1]
                                                      // repeat for each head: [n_heads, d, 2T-1] → unsqueeze → [1, n_heads, d, 2T-1]
            let kre_t_exp = kre_t.repeat(vec![self.n_heads, 1, 1])?.unsqueeze(0)?;
            // q_scaled [b, n_h, T, d] × [1, n_h, d, 2T-1] → [b, n_h, T, 2T-1]
            let rel_logits = q_scaled.matmul(&kre_t_exp)?;
            let scores_local = Self::relative_to_absolute(rel_logits)?;
            scores = scores.add(&scores_local)?;
        }

        // Apply attention mask: masked_fill(mask == 0, -1e4)
        // x_mask: [batch, 1, T]; expand to [batch, n_heads, T, T] masking key positions
        let mask_2d = x_mask.squeeze(1)?;
        let mask_4d = mask_2d.unsqueeze(1)?.unsqueeze(2)?; // [b, 1, 1, T] (key dim)
        let mask_bc = mask_4d.broadcast_as(scores.dims())?;
        let ones = Tensor::ones(mask_bc.dims(), scores.dtype(), x.device())?;
        let inv_mask = ones.sub(&mask_bc)?.broadcast_mul(
            &Tensor::full(-1e4f32, mask_bc.dims(), x.device())?.to_dtype(scores.dtype())?,
        )?;
        scores = scores.broadcast_mul(&mask_bc)?.add(&inv_mask)?;

        let attn_probs =
            candle_nn::ops::softmax(&scores.to_dtype(DType::F32)?, candle_core::D::Minus1)?
                .to_dtype(scores.dtype())?;
        let mut attn_out = attn_probs.matmul(&v)?; // [b, n_heads, T, head_dim]

        // Relative value attention
        if let Some(emb_rel_v) = &self.emb_rel_v {
            let rel_weights = Self::absolute_to_relative(attn_probs)?;
            let val_rel_emb = Self::get_relative_embeddings(emb_rel_v, seq_len)?;
            // val_rel_emb: [1, 2T-1, d]. Expand across heads: [1, n_heads, 2T-1, d]
            let vre_exp = val_rel_emb.repeat(vec![self.n_heads, 1, 1])?.unsqueeze(0)?;
            // [b, n_h, T, 2T-1] × [1, n_h, 2T-1, d] → [b, n_h, T, d]
            let value_local = rel_weights.matmul(&vre_exp)?;
            attn_out = attn_out.add(&value_local)?;
        }

        // Reshape back: [b, n_heads, T, head_dim] → [b, channels, T]
        let attn_out = attn_out
            .transpose(2, 3)?
            .contiguous()?
            .reshape((batch, channels, seq_len))?;
        Ok(self.conv_o.forward(&attn_out)?)
    }

    /// Get relative position embeddings for sequence length L.
    /// emb: [1, 2*window+1, k_ch] → output: [1, 2*L-1, k_ch]
    fn get_relative_embeddings(emb: &Tensor, length: usize) -> Result<Tensor> {
        let emb_len = emb.dims()[1];
        let window_size = (emb_len - 1) / 2;
        let k_ch = emb.dims()[2];
        let device = emb.device();

        let pad_length = if length > window_size + 1 {
            length - window_size - 1
        } else {
            0
        };
        let slice_start = if window_size + 1 > length {
            window_size + 1 - length
        } else {
            0
        };
        let slice_end = slice_start + 2 * length - 1;

        // Pad emb on dim 1 with zeros on both sides
        let padded = if pad_length > 0 {
            let zeros = Tensor::zeros((1, pad_length, k_ch), emb.dtype(), device)?;
            Tensor::cat(&[&zeros, emb, &zeros], 1)?
        } else {
            emb.clone()
        };
        Ok(padded.narrow(1, slice_start, slice_end - slice_start)?)
    }

    /// Convert relative logits [b, h, L, 2L-1] to absolute [b, h, L, L].
    /// Python: _relative_position_to_absolute_position
    fn relative_to_absolute(x: Tensor) -> Result<Tensor> {
        let (b, h, l, _) = x.dims4()?;
        let device = x.device();
        let zeros = Tensor::zeros((b, h, l, 1), x.dtype(), device)?;
        let x = Tensor::cat(&[&x, &zeros], 3)?;
        let x = x.reshape((b, h, 2 * l * l))?;
        let zeros2 = Tensor::zeros((b, h, l - 1), x.dtype(), device)?;
        let x = Tensor::cat(&[&x, &zeros2], 2)?;
        let x = x.reshape((b, h, l + 1, 2 * l - 1))?;
        Ok(x.narrow(2, 0, l)?.narrow(3, l - 1, l)?.contiguous()?)
    }

    /// Convert absolute attn weights [b, h, L, L] to relative [b, h, L, 2L-1].
    /// Python: _absolute_position_to_relative_position
    fn absolute_to_relative(x: Tensor) -> Result<Tensor> {
        let (b, h, l, _) = x.dims4()?;
        let device = x.device();
        let zeros = Tensor::zeros((b, h, l, l - 1), x.dtype(), device)?;
        let x = Tensor::cat(&[&x, &zeros], 3)?;
        let x = x.reshape((b, h, 2 * l * l - l))?;
        let zeros2 = Tensor::zeros((b, h, l), x.dtype(), device)?;
        let x = Tensor::cat(&[&zeros2, &x], 2)?;
        let x = x.reshape((b, h, l, 2 * l))?;
        Ok(x.narrow(3, 1, 2 * l - 1)?.contiguous()?)
    }
}

/// Feed-forward network
#[derive(Debug, Clone)]
pub struct FeedForward {
    conv_1: candle_nn::Conv1d,
    conv_2: candle_nn::Conv1d,
}

impl FeedForward {
    pub fn load(
        state_dict: &StateDict,
        prefix: &str,
        device: &Device,
        layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let conv_1 = load_conv1d(
            state_dict,
            &format!("{}.ffn_layers.{}.conv_1", prefix, layer_idx),
            device,
            dtype,
        )?;
        let conv_2 = load_conv1d(
            state_dict,
            &format!("{}.ffn_layers.{}.conv_2", prefix, layer_idx),
            device,
            dtype,
        )?;

        Ok(Self { conv_1, conv_2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_1.forward(x)?;
        let x = x.relu()?; // Python FFN default activation is ReLU (not GELU)
        Ok(self.conv_2.forward(&x)?)
    }
}

fn load_conv1d(
    state_dict: &StateDict,
    prefix: &str,
    device: &Device,
    dtype: DType,
) -> Result<candle_nn::Conv1d> {
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

    let config = candle_nn::Conv1dConfig {
        padding,
        stride: 1,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: Default::default(),
    };

    Ok(candle_nn::Conv1d::new(weight, bias, config))
}

/// Multi-Head Cross-Attention
#[derive(Debug, Clone)]
struct MultiHeadAttention {
    conv_q: candle_nn::Conv1d,
    conv_k: candle_nn::Conv1d,
    conv_v: candle_nn::Conv1d,
    conv_o: candle_nn::Conv1d,
    n_heads: usize,
    k_channels: usize,
}

impl MultiHeadAttention {
    fn load(
        state_dict: &StateDict,
        prefix: &str,
        device: &Device,
        n_heads: usize,
        dtype: DType,
    ) -> Result<Self> {
        let conv_q = load_conv1d(state_dict, &format!("{}.conv_q", prefix), device, dtype)?;
        let conv_k = load_conv1d(state_dict, &format!("{}.conv_k", prefix), device, dtype)?;
        let conv_v = load_conv1d(state_dict, &format!("{}.conv_v", prefix), device, dtype)?;
        let conv_o = load_conv1d(state_dict, &format!("{}.conv_o", prefix), device, dtype)?;

        let channels = conv_q.weight().dims()[0];
        let k_channels = channels / n_heads;

        Ok(Self {
            conv_q,
            conv_k,
            conv_v,
            conv_o,
            n_heads,
            k_channels,
        })
    }

    /// Cross-attention: query from x, key/value from c
    /// x: [batch, channels, seq_x], c: [batch, channels, seq_c]
    /// attn_mask: [batch, 1, seq_x, seq_c] (broadcast mask)
    fn forward(&self, x: &Tensor, c: &Tensor, attn_mask: &Tensor) -> Result<Tensor> {
        let q = self.conv_q.forward(x)?;
        let k = self.conv_k.forward(c)?;
        let v = self.conv_v.forward(c)?;

        let batch = q.dims()[0];
        let channels = q.dims()[1];
        let seq_q = q.dims()[2];
        let seq_k = k.dims()[2];
        let n_heads = self.n_heads;
        let k_ch = self.k_channels;

        // Reshape: [batch, n_heads*k_ch, seq] -> [batch, n_heads, k_ch, seq] -> transpose -> [batch, n_heads, seq, k_ch]
        // Python: q.view(b, n_heads, k_ch, seq).transpose(2, 3)
        let q_heads = q.reshape((batch, n_heads, k_ch, seq_q))?.transpose(2, 3)?;
        let k_heads = k.reshape((batch, n_heads, k_ch, seq_k))?.transpose(2, 3)?;
        let v_heads = v.reshape((batch, n_heads, k_ch, seq_k))?.transpose(2, 3)?;
        // All: [batch, n_heads, seq, k_ch]

        // Scaled dot-product attention: Q @ K^T
        let k_t = k_heads.transpose(2, 3)?; // [batch, n_heads, k_ch, seq_k]
        let scores_raw = q_heads.matmul(&k_t)?; // [batch, n_heads, seq_q, seq_k]
        let scale = Tensor::full(
            (k_ch as f32).sqrt().recip(),
            scores_raw.dims(),
            scores_raw.device(),
        )?
        .to_dtype(scores_raw.dtype())?;
        let scores = scores_raw.broadcast_mul(&scale)?;

        // Apply mask: scores * mask + (1 - mask) * (-1e9)
        // attn_mask: [batch, 1, seq_q, seq_k] -> broadcast to [batch, n_heads, seq_q, seq_k]
        let mask_bc = attn_mask.broadcast_as(scores.dims())?;
        let dims = scores.dims();
        let ones = Tensor::ones(dims, scores.dtype(), scores.device())?;
        let neg_inf = ones.broadcast_sub(&mask_bc)?.broadcast_mul(
            &Tensor::full(-1e9f32, dims, scores.device())?.to_dtype(scores.dtype())?,
        )?;
        let masked_scores = scores.broadcast_mul(&mask_bc)?.add(&neg_inf)?;

        // Softmax over last dimension (seq_k)
        let attn_probs =
            candle_nn::ops::softmax(&masked_scores.to_dtype(DType::F32)?, candle_core::D::Minus1)?
                .to_dtype(masked_scores.dtype())?;

        // Attention output: attn_probs @ V, then transpose+reshape back to [batch, channels, seq_q]
        let attn_out = attn_probs.matmul(&v_heads)?; // [batch, n_heads, seq_q, k_ch]
        let attn_out = attn_out
            .transpose(2, 3)?
            .contiguous()?
            .reshape((batch, channels, seq_q))?;

        // Output projection
        Ok(self.conv_o.forward(&attn_out)?)
    }
}

/// MRTE (Multi-Reference Timbre Encoder)
#[derive(Debug, Clone)]
pub struct MRTE {
    c_pre: Option<candle_nn::Conv1d>,
    c_post: Option<candle_nn::Conv1d>,
    text_pre: Option<candle_nn::Conv1d>,
    cross_attention: Option<MultiHeadAttention>,
}

impl MRTE {
    pub fn load(
        state_dict: &StateDict,
        prefix: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let c_pre = if state_dict.contains(&format!("{}.c_pre.weight", prefix)) {
            Some(load_conv1d(
                state_dict,
                &format!("{}.c_pre", prefix),
                device,
                dtype,
            )?)
        } else {
            None
        };

        let c_post = if state_dict.contains(&format!("{}.c_post.weight", prefix)) {
            Some(load_conv1d(
                state_dict,
                &format!("{}.c_post", prefix),
                device,
                dtype,
            )?)
        } else {
            None
        };

        let text_pre = if state_dict.contains(&format!("{}.text_pre.weight", prefix)) {
            Some(load_conv1d(
                state_dict,
                &format!("{}.text_pre", prefix),
                device,
                dtype,
            )?)
        } else {
            None
        };

        let cross_attention =
            if state_dict.contains(&format!("{}.cross_attention.conv_q.weight", prefix)) {
                Some(MultiHeadAttention::load(
                    state_dict,
                    &format!("{}.cross_attention", prefix),
                    device,
                    4,
                    dtype,
                )?)
            } else {
                None
            };

        Ok(Self {
            c_pre,
            c_post,
            text_pre,
            cross_attention,
        })
    }

    pub fn forward(
        &self,
        ssl_enc: &Tensor,
        ssl_mask: &Tensor,
        text: &Tensor,
        text_mask: &Tensor,
        ge: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Project SSL features to hidden size
        let ssl_proj_out = if let Some(c_pre) = &self.c_pre {
            c_pre.forward(&ssl_enc.broadcast_mul(ssl_mask)?)?
        } else {
            ssl_enc.clone()
        };

        // Project text features to hidden size
        let text_proj = if let Some(text_pre) = &self.text_pre {
            text_pre.forward(&text.broadcast_mul(text_mask)?)?
        } else {
            text.clone()
        };

        // Build cross-attention mask: text_mask.unsqueeze(2) * ssl_mask.unsqueeze(-1)
        // [batch, 1, seq_text] * [batch, 1, 1, seq_ssl] -> [batch, 1, seq_ssl, seq_text]
        let text_mask_3d = text_mask.reshape((text_mask.dims()[0], 1, text_mask.dims()[2]))?;
        let ssl_mask_4d = ssl_mask.reshape((ssl_mask.dims()[0], 1, ssl_mask.dims()[2], 1))?;
        let attn_mask = text_mask_3d.broadcast_mul(&ssl_mask_4d)?;

        // Cross-attention: SSL queries, text keys/values
        let x = if let Some(attn) = &self.cross_attention {
            let ssl_masked = ssl_proj_out.broadcast_mul(ssl_mask)?;
            let text_masked = text_proj.broadcast_mul(text_mask)?;
            attn.forward(&ssl_masked, &text_masked, &attn_mask)?
        } else {
            ssl_proj_out.clone()
        };

        // Residual connection: use POST-c_pre features (matching Python MRTE line 42)
        // Python: ssl_enc = self.c_pre(ssl_enc * ssl_mask); then x = cross_attn(...) + ssl_enc + ge
        let x = x.add(&ssl_proj_out)?;
        let ge = match ge {
            Some(g) => g.clone(),
            None => Tensor::zeros((x.dims()[0], x.dims()[1], 1), x.dtype(), x.device())?,
        };
        let ge_broadcasted = if ge.dims()[2] == 1 && x.dims()[2] != 1 {
            ge.broadcast_as(x.dims())?
        } else {
            ge.clone()
        };
        let x = x.add(&ge_broadcasted)?;

        // Output projection: 512 -> 192
        let x = if let Some(c_post) = &self.c_post {
            let masked = x.broadcast_mul(ssl_mask)?;
            c_post.forward(&masked)?
        } else {
            x
        };

        Ok(x)
    }
}

#[allow(dead_code)]
fn load_linear(
    state_dict: &StateDict,
    prefix: &str,
    device: &Device,
    dtype: DType,
) -> Result<candle_nn::Linear> {
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

    Ok(candle_nn::Linear::new(weight, bias))
}

/// Encoder P for processing semantic tokens
#[derive(Debug, Clone)]
pub struct EncP {
    ssl_proj: candle_nn::Conv1d,
    encoder_ssl: Vec<EncoderLayer>,
    encoder_text: Vec<EncoderLayer>,
    text_embedding: Tensor,
    mrte: Option<MRTE>,
    encoder2: Vec<EncoderLayer>,
    proj: candle_nn::Conv1d,
    out_channels: usize,
}

impl EncP {
    /// Load EncP from SoVITS state dict
    pub fn load(
        state_dict: &StateDict,
        device: &Device,
        _hidden_channels: usize,
        n_layers: usize,
        out_channels: usize,
        dtype: DType,
    ) -> Result<Self> {
        // Load SSL projection: [192, 768, 1]
        let ssl_proj_weight = state_dict
            .get("enc_p.ssl_proj.weight")?
            .to_device(device)?
            .to_dtype(dtype)?;
        let ssl_proj_bias = state_dict
            .get("enc_p.ssl_proj.bias")
            .ok()
            .cloned()
            .map(|t| t.to_device(device).and_then(|t| t.to_dtype(dtype)))
            .transpose()?;

        let ssl_proj_config = candle_nn::Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: Default::default(),
        };
        let ssl_proj = candle_nn::Conv1d::new(ssl_proj_weight, ssl_proj_bias, ssl_proj_config);

        // Load text embedding
        let text_embedding = state_dict
            .get("enc_p.text_embedding.weight")?
            .to_device(device)?
            .to_dtype(dtype)?;

        // Load encoder_ssl layers (model uses 3 layers)
        let mut encoder_ssl = Vec::new();
        for i in 0..(n_layers / 2) {
            let prefix = "enc_p.encoder_ssl";
            if state_dict.contains(&format!("{}.attn_layers.{}.conv_q.weight", prefix, i)) {
                let layer = EncoderLayer::load(state_dict, prefix, device, i, 8, dtype)?;
                encoder_ssl.push(layer);
            }
        }

        // Load encoder_text layers
        let mut encoder_text = Vec::new();
        for i in 0..n_layers {
            let prefix = "enc_p.encoder_text";
            if state_dict.contains(&format!("{}.attn_layers.{}.conv_q.weight", prefix, i)) {
                let layer = EncoderLayer::load(state_dict, prefix, device, i, 8, dtype)?;
                encoder_text.push(layer);
            }
        }

        // Load MRTE (optional)
        let mrte = if state_dict.contains("enc_p.mrte.cross_attention.conv_q.weight") {
            Some(MRTE::load(state_dict, "enc_p.mrte", device, dtype)?)
        } else {
            None
        };

        // Load encoder2 layers
        let mut encoder2 = Vec::new();
        for i in 0..(n_layers / 2) {
            let prefix = "enc_p.encoder2";
            if state_dict.contains(&format!("{}.attn_layers.{}.conv_q.weight", prefix, i)) {
                let layer = EncoderLayer::load(state_dict, prefix, device, i, 8, dtype)?;
                encoder2.push(layer);
            }
        }

        // Load output projection: [out_channels * 2, hidden_channels, 1]
        let proj_weight = state_dict
            .get("enc_p.proj.weight")?
            .to_device(device)?
            .to_dtype(dtype)?;
        let proj_bias = state_dict
            .get("enc_p.proj.bias")
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
            ssl_proj,
            encoder_ssl,
            encoder_text,
            text_embedding,
            mrte,
            encoder2,
            proj,
            out_channels,
        })
    }

    /// Forward pass
    pub fn forward(
        &self,
        quantized: &Tensor,
        y_lengths: &[i64],
        text: &Tensor,
        text_lengths: &[i64],
        ge: &Tensor,
        _speed: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let device = quantized.device();

        // Create mask for quantized — cast to match quantized dtype (F16 in FP16 mode)
        let y_max_len = quantized.dims()[2] as i64;
        let y_mask = self
            .sequence_mask(y_lengths, y_max_len, device)?
            .to_dtype(quantized.dtype())?;
        let y_mask_expanded = y_mask.unsqueeze(1)?;

        // SSL projection: matches Python y = self.ssl_proj(y * y_mask) * y_mask
        let mut y = self
            .ssl_proj
            .forward(&quantized.broadcast_mul(&y_mask_expanded)?)?;
        y = y.broadcast_mul(&y_mask_expanded)?;

        // Create text mask — cast to match embedding dtype
        let text_max_len = text.dims()[1] as i64;
        let text_mask_raw = self.sequence_mask(text_lengths, text_max_len, device)?;

        // Text embedding lookup
        let text_emb = self.lookup_embeddings(text)?;

        let mut text_emb = text_emb.transpose(1, 2)?;
        // Cast mask to match text_emb dtype
        let text_mask_expanded = text_mask_raw.unsqueeze(1)?.to_dtype(text_emb.dtype())?;
        text_emb = text_emb.broadcast_mul(&text_mask_expanded)?;

        // Pass through encoder_text
        for layer in self.encoder_text.iter() {
            text_emb = layer.forward(&text_emb, &text_mask_expanded)?;
        }

        // Pass through encoder_ssl
        for layer in self.encoder_ssl.iter() {
            y = layer.forward(&y, &y_mask_expanded)?;
        }

        // MRTE fusion (if available)
        if let Some(mrte) = &self.mrte {
            y = mrte.forward(
                &y,
                &y_mask_expanded,
                &text_emb,
                &text_mask_expanded,
                Some(ge),
            )?;
        } else {
            // Simple fusion: project ge to 192 channels and add
            let ge_192 = if ge.dims()[1] == y.dims()[1] {
                // Same channels, just broadcast
                ge.broadcast_as(y.dims())?
            } else {
                // Narrow to match (fallback)
                ge.narrow(1, 0, y.dims()[1])?.broadcast_as(y.dims())?
            };
            y = y.add(&ge_192)?;
        }

        // Pass through encoder2
        for layer in self.encoder2.iter() {
            y = layer.forward(&y, &y_mask_expanded)?;
        }

        // Output projection: split into m and logs
        let stats = self.proj.forward(&y)?;
        let stats = stats.broadcast_mul(&y_mask_expanded)?;

        let m = stats.narrow(1, 0, self.out_channels)?;
        let logs = stats.narrow(1, self.out_channels, self.out_channels)?;
        // Don't clamp - model was trained without clamping
        let logs = logs.clamp(-5.0, 2.0)?;

        Ok((y, m, logs, y_mask))
    }

    fn sequence_mask(&self, lengths: &[i64], max_len: i64, device: &Device) -> Result<Tensor> {
        let batch_size = lengths.len();
        let mut mask = Vec::with_capacity(batch_size * max_len as usize);

        for (_i, &len) in lengths.iter().enumerate() {
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

    fn lookup_embeddings(&self, text: &Tensor) -> Result<Tensor> {
        let dims = text.dims();
        let batch = dims[0];
        let seq_len = dims[1];

        let indices: Vec<i64> = text.flatten_all()?.to_vec1()?;
        let mut embeddings = Vec::new();

        for &idx in &indices {
            let emb = self.text_embedding.get(idx as usize)?;
            embeddings.push(emb);
        }

        let stacked = Tensor::stack(&embeddings, 0)?;
        Ok(stacked.reshape((batch, seq_len, self.text_embedding.dims()[1]))?)
    }
}
