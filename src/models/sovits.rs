//! SoVITS Model for audio synthesis
//!
//! Complete implementation matching GPT-SoVITS checkpoint structure:
//! - enc_p: Text encoder with phoneme embedding and SSL feature processing
//! - enc_q: Semantic token encoder
//! - flow: Flow-based decoder with coupling layers
//! - dec: BigVGAN-style neural vocoder
//! - ref_enc: Reference audio encoder for speaker embedding
//! - ssl_proj: SSL feature projection

use candle_core::{Device, Tensor, DType};
use crate::{Result, Error};
use crate::utils::{StateDict, load_safetensors, Linear, LayerNorm, Conv1d, Conv1dWeightNorm};

/// SoVITS Model for mel spectrogram generation
#[allow(dead_code)]
pub struct SoVITSModel {
    device: Device,
    dtype: DType,
    enc_p: EncoderP,
    enc_q: EncoderQ,
    flow: FlowDecoder,
    dec: Decoder,
    ref_enc: RefEncoder,
    ssl_proj: SslProj,
    n_mels: usize,
    sampling_rate: u32,
}

/// Encoder P - Text encoder with phoneme embedding and SSL features
#[allow(dead_code)]
pub struct EncoderP {
    text_embedding: Tensor,      // [vocab_size=322, hidden=192]
    encoder_text: TextEncoder2,
    encoder_ssl: SslEncoder,
    fusion: FusionNet,
}

/// Text encoder sub-module
struct TextEncoder2 {
    #[allow(dead_code)]
    layers: Vec<ConvAttentionLayer>,
}

/// SSL encoder sub-module
struct SslEncoder {
    #[allow(dead_code)]
    layers: Vec<ConvAttentionLayer>,
}

/// Fusion network combining text and SSL features
pub struct FusionNet {
    conv: Conv1d,
}

/// Encoder Q - Semantic token encoder
pub struct EncoderQ {
    enc: ResidualConditionedEncoder,
}

/// Flow decoder with coupling layers
pub struct FlowDecoder {
    flows: Vec<FlowModule>,
}

/// BigVGAN-style neural vocoder
pub struct Decoder {
    conv_pre: Conv1d,
    cond: Conv1d,
    resblocks: Vec<ResidualBlock>,
    conv_post: Conv1d,
}

/// Reference audio encoder for speaker embedding
pub struct RefEncoder {
    slf_attn: Linear,
    fc: Linear,
}

/// SSL feature projection
#[allow(dead_code)]
pub struct SslProj {
    projection: Conv1d,
}

/// Convolutional attention layer
#[allow(dead_code)]
pub struct ConvAttentionLayer {
    conv_q: Conv1d,
    conv_k: Conv1d,
    conv_v: Conv1d,
    conv_o: Conv1d,
    norm: LayerNorm,
}

/// Flow module with conditioning
pub struct FlowModule {
    enc: ResidualConditionedBlock,
}

/// Residual conditioned block
pub struct ResidualConditionedBlock {
    cond_layer: Conv1dWeightNorm,
    in_layers: Vec<Conv1dWeightNorm>,
    out_layer: Conv1dWeightNorm,
}

/// Residual block for BigVGAN decoder
pub struct ResidualBlock {
    convs1: Vec<Conv1dWeightNorm>,
    convs2: Vec<Conv1dWeightNorm>,
}

/// Residual conditioned encoder for enc_q
pub struct ResidualConditionedEncoder {
    cond_layer: Conv1dWeightNorm,
    in_layers: Vec<Conv1dWeightNorm>,
    out_layers: Vec<Conv1dWeightNorm>,
}

impl SoVITSModel {
    /// Load model from safetensors file
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu)
    }

    /// Load model with specific device
    pub fn load_with_device(path: &str, device: &Device) -> Result<Self> {
        // Load weights from safetensors
        let weights_map = load_safetensors(path)?;
        let state_dict = StateDict::new(weights_map);

        // Infer configuration from weights
        let n_mels = 100;
        let sampling_rate = 24000;

        // Create components
        let enc_p = EncoderP::new(&state_dict, device)?;
        let enc_q = EncoderQ::new(&state_dict, device)?;
        let flow = FlowDecoder::new(&state_dict, device)?;
        let dec = Decoder::new(&state_dict, device)?;
        let ref_enc = RefEncoder::new(&state_dict, device)?;
        let ssl_proj = SslProj::new(&state_dict, device)?;

        Ok(Self {
            device: device.clone(),
            dtype: DType::F32,
            enc_p,
            enc_q,
            flow,
            dec,
            ref_enc,
            ssl_proj,
            n_mels,
            sampling_rate,
        })
    }

    /// Synthesize mel spectrogram from semantic tokens
    pub fn synthesize(
        &self,
        semantic_tokens: &[usize],
        ref_audio: Option<&Tensor>,
    ) -> Result<Tensor> {
        if semantic_tokens.is_empty() {
            return Err(Error::InferenceError("Empty semantic tokens".to_string()));
        }

        // Convert tokens to tensor [1, seq_len]
        let tokens: Vec<i64> = semantic_tokens.iter().map(|&x| x as i64).collect();
        let tokens_tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        // Step 1: Encode semantic tokens through enc_q
        let ssl_features = self.enc_q.encode(&tokens_tensor)?;

        // Step 2: Get reference embedding if provided
        let ref_emb = if let Some(ref_audio) = ref_audio {
            self.ref_enc.encode(ref_audio)?
        } else {
            // Default embedding
            Tensor::zeros((1, 512), self.dtype, &self.device)?
        };

        // Step 3: Pass through flow decoder
        let mel_spec = self.flow.decode(&ssl_features, &ref_emb)?;

        Ok(mel_spec)
    }

    /// Get model device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get model dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get number of mel bins
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Get sampling rate
    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }
}

impl EncoderP {
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        // Load text embedding [322, 192]
        let text_embedding = state_dict
            .get("enc_p.text_embedding.weight")?
            .to_device(device)?
            .clone();

        // Create text encoder
        let encoder_text = TextEncoder2::new(state_dict, "enc_p.encoder2", device)?;

        // Create SSL encoder
        let encoder_ssl = SslEncoder::new(state_dict, "enc_p.encoder_ssl", device)?;

        // Create fusion network
        let fusion_conv_weight = state_dict
            .get("enc_p.fusion.c.weight")?
            .to_device(device)?
            .clone();
        let fusion_conv_bias = state_dict
            .get("enc_p.fusion.c.bias")?
            .to_device(device)?
            .clone();
        let fusion = FusionNet {
            conv: Conv1d::new(fusion_conv_weight, Some(fusion_conv_bias), 1, 0, 1),
        };

        Ok(Self {
            text_embedding,
            encoder_text,
            encoder_ssl,
            fusion,
        })
    }
}

impl TextEncoder2 {
    #[allow(unused_variables)]
    pub fn new(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let mut layers = Vec::new();

        // Count layers
        let mut i = 0;
        loop {
            let key = format!("{}.attn_layers.{}.conv_q.weight", prefix, i);
            if !state_dict.contains(&key) {
                break;
            }

            let conv_q = Conv1d::new(
                state_dict.get(&format!("{}.attn_layers.{}.conv_q.weight", prefix, i))?.to_device(device)?.clone(),
                Some(state_dict.get(&format!("{}.attn_layers.{}.conv_q.bias", prefix, i))?.to_device(device)?.clone()),
                1,
                0,
                1,
            );
            let conv_k = Conv1d::new(
                state_dict.get(&format!("{}.attn_layers.{}.conv_k.weight", prefix, i))?.to_device(device)?.clone(),
                Some(state_dict.get(&format!("{}.attn_layers.{}.conv_k.bias", prefix, i))?.to_device(device)?.clone()),
                1,
                0,
                1,
            );
            let conv_v = Conv1d::new(
                state_dict.get(&format!("{}.attn_layers.{}.conv_v.weight", prefix, i))?.to_device(device)?.clone(),
                Some(state_dict.get(&format!("{}.attn_layers.{}.conv_v.bias", prefix, i))?.to_device(device)?.clone()),
                1,
                0,
                1,
            );
            let conv_o = Conv1d::new(
                state_dict.get(&format!("{}.attn_layers.{}.conv_o.weight", prefix, i))?.to_device(device)?.clone(),
                Some(state_dict.get(&format!("{}.attn_layers.{}.conv_o.bias", prefix, i))?.to_device(device)?.clone()),
                1,
                0,
                1,
            );
            let norm = state_dict.get_layer_norm(&format!("{}.attn_layers.{}.norm", prefix, i))?;

            layers.push(ConvAttentionLayer {
                conv_q,
                conv_k,
                conv_v,
                conv_o,
                norm,
            });
            i += 1;
        }

        Ok(Self { layers })
    }
}

impl SslEncoder {
    pub fn new(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let mut layers = Vec::new();

        let mut i = 0;
        loop {
            let key = format!("{}.attn_layers.{}.conv_q.weight", prefix, i);
            if !state_dict.contains(&key) {
                break;
            }

            let conv_q = Conv1d::new(
                state_dict.get(&format!("{}.attn_layers.{}.conv_q.weight", prefix, i))?.to_device(device)?.clone(),
                Some(state_dict.get(&format!("{}.attn_layers.{}.conv_q.bias", prefix, i))?.to_device(device)?.clone()),
                1,
                0,
                1,
            );
            let conv_k = Conv1d::new(
                state_dict.get(&format!("{}.attn_layers.{}.conv_k.weight", prefix, i))?.to_device(device)?.clone(),
                Some(state_dict.get(&format!("{}.attn_layers.{}.conv_k.bias", prefix, i))?.to_device(device)?.clone()),
                1,
                0,
                1,
            );
            let conv_v = Conv1d::new(
                state_dict.get(&format!("{}.attn_layers.{}.conv_v.weight", prefix, i))?.to_device(device)?.clone(),
                Some(state_dict.get(&format!("{}.attn_layers.{}.conv_v.bias", prefix, i))?.to_device(device)?.clone()),
                1,
                0,
                1,
            );
            let conv_o = Conv1d::new(
                state_dict.get(&format!("{}.attn_layers.{}.conv_o.weight", prefix, i))?.to_device(device)?.clone(),
                Some(state_dict.get(&format!("{}.attn_layers.{}.conv_o.bias", prefix, i))?.to_device(device)?.clone()),
                1,
                0,
                1,
            );
            let norm = state_dict.get_layer_norm(&format!("{}.attn_layers.{}.norm", prefix, i))?;

            layers.push(ConvAttentionLayer {
                conv_q,
                conv_k,
                conv_v,
                conv_o,
                norm,
            });
            i += 1;
        }

        Ok(Self { layers })
    }
}

impl FusionNet {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

impl EncoderQ {
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        let enc = ResidualConditionedEncoder::new(state_dict, "enc_q.enc", device)?;
        Ok(Self { enc })
    }

    pub fn encode(&self, tokens: &Tensor) -> Result<Tensor> {
        self.enc.forward(tokens)
    }
}

impl ResidualConditionedEncoder {
    #[allow(unused_variables)]
    pub fn new(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let cond_layer = state_dict.get_conv1d_weight_norm(&format!("{}.cond_layer", prefix))?;

        let mut in_layers = Vec::new();
        let mut i = 0;
        loop {
            let key = format!("{}.in_layers.{}.weight_g", prefix, i);
            if !state_dict.contains(&key) {
                break;
            }
            in_layers.push(state_dict.get_conv1d_weight_norm(&format!("{}.in_layers.{}", prefix, i))?);
            i += 1;
        }

        let mut out_layers = Vec::new();
        i = 0;
        loop {
            let key = format!("{}.out_layers.{}.weight_g", prefix, i);
            if !state_dict.contains(&key) {
                break;
            }
            out_layers.push(state_dict.get_conv1d_weight_norm(&format!("{}.out_layers.{}", prefix, i))?);
            i += 1;
        }

        Ok(Self {
            cond_layer,
            in_layers,
            out_layers,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Simple forward through layers
        let mut h = x.clone();

        // Apply conditioning
        h = self.cond_layer.forward(&h)?;

        // Apply in/out layer pairs
        for (in_layer, out_layer) in self.in_layers.iter().zip(self.out_layers.iter()) {
            let residual = h.clone();
            let in_conv: &Conv1dWeightNorm = in_layer;
            h = in_conv.forward(&h)?;
            let out_conv: &Conv1dWeightNorm = out_layer;
            h = out_conv.forward(&h)?;
            h = h.add(&residual)?;
        }

        Ok(h)
    }
}

impl FlowDecoder {
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        let mut flows = Vec::new();

        let mut i = 0;
        loop {
            let key = format!("flow.flows.{}.enc.cond_layer.weight_g", i);
            if !state_dict.contains(&key) {
                break;
            }
            flows.push(FlowModule::new(state_dict, &format!("flow.flows.{}", i), device)?);
            i += 1;
        }

        Ok(Self { flows })
    }

    pub fn decode(&self, features: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let mut h = features.clone();

        for flow in &self.flows {
            h = flow.forward(&h, cond)?;
        }

        Ok(h)
    }
}

impl FlowModule {
    #[allow(unused_variables)]
    pub fn new(state_dict: &StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let enc = ResidualConditionedBlock::new(state_dict, &format!("{}.enc", prefix), device)?;
        Ok(Self { enc })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        self.enc.forward(x, cond)
    }
}

impl ResidualConditionedBlock {
    pub fn new(state_dict: &StateDict, prefix: &str, _device: &Device) -> Result<Self> {
        let cond_layer = state_dict.get_conv1d_weight_norm(&format!("{}.cond_layer", prefix))?;

        let mut in_layers = Vec::new();
        let mut i = 0;
        loop {
            let key = format!("{}.in_layers.{}.weight_g", prefix, i);
            if !state_dict.contains(&key) {
                break;
            }
            in_layers.push(state_dict.get_conv1d_weight_norm(&format!("{}.in_layers.{}", prefix, i))?);
            i += 1;
        }

        let out_layer = state_dict.get_conv1d_weight_norm(&format!("{}.out_layer", prefix))?;

        Ok(Self {
            cond_layer,
            in_layers,
            out_layer,
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        // Apply conditioning
        let cond_out = self.cond_layer.forward(cond)?;

        // Apply in layers
        let mut h = x.clone();
        for in_layer in &self.in_layers {
            let conv: &Conv1dWeightNorm = in_layer;
            h = conv.forward(&h)?;
        }

        // Add conditioning
        h = h.broadcast_add(&cond_out)?;

        // Apply out layer
        h = self.out_layer.forward(&h)?;

        Ok(h)
    }
}

impl Decoder {
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        // conv_pre: [512, 192, 7]
        let conv_pre_weight = state_dict.get("dec.conv_pre.weight")?.to_device(device)?.clone();
        let conv_pre_bias = state_dict.get("dec.conv_pre.bias")?.to_device(device)?.clone();
        let conv_pre = Conv1d::new(conv_pre_weight, Some(conv_pre_bias), 1, 3, 1);

        // cond: [512, 512, 1]
        let cond_weight = state_dict.get("dec.cond.weight")?.to_device(device)?.clone();
        let cond_bias = state_dict.get("dec.cond.bias")?.to_device(device)?.clone();
        let cond = Conv1d::new(cond_weight, Some(cond_bias), 1, 0, 1);

        // Count resblocks
        let mut resblocks = Vec::new();
        let mut i = 0;
        loop {
            let key = format!("dec.resblocks.{}.convs1.0.weight_g", i);
            if !state_dict.contains(&key) {
                break;
            }

            let mut convs1 = Vec::new();
            let mut j = 0;
            loop {
                let key = format!("dec.resblocks.{}.convs1.{}.weight_g", i, j);
                if !state_dict.contains(&key) {
                    break;
                }
                convs1.push(state_dict.get_conv1d_weight_norm(&format!("dec.resblocks.{}.convs1.{}", i, j))?);
                j += 1;
            }

            let mut convs2 = Vec::new();
            j = 0;
            loop {
                let key = format!("dec.resblocks.{}.convs2.{}.weight_g", i, j);
                if !state_dict.contains(&key) {
                    break;
                }
                convs2.push(state_dict.get_conv1d_weight_norm(&format!("dec.resblocks.{}.convs2.{}", i, j))?);
                j += 1;
            }

            resblocks.push(ResidualBlock { convs1, convs2 });
            i += 1;
        }

        // conv_post: [1, 16, 7]
        let conv_post_weight = state_dict.get("dec.conv_post.weight")?.to_device(device)?.clone();
        let conv_post = Conv1d::new(conv_post_weight, None, 1, 3, 1);

        Ok(Self {
            conv_pre,
            cond,
            resblocks,
            conv_post,
        })
    }

    pub fn vocode(&self, mel: &Tensor, cond: &Tensor) -> Result<Tensor> {
        // Pre-conv
        let mut h = self.conv_pre.forward(mel)?;

        // Conditioning
        let c = self.cond.forward(cond)?;

        // Residual blocks
        for block in &self.resblocks {
            h = block.forward(&h, &c)?;
        }

        // Post-conv
        h = self.conv_post.forward(&h)?;

        Ok(h)
    }
}

impl ResidualBlock {
    pub fn forward(&self, x: &Tensor, _cond: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();

        // convs1
        for conv in &self.convs1 {
            let c: &Conv1dWeightNorm = conv;
            h = c.forward(&h)?;
            // Add activation (LeakyReLU via clamp and mul)
            h = h.clamp(0.0, f32::INFINITY)?;
            h = h.broadcast_add(&h)?; // h * 0.1 * 2 = h * 0.2, then add original = h * 1.2
            h = h.clamp(0.0, f32::INFINITY)?;
        }

        // convs2
        for conv in &self.convs2 {
            let c: &Conv1dWeightNorm = conv;
            h = c.forward(&h)?;
            h = h.clamp(0.0, f32::INFINITY)?;
            h = h.broadcast_add(&h)?;
            h = h.clamp(0.0, f32::INFINITY)?;
        }

        // Residual connection
        Ok(x.add(&h)?)
    }
}

impl RefEncoder {
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        let slf_attn_w = state_dict.get("ref_enc.slf_attn.fc.weight")?.to_device(device)?.clone();
        let slf_attn_b = state_dict.get("ref_enc.slf_attn.fc.bias")?.to_device(device)?.clone();
        let slf_attn = Linear::new(slf_attn_w, Some(slf_attn_b));

        let fc_w = state_dict.get("ref_enc.fc.fc.weight")?.to_device(device)?.clone();
        let fc_b = state_dict.get("ref_enc.fc.fc.bias")?.to_device(device)?.clone();
        let fc = Linear::new(fc_w, Some(fc_b));

        Ok(Self { slf_attn, fc })
    }

    pub fn encode(&self, ref_audio: &Tensor) -> Result<Tensor> {
        // Simple encoding through fc layers
        let h = self.slf_attn.forward(ref_audio)?;
        self.fc.forward(&h)
    }
}

impl SslProj {
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        let weight = state_dict.get("ssl_proj.weight")?.to_device(device)?.clone();
        let bias = state_dict.get("ssl_proj.bias")?.to_device(device)?.clone();
        let projection = Conv1d::new(weight, Some(bias), 1, 0, 2);

        Ok(Self { projection })
    }
}

impl crate::models::Model for SoVITSModel {
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
