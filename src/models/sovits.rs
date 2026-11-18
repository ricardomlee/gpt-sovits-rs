//! SoVITS Model - Complete Implementation
//!
//! This module implements the complete SoVITS model for audio synthesis:
//! semantic tokens → quantizer → enc_p → flow → decoder → waveform
//!
//! Inference path: semantic tokens + text + reference mel speaker embedding
//! → enc_p → flow → decoder

use crate::utils::profiling::{sync_profile_enabled, sync_profile_stage, Instant};
use crate::utils::sampling::SamplingRng;
use crate::utils::weights::decode_safetensors;
use crate::utils::{load_safetensors, Linear, StateDict};
use crate::{Error, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

use crate::models::sovits_decoder::Decoder;
use crate::models::sovits_encp::EncP;
use crate::models::sovits_encq::EncQ;
use crate::models::sovits_flow::ResidualCouplingBlock;
use crate::models::sovits_ref_enc::RefEnc;

/// SoVITS Model for audio synthesis
pub struct SoVITSModel {
    device: Device,
    dtype: DType,

    // Quantizer for semantic tokens
    quantizer: Quantizer,

    // Encoder P for processing semantic and text features (training teacher)
    enc_p: EncP,

    // Optional training-time posterior encoder. Some inference checkpoints omit it.
    enc_q: Option<EncQ>,

    // Flow model for latent variable transformation
    flow: ResidualCouplingBlock,

    // Decoder (vocoder)
    decoder: Decoder,

    // Reference encoder for speaker embedding (MelStyleEncoder)
    ref_enc: RefEnc,

    // v2Pro conditioning layers. The reference encoder produces a 1024-channel
    // speaker condition, which is projected to 512 only for enc_p.
    ge_to512: Option<Linear>,
    sv_emb: Option<Linear>,
    prelu_weight: Option<Tensor>,

    // Configuration
    n_mels: usize,
    sampling_rate: u32,
    gin_channels: usize,
    sampling_rng: SamplingRng,
}

/// Simple quantizer for semantic tokens
#[derive(Debug, Clone)]
pub struct Quantizer {
    dimension: usize,
    codebook: Tensor,
}

impl Quantizer {
    pub fn new(dimension: usize, _bins: usize, codebook: Tensor) -> Self {
        Self {
            dimension,
            codebook,
        }
    }

    /// Decode codes to continuous features
    /// codes: [batch, seq_len] - semantic token IDs
    /// Returns: [batch, dimension, seq_len]
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let dims = codes.dims();
        let batch = dims[0];
        let seq_len = dims[1];

        let indices: Vec<i64> = codes.flatten_all()?.to_vec1()?;
        let mut embeddings = Vec::with_capacity(indices.len());
        for &index in &indices {
            embeddings.push(self.codebook.get(index as usize)?);
        }
        let stacked = Tensor::stack(&embeddings, 0)?;
        let result = stacked.reshape((batch, seq_len, self.dimension))?;
        Ok(result.transpose(1, 2)?) // [batch, dimension, seq_len]
    }
}

/// Nearest-neighbor 2x upsampling along the time dimension
/// Input: [batch, channels, time] → Output: [batch, channels, time*2]
fn nearest_upsample_2x(x: &Tensor) -> Result<Tensor> {
    let (batch, channels, time) = x.dims3()?;
    if x.device().is_cpu() {
        return Ok(x.upsample_nearest1d(time * 2)?);
    }

    if x.elem_count() == 0 {
        return Ok(x.reshape((batch, channels, time * 2))?);
    }

    // Height 1 lets Candle's 2D kernel repeat only the time axis on device.
    Ok(x.unsqueeze(2)?
        .upsample_nearest2d(1, time * 2)?
        .squeeze(2)?)
}

impl SoVITSModel {
    /// Load model from safetensors file
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu, DType::F32)
    }

    /// Load model with specific device
    pub fn load_with_device(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let weights_map = load_safetensors(path)?;
        let state_dict = StateDict::new(weights_map);
        Self::from_state_dict(&state_dict, device, dtype)
    }

    /// Load a CPU F32 model from a safetensors buffer.
    pub fn from_safetensors(buffer: &[u8]) -> Result<Self> {
        let state_dict = StateDict::new(decode_safetensors(buffer)?);
        Self::from_state_dict(&state_dict, &Device::Cpu, DType::F32)
    }

    pub(crate) fn from_state_dict(
        state_dict: &StateDict,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Configuration
        let hidden_channels = 192;
        let n_layers = 6;
        let enc_out_channels = 192;

        // Load quantizer (dimension=768 matches codebook embedding size)
        let codebook = state_dict
            .get("quantizer.vq.layers.0._codebook.embed")?
            .to_device(device)?
            .to_dtype(dtype)?;
        let quantizer = Quantizer::new(768, 1024, codebook);

        // Load EncP (text + semantic token encoder)
        let enc_p = EncP::load(
            state_dict,
            device,
            hidden_channels,
            n_layers,
            enc_out_channels,
            dtype,
        )?;

        // Load EncQ when present. It is not used by the normal TTS inference
        // path, and some exported inference checkpoints omit all enc_q weights.
        let enc_q = if state_dict.contains("enc_q.pre.weight") {
            Some(EncQ::load(
                state_dict,
                device,
                hidden_channels,
                enc_out_channels,
                dtype,
            )?)
        } else {
            None
        };

        // Load Flow (ResidualCouplingBlock)
        let flow = ResidualCouplingBlock::load(state_dict, "flow.flows", device, 4, dtype)?;

        // Load Decoder
        let decoder = Decoder::load(state_dict, device, dtype)?;

        // Load RefEnc (MelStyleEncoder for speaker embedding)
        let ref_enc = RefEnc::load(state_dict, device, dtype)?;
        let gin_channels = state_dict
            .get("ref_enc.fc.fc.weight")?
            .dims()
            .first()
            .copied()
            .unwrap_or(512);

        let ge_to512 = if state_dict.contains("ge_to512.weight") {
            let weight = state_dict
                .get("ge_to512.weight")?
                .to_device(device)?
                .to_dtype(dtype)?;
            let bias = state_dict
                .get("ge_to512.bias")
                .ok()
                .cloned()
                .map(|t| t.to_device(device).and_then(|t| t.to_dtype(dtype)))
                .transpose()?;
            Some(Linear::new(weight, bias))
        } else {
            None
        };

        let sv_emb = if state_dict.contains("sv_emb.weight") {
            let weight = state_dict
                .get("sv_emb.weight")?
                .to_device(device)?
                .to_dtype(dtype)?;
            let bias = state_dict
                .get("sv_emb.bias")
                .ok()
                .cloned()
                .map(|t| t.to_device(device).and_then(|t| t.to_dtype(dtype)))
                .transpose()?;
            Some(Linear::new(weight, bias))
        } else {
            None
        };

        let prelu_weight = if state_dict.contains("prelu.weight") {
            Some(
                state_dict
                    .get("prelu.weight")?
                    .to_device(device)?
                    .to_dtype(dtype)?,
            )
        } else {
            None
        };

        Ok(Self {
            device: device.clone(),
            dtype,
            quantizer,
            enc_p,
            enc_q,
            flow,
            decoder,
            ref_enc,
            ge_to512,
            sv_emb,
            prelu_weight,
            n_mels: 100,
            sampling_rate: 32000,
            gin_channels,
            sampling_rng: SamplingRng::default(),
        })
    }

    /// Synthesize audio from semantic tokens and text tokens
    ///
    /// # Arguments
    /// * `semantic_tokens` - GPT-generated semantic token IDs
    /// * `text_tokens` - Phoneme IDs for target text
    /// * `ref_audio_mel` - Optional reference STFT magnitude [1, 704, time] (first 704 of 1025 bins)
    /// * `noise_scale` - Sampling randomness (Python default: 0.5, higher = more variation)
    ///
    /// The pipeline ALWAYS uses enc_p (text-driven path) for synthesis.
    /// When ref_audio_mel is provided, it is fed to ref_enc (MelStyleEncoder) to compute
    /// the speaker embedding (ge) via mean-pooling, which conditions enc_p.
    pub fn synthesize(
        &self,
        semantic_tokens: &[usize],
        text_tokens: &[usize],
        ref_audio_mel: Option<&Tensor>,
        noise_scale: f32,
    ) -> Result<Vec<f32>> {
        self.synthesize_with_speed(
            semantic_tokens,
            text_tokens,
            ref_audio_mel,
            noise_scale,
            1.0,
        )
    }

    /// Synthesize audio while controlling duration in EncP, matching Python's `speed_factor`.
    pub fn synthesize_with_speed(
        &self,
        semantic_tokens: &[usize],
        text_tokens: &[usize],
        ref_audio_mel: Option<&Tensor>,
        noise_scale: f32,
        speed: f32,
    ) -> Result<Vec<f32>> {
        self.synthesize_with_speed_and_sv(
            semantic_tokens,
            text_tokens,
            ref_audio_mel,
            None,
            noise_scale,
            speed,
        )
    }

    /// Synthesize audio with an optional v2Pro speaker-verification embedding.
    ///
    /// `sv_embedding` must be `[1, 20480]` or `[20480]` and is only used by v2Pro/v2ProPlus
    /// checkpoints that contain `sv_emb.*` weights. When omitted, a zero embedding is used so
    /// v2Pro checkpoints can still run from the reference mel alone.
    pub fn synthesize_with_speed_and_sv(
        &self,
        semantic_tokens: &[usize],
        text_tokens: &[usize],
        ref_audio_mel: Option<&Tensor>,
        sv_embedding: Option<&Tensor>,
        noise_scale: f32,
        speed: f32,
    ) -> Result<Vec<f32>> {
        self.synthesize_internal(
            semantic_tokens,
            text_tokens,
            ref_audio_mel,
            sv_embedding,
            noise_scale,
            speed,
            None,
        )
        .map(|(_, audio)| audio)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn synthesize_internal(
        &self,
        semantic_tokens: &[usize],
        text_tokens: &[usize],
        ref_audio_mel: Option<&Tensor>,
        sv_embedding: Option<&Tensor>,
        noise_scale: f32,
        speed: f32,
        trace_dir: Option<&Path>,
    ) -> Result<(Tensor, Vec<f32>)> {
        let profile_start = Instant::now();
        if semantic_tokens.is_empty() {
            return Err(Error::InferenceError("Empty semantic tokens".to_string()));
        }
        if text_tokens.is_empty() {
            return Err(Error::InferenceError("Empty text tokens".to_string()));
        }

        let trace_tensor = |name: &str, tensor: &Tensor| match trace_dir {
            Some(directory) => self.save_tensor(directory, name, tensor),
            None => Ok(()),
        };
        if let Some(directory) = trace_dir {
            self.save_tokens(directory, "sovits_debug_semantic_tokens", semantic_tokens)?;
            self.save_tokens(directory, "sovits_debug_text_tokens", text_tokens)?;
        }

        // Convert semantic tokens to tensor [1, seq_len]
        let codes_vec: Vec<i64> = semantic_tokens.iter().map(|&x| x as i64).collect();
        let codes = Tensor::from_vec(codes_vec, (1, semantic_tokens.len()), &self.device)?;

        // Convert text tokens to tensor [1, seq_len]
        let text_vec: Vec<i64> = text_tokens.iter().map(|&x| x as i64).collect();
        let text = Tensor::from_vec(text_vec, (1, text_tokens.len()), &self.device)?;

        let ge = self.compute_ge(ref_audio_mel, sv_embedding)?;
        let ge_enc = self.ge_for_enc_p(&ge)?;
        trace_tensor("sovits_debug_ge", &ge)?;
        trace_tensor("sovits_debug_ge_enc", &ge_enc)?;
        sync_profile_stage(&self.device)?;
        let ref_enc_ms = profile_start.elapsed().as_millis();

        // Decode semantic codes using quantizer
        let stage_start = Instant::now();
        let quantized = self.quantizer.decode(&codes)?;
        trace_tensor("sovits_debug_quantized", &quantized)?;

        // 2x upsampling to match frame rate
        let quantized_up = nearest_upsample_2x(&quantized)?;
        trace_tensor("sovits_debug_quantized_up", &quantized_up)?;

        // Create length tensors
        let y_lengths = vec![quantized_up.dims()[2] as i64];
        let text_lengths = vec![text.dims()[1] as i64];

        sync_profile_stage(&self.device)?;
        let prepare_ms = stage_start.elapsed().as_millis();

        // Pass through enc_p
        let stage_start = Instant::now();
        let (_y, m_p, logs_p, y_mask) = self.enc_p.forward(
            &quantized_up,
            &y_lengths,
            &text,
            &text_lengths,
            &ge_enc,
            speed,
        )?;
        trace_tensor("sovits_debug_encp_m", &m_p)?;
        trace_tensor("sovits_debug_encp_logs", &logs_p)?;
        sync_profile_stage(&self.device)?;
        let enc_p_ms = stage_start.elapsed().as_millis();

        // Sample from N(m, exp(logs)) to get latent z_p (matching Python: noise_scale=0.5)
        let stage_start = Instant::now();
        let noise = self.sample_noise(&m_p)?;
        let logs_exp = logs_p.exp()?;
        let z_p = m_p.add(&noise.broadcast_mul(&logs_exp)?.broadcast_mul(
            &Tensor::full(noise_scale, m_p.dims(), &self.device)?.to_dtype(m_p.dtype())?,
        )?)?;
        trace_tensor("sovits_debug_zp", &z_p)?;
        sync_profile_stage(&self.device)?;
        let sample_ms = stage_start.elapsed().as_millis();

        // Invert flow transform: z_p → z (with ge conditioning, matching Python)
        let stage_start = Instant::now();
        let z = self.flow.forward(&z_p, &y_mask, Some(&ge), true)?;
        trace_tensor("sovits_debug_flow_z", &z)?;

        // Apply mask
        let z_masked = z.broadcast_mul(&y_mask)?;
        trace_tensor("sovits_debug_dec_input", &z_masked)?;
        sync_profile_stage(&self.device)?;
        let flow_ms = stage_start.elapsed().as_millis();

        // Pass through decoder with full 512-dim ge (matching Python: o = self.dec((z * y_mask), g=ge))
        let stage_start = Instant::now();
        let output = self.decoder.forward(&z_masked, Some(&ge))?;
        if let Some(directory) = trace_dir {
            self.save_f32_file(directory, "sovits_debug_audio", &output)?;
        }
        sync_profile_stage(&self.device)?;
        let decoder_ms = stage_start.elapsed().as_millis();

        if self.device.is_cpu() || sync_profile_enabled() {
            tracing::debug!(
                "profile sovits ref_enc={}ms prepare={}ms enc_p={}ms sample={}ms flow={}ms decoder={}ms total={}ms",
                ref_enc_ms,
                prepare_ms,
                enc_p_ms,
                sample_ms,
                flow_ms,
                decoder_ms,
                profile_start.elapsed().as_millis()
            );
        }

        Ok((z_masked, output))
    }

    fn compute_ge(
        &self,
        ref_audio_mel: Option<&Tensor>,
        sv_embedding: Option<&Tensor>,
    ) -> Result<Tensor> {
        let ge = if let Some(mel) = ref_audio_mel {
            let mel_in = mel.to_dtype(self.dtype)?;
            let time = mel_in.dims()[2];
            let refer_mask =
                Tensor::full(1.0f32, &[1, 1, time], &self.device)?.to_dtype(mel_in.dtype())?;
            let mel_masked = mel_in.broadcast_mul(&refer_mask)?;
            self.ref_enc.forward(&mel_masked, &refer_mask)?
        } else {
            Tensor::zeros((1, self.gin_channels, 1), self.dtype, &self.device)?
        };

        if let Some(sv_emb) = &self.sv_emb {
            let sv = match sv_embedding {
                Some(tensor) if tensor.dims() == [20480] => tensor.unsqueeze(0)?,
                Some(tensor) => tensor.clone(),
                None => Tensor::zeros((1, 20480), self.dtype, &self.device)?,
            }
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
            let sv = sv_emb.forward(&sv)?.unsqueeze(2)?;
            let ge = ge.broadcast_add(&sv)?;
            if let Some(weight) = &self.prelu_weight {
                return self.prelu(&ge, weight);
            }
            return Ok(ge);
        }

        Ok(ge)
    }

    fn ge_for_enc_p(&self, ge: &Tensor) -> Result<Tensor> {
        if let Some(project) = &self.ge_to512 {
            let ge = ge.transpose(1, 2)?;
            return Ok(project.forward(&ge)?.transpose(1, 2)?);
        }
        Ok(ge.clone())
    }

    fn prelu(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let zeros = Tensor::zeros_like(x)?;
        let positive = x.maximum(&zeros)?;
        let negative = x.minimum(&zeros)?;
        let alpha = weight.reshape((1, weight.dims()[0], 1))?;
        Ok(positive.add(&negative.broadcast_mul(&alpha)?)?)
    }

    #[cfg(any(test, not(target_arch = "wasm32")))]
    pub(crate) fn set_sampling_seed(&self, seed: Option<u64>) -> Result<()> {
        self.sampling_rng.set_seed(seed)
    }

    fn sample_noise(&self, mean: &Tensor) -> Result<Tensor> {
        self.sampling_rng.normal_like(mean)
    }

    /// Synthesize audio and return (decoder_input, audio) for debugging
    pub fn synthesize_debug(
        &self,
        semantic_tokens: &[usize],
        text_tokens: &[usize],
        ref_audio_mel: Option<&Tensor>,
        noise_scale: f32,
    ) -> Result<(Tensor, Vec<f32>)> {
        self.synthesize_internal(
            semantic_tokens,
            text_tokens,
            ref_audio_mel,
            None,
            noise_scale,
            1.0,
            None,
        )
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

    /// Get ref_enc
    pub fn ref_enc(&self) -> &RefEnc {
        &self.ref_enc
    }

    /// Get quantizer
    pub fn quantizer(&self) -> &Quantizer {
        &self.quantizer
    }

    /// Get enc_p
    pub fn enc_p(&self) -> &EncP {
        &self.enc_p
    }

    /// Get flow
    pub fn flow(&self) -> &ResidualCouplingBlock {
        &self.flow
    }

    /// Get decoder
    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }

    /// Get enc_q when the checkpoint includes the training-time posterior encoder.
    pub fn enc_q(&self) -> Option<&EncQ> {
        self.enc_q.as_ref()
    }

    /// Run pipeline and save all intermediates for debugging
    pub fn debug_pipeline(
        &self,
        semantic_tokens: &[usize],
        text_tokens: &[usize],
        ref_audio_mel: Option<&Tensor>,
        noise_scale: f32,
    ) -> Result<()> {
        self.synthesize_internal(
            semantic_tokens,
            text_tokens,
            ref_audio_mel,
            None,
            noise_scale,
            1.0,
            Some(Path::new(".")),
        )
        .map(|_| ())
    }

    fn save_tensor(&self, directory: &Path, name: &str, t: &Tensor) -> Result<()> {
        let flat: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let dims = t.dims();
        if !flat.is_empty() {
            let mean = flat.iter().sum::<f32>() / flat.len() as f32;
            let variance =
                flat.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / flat.len() as f32;
            let min = flat.iter().copied().fold(f32::INFINITY, f32::min);
            let max = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            tracing::debug!(
                name,
                ?dims,
                mean,
                std = variance.sqrt(),
                min,
                max,
                "SoVITS intermediate"
            );
        }
        let header = dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let data = flat
            .iter()
            .map(|v| format!("{:.10}", v))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(
            directory.join(format!("{}.txt", name)),
            format!("{}\n{}\n", header, data),
        )
        .map_err(|e| crate::Error::InferenceError(format!("Failed to save {}: {}", name, e)))
    }

    fn save_tokens(&self, directory: &Path, name: &str, tokens: &[usize]) -> Result<()> {
        let data = tokens
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(
            directory.join(format!("{name}.txt")),
            format!("{}\n{data}\n", tokens.len()),
        )
        .map_err(|e| Error::InferenceError(format!("Failed to save {name}: {e}")))
    }

    fn save_f32_file(&self, directory: &Path, name: &str, data: &[f32]) -> Result<()> {
        let content = data
            .iter()
            .map(|v| format!("{:.10}", v))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(
            directory.join(format!("{}.txt", name)),
            format!("{}\n", content),
        )
        .map_err(|e| Error::InferenceError(format!("Failed to save {name}: {e}")))
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
}

#[cfg(test)]
mod tests;
