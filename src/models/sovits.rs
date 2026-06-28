//! SoVITS Model - Complete Implementation
//!
//! This module implements the complete SoVITS model for audio synthesis:
//! semantic tokens → quantizer → enc_p → flow → decoder → waveform
//!
//! Two inference paths are supported:
//! 1. Text-driven synthesis: semantic tokens + text → enc_p → flow → decoder
//! 2. Reference-driven synthesis: reference mel → enc_q → flow → decoder

use crate::utils::{load_safetensors, StateDict};
use crate::{Error, Result};
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

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

    // Encoder Q for processing reference audio mel spectrograms (inference)
    enc_q: EncQ,

    // Flow model for latent variable transformation
    flow: ResidualCouplingBlock,

    // Decoder (vocoder)
    decoder: Decoder,

    // Reference encoder for speaker embedding (MelStyleEncoder)
    ref_enc: RefEnc,

    // Configuration
    n_mels: usize,
    sampling_rate: u32,
    #[allow(dead_code)]
    gin_channels: usize,
}

/// Simple quantizer for semantic tokens
#[derive(Debug, Clone)]
pub struct Quantizer {
    #[allow(dead_code)]
    bins: usize,
    dimension: usize,
    codebook: Tensor,
}

impl Quantizer {
    pub fn new(dimension: usize, bins: usize, codebook: Tensor) -> Self {
        Self {
            bins,
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

/// Build sequence mask from lengths
/// Returns [batch, 1, time] where positions < length are 1 and >= length are 0
fn build_sequence_mask(
    lengths: &[i64],
    time: usize,
    batch: usize,
    device: &Device,
) -> Result<Tensor> {
    let indices: Vec<i64> = (0..time as i64).collect();
    let idx_tensor = Tensor::from_vec(indices, (1, 1, time), device)?;
    let len_tensor = Tensor::from_vec(lengths.to_vec(), (batch, 1, 1), device)?;
    let lengths_b = len_tensor.broadcast_as((batch, 1, time))?;
    // mask = idx < length — always F32; caller casts to model dtype as needed
    let mask = idx_tensor.broadcast_lt(&lengths_b)?;
    mask.to_dtype(DType::F32)
        .map_err(|e| crate::Error::InferenceError(e.to_string()))
}

/// Build sequence mask with specified dtype
fn build_sequence_mask_typed(
    lengths: &[i64],
    time: usize,
    batch: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mask = build_sequence_mask(lengths, time, batch, device)?;
    mask.to_dtype(dtype)
        .map_err(|e| crate::Error::InferenceError(e.to_string()))
}

/// Nearest-neighbor 2x upsampling along the time dimension
/// Input: [batch, channels, time] → Output: [batch, channels, time*2]
fn nearest_upsample_2x(x: &Tensor) -> Result<Tensor> {
    let (batch, channels, time) = x.dims3()?;
    if x.device().is_cpu() {
        return Ok(x.upsample_nearest1d(time * 2)?);
    }

    // Candle 0.10 has no CUDA nearest-1d kernel, and composing repeat/broadcast
    // produces invalid output on this path. Use the small, proven host fallback.
    let original_dtype = x.dtype();
    let values = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut repeated = Vec::with_capacity(batch * channels * time * 2);
    for value in values {
        repeated.push(value);
        repeated.push(value);
    }
    Ok(
        Tensor::from_vec(repeated, (batch, channels, time * 2), x.device())?
            .to_dtype(original_dtype)?,
    )
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

        // Configuration
        let hidden_channels = 192;
        let n_layers = 6;
        let gin_channels = 512;
        let enc_out_channels = 192;

        // Load quantizer (dimension=768 matches codebook embedding size)
        let codebook = state_dict
            .get("quantizer.vq.layers.0._codebook.embed")?
            .to_device(device)?
            .to_dtype(dtype)?;
        let quantizer = Quantizer::new(768, 1024, codebook);

        // Load EncP (text + semantic token encoder)
        let enc_p = EncP::load(
            &state_dict,
            device,
            hidden_channels,
            n_layers,
            enc_out_channels,
            dtype,
        )?;

        // Load EncQ (reference audio mel encoder)
        let enc_q = EncQ::load(
            &state_dict,
            device,
            hidden_channels,
            enc_out_channels,
            dtype,
        )?;

        // Load Flow (ResidualCouplingBlock)
        let flow = ResidualCouplingBlock::load(&state_dict, "flow.flows", device, 4, dtype)?;

        // Load Decoder
        let decoder = Decoder::load(&state_dict, device, dtype)?;

        // Load RefEnc (MelStyleEncoder for speaker embedding)
        let ref_enc = RefEnc::load(&state_dict, device, dtype)?;

        Ok(Self {
            device: device.clone(),
            dtype,
            quantizer,
            enc_p,
            enc_q,
            flow,
            decoder,
            ref_enc,
            n_mels: 100,
            sampling_rate: 32000,
            gin_channels,
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
        let profile_start = Instant::now();
        if semantic_tokens.is_empty() {
            return Err(Error::InferenceError("Empty semantic tokens".to_string()));
        }
        if text_tokens.is_empty() {
            return Err(Error::InferenceError("Empty text tokens".to_string()));
        }

        // Convert semantic tokens to tensor [1, seq_len]
        let codes_vec: Vec<i64> = semantic_tokens.iter().map(|&x| x as i64).collect();
        let codes = Tensor::from_vec(codes_vec, (1, semantic_tokens.len()), &self.device)?;

        // Convert text tokens to tensor [1, seq_len]
        let text_vec: Vec<i64> = text_tokens.iter().map(|&x| x as i64).collect();
        let text = Tensor::from_vec(text_vec, (1, text_tokens.len()), &self.device)?;

        // Compute speaker embedding using ref_enc (MelStyleEncoder)
        // Python: ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        let ge = if let Some(mel) = ref_audio_mel {
            // Caller must pre-truncate to 704 bins: mel[:, :704, :]
            let mel_in = mel.clone();

            // Build refer_mask from time dimension (all valid since we have full audio)
            let time = mel_in.dims()[2];
            let refer_mask =
                Tensor::full(1.0f32, &[1, 1, time], &self.device)?.to_dtype(mel_in.dtype())?;

            // Apply mask and compute ge
            let mel_masked = mel_in.broadcast_mul(&refer_mask)?;
            let ge = self.ref_enc.forward(&mel_masked, &refer_mask)?;
            ge
        } else {
            Tensor::zeros((1, 512, 1), self.dtype, &self.device)?
        };
        let ref_enc_ms = profile_start.elapsed().as_millis();

        // Decode semantic codes using quantizer
        let stage_start = Instant::now();
        let quantized = self.quantizer.decode(&codes)?;

        // 2x upsampling to match frame rate
        let quantized_up = nearest_upsample_2x(&quantized)?;

        // Create length tensors
        let y_lengths = vec![quantized_up.dims()[2] as i64];
        let text_lengths = vec![text.dims()[1] as i64];

        // Build y_mask from y_lengths
        let time_len = quantized_up.dims()[2];
        let y_mask = build_sequence_mask_typed(&y_lengths, time_len, 1, &self.device, self.dtype)?;
        let prepare_ms = stage_start.elapsed().as_millis();

        // Pass through enc_p
        let stage_start = Instant::now();
        let (_y, m_p, logs_p, _y_mask_enc) =
            self.enc_p
                .forward(&quantized_up, &y_lengths, &text, &text_lengths, &ge, 1.0)?;
        let enc_p_ms = stage_start.elapsed().as_millis();

        // Sample from N(m, exp(logs)) to get latent z_p (matching Python: noise_scale=0.5)
        let stage_start = Instant::now();
        let noise = self.sample_noise(&m_p)?;
        let logs_exp = logs_p.exp()?;
        let z_p = m_p.add(&noise.broadcast_mul(&logs_exp)?.broadcast_mul(
            &Tensor::full(noise_scale, m_p.dims(), &self.device)?.to_dtype(m_p.dtype())?,
        )?)?;
        let sample_ms = stage_start.elapsed().as_millis();

        // Invert flow transform: z_p → z (with ge conditioning, matching Python)
        let stage_start = Instant::now();
        let z = self.flow.forward(&z_p, &y_mask, Some(&ge), true)?;

        // Apply mask
        let z_masked = z.broadcast_mul(&y_mask)?;
        let flow_ms = stage_start.elapsed().as_millis();

        // Pass through decoder with full 512-dim ge (matching Python: o = self.dec((z * y_mask), g=ge))
        let stage_start = Instant::now();
        let output = self.decoder.forward(&z_masked, Some(&ge))?;
        let decoder_ms = stage_start.elapsed().as_millis();

        if self.device.is_cpu() {
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

        Ok(output)
    }

    fn sample_noise(&self, mean: &Tensor) -> Result<Tensor> {
        Ok(Tensor::randn(0.0f32, 1.0, mean.dims(), &self.device)?.to_dtype(mean.dtype())?)
    }

    /// Synthesize audio and return (decoder_input, audio) for debugging
    pub fn synthesize_debug(
        &self,
        semantic_tokens: &[usize],
        text_tokens: &[usize],
        ref_audio_mel: Option<&Tensor>,
        noise_scale: f32,
    ) -> Result<(Tensor, Vec<f32>)> {
        if semantic_tokens.is_empty() {
            return Err(Error::InferenceError("Empty semantic tokens".to_string()));
        }
        if text_tokens.is_empty() {
            return Err(Error::InferenceError("Empty text tokens".to_string()));
        }

        let codes_vec: Vec<i64> = semantic_tokens.iter().map(|&x| x as i64).collect();
        let codes = Tensor::from_vec(codes_vec, (1, semantic_tokens.len()), &self.device)?;

        let text_vec: Vec<i64> = text_tokens.iter().map(|&x| x as i64).collect();
        let text = Tensor::from_vec(text_vec, (1, text_tokens.len()), &self.device)?;

        let ge = if let Some(mel) = ref_audio_mel {
            let mel_in = mel.clone();
            let time = mel_in.dims()[2];
            let refer_mask =
                Tensor::full(1.0f32, &[1, 1, time], &self.device)?.to_dtype(mel_in.dtype())?;
            let mel_masked = mel_in.broadcast_mul(&refer_mask)?;
            let ge = self.ref_enc.forward(&mel_masked, &refer_mask)?;
            ge
        } else {
            Tensor::zeros((1, 512, 1), self.dtype, &self.device)?
        };

        let quantized = self.quantizer.decode(&codes)?;
        let quantized_up = nearest_upsample_2x(&quantized)?;

        let y_lengths = vec![quantized_up.dims()[2] as i64];
        let text_lengths = vec![text.dims()[1] as i64];

        let time_len = quantized_up.dims()[2];
        let y_mask = build_sequence_mask_typed(&y_lengths, time_len, 1, &self.device, self.dtype)?;

        let (_y, m_p, logs_p, _y_mask_enc) =
            self.enc_p
                .forward(&quantized_up, &y_lengths, &text, &text_lengths, &ge, 1.0)?;

        // enc_p.forward() already clamps logs to [-5.0, 2.0]
        let noise = self.sample_noise(&m_p)?;
        let logs_exp = logs_p.exp()?;
        let z_p = m_p.add(&noise.broadcast_mul(&logs_exp)?.broadcast_mul(
            &Tensor::full(noise_scale, m_p.dims(), &self.device)?.to_dtype(m_p.dtype())?,
        )?)?;

        let z = self.flow.forward(&z_p, &y_mask, Some(&ge), true)?;
        let z_masked = z.broadcast_mul(&y_mask)?;

        let dec_input = z_masked.clone();
        let output = self.decoder.forward(&z_masked, Some(&ge))?;

        Ok((dec_input, output))
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

    /// Get enc_q
    pub fn enc_q(&self) -> &EncQ {
        &self.enc_q
    }

    /// Run pipeline and save all intermediates for debugging
    pub fn debug_pipeline(
        &self,
        semantic_tokens: &[usize],
        text_tokens: &[usize],
        ref_audio_mel: Option<&Tensor>,
        noise_scale: f32,
    ) -> Result<()> {
        let codes_vec: Vec<i64> = semantic_tokens.iter().map(|&x| x as i64).collect();
        let codes = Tensor::from_vec(codes_vec.clone(), (1, semantic_tokens.len()), &self.device)?;

        let text_vec: Vec<i64> = text_tokens.iter().map(|&x| x as i64).collect();
        let text = Tensor::from_vec(text_vec.clone(), (1, text_tokens.len()), &self.device)?;

        // Save tokens to files for Python comparison
        {
            use std::io::Write;
            if let Ok(mut f) = std::fs::File::create("sovits_debug_semantic_tokens.txt") {
                writeln!(f, "{}", codes_vec.len()).ok();
                for t in &codes_vec {
                    writeln!(f, "{}", t).ok();
                }
            }
            if let Ok(mut f) = std::fs::File::create("sovits_debug_text_tokens.txt") {
                writeln!(f, "{}", text_vec.len()).ok();
                for t in &text_vec {
                    writeln!(f, "{}", t).ok();
                }
            }
        }

        let ge = if let Some(mel) = ref_audio_mel {
            let mel_in = mel.to_dtype(self.dtype)?;
            let refer_mask_m = Tensor::full(1.0f32, &[1, 1, mel_in.dims()[2]], &self.device)?
                .to_dtype(self.dtype)?;
            let mel_masked = mel_in.broadcast_mul(&refer_mask_m)?;
            let ge = self.ref_enc.forward(&mel_masked, &refer_mask_m)?;
            self.save_tensor("sovits_debug_ge", &ge)?;
            ge
        } else {
            Tensor::zeros((1, 512, 1), self.dtype, &self.device)?
        };

        // Quantizer
        let quantized = self.quantizer.decode(&codes)?;
        self.save_tensor("sovits_debug_quantized", &quantized)?;

        // Upsample
        let quantized_up = nearest_upsample_2x(&quantized)?;
        self.save_tensor("sovits_debug_quantized_up", &quantized_up)?;

        let y_lengths = vec![quantized_up.dims()[2] as i64];
        let text_lengths = vec![text.dims()[1] as i64];
        let time_len = quantized_up.dims()[2];
        let y_mask = build_sequence_mask_typed(&y_lengths, time_len, 1, &self.device, self.dtype)?;

        // enc_p
        let (_y, m_p, logs_p, _y_mask_enc) =
            self.enc_p
                .forward(&quantized_up, &y_lengths, &text, &text_lengths, &ge, 1.0)?;
        self.save_tensor("sovits_debug_encp_m", &m_p)?;
        self.save_tensor("sovits_debug_encp_logs", &logs_p)?;

        // Sampling - enc_p.forward() already clamps logs to [-5.0, 2.0]
        let noise = self.sample_noise(&m_p)?;
        let logs_exp = logs_p.exp()?;
        let z_p = m_p.add(&noise.broadcast_mul(&logs_exp)?.broadcast_mul(
            &Tensor::full(noise_scale, m_p.dims(), &self.device)?.to_dtype(m_p.dtype())?,
        )?)?;
        self.save_tensor("sovits_debug_zp", &z_p)?;

        // Flow inverse
        let z = self.flow.forward(&z_p, &y_mask, Some(&ge), true)?;
        self.save_tensor("sovits_debug_flow_z", &z)?;
        let z_masked = z.broadcast_mul(&y_mask)?;
        self.save_tensor("sovits_debug_dec_input", &z_masked)?;

        // Decoder
        let output = self.decoder.forward(&z_masked, Some(&ge))?;
        self.save_f32_file("sovits_debug_audio", &output);

        // Print stats
        for name in &[
            "quantized",
            "quantized_up",
            "encp_m",
            "encp_logs",
            "zp",
            "flow_z",
            "dec_input",
        ] {
            let key = format!("sovits_debug_{}", name);
            let path = format!("{}.txt", key);
            if let Ok(content) = std::fs::read_to_string(&path) {
                let mut lines = content.lines();
                if let Some(dims_line) = lines.next() {
                    let dims: Vec<usize> = dims_line
                        .split(',')
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    let data: Vec<f32> = lines.filter_map(|l| l.trim().parse().ok()).collect();
                    if !data.is_empty() {
                        let mean = data.iter().sum::<f32>() / data.len() as f32;
                        let sq_sum = data.iter().map(|v| v * v).sum::<f32>();
                        let std = (sq_sum / data.len() as f32 - mean * mean).sqrt();
                        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        let mean_f = mean as f64;
                        let std_f = std as f64;
                        let min_f = min as f64;
                        let max_f = max as f64;
                        tracing::debug!(
                            "  {}: {:?} mean={:.4} std={:.4} min={:.4} max={:.4}",
                            name,
                            dims,
                            mean_f,
                            std_f,
                            min_f,
                            max_f
                        );
                    }
                }
            }
        }

        Ok(())
    }

    fn save_tensor(&self, name: &str, t: &Tensor) -> Result<()> {
        let flat: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let dims = t.dims();
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
        std::fs::write(format!("{}.txt", name), format!("{}\n{}\n", header, data))
            .map_err(|e| crate::Error::InferenceError(format!("Failed to save {}: {}", name, e)))
    }

    fn save_f32_file(&self, name: &str, data: &[f32]) {
        let content = data
            .iter()
            .map(|v| format!("{:.10}", v))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(format!("{}.txt", name), format!("{}\n", content)).unwrap();
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
            "cuda" => Device::new_cuda_with_stream(0),
            "mps" => Device::new_metal(0),
            _ => Ok(Device::Cpu),
        }
        .map_err(|e| Error::ModelLoadError(e.to_string()))?;

        self.device = new_device;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_upsample_repeats_each_frame() -> Result<()> {
        let input = Tensor::new(&[[[1f32, 2., 3.], [4., 5., 6.]]], &Device::Cpu)?;
        let output = nearest_upsample_2x(&input)?;

        assert_eq!(output.dims(), &[1, 2, 6]);
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            [1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6.]
        );
        Ok(())
    }

    #[test]
    fn quantizer_decodes_codes_without_changing_layout() -> Result<()> {
        let device = Device::Cpu;
        let codebook = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &device)?;
        let quantizer = Quantizer::new(2, 3, codebook);
        let codes = Tensor::new(&[[2i64, 0], [1, 2]], &device)?;

        let output = quantizer.decode(&codes)?;
        assert_eq!(output.dims(), &[2, 2, 2]);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            [5., 1., 6., 2., 3., 5., 4., 6.]
        );
        Ok(())
    }
}
