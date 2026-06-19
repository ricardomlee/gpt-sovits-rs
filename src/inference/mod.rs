//! Inference Module
//!
//! Main pipeline for TTS inference

use candle_core::Device;
use crate::config::Config;
use crate::models::{BertModel, BigVGAN, GPTModel, HubertModel, SemanticTokenizer, SoVITSModel};
use crate::text_frontend::TextFrontend;
use crate::utils::{AudioBuffer, SpectrogramExtractor};
use crate::{Error, Language, Result};
use std::path::Path;

/// Inference options
#[derive(Debug, Clone)]
pub struct InferenceOptions {
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub speed: f32,
    pub language: Language,
    pub max_tokens: usize,
    pub repetition_penalty: f32,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self {
            top_k: 15,
            top_p: 0.95,
            temperature: 0.8,
            speed: 1.0,
            language: Language::Chinese,
            max_tokens: 500,
            repetition_penalty: 1.35,
        }
    }
}

impl InferenceOptions {
    pub fn builder() -> InferenceOptionsBuilder {
        InferenceOptionsBuilder::default()
    }
}

#[derive(Default)]
pub struct InferenceOptionsBuilder {
    top_k: Option<usize>,
    top_p: Option<f32>,
    temperature: Option<f32>,
    speed: Option<f32>,
    language: Option<Language>,
    max_tokens: Option<usize>,
    repetition_penalty: Option<f32>,
}

impl InferenceOptionsBuilder {
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn speed(mut self, s: f32) -> Self {
        self.speed = Some(s);
        self
    }

    pub fn language(mut self, lang: Language) -> Self {
        self.language = Some(lang);
        self
    }

    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = Some(n);
        self
    }

    pub fn repetition_penalty(mut self, p: f32) -> Self {
        self.repetition_penalty = Some(p);
        self
    }

    pub fn build(self) -> InferenceOptions {
        InferenceOptions {
            top_k: self.top_k.unwrap_or(15),
            top_p: self.top_p.unwrap_or(0.95),
            temperature: self.temperature.unwrap_or(0.8),
            speed: self.speed.unwrap_or(1.0),
            language: self.language.unwrap_or(Language::Chinese),
            max_tokens: self.max_tokens.unwrap_or(500),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.35),
        }
    }
}

/// Main TTS inference pipeline
pub struct Pipeline {
    #[allow(dead_code)]
    config: Config,
    text_frontend: TextFrontend,
    device: Device,
    gpt_model: Option<GPTModel>,
    sovits_model: Option<SoVITSModel>,
    bert_model: Option<BertModel>,
    hubert_model: Option<HubertModel>,
    bigvgan_model: Option<BigVGAN>,
    semantic_tokenizer: Option<SemanticTokenizer>,
}

impl Pipeline {
    /// Create a new pipeline with configuration
    pub fn new(config: Config) -> Result<Self> {
        let device = config.candle_device();
        Ok(Self {
            config,
            text_frontend: TextFrontend::new()?,
            device,
            gpt_model: None,
            sovits_model: None,
            bert_model: None,
            hubert_model: None,
            bigvgan_model: None,
            semantic_tokenizer: None,
        })
    }

    /// Load GPT model
    pub fn load_gpt<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let model = GPTModel::load_with_device(path.as_ref().to_str().unwrap(), &self.device)?;
        self.gpt_model = Some(model);
        Ok(())
    }

    /// Load SoVITS model
    pub fn load_sovits<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let model = SoVITSModel::load_with_device(path.as_ref().to_str().unwrap(), &self.device)?;
        self.sovits_model = Some(model);
        Ok(())
    }

    /// Load BERT model (optional, uses ONNX)
    pub fn load_bert<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let device_str = match self.device {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Metal(_) => "mps",
        };
        let model = BertModel::load_with_device(path.as_ref().to_str().unwrap(), device_str)?;
        self.bert_model = Some(model);
        Ok(())
    }

    /// Get mutable reference to text frontend
    pub fn text_frontend_mut(&mut self) -> &mut TextFrontend {
        &mut self.text_frontend
    }

    /// Get reference to Hubert model
    pub fn hubert_model(&mut self) -> &mut Option<HubertModel> {
        &mut self.hubert_model
    }

    /// Get reference to BERT model
    pub fn bert_model(&mut self) -> &mut Option<BertModel> {
        &mut self.bert_model
    }

    /// Get reference to GPT model
    pub fn gpt_model(&self) -> &Option<GPTModel> {
        &self.gpt_model
    }

    /// Get reference to SoVITS model
    pub fn sovits_model(&self) -> &Option<SoVITSModel> {
        &self.sovits_model
    }

    /// Get reference to BigVGAN model
    pub fn bigvgan_model(&self) -> &Option<BigVGAN> {
        &self.bigvgan_model
    }

    /// Load Hubert model (optional, uses ONNX)
    pub fn load_hubert<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let device_str = match self.device {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Metal(_) => "mps",
        };
        let model = HubertModel::load_with_device(path.as_ref().to_str().unwrap(), device_str)?;
        self.hubert_model = Some(model);
        Ok(())
    }

    /// Load semantic tokenizer from SoVITS weights (for prompt token extraction)
    pub fn load_semantic_tokenizer<P: AsRef<Path>>(&mut self, sovits_path: P) -> Result<()> {
        let tokenizer = SemanticTokenizer::load_with_device(
            sovits_path.as_ref().to_str().unwrap(),
            &self.device,
        )?;
        self.semantic_tokenizer = Some(tokenizer);
        Ok(())
    }

    /// Load BigVGAN model
    pub fn load_bigvgan<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let model = BigVGAN::load_with_device(path.as_ref().to_str().unwrap(), &self.device)?;
        self.bigvgan_model = Some(model);
        Ok(())
    }

    /// Run TTS inference
    ///
    /// # Arguments
    /// * `text` - Input text to synthesize
    /// * `reference_audio` - Path to reference audio file
    /// * `reference_text` - Text content of reference audio
    /// * `options` - Inference options
    ///
    /// # Returns
    /// Audio buffer containing synthesized speech
    pub fn inference<P: AsRef<Path>>(
        &mut self,
        text: &str,
        reference_audio: P,
        _reference_text: &str,  // TODO: Use for prosody alignment in future
        options: &InferenceOptions,
    ) -> Result<AudioBuffer> {
        // Validate models are loaded
        let gpt = self.gpt_model.as_ref().ok_or_else(|| {
            Error::ModelLoadError("GPT model not loaded".to_string())
        })?;
        let sovits = self.sovits_model.as_ref().ok_or_else(|| {
            Error::ModelLoadError("SoVITS model not loaded".to_string())
        })?;

        // Step 1: Process text through frontend to get phoneme IDs
        let phoneme_ids = self.text_frontend.process(text, options.language)?;

        // Step 2: Extract Hubert features from reference audio (used for both prompt tokens and MRTE fusion)
        let hubert_features = if let Some(hubert) = &mut self.hubert_model {
            match hubert.extract(reference_audio.as_ref()) {
                Ok(features) => {
                    tracing::info!("Extracted Hubert features: {:?}", features.dims());
                    Some(features)
                }
                Err(e) => {
                    tracing::warn!("Failed to extract Hubert features: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Step 3: Extract BERT features from text
        let bert_features = if let Some(bert) = &mut self.bert_model {
            match bert.extract(text) {
                Ok(features) => {
                    tracing::info!("Extracted BERT features: {:?}", features.dims());
                    Some(features)
                }
                Err(e) => {
                    tracing::warn!("Failed to extract BERT features: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Step 4: Extract prompt tokens from reference audio (reusing Hubert features)
        let prompt_tokens = if let Some(tokenizer) = &self.semantic_tokenizer {
            if let Some(ref hubert_feats) = hubert_features {
                // Transpose hubert features: [1, T, 768] -> [1, 768, T]
                let hubert_t = hubert_feats.transpose(1, 2)?;
                // Move to same device as the rest of the pipeline
                let hubert_t = hubert_t.to_device(&self.device)?;
                match tokenizer.extract(&hubert_t) {
                    Ok(tokens) => {
                        tracing::info!("Extracted {} prompt tokens from reference audio", tokens.len());
                        Some(tokens)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to extract prompt tokens: {}", e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        // Step 5: Generate semantic tokens with GPT
        let semantic_tokens = if let Some(ref prompts) = prompt_tokens {
            gpt.generate_with_prompts(
                &phoneme_ids,
                prompts,
                bert_features.as_ref(),
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
            )?
        } else {
            gpt.generate_with_features(
                &phoneme_ids,
                bert_features.as_ref(),
                hubert_features.as_ref(),
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
            )?
        };

        tracing::info!("Generated {} semantic tokens", semantic_tokens.len());

        // Step 5: Extract mel spectrogram from reference audio for speaker conditioning
        let ref_mel = self.extract_ref_mel(reference_audio.as_ref(), sovits)?;

        // Step 6: Synthesize audio with SoVITS (directly outputs waveform)
        let audio_samples = sovits.synthesize(&semantic_tokens, &phoneme_ids, ref_mel.as_ref(), 0.5)?;

        // Create audio buffer
        let audio = AudioBuffer::new(
            audio_samples,
            sovits.sampling_rate(),
            1,
        );

        tracing::info!("Generated audio: {} samples", audio.samples.len());

        Ok(audio)
    }

    /// Run TTS inference with KV cache optimization
    pub fn inference_kv_cache<P: AsRef<Path>>(
        &mut self,
        text: &str,
        reference_audio: P,
        _reference_text: &str,
        options: &InferenceOptions,
    ) -> Result<AudioBuffer> {
        // Validate models are loaded
        let gpt = self.gpt_model.as_ref().ok_or_else(|| {
            Error::ModelLoadError("GPT model not loaded".to_string())
        })?;
        let sovits = self.sovits_model.as_ref().ok_or_else(|| {
            Error::ModelLoadError("SoVITS model not loaded".to_string())
        })?;

        // Step 1: Process text through frontend to get phoneme IDs
        let phoneme_ids = self.text_frontend.process(text, options.language)?;

        // Step 2: Extract Hubert features from reference audio
        let hubert_features = if let Some(hubert) = &mut self.hubert_model {
            match hubert.extract(reference_audio.as_ref()) {
                Ok(features) => Some(features),
                Err(_) => None
            }
        } else {
            None
        };

        // Step 3: Extract BERT features from text
        let bert_features = if let Some(bert) = &mut self.bert_model {
            match bert.extract(text) {
                Ok(features) => Some(features),
                Err(_) => None
            }
        } else {
            None
        };

        // Step 4: Generate semantic tokens with GPT using KV cache
        let semantic_tokens = gpt.generate_with_features_kv_cache(
            &phoneme_ids,
            bert_features.as_ref(),
            hubert_features.as_ref(),
            options.top_k,
            options.top_p,
            options.temperature,
            options.repetition_penalty,
        )?;

        tracing::info!("Generated {} semantic tokens (with KV cache)", semantic_tokens.len());

        // Step 5: Extract mel spectrogram from reference audio for speaker conditioning
        let ref_mel = self.extract_ref_mel(reference_audio.as_ref(), sovits)?;

        // Step 6: Synthesize audio with SoVITS (directly outputs waveform)
        let audio_samples = sovits.synthesize(&semantic_tokens, &phoneme_ids, ref_mel.as_ref(), 0.5)?;

        // Create audio buffer
        let audio = AudioBuffer::new(
            audio_samples,
            sovits.sampling_rate(),
            1,
        );

        Ok(audio)
    }

    /// Check if pipeline is ready for inference
    pub fn is_ready(&self) -> bool {
        self.gpt_model.is_some() && self.sovits_model.is_some()
    }

    /// Extract mel spectrogram from reference audio for SoVITS enc_q conditioning
    fn extract_ref_mel<P: AsRef<Path>>(&self, ref_audio: P, sovits: &SoVITSModel) -> Result<Option<candle_core::Tensor>> {
        use hound::WavReader;

        let device = sovits.device();

        // Load WAV file
        let mut reader = WavReader::open(ref_audio.as_ref())
            .map_err(|e| Error::AudioError(format!("Failed to open reference audio: {}", e)))?;

        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max = match spec.bits_per_sample {
                    8 => i8::MAX as f32,
                    16 => i16::MAX as f32,
                    24 => (1 << 23) as f32,
                    _ => i16::MAX as f32,
                };
                reader.samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / max)
                    .collect()
            }
            hound::SampleFormat::Float => {
                reader.samples::<f32>()
                    .filter_map(|s| s.ok())
                    .collect()
            }
        };

        // Convert to mono if stereo
        let samples = if spec.channels > 1 {
            let mut mono = Vec::with_capacity(samples.len() / spec.channels as usize);
            for chunk in samples.chunks(spec.channels as usize) {
                mono.push(chunk.iter().sum::<f32>() / spec.channels as f32);
            }
            mono
        } else {
            samples
        };

        // Resample to 24kHz if needed
        let target_sr = sovits.sampling_rate();
        let samples = if spec.sample_rate != target_sr {
            let ratio = target_sr as f64 / spec.sample_rate as f64;
            let new_len = (samples.len() as f64 * ratio) as usize;
            let mut resampled = vec![0.0f32; new_len];
            for i in 0..new_len {
                let src_idx = i as f64 / ratio;
                let idx = src_idx.floor() as usize;
                let frac = src_idx - idx as f64;
                let v0 = samples.get(idx).copied().unwrap_or(0.0);
                let v1 = samples.get(idx + 1).copied().unwrap_or(0.0);
                resampled[i] = v0 + (v1 - v0) * frac as f32;
            }
            resampled
        } else {
            samples
        };

        // Extract STFT magnitude spectrum matching Python's spectrogram_torch
        // Model training uses n_fft=2048, hop=512 with center=False + reflect padding
        let n_fft = 2048;
        let hop_length = 512;
        let _n_mels = sovits.n_mels();
        let extractor = SpectrogramExtractor::new(target_sr, n_fft, hop_length, _n_mels);
        let stft_mag = extractor.extract_spectrogram_batched(&samples, device)?;

        tracing::info!("Extracted reference STFT magnitude: {:?}", stft_mag.dims());

        Ok(Some(stft_mag))
    }
}
