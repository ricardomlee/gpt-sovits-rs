//! Inference Module
//!
//! Main pipeline for TTS inference

use candle_core::Device;
use crate::config::Config;
use crate::models::{BertModel, BigVGAN, GPTModel, HubertModel, SoVITSModel};
use crate::text_frontend::TextFrontend;
use crate::utils::AudioBuffer;
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

    pub fn build(self) -> InferenceOptions {
        InferenceOptions {
            top_k: self.top_k.unwrap_or(15),
            top_p: self.top_p.unwrap_or(0.95),
            temperature: self.temperature.unwrap_or(0.8),
            speed: self.speed.unwrap_or(1.0),
            language: self.language.unwrap_or(Language::Chinese),
            max_tokens: self.max_tokens.unwrap_or(500),
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
        reference_text: &str,
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
                Ok(features) => {
                    tracing::info!("Extracted Hubert features: {:?}", features.dims());
                    Some(features)
                }
                Err(e) => {
                    tracing::warn!("Hubert extraction failed: {}, using zero tensor", e);
                    None
                }
            }
        } else {
            None
        };

        // Step 3: Get BERT features from text
        let bert_features = if let Some(bert) = &mut self.bert_model {
            match bert.extract(text) {
                Ok(features) => {
                    tracing::info!("Extracted BERT features: {:?}", features.dims());
                    Some(features)
                }
                Err(e) => {
                    tracing::warn!("BERT extraction failed: {}, using zero tensor", e);
                    None
                }
            }
        } else {
            None
        };

        // Step 4: Run GPT model to generate semantic tokens (with BERT/Hubert features if available)
        let semantic_tokens = gpt.generate_with_features(
            &phoneme_ids,
            bert_features.as_ref(),
            hubert_features.as_ref(),
            options.top_k,
            options.top_p,
            options.temperature,
        )?;
        tracing::info!("Generated {} semantic tokens", semantic_tokens.len());

        // Step 5: Run SoVITS to generate mel spectrogram
        let mel_spec = sovits.synthesize(&semantic_tokens, None)?;
        let mel_dims = mel_spec.dims();
        tracing::info!("SoVITS output mel spectrogram: {:?}", mel_dims);

        // Step 6: Run BigVGAN to generate waveform
        let waveform = if let Some(bigvgan) = &self.bigvgan_model {
            tracing::info!("Running BigVGAN with input: {:?}", mel_dims);
            match bigvgan.generate(&mel_spec) {
                Ok(w) => {
                    tracing::info!("BigVGAN output: {} samples", w.len());
                    w
                }
                Err(e) => {
                    tracing::error!("BigVGAN failed: {}", e);
                    vec![0.0f32; 24000] // 1 second silence
                }
            }
        } else {
            // Fallback: generate simple waveform from mel spec dimensions
            // This produces silence but with correct duration
            let frames = mel_dims[2];
            let hop_length = 256;
            tracing::info!("Using fallback: {} frames * {} hop = {} samples", frames, hop_length, frames * hop_length);
            vec![0.0f32; frames * hop_length]
        };

        // Create audio buffer
        let audio = AudioBuffer::new(waveform, 24000, 1);

        Ok(audio)
    }

    /// Check if pipeline is ready for inference
    pub fn is_ready(&self) -> bool {
        self.gpt_model.is_some() && self.sovits_model.is_some()
    }
}
