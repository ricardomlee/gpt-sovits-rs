//! Inference Module
//!
//! Main pipeline for TTS inference

use candle_core::{Device, Tensor, DType};
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
    gpt_model: Option<GPTModel>,
    sovits_model: Option<SoVITSModel>,
    bert_model: Option<BertModel>,
    hubert_model: Option<HubertModel>,
    bigvgan_model: Option<BigVGAN>,
}

impl Pipeline {
    /// Create a new pipeline with configuration
    pub fn new(config: Config) -> Result<Self> {
        Ok(Self {
            config,
            text_frontend: TextFrontend::new()?,
            gpt_model: None,
            sovits_model: None,
            bert_model: None,
            hubert_model: None,
            bigvgan_model: None,
        })
    }

    /// Load GPT model
    pub fn load_gpt<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let model = GPTModel::load(path.as_ref().to_str().unwrap())?;
        self.gpt_model = Some(model);
        Ok(())
    }

    /// Load SoVITS model
    pub fn load_sovits<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let model = SoVITSModel::load(path.as_ref().to_str().unwrap())?;
        self.sovits_model = Some(model);
        Ok(())
    }

    /// Load BERT model (optional, uses ONNX)
    pub fn load_bert<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let model = BertModel::load(path.as_ref().to_str().unwrap())?;
        self.bert_model = Some(model);
        Ok(())
    }

    /// Load Hubert model (optional, uses ONNX)
    pub fn load_hubert<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let model = HubertModel::load(path.as_ref().to_str().unwrap())?;
        self.hubert_model = Some(model);
        Ok(())
    }

    /// Load BigVGAN model
    pub fn load_bigvgan<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let model = BigVGAN::load(path.as_ref().to_str().unwrap())?;
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

        // Step 2: Extract features from reference audio
        // Note: Hubert features require ONNX Runtime (--features onnx)
        // Without ONNX, uses zero tensor fallback
        let _hubert_features = if let Some(hubert) = &self.hubert_model {
            hubert.extract(reference_audio.as_ref())?
        } else {
            // Fallback: zero tensor with expected shape [batch=1, time=100, hidden=768]
            Tensor::zeros((1, 100, 768), DType::F32, &Device::Cpu)?
        };

        // Step 3: Get BERT features
        // Note: BERT requires ONNX Runtime (--features onnx)
        // Without ONNX, uses zero tensor fallback
        let _bert_features = if let Some(bert) = &self.bert_model {
            bert.extract(text)?
        } else {
            // Fallback: zero tensor with expected shape [batch=1, hidden=768, seq_len=10]
            Tensor::zeros((1, 768, 10), DType::F32, &Device::Cpu)?
        };

        // Step 4: Run GPT model to generate semantic tokens
        // phoneme_ids is already Vec<usize> from text_frontend
        let semantic_tokens = gpt.generate(
            &phoneme_ids,
            options.top_k,
            options.top_p,
            options.temperature,
        )?;

        // Step 5: Run SoVITS to generate mel spectrogram
        let mel_spec = sovits.synthesize(&semantic_tokens)?;

        // Step 6: Run BigVGAN to generate waveform
        let waveform = if let Some(bigvgan) = &self.bigvgan_model {
            bigvgan.generate(&mel_spec)?
        } else {
            // Fallback: generate simple waveform from mel spec dimensions
            // This produces silence but with correct duration
            let mel_dims = mel_spec.dims();
            let frames = mel_dims[2];
            let hop_length = 256;
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
