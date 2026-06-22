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

    /// Load SoVITS model (also initializes semantic tokenizer from same weights)
    pub fn load_sovits<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path_str = path.as_ref().to_str().unwrap();
        let model = SoVITSModel::load_with_device(path_str, &self.device)?;
        self.sovits_model = Some(model);
        // Semantic tokenizer shares the same quantizer weights
        let tokenizer = SemanticTokenizer::load_with_device(path_str, &self.device)?;
        self.semantic_tokenizer = Some(tokenizer);
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

        // Step 1: Process target text → phoneme IDs + word2ph
        let (target_phoneme_ids, target_word2ph) = self.text_frontend.process_with_word2ph(text, options.language)?;
        tracing::debug!("target word2ph ({} entries): {:?}", target_word2ph.len(), &target_word2ph[..target_word2ph.len().min(15)]);

        // Step 1b: Process reference text (if provided) → ref phoneme IDs + word2ph
        let (ref_phoneme_ids, ref_word2ph) = if !reference_text.is_empty() {
            self.text_frontend.process_with_word2ph(reference_text, options.language)?
        } else {
            (vec![], vec![])
        };

        // Concatenate: [ref_phonemes + target_phonemes]
        let phoneme_ids: Vec<usize> = ref_phoneme_ids.iter().chain(target_phoneme_ids.iter()).cloned().collect();
        tracing::debug!("phoneme_ids: ref={}, target={}, total={}", ref_phoneme_ids.len(), target_phoneme_ids.len(), phoneme_ids.len());

        // Step 2: Extract Hubert features from reference audio
        let hubert_features = if let Some(hubert) = &mut self.hubert_model {
            match hubert.extract(reference_audio.as_ref()) {
                Ok(features) => {
                    let features = features.to_device(&self.device).unwrap_or(features);
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

        // Step 3: Extract and align BERT features
        // If ref_text provided: align ref+target separately then concatenate (matches Python's bert = cat([bert1, bert2], 1))
        let combined_bert = if let Some(bert) = &mut self.bert_model {
            let has_ref = !ref_phoneme_ids.is_empty();
            let target_bert = bert.extract(text).ok()
                .and_then(|f| f.to_device(&self.device).ok().or(Some(f)));
            let ref_bert = if has_ref {
                bert.extract(reference_text).ok()
                    .and_then(|f| f.to_device(&self.device).ok().or(Some(f)))
            } else {
                None
            };
            // Pre-align both to phone level using GPT's bert_proj and word2ph
            let target_aligned = target_bert.as_ref().and_then(|tb| {
                gpt.project_and_align_bert(tb, &target_word2ph, target_phoneme_ids.len()).ok()
            });
            let ref_aligned = if has_ref {
                ref_bert.as_ref().and_then(|rb| {
                    gpt.project_and_align_bert(rb, &ref_word2ph, ref_phoneme_ids.len()).ok()
                })
            } else {
                None
            };
            match (ref_aligned, target_aligned) {
                (Some(ra), Some(ta)) => {
                    candle_core::Tensor::cat(&[&ra, &ta], 1).ok()
                }
                (None, Some(ta)) => Some(ta),
                _ => None,
            }
        } else {
            None
        };

        // Step 4: Extract prompt tokens from reference audio (reusing Hubert features)
        let prompt_tokens = if let Some(tokenizer) = &self.semantic_tokenizer {
            if let Some(ref hubert_feats) = hubert_features {
                let hubert_t = hubert_feats.transpose(1, 2)?;
                let hubert_t = hubert_t.to_device(&self.device)?;
                match tokenizer.extract(&hubert_t) {
                    Ok(tokens) => {
                        tracing::debug!("Extracted {} prompt tokens", tokens.len());
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
            gpt.generate_with_prompts_aligned_bert(
                &phoneme_ids,
                prompts,
                combined_bert.as_ref(),
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
                options.max_tokens,
            )?
        } else {
            gpt.generate_with_features(
                &phoneme_ids,
                combined_bert.as_ref(),
                hubert_features.as_ref(),
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
                options.max_tokens,
            )?
        };

        tracing::info!("Generated {} semantic tokens", semantic_tokens.len());

        // Step 6: Extract mel spectrogram from reference audio for speaker conditioning
        let ref_mel = self.extract_ref_mel(reference_audio.as_ref(), sovits)?;

        // Step 7: Synthesize audio with SoVITS — use only TARGET phoneme IDs
        let audio_samples = sovits.synthesize(&semantic_tokens, &target_phoneme_ids, ref_mel.as_ref(), 0.5)?;

        // Create audio buffer
        let audio = AudioBuffer::new(
            audio_samples,
            sovits.sampling_rate(),
            1,
        );

        tracing::info!("Generated audio: {} samples", audio.samples.len());

        Ok(audio)
    }

    /// Run TTS inference with KV cache optimization.
    ///
    /// Functionally identical to `inference()` but uses a prefill+single-token-decode
    /// strategy: the full text+prompt sequence is processed in one forward pass to fill
    /// the KV cache, and each subsequent audio token only runs a single-token forward.
    pub fn inference_kv_cache<P: AsRef<Path>>(
        &mut self,
        text: &str,
        reference_audio: P,
        reference_text: &str,
        options: &InferenceOptions,
    ) -> Result<AudioBuffer> {
        let gpt = self.gpt_model.as_ref().ok_or_else(|| {
            Error::ModelLoadError("GPT model not loaded".to_string())
        })?;
        let sovits = self.sovits_model.as_ref().ok_or_else(|| {
            Error::ModelLoadError("SoVITS model not loaded".to_string())
        })?;

        // Step 1: Process target text
        let (target_phoneme_ids, target_word2ph) =
            self.text_frontend.process_with_word2ph(text, options.language)?;

        // Step 1b: Process reference text
        let (ref_phoneme_ids, ref_word2ph) = if !reference_text.is_empty() {
            self.text_frontend.process_with_word2ph(reference_text, options.language)?
        } else {
            (vec![], vec![])
        };

        // Concatenate [ref_phonemes + target_phonemes]
        let phoneme_ids: Vec<usize> = ref_phoneme_ids
            .iter()
            .chain(target_phoneme_ids.iter())
            .cloned()
            .collect();
        tracing::debug!("kv_cache phoneme_ids: ref={}, target={}, total={}",
            ref_phoneme_ids.len(), target_phoneme_ids.len(), phoneme_ids.len());

        // Step 2: Extract HuBERT features
        let hubert_features = if let Some(hubert) = &mut self.hubert_model {
            match hubert.extract(reference_audio.as_ref()) {
                Ok(f) => {
                    let f = f.to_device(&self.device).unwrap_or(f);
                    tracing::info!("HuBERT features: {:?}", f.dims());
                    Some(f)
                }
                Err(e) => { tracing::warn!("HuBERT extraction failed: {}", e); None }
            }
        } else {
            None
        };

        // Step 3: Extract and align BERT (ref + target separately, then cat)
        let combined_bert = if let Some(bert) = &mut self.bert_model {
            let has_ref = !ref_phoneme_ids.is_empty();
            let target_bert = bert.extract(text).ok()
                .and_then(|f| f.to_device(&self.device).ok().or(Some(f)));
            let ref_bert = if has_ref {
                bert.extract(reference_text).ok()
                    .and_then(|f| f.to_device(&self.device).ok().or(Some(f)))
            } else {
                None
            };
            let target_aligned = target_bert.as_ref().and_then(|tb| {
                gpt.project_and_align_bert(tb, &target_word2ph, target_phoneme_ids.len()).ok()
            });
            let ref_aligned = if has_ref {
                ref_bert.as_ref().and_then(|rb| {
                    gpt.project_and_align_bert(rb, &ref_word2ph, ref_phoneme_ids.len()).ok()
                })
            } else {
                None
            };
            match (ref_aligned, target_aligned) {
                (Some(ra), Some(ta)) => candle_core::Tensor::cat(&[&ra, &ta], 1).ok(),
                (None, Some(ta)) => Some(ta),
                _ => None,
            }
        } else {
            None
        };

        // Step 4: Extract VQ prompt tokens from HuBERT features
        let prompt_tokens = if let Some(tokenizer) = &self.semantic_tokenizer {
            if let Some(ref hf) = hubert_features {
                let hf_t = hf.transpose(1, 2)?.to_device(&self.device)?;
                tokenizer.extract(&hf_t).ok()
                    .map(|t| { tracing::debug!("Prompt tokens: {}", t.len()); t })
            } else { None }
        } else { None };

        // Step 5: Generate with KV-cache prefill strategy
        let semantic_tokens = if let Some(ref prompts) = prompt_tokens {
            gpt.generate_with_prompts_aligned_bert_kv_cache(
                &phoneme_ids,
                prompts,
                combined_bert.as_ref(),
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
                options.max_tokens,
            )?
        } else {
            // Fallback: no prompt tokens → use non-cached path
            gpt.generate_with_features(
                &phoneme_ids,
                combined_bert.as_ref(),
                hubert_features.as_ref(),
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
                options.max_tokens,
            )?
        };

        tracing::info!("Generated {} semantic tokens (kv_cache)", semantic_tokens.len());

        // Step 6: Reference mel for speaker conditioning
        let ref_mel = self.extract_ref_mel(reference_audio.as_ref(), sovits)?;

        // Step 7: SoVITS synthesis — target phonemes only
        let audio_samples = sovits.synthesize(
            &semantic_tokens, &target_phoneme_ids, ref_mel.as_ref(), 0.5
        )?;

        Ok(AudioBuffer::new(audio_samples, sovits.sampling_rate(), 1))
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

        // Resample to target sample rate (24kHz) using soxr HQ, matching librosa quality
        let target_sr = sovits.sampling_rate();
        let samples = if spec.sample_rate != target_sr {
            use soxr::{Soxr, format::Mono};
            let ratio = target_sr as f64 / spec.sample_rate as f64;
            let out_capacity = (samples.len() as f64 * ratio).ceil() as usize + 64;
            let mut output = vec![0.0f32; out_capacity];
            let mut resampler = Soxr::<Mono<f32>>::new(spec.sample_rate as f64, target_sr as f64)
                .map_err(|e| Error::AudioError(format!("soxr init: {}", e)))?;
            let proc = resampler.process(&samples, &mut output)
                .map_err(|e| Error::AudioError(format!("soxr process: {}", e)))?;
            let mut tail = vec![0.0f32; out_capacity];
            let tail_n = resampler.drain(&mut tail)
                .map_err(|e| Error::AudioError(format!("soxr drain: {}", e)))?;
            output.truncate(proc.output_frames);
            output.extend_from_slice(&tail[..tail_n]);
            output
        } else {
            samples
        };

        // Extract STFT magnitude spectrum matching Python's spectrogram_torch
        // n_fft=2048, hop=640, win=2048, center=False + reflect padding
        let n_fft = 2048;
        let hop_length = 640;
        let _n_mels = sovits.n_mels();
        let extractor = SpectrogramExtractor::new(target_sr, n_fft, hop_length, _n_mels);
        let stft_mag = extractor.extract_spectrogram_batched(&samples, device)?;

        // Truncate to first 704 frequency bins (matching Python's y[:, :704])
        let stft_mag = stft_mag.narrow(1, 0, 704)?;

        tracing::info!("Extracted reference STFT magnitude: {:?}", stft_mag.dims());

        Ok(Some(stft_mag))
    }
}
