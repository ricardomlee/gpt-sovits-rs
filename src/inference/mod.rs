//! Inference Module
//!
//! Main pipeline for TTS inference

use crate::config::Config;
use crate::models::{BertModel, BigVGAN, GPTModel, HubertModel, SemanticTokenizer, SoVITSModel};
use crate::text_frontend::TextFrontend;
use crate::utils::{AudioBuffer, SpectrogramExtractor};
use crate::{Error, Language, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Once;
use std::time::Instant;

/// Split text into sentence-level chunks for streaming inference.
/// Splits on Chinese/Japanese/English sentence-ending punctuation.
/// Chunks shorter than `min_chars` are merged with the next chunk.
pub fn split_sentences(text: &str, min_chars: usize) -> Vec<String> {
    // Sentence-ending punctuation (keep the delimiter attached to preceding text)
    const DELIMITERS: &[char] = &['。', '！', '？', '…', '!', '?', '.', '\n'];

    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if DELIMITERS.contains(&ch) && !current.trim().is_empty() {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    // Remaining text without terminator
    let tail = current.trim().to_string();
    if !tail.is_empty() {
        sentences.push(tail);
    }

    // Merge short chunks into next chunk (avoid tiny inference calls)
    let mut merged: Vec<String> = Vec::new();
    let mut acc = String::new();
    for s in sentences {
        acc.push_str(&s);
        if acc.chars().count() >= min_chars {
            merged.push(acc.trim().to_string());
            acc.clear();
        }
    }
    if !acc.trim().is_empty() {
        if let Some(last) = merged.last_mut() {
            last.push_str(&acc);
        } else {
            merged.push(acc.trim().to_string());
        }
    }

    merged
}

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

/// Cached features derived from a (reference_audio, reference_text) pair.
/// All fields are clone-cheap (Vec or Tensor with Arc-backed storage).
#[derive(Clone)]
struct CachedSpeaker {
    /// VQ semantic tokens from HuBERT — used as GPT prefix
    prompt_tokens: Vec<usize>,
    /// STFT magnitude of reference audio — used for SoVITS enc_q conditioning
    ref_mel: Option<Tensor>,
    /// Phone IDs for reference text
    ref_phoneme_ids: Vec<usize>,
    /// BERT features aligned to reference phone level
    ref_bert_aligned: Option<Tensor>,
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
    /// Cache keyed by (ref_audio_path, ref_text)
    ref_cache: HashMap<(String, String), CachedSpeaker>,
}

impl Pipeline {
    pub fn new(config: Config) -> Result<Self> {
        if config.half_precision {
            tracing::warn!(
                "FP16 SoVITS is disabled because it can produce silent audio; using F32"
            );
        }
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
            ref_cache: HashMap::new(),
        })
    }

    pub fn load_gpt<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let dtype = self.config.gpt_dtype();
        let model =
            GPTModel::load_with_device(path.as_ref().to_str().unwrap(), &self.device, dtype)?;
        self.gpt_model = Some(model);
        Ok(())
    }

    pub fn load_sovits<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path_str = path.as_ref().to_str().unwrap();
        let dtype = self.config.candle_dtype();
        let model = SoVITSModel::load_with_device(path_str, &self.device, dtype)?;
        self.sovits_model = Some(model);
        let tokenizer = SemanticTokenizer::load_with_device(path_str, &self.device)?;
        self.semantic_tokenizer = Some(tokenizer);
        Ok(())
    }

    pub fn load_bert<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let model = BertModel::load_with_device(path.as_ref().to_str().unwrap(), &self.device)?;
        self.bert_model = Some(model);
        Ok(())
    }

    pub fn load_hubert<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let model = HubertModel::load_with_device(path.as_ref().to_str().unwrap(), &self.device)?;
        self.hubert_model = Some(model);
        Ok(())
    }

    pub fn load_semantic_tokenizer<P: AsRef<Path>>(&mut self, sovits_path: P) -> Result<()> {
        let tokenizer = SemanticTokenizer::load_with_device(
            sovits_path.as_ref().to_str().unwrap(),
            &self.device,
        )?;
        self.semantic_tokenizer = Some(tokenizer);
        Ok(())
    }

    pub fn load_bigvgan<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let model = BigVGAN::load_with_device(path.as_ref().to_str().unwrap(), &self.device)?;
        self.bigvgan_model = Some(model);
        Ok(())
    }

    pub fn text_frontend_mut(&mut self) -> &mut TextFrontend {
        &mut self.text_frontend
    }
    pub fn hubert_model(&mut self) -> &mut Option<HubertModel> {
        &mut self.hubert_model
    }
    pub fn bert_model(&mut self) -> &mut Option<BertModel> {
        &mut self.bert_model
    }
    pub fn gpt_model(&self) -> &Option<GPTModel> {
        &self.gpt_model
    }
    pub fn sovits_model(&self) -> &Option<SoVITSModel> {
        &self.sovits_model
    }
    pub fn bigvgan_model(&self) -> &Option<BigVGAN> {
        &self.bigvgan_model
    }
    pub fn is_ready(&self) -> bool {
        self.gpt_model.is_some() && self.sovits_model.is_some()
    }

    /// Stream inference sentence-by-sentence.
    /// Splits `text` into sentences, yields one `AudioBuffer` per sentence as it's ready.
    /// Caller should preload speaker features first for best performance.
    pub fn inference_sentences<'a, P: AsRef<Path> + 'a>(
        &'a mut self,
        text: &'a str,
        reference_audio: P,
        reference_text: &'a str,
        options: &'a InferenceOptions,
        mode: &str,
        min_sentence_chars: usize,
    ) -> impl Iterator<Item = Result<AudioBuffer>> + 'a {
        let sentences = split_sentences(text, min_sentence_chars);
        SentenceIterator {
            pipeline: self,
            sentences,
            index: 0,
            reference_audio: reference_audio.as_ref().to_path_buf(),
            reference_text,
            options,
            mode: mode.to_string(),
        }
    }

    /// Run synthesis through the selected GPT decoding implementation.
    pub fn inference_with_mode<P: AsRef<Path>>(
        &mut self,
        mode: &str,
        text: &str,
        reference_audio: P,
        reference_text: &str,
        options: &InferenceOptions,
    ) -> Result<AudioBuffer> {
        match mode {
            "plain" => self.inference(text, reference_audio, reference_text, options),
            "kv" => self.inference_kv_cache(text, reference_audio, reference_text, options),
            "cuda-graph" => {
                self.inference_cuda_graph(text, reference_audio, reference_text, options)
            }
            _ => Err(Error::ConfigError(format!(
                "Unsupported inference mode: {mode}"
            ))),
        }
    }

    /// Pre-compute and cache reference speaker features.
    /// Call once per speaker before batch inference to avoid repeated HuBERT/BERT runs.
    pub fn preload_speaker<P: AsRef<Path>>(
        &mut self,
        ref_audio: P,
        ref_text: &str,
        language: Language,
    ) -> Result<()> {
        let key = (
            ref_audio.as_ref().to_string_lossy().into_owned(),
            ref_text.to_owned(),
        );
        if !self.ref_cache.contains_key(&key) {
            let sovits = self
                .sovits_model
                .as_ref()
                .ok_or_else(|| Error::ModelLoadError("SoVITS model not loaded".to_string()))?;
            let sr = sovits.sampling_rate();
            let n_mels = sovits.n_mels();
            let cached = Self::compute_ref_features(
                &mut self.hubert_model,
                &mut self.bert_model,
                &mut self.text_frontend,
                &self.semantic_tokenizer,
                self.gpt_model.as_ref(),
                ref_audio.as_ref(),
                ref_text,
                language,
                &self.device,
                sr,
                n_mels,
            )?;
            self.ref_cache.insert(key, cached);
        }
        Ok(())
    }

    /// Drop all cached speaker features.
    pub fn clear_speaker_cache(&mut self) {
        self.ref_cache.clear();
    }

    /// Compute all features that depend only on (ref_audio, ref_text).
    /// This is the shared hot path called by both inference() and inference_kv_cache().
    #[allow(clippy::too_many_arguments)]
    fn compute_ref_features(
        hubert_model: &mut Option<HubertModel>,
        bert_model: &mut Option<BertModel>,
        text_frontend: &mut TextFrontend,
        semantic_tokenizer: &Option<SemanticTokenizer>,
        gpt_model: Option<&GPTModel>,
        ref_audio: &Path,
        ref_text: &str,
        language: Language,
        device: &Device,
        sovits_sr: u32,
        sovits_n_mels: usize,
    ) -> Result<CachedSpeaker> {
        // Ref text → phoneme IDs + word2ph
        let (ref_phoneme_ids, ref_word2ph) = if !ref_text.is_empty() {
            text_frontend.process_with_word2ph(ref_text, language)?
        } else {
            (vec![], vec![])
        };

        // HuBERT features → prompt tokens
        let (prompt_tokens, hubert_features) = if let Some(hubert) = hubert_model {
            match hubert.extract(ref_audio) {
                Ok(features) => {
                    let features = features.to_device(device).unwrap_or(features);
                    tracing::info!("Extracted Hubert features: {:?}", features.dims());
                    let tokens = if let Some(tokenizer) = semantic_tokenizer {
                        let hf_t = features.transpose(1, 2)?.to_device(device)?;
                        tokenizer.extract(&hf_t).ok().inspect(|t| {
                            tracing::debug!("Prompt tokens: {}", t.len());
                        })
                    } else {
                        None
                    };
                    (tokens.unwrap_or_default(), Some(features))
                }
                Err(e) => {
                    tracing::warn!("HuBERT extraction failed: {}", e);
                    (vec![], None)
                }
            }
        } else {
            (vec![], None)
        };
        let _ = hubert_features; // only needed for tokenization above

        // BERT features aligned to ref phone level
        let ref_bert_aligned = if let (Some(bert), Some(gpt), false) =
            (bert_model.as_mut(), gpt_model, ref_phoneme_ids.is_empty())
        {
            bert.extract(ref_text)
                .ok()
                .and_then(|f| f.to_device(device).ok().or(Some(f)))
                .and_then(|rb| {
                    gpt.project_and_align_bert(&rb, &ref_word2ph, ref_phoneme_ids.len())
                        .ok()
                })
        } else {
            None
        };

        // Reference mel spectrogram for SoVITS enc_q conditioning
        let ref_mel = Self::extract_ref_mel_static(ref_audio, device, sovits_sr, sovits_n_mels)?;

        Ok(CachedSpeaker {
            prompt_tokens,
            ref_mel,
            ref_phoneme_ids,
            ref_bert_aligned,
        })
    }

    /// Get cached ref features (compute and cache on miss).
    fn get_ref_features<P: AsRef<Path>>(
        &mut self,
        ref_audio: P,
        ref_text: &str,
        language: Language,
    ) -> Result<CachedSpeaker> {
        let key = (
            ref_audio.as_ref().to_string_lossy().into_owned(),
            ref_text.to_owned(),
        );
        if let Some(cached) = self.ref_cache.get(&key) {
            tracing::debug!("Speaker cache hit: {:?}", key.0);
            return Ok(cached.clone());
        }
        tracing::debug!("Speaker cache miss: {:?}", key.0);
        let sovits = self
            .sovits_model
            .as_ref()
            .ok_or_else(|| Error::ModelLoadError("SoVITS model not loaded".to_string()))?;
        let sr = sovits.sampling_rate();
        let n_mels = sovits.n_mels();
        let cached = Self::compute_ref_features(
            &mut self.hubert_model,
            &mut self.bert_model,
            &mut self.text_frontend,
            &self.semantic_tokenizer,
            self.gpt_model.as_ref(),
            ref_audio.as_ref(),
            ref_text,
            language,
            &self.device,
            sr,
            n_mels,
        )?;
        self.ref_cache.insert(key, cached.clone());
        Ok(cached)
    }

    pub fn inference<P: AsRef<Path>>(
        &mut self,
        text: &str,
        reference_audio: P,
        reference_text: &str,
        options: &InferenceOptions,
    ) -> Result<AudioBuffer> {
        if self.gpt_model.is_none() {
            return Err(Error::ModelLoadError("GPT model not loaded".to_string()));
        }
        if self.sovits_model.is_none() {
            return Err(Error::ModelLoadError("SoVITS model not loaded".to_string()));
        }
        let total_start = Instant::now();

        // Target text features (not cached — depend on synthesis text)
        let target_start = Instant::now();
        let (target_phoneme_ids, target_word2ph) = self
            .text_frontend
            .process_with_word2ph(text, options.language)?;
        let target_ms = target_start.elapsed().as_millis();

        // Reference features (cached after first call)
        let ref_start = Instant::now();
        let ref_feats =
            self.get_ref_features(&reference_audio, reference_text, options.language)?;
        let ref_ms = ref_start.elapsed().as_millis();

        let gpt = self.gpt_model.as_ref().unwrap();
        let sovits = self.sovits_model.as_ref().unwrap();

        // Target BERT aligned
        let bert_start = Instant::now();
        let target_bert_aligned = if let Some(bert) = self.bert_model.as_mut() {
            bert.extract(text)
                .ok()
                .and_then(|f| f.to_device(&self.device).ok().or(Some(f)))
                .and_then(|tb| {
                    gpt.project_and_align_bert(&tb, &target_word2ph, target_phoneme_ids.len())
                        .ok()
                })
        } else {
            None
        };
        let bert_ms = bert_start.elapsed().as_millis();

        // Combined phoneme IDs and BERT: [ref | target]
        let phoneme_ids: Vec<usize> = ref_feats
            .ref_phoneme_ids
            .iter()
            .chain(target_phoneme_ids.iter())
            .cloned()
            .collect();

        let combined_bert = match (
            ref_feats.ref_bert_aligned.as_ref(),
            target_bert_aligned.as_ref(),
        ) {
            (Some(ra), Some(ta)) => Tensor::cat(&[ra, ta], 1).ok(),
            (None, Some(ta)) => Some(ta.clone()),
            _ => None,
        };

        // GPT generation
        let gpt_start = Instant::now();
        let semantic_tokens = if !ref_feats.prompt_tokens.is_empty() {
            gpt.generate_with_prompts_aligned_bert(
                &phoneme_ids,
                &ref_feats.prompt_tokens,
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
                None,
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
                options.max_tokens,
            )?
        };
        let gpt_ms = gpt_start.elapsed().as_millis();

        tracing::info!("Generated {} semantic tokens", semantic_tokens.len());
        if semantic_tokens.len() >= options.max_tokens {
            tracing::warn!(
                "Generated token count reached max_tokens={} in plain mode; output may be truncated",
                options.max_tokens
            );
        }

        let sovits_start = Instant::now();
        let audio_samples = sovits.synthesize(
            &semantic_tokens,
            &target_phoneme_ids,
            ref_feats.ref_mel.as_ref(),
            0.5,
        )?;
        let sovits_ms = sovits_start.elapsed().as_millis();
        tracing::info!(
            "profile mode=plain target={}ms ref={}ms target_bert={}ms gpt={}ms sovits={}ms total={}ms tokens={} audio_samples={}",
            target_ms,
            ref_ms,
            bert_ms,
            gpt_ms,
            sovits_ms,
            total_start.elapsed().as_millis(),
            semantic_tokens.len(),
            audio_samples.len()
        );

        Ok(AudioBuffer::new(audio_samples, sovits.sampling_rate(), 1))
    }

    pub fn inference_kv_cache<P: AsRef<Path>>(
        &mut self,
        text: &str,
        reference_audio: P,
        reference_text: &str,
        options: &InferenceOptions,
    ) -> Result<AudioBuffer> {
        let total_start = Instant::now();
        // Target text features
        let target_start = Instant::now();
        let (target_phoneme_ids, target_word2ph) = self
            .text_frontend
            .process_with_word2ph(text, options.language)?;
        let target_ms = target_start.elapsed().as_millis();

        // Reference features (cached after first call)
        let ref_start = Instant::now();
        let ref_feats =
            self.get_ref_features(&reference_audio, reference_text, options.language)?;
        let ref_ms = ref_start.elapsed().as_millis();

        if self.sovits_model.is_none() {
            return Err(Error::ModelLoadError("SoVITS model not loaded".to_string()));
        }
        let gpt = self
            .gpt_model
            .as_ref()
            .ok_or_else(|| Error::ModelLoadError("GPT model not loaded".to_string()))?;

        // Target BERT aligned
        let bert_start = Instant::now();
        let target_bert_aligned = if let Some(bert) = self.bert_model.as_mut() {
            bert.extract(text)
                .ok()
                .and_then(|f| f.to_device(&self.device).ok().or(Some(f)))
                .and_then(|tb| {
                    gpt.project_and_align_bert(&tb, &target_word2ph, target_phoneme_ids.len())
                        .ok()
                })
        } else {
            None
        };
        let bert_ms = bert_start.elapsed().as_millis();

        let gpt = self.gpt_model.as_ref().unwrap();
        let sovits = self.sovits_model.as_ref().unwrap();

        let phoneme_ids: Vec<usize> = ref_feats
            .ref_phoneme_ids
            .iter()
            .chain(target_phoneme_ids.iter())
            .cloned()
            .collect();

        let combined_bert = match (
            ref_feats.ref_bert_aligned.as_ref(),
            target_bert_aligned.as_ref(),
        ) {
            (Some(ra), Some(ta)) => Tensor::cat(&[ra, ta], 1).ok(),
            (None, Some(ta)) => Some(ta.clone()),
            _ => None,
        };

        let gpt_start = Instant::now();
        let semantic_tokens = if !ref_feats.prompt_tokens.is_empty() {
            gpt.generate_with_prompts_aligned_bert_kv_cache(
                &phoneme_ids,
                &ref_feats.prompt_tokens,
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
                None,
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
                options.max_tokens,
            )?
        };
        let gpt_ms = gpt_start.elapsed().as_millis();

        tracing::info!(
            "Generated {} semantic tokens (kv_cache)",
            semantic_tokens.len()
        );
        if semantic_tokens.len() >= options.max_tokens {
            tracing::warn!(
                "Generated token count reached max_tokens={} in kv mode; output may be truncated",
                options.max_tokens
            );
        }

        let sovits_start = Instant::now();
        let audio_samples = sovits.synthesize(
            &semantic_tokens,
            &target_phoneme_ids,
            ref_feats.ref_mel.as_ref(),
            0.5,
        )?;
        let sovits_ms = sovits_start.elapsed().as_millis();
        tracing::info!(
            "profile mode=kv target={}ms ref={}ms target_bert={}ms gpt={}ms sovits={}ms total={}ms tokens={} audio_samples={}",
            target_ms,
            ref_ms,
            bert_ms,
            gpt_ms,
            sovits_ms,
            total_start.elapsed().as_millis(),
            semantic_tokens.len(),
            audio_samples.len()
        );

        Ok(AudioBuffer::new(audio_samples, sovits.sampling_rate(), 1))
    }

    /// Like `inference_kv_cache` but accelerated with a CUDA graph for the decode loop.
    ///
    /// The experimental CUDA path validates its first graph result before sampling and
    /// continues from a guarded KV state if capture is not numerically correct. Non-CUDA
    /// devices use static KV without graph capture.
    pub fn inference_cuda_graph<P: AsRef<Path>>(
        &mut self,
        text: &str,
        reference_audio: P,
        reference_text: &str,
        options: &InferenceOptions,
    ) -> Result<AudioBuffer> {
        if std::env::var("GPT_SOVITS_EXPERIMENTAL_CUDA_GRAPH").as_deref() != Ok("1") {
            static WARNED: Once = Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    "CUDA Graph is disabled because its long-generation output is not yet stable; using KV cache"
                );
            });
            return self.inference_kv_cache(text, reference_audio, reference_text, options);
        }

        let total_start = Instant::now();
        let target_start = Instant::now();
        let (target_phoneme_ids, target_word2ph) = self
            .text_frontend
            .process_with_word2ph(text, options.language)?;
        let target_ms = target_start.elapsed().as_millis();

        let ref_start = Instant::now();
        let ref_feats =
            self.get_ref_features(&reference_audio, reference_text, options.language)?;
        let ref_ms = ref_start.elapsed().as_millis();

        if self.sovits_model.is_none() {
            return Err(Error::ModelLoadError("SoVITS model not loaded".to_string()));
        }
        let gpt = self
            .gpt_model
            .as_ref()
            .ok_or_else(|| Error::ModelLoadError("GPT model not loaded".to_string()))?;

        let bert_start = Instant::now();
        let target_bert_aligned = if let Some(bert) = self.bert_model.as_mut() {
            bert.extract(text)
                .ok()
                .and_then(|f| f.to_device(&self.device).ok().or(Some(f)))
                .and_then(|tb| {
                    gpt.project_and_align_bert(&tb, &target_word2ph, target_phoneme_ids.len())
                        .ok()
                })
        } else {
            None
        };
        let bert_ms = bert_start.elapsed().as_millis();

        let gpt = self.gpt_model.as_ref().unwrap();
        let sovits = self.sovits_model.as_ref().unwrap();

        let phoneme_ids: Vec<usize> = ref_feats
            .ref_phoneme_ids
            .iter()
            .chain(target_phoneme_ids.iter())
            .cloned()
            .collect();

        let combined_bert = match (
            ref_feats.ref_bert_aligned.as_ref(),
            target_bert_aligned.as_ref(),
        ) {
            (Some(ra), Some(ta)) => Tensor::cat(&[ra, ta], 1).ok(),
            (None, Some(ta)) => Some(ta.clone()),
            _ => None,
        };

        // max_kv_len covers prefill + all generated tokens with a small safety margin
        let max_kv_len =
            phoneme_ids.len() + ref_feats.prompt_tokens.len() + options.max_tokens + 32;

        let gpt_start = Instant::now();
        let semantic_tokens = if !ref_feats.prompt_tokens.is_empty() {
            gpt.generate_with_cuda_graph(
                &phoneme_ids,
                &ref_feats.prompt_tokens,
                combined_bert.as_ref(),
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
                options.max_tokens,
                max_kv_len,
            )?
        } else {
            gpt.generate_with_features(
                &phoneme_ids,
                combined_bert.as_ref(),
                None,
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
                options.max_tokens,
            )?
        };
        let gpt_ms = gpt_start.elapsed().as_millis();

        tracing::info!(
            "Generated {} semantic tokens (cuda-graph)",
            semantic_tokens.len()
        );
        if semantic_tokens.len() >= options.max_tokens {
            tracing::warn!(
                "Generated token count reached max_tokens={} in cuda-graph mode; output may be truncated",
                options.max_tokens
            );
        }

        if std::env::var("SOVITS_DEBUG").is_ok() {
            sovits.debug_pipeline(
                &semantic_tokens,
                &target_phoneme_ids,
                ref_feats.ref_mel.as_ref(),
                0.5,
            )?;
        }

        let sovits_start = Instant::now();
        let audio_samples = sovits.synthesize(
            &semantic_tokens,
            &target_phoneme_ids,
            ref_feats.ref_mel.as_ref(),
            0.5,
        )?;
        let sovits_ms = sovits_start.elapsed().as_millis();
        tracing::info!(
            "profile mode=cuda-graph target={}ms ref={}ms target_bert={}ms gpt={}ms sovits={}ms total={}ms tokens={} audio_samples={}",
            target_ms,
            ref_ms,
            bert_ms,
            gpt_ms,
            sovits_ms,
            total_start.elapsed().as_millis(),
            semantic_tokens.len(),
            audio_samples.len()
        );

        Ok(AudioBuffer::new(audio_samples, sovits.sampling_rate(), 1))
    }

    /// Static version of extract_ref_mel (no &self needed, called from compute_ref_features)
    fn extract_ref_mel_static(
        ref_audio: &Path,
        device: &Device,
        sovits_sr: u32,
        sovits_n_mels: usize,
    ) -> Result<Option<Tensor>> {
        use hound::WavReader;

        let mut reader = WavReader::open(ref_audio)
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
                reader
                    .samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / max)
                    .collect()
            }
            hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        };

        let samples = if spec.channels > 1 {
            samples
                .chunks(spec.channels as usize)
                .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        } else {
            samples
        };

        let samples = if spec.sample_rate != sovits_sr {
            use soxr::{format::Mono, Soxr};
            let ratio = sovits_sr as f64 / spec.sample_rate as f64;
            let out_cap = (samples.len() as f64 * ratio).ceil() as usize + 64;
            let mut output = vec![0.0f32; out_cap];
            let mut resampler = Soxr::<Mono<f32>>::new(spec.sample_rate as f64, sovits_sr as f64)
                .map_err(|e| Error::AudioError(format!("soxr init: {}", e)))?;
            let proc = resampler
                .process(&samples, &mut output)
                .map_err(|e| Error::AudioError(format!("soxr process: {}", e)))?;
            let mut tail = vec![0.0f32; out_cap];
            let tail_n = resampler
                .drain(&mut tail)
                .map_err(|e| Error::AudioError(format!("soxr drain: {}", e)))?;
            output.truncate(proc.output_frames);
            output.extend_from_slice(&tail[..tail_n]);
            output
        } else {
            samples
        };

        let n_fft = 2048;
        let hop_length = 640;
        let extractor = SpectrogramExtractor::new(sovits_sr, n_fft, hop_length, sovits_n_mels);
        let stft_mag = extractor.extract_spectrogram_batched(&samples, device)?;
        let stft_mag = stft_mag.narrow(1, 0, 704)?;

        tracing::info!("Extracted reference STFT magnitude: {:?}", stft_mag.dims());
        Ok(Some(stft_mag))
    }
}

/// Iterator that yields one `AudioBuffer` per sentence, advancing as each sentence is synthesized.
struct SentenceIterator<'a> {
    pipeline: &'a mut Pipeline,
    sentences: Vec<String>,
    index: usize,
    reference_audio: std::path::PathBuf,
    reference_text: &'a str,
    options: &'a InferenceOptions,
    mode: String,
}

impl<'a> Iterator for SentenceIterator<'a> {
    type Item = Result<AudioBuffer>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.sentences.len() {
            return None;
        }
        let text = self.sentences[self.index].clone();
        self.index += 1;
        tracing::info!(
            "Streaming sentence {}/{}: {:?}",
            self.index,
            self.sentences.len(),
            text
        );
        Some(self.pipeline.inference_with_mode(
            &self.mode,
            &text,
            &self.reference_audio,
            self.reference_text,
            self.options,
        ))
    }
}
