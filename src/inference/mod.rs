//! Inference Module
//!
//! Main pipeline for TTS inference

use crate::config::Config;
use crate::models::{BertModel, BigVGAN, GPTModel, HubertModel, SemanticTokenizer, SoVITSModel};
use crate::text_frontend::TextFrontend;
use crate::utils::AudioBuffer;
use crate::{Error, Result};
use candle_core::Device;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

mod options;
mod ref_audio;
mod speaker;
mod split;

pub use options::{InferenceOptions, InferenceOptionsBuilder};
use speaker::{CachedSpeaker, PreparedTarget};
use split::split_text;
pub use split::{
    split_cut5_for_language, split_sentences, split_sentences_for_language, SplitMethod,
};

#[derive(Debug, Clone, Copy)]
enum DecodeBackend {
    Plain,
    KvCache,
    CudaGraph,
}

impl DecodeBackend {
    fn profile_mode(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::KvCache => "kv",
            Self::CudaGraph => "cuda-graph",
        }
    }

    fn generated_suffix(self) -> &'static str {
        match self {
            Self::Plain => "",
            Self::KvCache => " (kv_cache)",
            Self::CudaGraph => " (cuda-graph)",
        }
    }

    fn generate_semantic_tokens(
        self,
        gpt: &GPTModel,
        ref_feats: &CachedSpeaker,
        prepared: &PreparedTarget,
        options: &InferenceOptions,
    ) -> Result<Vec<usize>> {
        if ref_feats.prompt_tokens.is_empty() {
            return gpt.generate_with_features(
                &prepared.phoneme_ids,
                prepared.combined_bert.as_ref(),
                None,
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
                options.max_tokens,
            );
        }

        match self {
            Self::Plain => gpt.generate_with_prompts_aligned_bert(
                &prepared.phoneme_ids,
                &ref_feats.prompt_tokens,
                prepared.combined_bert.as_ref(),
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
                options.max_tokens,
            ),
            Self::KvCache => gpt.generate_with_prompts_aligned_bert_kv_cache(
                &prepared.phoneme_ids,
                &ref_feats.prompt_tokens,
                prepared.combined_bert.as_ref(),
                options.top_k,
                options.top_p,
                options.temperature,
                options.repetition_penalty,
                options.max_tokens,
            ),
            Self::CudaGraph => {
                let max_kv_len = prepared.phoneme_ids.len()
                    + ref_feats.prompt_tokens.len()
                    + options.max_tokens
                    + 32;
                gpt.generate_with_cuda_graph(
                    &prepared.phoneme_ids,
                    &ref_feats.prompt_tokens,
                    prepared.combined_bert.as_ref(),
                    options.top_k,
                    options.top_p,
                    options.temperature,
                    options.repetition_penalty,
                    options.max_tokens,
                    max_kv_len,
                )
            }
        }
    }

    fn log_generated(self, token_count: usize, max_tokens: usize) {
        tracing::info!(
            "Generated {} semantic tokens{}",
            token_count,
            self.generated_suffix()
        );
        if token_count >= max_tokens {
            tracing::warn!(
                "Generated token count reached max_tokens={} in {} mode; output may be truncated",
                max_tokens,
                self.profile_mode()
            );
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
        self.inference_sentences_with_method(
            text,
            reference_audio,
            reference_text,
            options,
            mode,
            min_sentence_chars,
            SplitMethod::Sentence,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inference_sentences_with_method<'a, P: AsRef<Path> + 'a>(
        &'a mut self,
        text: &'a str,
        reference_audio: P,
        reference_text: &'a str,
        options: &'a InferenceOptions,
        mode: &str,
        min_sentence_chars: usize,
        split_method: SplitMethod,
    ) -> impl Iterator<Item = Result<AudioBuffer>> + 'a {
        let sentences = split_text(text, min_sentence_chars, options.language, split_method);
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
            "auto" if self.automatic_inference_mode() == "cuda-graph" => {
                self.inference_cuda_graph(text, reference_audio, reference_text, options)
            }
            "auto" => self.inference_kv_cache(text, reference_audio, reference_text, options),
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

    /// Split text with the Python-compatible policy and concatenate synthesized chunks.
    #[allow(clippy::too_many_arguments)]
    pub fn inference_split<P: AsRef<Path>>(
        &mut self,
        text: &str,
        reference_audio: P,
        reference_text: &str,
        options: &InferenceOptions,
        mode: &str,
        min_chars: usize,
        gap_ms: u32,
        fade_ms: u32,
    ) -> Result<AudioBuffer> {
        self.inference_split_with_method(
            text,
            reference_audio,
            reference_text,
            options,
            mode,
            min_chars,
            gap_ms,
            fade_ms,
            SplitMethod::Sentence,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inference_split_with_method<P: AsRef<Path>>(
        &mut self,
        text: &str,
        reference_audio: P,
        reference_text: &str,
        options: &InferenceOptions,
        mode: &str,
        min_chars: usize,
        gap_ms: u32,
        fade_ms: u32,
        split_method: SplitMethod,
    ) -> Result<AudioBuffer> {
        let chunks = split_text(text, min_chars, options.language, split_method);
        let reference_audio = reference_audio.as_ref().to_path_buf();
        self.preload_speaker(&reference_audio, reference_text, options.language)?;

        let mut output: Option<AudioBuffer> = None;
        for chunk in chunks {
            let mut audio =
                self.inference_with_mode(mode, &chunk, &reference_audio, reference_text, options)?;
            if fade_ms > 0 {
                audio.fade_in(fade_ms);
                audio.fade_out(fade_ms);
            }
            if let Some(current) = output.as_mut() {
                if gap_ms > 0 {
                    let gap_samples = (gap_ms as f32 * current.sample_rate as f32 / 1000.0)
                        as usize
                        * current.channels as usize;
                    current.concat(&AudioBuffer::new(
                        vec![0.0; gap_samples],
                        current.sample_rate,
                        current.channels,
                    ))?;
                }
                current.concat(&audio)?;
            } else {
                output = Some(audio);
            }
        }
        output.ok_or_else(|| Error::InferenceError("No sentence chunks generated".to_string()))
    }

    /// Select the production decode path for the active device and model dtype.
    pub fn automatic_inference_mode(&self) -> &'static str {
        let graph_disabled = std::env::var("GPT_SOVITS_DISABLE_CUDA_GRAPH").as_deref() == Ok("1");
        let graph_supported = matches!(self.device, Device::Cuda(_))
            && self
                .gpt_model
                .as_ref()
                .is_none_or(|model| model.dtype() == candle_core::DType::F32);
        if graph_supported && !graph_disabled {
            "cuda-graph"
        } else {
            "kv"
        }
    }

    /// Reset the backend RNG for reproducible synthesis and quality comparisons.
    pub fn set_seed(&self, seed: u64) -> Result<()> {
        self.device.set_seed(seed)?;
        Ok(())
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

        // Reference features (cached after first call)
        let ref_start = Instant::now();
        let ref_feats =
            self.get_ref_features(&reference_audio, reference_text, options.language)?;
        let ref_ms = ref_start.elapsed().as_millis();

        let gpt = self.gpt_model.as_ref().unwrap();
        let sovits = self.sovits_model.as_ref().unwrap();
        let prepared = Self::prepare_target_features(
            &mut self.text_frontend,
            &mut self.bert_model,
            &self.device,
            gpt,
            &ref_feats,
            text,
            options.language,
        )?;

        // GPT generation
        let gpt_start = Instant::now();
        let backend = DecodeBackend::Plain;
        let semantic_tokens =
            backend.generate_semantic_tokens(gpt, &ref_feats, &prepared, options)?;
        let gpt_ms = gpt_start.elapsed().as_millis();
        backend.log_generated(semantic_tokens.len(), options.max_tokens);

        let sovits_start = Instant::now();
        let audio_samples = sovits.synthesize_with_speed(
            &semantic_tokens,
            &prepared.target_phoneme_ids,
            ref_feats.ref_mel.as_ref(),
            0.5,
            options.speed,
        )?;
        let sovits_ms = sovits_start.elapsed().as_millis();
        tracing::info!(
            "profile mode=plain target={}ms ref={}ms target_bert={}ms gpt={}ms sovits={}ms total={}ms tokens={} audio_samples={}",
            prepared.target_ms,
            ref_ms,
            prepared.bert_ms,
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
        let prepared = Self::prepare_target_features(
            &mut self.text_frontend,
            &mut self.bert_model,
            &self.device,
            gpt,
            &ref_feats,
            text,
            options.language,
        )?;

        let gpt = self.gpt_model.as_ref().unwrap();
        let sovits = self.sovits_model.as_ref().unwrap();

        let gpt_start = Instant::now();
        let backend = DecodeBackend::KvCache;
        let semantic_tokens =
            backend.generate_semantic_tokens(gpt, &ref_feats, &prepared, options)?;
        let gpt_ms = gpt_start.elapsed().as_millis();
        backend.log_generated(semantic_tokens.len(), options.max_tokens);

        let sovits_start = Instant::now();
        let audio_samples = sovits.synthesize_with_speed(
            &semantic_tokens,
            &prepared.target_phoneme_ids,
            ref_feats.ref_mel.as_ref(),
            0.5,
            options.speed,
        )?;
        let sovits_ms = sovits_start.elapsed().as_millis();
        tracing::info!(
            "profile mode=kv target={}ms ref={}ms target_bert={}ms gpt={}ms sovits={}ms total={}ms tokens={} audio_samples={}",
            prepared.target_ms,
            ref_ms,
            prepared.bert_ms,
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
    /// The CUDA path validates its first graph result before sampling and continues from
    /// a guarded KV state if capture is not numerically correct. Non-CUDA devices use
    /// static KV without graph capture when this method is called explicitly.
    pub fn inference_cuda_graph<P: AsRef<Path>>(
        &mut self,
        text: &str,
        reference_audio: P,
        reference_text: &str,
        options: &InferenceOptions,
    ) -> Result<AudioBuffer> {
        let total_start = Instant::now();

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
        let prepared = Self::prepare_target_features(
            &mut self.text_frontend,
            &mut self.bert_model,
            &self.device,
            gpt,
            &ref_feats,
            text,
            options.language,
        )?;

        let gpt = self.gpt_model.as_ref().unwrap();
        let sovits = self.sovits_model.as_ref().unwrap();

        let gpt_start = Instant::now();
        let backend = DecodeBackend::CudaGraph;
        let semantic_tokens =
            backend.generate_semantic_tokens(gpt, &ref_feats, &prepared, options)?;
        let gpt_ms = gpt_start.elapsed().as_millis();
        backend.log_generated(semantic_tokens.len(), options.max_tokens);

        if std::env::var("SOVITS_DEBUG").is_ok() {
            sovits.debug_pipeline(
                &semantic_tokens,
                &prepared.target_phoneme_ids,
                ref_feats.ref_mel.as_ref(),
                0.5,
            )?;
        }

        let sovits_start = Instant::now();
        let audio_samples = sovits.synthesize_with_speed(
            &semantic_tokens,
            &prepared.target_phoneme_ids,
            ref_feats.ref_mel.as_ref(),
            0.5,
            options.speed,
        )?;
        let sovits_ms = sovits_start.elapsed().as_millis();
        tracing::info!(
            "profile mode=cuda-graph target={}ms ref={}ms target_bert={}ms gpt={}ms sovits={}ms total={}ms tokens={} audio_samples={}",
            prepared.target_ms,
            ref_ms,
            prepared.bert_ms,
            gpt_ms,
            sovits_ms,
            total_start.elapsed().as_millis(),
            semantic_tokens.len(),
            audio_samples.len()
        );

        Ok(AudioBuffer::new(audio_samples, sovits.sampling_rate(), 1))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_mode_uses_kv_on_cpu_without_loaded_models() {
        let pipeline = Pipeline::new(Config::builder().with_device("cpu").build()).unwrap();

        assert_eq!(pipeline.automatic_inference_mode(), "kv");
    }

    #[test]
    fn inference_mode_rejects_unknown_mode_before_model_access() {
        let mut pipeline = Pipeline::new(Config::builder().with_device("cpu").build()).unwrap();
        let options = InferenceOptions::default();
        let error = pipeline
            .inference_with_mode("bogus", "hello", "missing.wav", "prompt", &options)
            .expect_err("unknown mode should fail");

        assert!(matches!(error, Error::ConfigError(_)));
        assert!(error.to_string().contains("Unsupported inference mode"));
    }

    #[test]
    fn plain_inference_reports_missing_gpt_model_before_audio_access() {
        let mut pipeline = Pipeline::new(Config::builder().with_device("cpu").build()).unwrap();
        let options = InferenceOptions::default();
        let error = pipeline
            .inference_with_mode("plain", "hello", "missing.wav", "prompt", &options)
            .expect_err("unloaded GPT should fail");

        assert!(matches!(error, Error::ModelLoadError(_)));
        assert!(error.to_string().contains("GPT model not loaded"));
    }
}
