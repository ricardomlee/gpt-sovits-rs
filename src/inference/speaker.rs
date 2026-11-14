//! Speaker/reference feature cache and target text preparation.

use super::ref_audio;
use super::Pipeline;
use crate::models::{BertModel, GPTModel, HubertModel, SemanticTokenizer};
use crate::text_frontend::TextFrontend;
use crate::utils::load_safetensors;
use crate::{Error, InferenceOptions, Language, Result};
use candle_core::{Device, Tensor};
use std::path::Path;
use std::time::Instant;

/// Cached features derived from a (reference_audio, reference_text) pair.
/// All fields are clone-cheap (Vec or Tensor with Arc-backed storage).
#[derive(Clone)]
pub(super) struct CachedSpeaker {
    /// VQ semantic tokens from HuBERT - used as GPT prefix.
    pub(super) prompt_tokens: Vec<usize>,
    /// STFT magnitude of reference audio - used for SoVITS ref_enc speaker conditioning.
    pub(super) ref_mel: Option<Tensor>,
    /// Optional v2Pro speaker-verification embedding, shape [1, 20480].
    pub(super) sv_embedding: Option<Tensor>,
    /// Phone IDs for reference text.
    pub(super) ref_phoneme_ids: Vec<usize>,
    /// BERT features aligned to reference phone level.
    pub(super) ref_bert_aligned: Option<Tensor>,
}

pub(super) struct PreparedTarget {
    pub(super) target_phoneme_ids: Vec<usize>,
    pub(super) phoneme_ids: Vec<usize>,
    pub(super) combined_bert: Option<Tensor>,
    pub(super) target_ms: u128,
    pub(super) bert_ms: u128,
}

impl Pipeline {
    fn cache_speaker(&mut self, key: (String, String, Option<String>), cached: CachedSpeaker) {
        self.ref_cache_order.retain(|candidate| candidate != &key);
        self.ref_cache.insert(key.clone(), cached);
        self.ref_cache_order.push_back(key);
        while self.ref_cache_order.len() > self.ref_cache_capacity {
            if let Some(oldest) = self.ref_cache_order.pop_front() {
                self.ref_cache.remove(&oldest);
            }
        }
    }

    fn touch_speaker_cache(&mut self, key: &(String, String, Option<String>)) {
        self.ref_cache_order.retain(|candidate| candidate != key);
        self.ref_cache_order.push_back(key.clone());
    }

    /// Pre-compute and cache reference speaker features.
    /// Call once per speaker before batch inference to avoid repeated HuBERT/BERT runs.
    pub fn preload_speaker<P: AsRef<Path>>(
        &mut self,
        ref_audio: P,
        ref_text: &str,
        language: Language,
    ) -> Result<()> {
        let options = InferenceOptions {
            language,
            ..InferenceOptions::default()
        };
        self.preload_speaker_with_options(ref_audio, ref_text, &options)
    }

    /// Pre-compute and cache reference speaker features with full inference options.
    /// This includes v2Pro SV embeddings when configured.
    pub fn preload_speaker_with_options<P: AsRef<Path>>(
        &mut self,
        ref_audio: P,
        ref_text: &str,
        options: &InferenceOptions,
    ) -> Result<()> {
        let sv_embedding_key = options
            .sv_embedding
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned());
        let key = (
            ref_audio.as_ref().to_string_lossy().into_owned(),
            ref_text.to_owned(),
            sv_embedding_key,
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
                options.language,
                &self.device,
                sr,
                n_mels,
                options.sv_embedding.as_deref(),
            )?;
            self.cache_speaker(key, cached);
        } else {
            self.touch_speaker_cache(&key);
        }
        Ok(())
    }

    /// Drop all cached speaker features.
    pub fn clear_speaker_cache(&mut self) {
        self.ref_cache.clear();
    }

    /// Get cached ref features (compute and cache on miss).
    pub(super) fn get_ref_features<P: AsRef<Path>>(
        &mut self,
        ref_audio: P,
        ref_text: &str,
        options: &InferenceOptions,
    ) -> Result<CachedSpeaker> {
        let sv_embedding_key = options
            .sv_embedding
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned());
        let key = (
            ref_audio.as_ref().to_string_lossy().into_owned(),
            ref_text.to_owned(),
            sv_embedding_key,
        );
        if let Some(cached) = self.ref_cache.get(&key).cloned() {
            tracing::debug!("Speaker cache hit: {:?}", key.0);
            self.touch_speaker_cache(&key);
            return Ok(cached);
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
            options.language,
            &self.device,
            sr,
            n_mels,
            options.sv_embedding.as_deref(),
        )?;
        self.cache_speaker(key, cached.clone());
        Ok(cached)
    }

    pub(super) fn prepare_target_features(
        text_frontend: &mut TextFrontend,
        bert_model: &mut Option<BertModel>,
        device: &Device,
        gpt: &GPTModel,
        ref_feats: &CachedSpeaker,
        text: &str,
        language: Language,
    ) -> Result<PreparedTarget> {
        let target_start = Instant::now();
        let (target_phoneme_ids, target_word2ph, normalized_text) =
            text_frontend.process_with_word2ph_and_text(text, language)?;
        let target_ms = target_start.elapsed().as_millis();

        let bert_start = Instant::now();
        let target_bert_aligned = if let Some(bert) = bert_model.as_mut() {
            Some(extract_bert_features(
                bert,
                gpt,
                &normalized_text,
                &target_word2ph,
                target_phoneme_ids.len(),
                device,
                "Target",
            )?)
        } else {
            None
        };
        let bert_ms = bert_start.elapsed().as_millis();

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
            (Some(ra), Some(ta)) => Some(Tensor::cat(&[ra, ta], 1).map_err(|e| {
                Error::InferenceError(format!("BERT reference/target concatenation failed: {e}"))
            })?),
            (None, Some(ta)) => Some(ta.clone()),
            _ => None,
        };

        Ok(PreparedTarget {
            target_phoneme_ids,
            phoneme_ids,
            combined_bert,
            target_ms,
            bert_ms,
        })
    }

    /// Compute all features that depend only on (ref_audio, ref_text).
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
        sv_embedding_path: Option<&Path>,
    ) -> Result<CachedSpeaker> {
        let (ref_phoneme_ids, ref_word2ph, normalized_ref_text) = if !ref_text.is_empty() {
            text_frontend.process_with_word2ph_and_text(ref_text, language)?
        } else {
            (vec![], vec![], String::new())
        };

        let prompt_tokens = if let Some(hubert) = hubert_model {
            let features = hubert.extract(ref_audio).map_err(|e| {
                Error::InferenceError(format!("Reference HuBERT extraction failed: {e}"))
            })?;
            let features = features.to_device(device).map_err(|e| {
                Error::InferenceError(format!("Reference HuBERT device transfer failed: {e}"))
            })?;
            tracing::info!("Extracted Hubert features: {:?}", features.dims());
            let tokenizer = semantic_tokenizer.as_ref().ok_or_else(|| {
                Error::ModelLoadError("Semantic tokenizer not loaded for HuBERT features".into())
            })?;
            let tokens = tokenizer.extract(&features.transpose(1, 2)?).map_err(|e| {
                Error::InferenceError(format!("Reference semantic token extraction failed: {e}"))
            })?;
            tracing::debug!("Prompt tokens: {}", tokens.len());
            tokens
        } else {
            vec![]
        };

        let ref_bert_aligned = if let (Some(bert), Some(gpt), false) =
            (bert_model.as_mut(), gpt_model, ref_phoneme_ids.is_empty())
        {
            Some(extract_bert_features(
                bert,
                gpt,
                &normalized_ref_text,
                &ref_word2ph,
                ref_phoneme_ids.len(),
                device,
                "Reference",
            )?)
        } else {
            None
        };

        let ref_mel = ref_audio::extract_ref_mel(ref_audio, device, sovits_sr, sovits_n_mels)?;
        let sv_embedding = sv_embedding_path
            .map(|path| load_sv_embedding(path, device))
            .transpose()?;

        Ok(CachedSpeaker {
            prompt_tokens,
            ref_mel,
            sv_embedding,
            ref_phoneme_ids,
            ref_bert_aligned,
        })
    }
}

fn extract_bert_features(
    bert: &mut BertModel,
    gpt: &GPTModel,
    text: &str,
    word2ph: &[usize],
    n_phones: usize,
    device: &Device,
    stage: &str,
) -> Result<Tensor> {
    let features = bert
        .extract(text)
        .map_err(|e| Error::InferenceError(format!("{stage} BERT extraction failed: {e}")))?;
    let features = features
        .to_device(device)
        .map_err(|e| Error::InferenceError(format!("{stage} BERT device transfer failed: {e}")))?;
    gpt.project_and_align_bert(&features, word2ph, n_phones)
        .map_err(|e| Error::InferenceError(format!("{stage} BERT alignment failed: {e}")))
}

fn load_sv_embedding(path: &Path, device: &Device) -> Result<Tensor> {
    let weights = load_safetensors(path)?;
    let tensor = weights
        .get("sv_embedding")
        .or_else(|| weights.get("embedding"))
        .or_else(|| weights.values().next())
        .ok_or_else(|| {
            Error::ModelLoadError(format!(
                "SV embedding safetensors contains no tensors: {}",
                path.display()
            ))
        })?
        .clone()
        .to_device(device)?;
    match tensor.dims() {
        [20480] => Ok(tensor.unsqueeze(0)?),
        [1, 20480] => Ok(tensor),
        other => Err(Error::ModelLoadError(format!(
            "SV embedding must have shape [20480] or [1, 20480], got {:?}: {}",
            other,
            path.display()
        ))),
    }
}
