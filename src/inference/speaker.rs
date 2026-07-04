//! Speaker/reference feature cache and target text preparation.

use super::ref_audio;
use super::Pipeline;
use crate::models::{BertModel, GPTModel, HubertModel, SemanticTokenizer};
use crate::text_frontend::TextFrontend;
use crate::{Error, Language, Result};
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

    /// Get cached ref features (compute and cache on miss).
    pub(super) fn get_ref_features<P: AsRef<Path>>(
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
            bert.extract(&normalized_text)
                .ok()
                .and_then(|f| f.to_device(device).ok().or(Some(f)))
                .and_then(|tb| {
                    gpt.project_and_align_bert(&tb, &target_word2ph, target_phoneme_ids.len())
                        .ok()
                })
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
            (Some(ra), Some(ta)) => Tensor::cat(&[ra, ta], 1).ok(),
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
    ) -> Result<CachedSpeaker> {
        let (ref_phoneme_ids, ref_word2ph, normalized_ref_text) = if !ref_text.is_empty() {
            text_frontend.process_with_word2ph_and_text(ref_text, language)?
        } else {
            (vec![], vec![], String::new())
        };

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
        let _ = hubert_features;

        let ref_bert_aligned = if let (Some(bert), Some(gpt), false) =
            (bert_model.as_mut(), gpt_model, ref_phoneme_ids.is_empty())
        {
            bert.extract(&normalized_ref_text)
                .ok()
                .and_then(|f| f.to_device(device).ok().or(Some(f)))
                .and_then(|rb| {
                    gpt.project_and_align_bert(&rb, &ref_word2ph, ref_phoneme_ids.len())
                        .ok()
                })
        } else {
            None
        };

        let ref_mel = ref_audio::extract_ref_mel(ref_audio, device, sovits_sr, sovits_n_mels)?;

        Ok(CachedSpeaker {
            prompt_tokens,
            ref_mel,
            ref_phoneme_ids,
            ref_bert_aligned,
        })
    }
}
