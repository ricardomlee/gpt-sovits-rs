//! Speaker/reference feature cache and target text preparation.

use super::ref_audio;
use super::Pipeline;
use crate::models::{BertModel, GPTModel, HubertModel, SemanticTokenizer, SvModel};
use crate::text_frontend::TextFrontend;
use crate::utils::load_safetensors;
use crate::{Error, InferenceOptions, Language, Result};
use candle_core::{DType, Device, Tensor};
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
        self.get_ref_features(ref_audio, ref_text, options)
            .map(|_| ())
    }

    /// Drop all cached speaker features.
    pub fn clear_speaker_cache(&mut self) {
        self.ref_cache.clear();
        self.ref_cache_order.clear();
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
        let sv_embedding = resolve_sv_embedding(
            self.sv_model.as_ref(),
            options.sv_embedding.as_deref(),
            sovits.requires_sv(),
            ref_audio.as_ref(),
            sr,
            &self.device,
        )?;
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
            sv_embedding,
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
        sv_embedding: Option<Tensor>,
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
            sv_embedding,
            ref_phoneme_ids,
            ref_bert_aligned,
        })
    }
}

fn resolve_sv_embedding(
    model: Option<&SvModel>,
    explicit: Option<&Path>,
    required: bool,
    audio: &Path,
    sample_rate: u32,
    device: &Device,
) -> Result<Option<Tensor>> {
    if let Some(path) = explicit {
        return load_sv_embedding(path, device).map(Some);
    }
    if !required {
        return Ok(None);
    }
    let model = model.ok_or_else(|| Error::ModelLoadError(
        "v2Pro requires SV features: provide sv_embedding, or convert the ERes2NetV2 checkpoint with `gpt-sovits-convert sv-model` and place it at models/sv/sv.safetensors (or use --sv-model)".into(),
    ))?;
    let started = Instant::now();
    let embedding = model.extract(audio, sample_rate)?;
    tracing::info!(
        elapsed_ms = started.elapsed().as_millis(),
        "Extracted native SV features"
    );
    Ok(Some(embedding))
}

fn load_sv_embedding(path: &Path, device: &Device) -> Result<Tensor> {
    let weights = load_safetensors(path)?;
    let tensor = weights
        .get("sv_embedding")
        .or_else(|| weights.get("embedding"))
        .or_else(|| (weights.len() == 1).then(|| weights.values().next()).flatten())
        .ok_or_else(|| {
            Error::ModelLoadError(format!(
                "SV embedding safetensors must contain sv_embedding, embedding, or exactly one tensor: {}",
                path.display()
            ))
        })?
        .clone()
        .to_device(device)?
        .to_dtype(DType::F32)?;
    if tensor
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err(Error::ModelLoadError(
            "SV embedding contains non-finite values".into(),
        ));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_needs_no_encoder_but_v2pro_fails_without_one() {
        let missing = Path::new("no-reference.wav");
        assert!(
            resolve_sv_embedding(None, None, false, missing, 32000, &Device::Cpu)
                .unwrap()
                .is_none()
        );
        let error = resolve_sv_embedding(None, None, true, missing, 32000, &Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(error.contains("sv-model"));
        assert!(error.contains("sv_embedding"));
    }

    #[test]
    fn explicit_embedding_works_without_encoder_or_reference_audio() -> Result<()> {
        let file = tempfile::NamedTempFile::new().unwrap();
        let embedding = Tensor::ones(20480, DType::F32, &Device::Cpu)?;
        candle_core::safetensors::save(
            &std::collections::HashMap::from([("sv_embedding", embedding)]),
            file.path(),
        )?;
        let result = resolve_sv_embedding(
            None,
            Some(file.path()),
            true,
            Path::new("missing.wav"),
            32000,
            &Device::Cpu,
        )?
        .unwrap();
        assert_eq!(result.dims(), &[1, 20480]);
        Ok(())
    }

    #[test]
    fn rejects_wrong_shape_and_non_finite_explicit_embeddings() -> Result<()> {
        let file = tempfile::NamedTempFile::new().unwrap();
        for embedding in [
            Tensor::zeros(192, DType::F32, &Device::Cpu)?,
            Tensor::full(f32::NAN, 20480, &Device::Cpu)?,
        ] {
            candle_core::safetensors::save(
                &std::collections::HashMap::from([("sv_embedding", embedding)]),
                file.path(),
            )?;
            assert!(load_sv_embedding(file.path(), &Device::Cpu).is_err());
        }
        Ok(())
    }
}
