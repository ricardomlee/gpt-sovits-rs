//! HTTP request models and request-to-inference resolution.

use gpt_sovits_rs::inference::validate_sentence_timing;
use gpt_sovits_rs::voice::{
    InferenceOptionOverrides, LoadedVoiceProfile, VoiceDefaults, VoiceModelPaths,
};
use gpt_sovits_rs::{InferenceOptions, Language, SplitMethod};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize)]
pub(super) struct TtsRequest {
    pub(super) seed: Option<u64>,
    pub(super) voice: Option<String>,
    #[serde(alias = "input")]
    pub(super) text: String,
    #[serde(alias = "language", alias = "lang", alias = "languageCode")]
    pub(super) text_language: Option<String>,
    #[serde(
        alias = "reference_audio",
        alias = "referenceAudio",
        alias = "prompt_wav_path",
        alias = "promptWavPath"
    )]
    pub(super) refer_wav_path: Option<String>,
    #[serde(alias = "reference_text", alias = "referenceText")]
    pub(super) prompt_text: Option<String>,
    #[serde(alias = "svEmbedding", alias = "speakerEmbedding")]
    pub(super) sv_embedding: Option<String>,
    pub(super) prompt_language: Option<String>,
    pub(super) top_k: Option<usize>,
    pub(super) top_p: Option<f32>,
    pub(super) temperature: Option<f32>,
    pub(super) speed: Option<f32>,
    #[serde(alias = "maxTokens")]
    pub(super) max_tokens: Option<usize>,
    #[serde(alias = "repetitionPenalty")]
    pub(super) repetition_penalty: Option<f32>,
    #[serde(alias = "splitSentences")]
    pub(super) split_sentences: Option<bool>,
    #[serde(alias = "splitMethod")]
    pub(super) split_method: Option<String>,
    #[serde(alias = "minSentenceChars")]
    pub(super) min_sentence_chars: Option<usize>,
    #[serde(alias = "sentenceGapMs")]
    pub(super) sentence_gap_ms: Option<u32>,
    #[serde(alias = "sentenceFadeMs")]
    pub(super) sentence_fade_ms: Option<u32>,
}

#[derive(Deserialize)]
pub(super) struct OpenAiSpeechRequest {
    pub(super) prompt_language: Option<String>,
    pub(super) seed: Option<u64>,
    #[allow(dead_code)]
    pub(super) model: Option<String>,
    #[serde(alias = "text")]
    pub(super) input: String,
    #[serde(alias = "speakerVoice", alias = "speakerVoiceId", alias = "voiceId")]
    pub(super) voice: String,
    #[serde(
        alias = "responseFormat",
        alias = "output_format",
        alias = "outputFormat"
    )]
    pub(super) response_format: Option<String>,
    #[serde(alias = "languageCode", alias = "lang", alias = "language")]
    pub(super) text_language: Option<String>,
    #[allow(dead_code)]
    pub(super) instructions: Option<String>,
    pub(super) speed: Option<f32>,
    #[serde(alias = "maxTokens")]
    pub(super) max_tokens: Option<usize>,
    #[serde(alias = "repetitionPenalty")]
    pub(super) repetition_penalty: Option<f32>,
    #[serde(alias = "splitSentences")]
    pub(super) split_sentences: Option<bool>,
    #[serde(alias = "splitMethod")]
    pub(super) split_method: Option<String>,
    #[serde(alias = "minSentenceChars")]
    pub(super) min_sentence_chars: Option<usize>,
    #[serde(alias = "sentenceGapMs")]
    pub(super) sentence_gap_ms: Option<u32>,
    #[serde(alias = "sentenceFadeMs")]
    pub(super) sentence_fade_ms: Option<u32>,
}

/// POST /tts/batch — synthesize multiple texts in one call.
/// Shared speaker features are computed once for all items.
/// Results stream back as NDJSON (one JSON line per item) as each completes.
#[derive(Deserialize)]
pub(super) struct TtsBatchRequest {
    pub(super) prompt_language: Option<String>,
    pub(super) seed: Option<u64>,
    /// List of texts to synthesize (processed sequentially on GPU).
    #[serde(alias = "inputs")]
    pub(super) texts: Vec<String>,
    pub(super) voice: Option<String>,
    #[serde(alias = "language", alias = "lang", alias = "languageCode")]
    pub(super) text_language: Option<String>,
    #[serde(
        alias = "reference_audio",
        alias = "referenceAudio",
        alias = "prompt_wav_path",
        alias = "promptWavPath"
    )]
    pub(super) refer_wav_path: Option<String>,
    #[serde(alias = "reference_text", alias = "referenceText")]
    pub(super) prompt_text: Option<String>,
    #[serde(alias = "svEmbedding", alias = "speakerEmbedding")]
    pub(super) sv_embedding: Option<String>,
    pub(super) top_k: Option<usize>,
    pub(super) top_p: Option<f32>,
    pub(super) temperature: Option<f32>,
    pub(super) speed: Option<f32>,
    #[serde(alias = "maxTokens")]
    pub(super) max_tokens: Option<usize>,
    #[serde(alias = "repetitionPenalty")]
    pub(super) repetition_penalty: Option<f32>,
    #[serde(alias = "splitSentences")]
    pub(super) split_sentences: Option<bool>,
    #[serde(alias = "splitMethod")]
    pub(super) split_method: Option<String>,
    #[serde(alias = "minSentenceChars")]
    pub(super) min_sentence_chars: Option<usize>,
    #[serde(alias = "sentenceGapMs")]
    pub(super) sentence_gap_ms: Option<u32>,
    #[serde(alias = "sentenceFadeMs")]
    pub(super) sentence_fade_ms: Option<u32>,
}

pub(super) struct ResolvedSynthesis {
    pub(super) voice: Option<String>,
    pub(super) models: VoiceModelPaths,
    pub(super) mode: String,
    pub(super) language: Language,
    pub(super) options: InferenceOptions,
    pub(super) split_sentences: bool,
    pub(super) split_method: SplitMethod,
    pub(super) min_sentence_chars: usize,
    pub(super) sentence_gap_ms: u32,
    pub(super) sentence_fade_ms: u32,
    pub(super) refer_path: String,
    pub(super) prompt_text: String,
}

#[derive(Clone, Copy, Default)]
pub(super) struct RequestPathPolicy {
    pub(super) allow_external_reference_paths: bool,
}

#[derive(Default)]
pub(super) struct SynthesisOverrides {
    pub(super) prompt_language: Option<String>,
    pub(super) seed: Option<u64>,
    pub(super) top_k: Option<usize>,
    pub(super) top_p: Option<f32>,
    pub(super) temperature: Option<f32>,
    pub(super) speed: Option<f32>,
    pub(super) max_tokens: Option<usize>,
    pub(super) repetition_penalty: Option<f32>,
    pub(super) split_sentences: Option<bool>,
    pub(super) split_method: Option<String>,
    pub(super) min_sentence_chars: Option<usize>,
    pub(super) sentence_gap_ms: Option<u32>,
    pub(super) sentence_fade_ms: Option<u32>,
    pub(super) sv_embedding: Option<String>,
}

impl SynthesisOverrides {
    pub(super) fn from_tts(req: &TtsRequest) -> Self {
        Self {
            prompt_language: req.prompt_language.clone(),
            seed: req.seed,
            top_k: req.top_k,
            top_p: req.top_p,
            temperature: req.temperature,
            speed: req.speed,
            max_tokens: req.max_tokens,
            repetition_penalty: req.repetition_penalty,
            split_sentences: req.split_sentences,
            split_method: req.split_method.clone(),
            min_sentence_chars: req.min_sentence_chars,
            sentence_gap_ms: req.sentence_gap_ms,
            sentence_fade_ms: req.sentence_fade_ms,
            sv_embedding: req.sv_embedding.clone(),
        }
    }

    pub(super) fn from_openai(req: &OpenAiSpeechRequest) -> Self {
        Self {
            prompt_language: req.prompt_language.clone(),
            seed: req.seed,
            speed: req.speed,
            max_tokens: req.max_tokens,
            repetition_penalty: req.repetition_penalty,
            split_sentences: req.split_sentences,
            split_method: req.split_method.clone(),
            min_sentence_chars: req.min_sentence_chars,
            sentence_gap_ms: req.sentence_gap_ms,
            sentence_fade_ms: req.sentence_fade_ms,
            sv_embedding: None,
            ..Default::default()
        }
    }

    pub(super) fn from_batch(req: &TtsBatchRequest) -> Self {
        Self {
            prompt_language: req.prompt_language.clone(),
            seed: req.seed,
            top_k: req.top_k,
            top_p: req.top_p,
            temperature: req.temperature,
            speed: req.speed,
            max_tokens: req.max_tokens,
            repetition_penalty: req.repetition_penalty,
            split_sentences: req.split_sentences,
            split_method: req.split_method.clone(),
            min_sentence_chars: req.min_sentence_chars,
            sentence_gap_ms: req.sentence_gap_ms,
            sentence_fade_ms: req.sentence_fade_ms,
            sv_embedding: req.sv_embedding.clone(),
        }
    }
}

#[derive(Serialize)]
pub(super) struct BatchItemResult {
    pub(super) index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) voice: Option<String>,
    pub(super) language: &'static str,
    pub(super) text_chars: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) wav_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) error: Option<String>,
    pub(super) sample_rate: u32,
    pub(super) duration_s: f32,
    pub(super) inference_ms: u64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_synthesis(
    voice_name: Option<&str>,
    text_language: Option<&str>,
    refer_wav_path: Option<String>,
    prompt_text: Option<String>,
    overrides: SynthesisOverrides,
    voices_dir: &Path,
    models_dir: &Path,
    path_policy: RequestPathPolicy,
) -> Result<ResolvedSynthesis, String> {
    let voice_name = voice_name.map(str::trim).filter(|name| !name.is_empty());
    let voice = voice_name
        .map(|name| LoadedVoiceProfile::load(name, voices_dir))
        .transpose()?;
    let defaults = VoiceDefaults::from_profile(voice.as_ref().map(|v| &v.profile));

    let language_text = text_language
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .unwrap_or(&defaults.language);
    let language = Language::parse(language_text)
        .ok_or_else(|| format!("unsupported text_language: {language_text}"))?;
    let prompt_language = match overrides.prompt_language.as_deref().map(str::trim) {
        None | Some("") => defaults.prompt_language,
        Some(text) => {
            Language::parse(text).ok_or_else(|| format!("unsupported prompt_language: {text}"))?
        }
    };

    let refer_path = match refer_wav_path {
        Some(path) => {
            validate_request_file_path("reference audio", &path, voices_dir, path_policy)?
        }
        None => voice
            .as_ref()
            .and_then(|v| v.reference_audio_path())
            .map(|p| p.to_string_lossy().into_owned())
            .ok_or_else(|| {
                "reference audio is required; select a configured voice or pass refer_wav_path"
                    .to_string()
            })?,
    };
    if !Path::new(&refer_path).is_file() {
        return Err(format!("reference audio not found: {refer_path}"));
    }
    let prompt_text = prompt_text
        .or_else(|| {
            voice
                .as_ref()
                .and_then(|v| v.reference_text().map(str::to_string))
        })
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| {
            "reference text is required; select a configured voice or pass prompt_text".to_string()
        })?;
    let request_sv_embedding = overrides
        .sv_embedding
        .filter(|path| !path.trim().is_empty());
    let sv_embedding = match request_sv_embedding {
        Some(path) => Some(validate_request_file_path(
            "SV embedding",
            &path,
            voices_dir,
            path_policy,
        )?),
        None => voice
            .as_ref()
            .and_then(|v| v.sv_embedding_path())
            .map(|p| p.to_string_lossy().into_owned()),
    };
    if let Some(path) = sv_embedding.as_deref()
        && !Path::new(path).is_file()
    {
        return Err(format!("SV embedding not found: {path}"));
    }

    let options = defaults.to_inference_options(
        language,
        InferenceOptionOverrides {
            top_k: overrides.top_k,
            top_p: overrides.top_p,
            temperature: overrides.temperature,
            speed: overrides.speed,
            max_tokens: overrides.max_tokens,
            repetition_penalty: overrides.repetition_penalty,
        },
    );
    let mut options = if let Some(path) = sv_embedding.as_ref() {
        InferenceOptions {
            sv_embedding: Some(path.into()),
            ..options
        }
    } else {
        options
    };
    options.seed = overrides.seed;
    options.prompt_language = prompt_language;
    let split_method = match overrides.split_method.as_deref().map(str::trim) {
        Some("") | None => defaults.split_method,
        Some(method) => SplitMethod::parse(method)
            .ok_or_else(|| format!("unsupported split_method: {method}"))?,
    };

    let models = voice
        .as_ref()
        .map(|v| v.model_paths(models_dir))
        .unwrap_or_default();
    validate_optional_model_path("GPT", models.gpt.as_deref())?;
    validate_optional_model_path("SoVITS", models.sovits.as_deref())?;
    validate_optional_model_path("BigVGAN", models.bigvgan.as_deref())?;

    let resolved = ResolvedSynthesis {
        voice: voice.map(|v| v.name),
        models,
        mode: defaults.mode,
        language,
        options,
        split_sentences: overrides
            .split_sentences
            .unwrap_or(defaults.split_sentences),
        split_method,
        min_sentence_chars: overrides
            .min_sentence_chars
            .unwrap_or(defaults.min_sentence_chars),
        sentence_gap_ms: overrides
            .sentence_gap_ms
            .unwrap_or(defaults.sentence_gap_ms),
        sentence_fade_ms: overrides
            .sentence_fade_ms
            .unwrap_or(defaults.sentence_fade_ms),
        refer_path,
        prompt_text,
    };
    resolved.options.validate().map_err(|e| e.to_string())?;
    validate_sentence_timing(resolved.sentence_gap_ms, resolved.sentence_fade_ms)
        .map_err(|e| e.to_string())?;
    Ok(resolved)
}

fn validate_request_file_path(
    label: &str,
    path: &str,
    voices_dir: &Path,
    policy: RequestPathPolicy,
) -> Result<String, String> {
    let path = Path::new(path);
    if !path.is_file() {
        return Err(format!("{label} not found: {}", path.display()));
    }
    let canonical = path
        .canonicalize()
        .map_err(|e| format!("Failed to resolve {label} {}: {e}", path.display()))?;
    if !policy.allow_external_reference_paths {
        let voices_root = voices_dir.canonicalize().map_err(|e| {
            format!(
                "Failed to resolve voices directory {}: {e}",
                voices_dir.display()
            )
        })?;
        if !canonical.starts_with(&voices_root) {
            return Err(format!(
                "{label} must be inside {} (start the server with --allow-external-reference-paths to override)",
                voices_dir.display()
            ));
        }
    }
    Ok(canonical.to_string_lossy().into_owned())
}

fn validate_optional_model_path(name: &str, path: Option<&Path>) -> Result<(), String> {
    if let Some(path) = path
        && !path.is_file()
    {
        return Err(format!("{} model not found: {}", name, path.display()));
    }
    Ok(())
}

pub(super) fn validate_text(text: &str, field: &str) -> Result<(), String> {
    if text.trim().is_empty() {
        Err(format!("{field} must not be empty"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsafe_numeric_settings_fail_after_resolution() {
        let temp = tempfile::tempdir().unwrap();
        let reference = temp.path().join("ref.wav");
        write_ref_audio(&reference);
        for (field, value) in [
            ("max_tokens", serde_json::json!(0)),
            ("max_tokens", serde_json::json!(2049)),
            ("max_tokens", serde_json::json!(usize::MAX)),
            ("speed", serde_json::json!(0.0)),
            ("speed", serde_json::json!(0.24)),
            ("speed", serde_json::json!(4.1)),
            ("top_p", serde_json::json!(-0.1)),
            ("top_p", serde_json::json!(1.1)),
            ("temperature", serde_json::json!(0.0)),
            ("temperature", serde_json::json!(-1.0)),
            ("repetition_penalty", serde_json::json!(0.0)),
            ("repetition_penalty", serde_json::json!(-1.0)),
            ("sentence_gap_ms", serde_json::json!(10_001)),
            ("sentence_gap_ms", serde_json::json!(u32::MAX)),
            ("sentence_fade_ms", serde_json::json!(1001)),
        ] {
            let mut body = serde_json::json!({"voice":"demo","text":"hello","texts":["hello"]});
            body[field] = value;
            let tts: TtsRequest = serde_json::from_value(body.clone()).unwrap();
            let batch: TtsBatchRequest = serde_json::from_value(body.clone()).unwrap();
            let mut overrides = vec![
                SynthesisOverrides::from_tts(&tts),
                SynthesisOverrides::from_batch(&batch),
            ];
            if !matches!(field, "top_p" | "temperature") {
                let speech: OpenAiSpeechRequest = serde_json::from_value(body).unwrap();
                overrides.push(SynthesisOverrides::from_openai(&speech));
            }
            for overrides in overrides {
                let error = resolve_synthesis(
                    None,
                    None,
                    Some(reference.to_string_lossy().into_owned()),
                    Some("hello".into()),
                    overrides,
                    temp.path(),
                    temp.path(),
                    RequestPathPolicy::default(),
                )
                .err()
                .expect("unsafe effective settings must be rejected");
                assert!(error.contains(field), "{field}: {error}");
            }
        }
    }

    #[test]
    fn validation_uses_effective_settings_including_voice_defaults() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("demo");
        std::fs::create_dir(&dir).unwrap();
        write_ref_audio(&dir.join("ref.wav"));
        std::fs::write(
            dir.join("voice.json"),
            r#"{"reference_audio":"ref.wav","reference_text":"hello","speed":0,"sentence_gap_ms":4294967295}"#,
        )
        .unwrap();
        let resolve = |overrides| {
            resolve_synthesis(
                Some("demo"),
                None,
                None,
                None,
                overrides,
                temp.path(),
                temp.path(),
                RequestPathPolicy::default(),
            )
        };
        assert!(resolve(SynthesisOverrides::default()).is_err());
        let resolved = resolve(SynthesisOverrides {
            speed: Some(1.0),
            sentence_gap_ms: Some(120),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(resolved.options.speed, 1.0);
        assert_eq!(resolved.sentence_gap_ms, 120);
    }

    #[test]
    fn seed_is_optional_and_must_be_an_unsigned_integer_on_every_endpoint() {
        for seed in [
            serde_json::json!(0),
            serde_json::json!(u64::MAX),
            serde_json::Value::Null,
        ] {
            let body =
                serde_json::json!({"voice":"demo","text":"hello","texts":["hello"],"seed":seed});
            let tts: TtsRequest = serde_json::from_value(body.clone()).unwrap();
            let speech: OpenAiSpeechRequest = serde_json::from_value(body.clone()).unwrap();
            let batch: TtsBatchRequest = serde_json::from_value(body).unwrap();
            assert_eq!(SynthesisOverrides::from_tts(&tts).seed, seed.as_u64());
            assert_eq!(SynthesisOverrides::from_openai(&speech).seed, seed.as_u64());
            assert_eq!(SynthesisOverrides::from_batch(&batch).seed, seed.as_u64());
        }
        let tts: TtsRequest = serde_json::from_str(r#"{"text":"hello"}"#).unwrap();
        assert_eq!(tts.seed, None);
        for seed in [
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("42"),
            serde_json::json!(true),
        ] {
            let body =
                serde_json::json!({"voice":"demo","text":"hello","texts":["hello"],"seed":seed});
            assert!(serde_json::from_value::<TtsRequest>(body.clone()).is_err());
            assert!(serde_json::from_value::<OpenAiSpeechRequest>(body.clone()).is_err());
            assert!(serde_json::from_value::<TtsBatchRequest>(body).is_err());
        }
    }

    fn write_ref_audio(path: &Path) {
        std::fs::write(
            path,
            b"not a real wav; resolve_synthesis only checks path existence",
        )
        .unwrap();
    }

    #[test]
    fn resolves_legacy_reference_fields_without_voice() {
        let temp = tempfile::tempdir().unwrap();
        let ref_path = temp.path().join("ref.wav");
        write_ref_audio(&ref_path);

        let resolved = resolve_synthesis(
            None,
            Some("zh"),
            Some(ref_path.to_string_lossy().into_owned()),
            Some("prompt".to_string()),
            SynthesisOverrides {
                top_k: Some(20),
                top_p: Some(0.9),
                temperature: Some(0.7),
                speed: Some(1.1),
                ..Default::default()
            },
            temp.path(),
            temp.path(),
            RequestPathPolicy::default(),
        )
        .unwrap();

        assert_eq!(resolved.language, Language::Chinese);
        assert_eq!(resolved.options.prompt_language, Language::Auto);
        assert_eq!(
            resolved.refer_path,
            ref_path.canonicalize().unwrap().to_string_lossy()
        );
        assert_eq!(resolved.prompt_text, "prompt");
        assert_eq!(resolved.options.top_k, 20);
        assert!((resolved.options.top_p - 0.9).abs() < 0.001);
        assert!((resolved.options.temperature - 0.7).abs() < 0.001);
        assert!((resolved.options.speed - 1.1).abs() < 0.001);
    }

    #[test]
    fn all_request_variants_keep_reference_language_separate() {
        let tts: TtsRequest =
            serde_json::from_str(r#"{"text":"你好","prompt_language":"en"}"#).unwrap();
        let speech: OpenAiSpeechRequest =
            serde_json::from_str(r#"{"input":"你好","voice":"demo","prompt_language":"en"}"#)
                .unwrap();
        let batch: TtsBatchRequest =
            serde_json::from_str(r#"{"texts":["你好"],"prompt_language":"en"}"#).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let reference = temp.path().join("ref.wav");
        write_ref_audio(&reference);
        for overrides in [
            SynthesisOverrides::from_tts(&tts),
            SynthesisOverrides::from_openai(&speech),
            SynthesisOverrides::from_batch(&batch),
        ] {
            let resolved = resolve_synthesis(
                None,
                Some("zh"),
                Some(reference.to_string_lossy().into_owned()),
                Some("hello".into()),
                overrides,
                temp.path(),
                temp.path(),
                RequestPathPolicy::default(),
            )
            .unwrap();
            assert_eq!(resolved.options.language, Language::Chinese);
            assert_eq!(resolved.options.prompt_language, Language::English);
        }
    }

    #[test]
    fn reference_language_request_overrides_profile_without_changing_target() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("demo");
        std::fs::create_dir(&dir).unwrap();
        write_ref_audio(&dir.join("ref.wav"));
        std::fs::write(dir.join("voice.json"),
            r#"{"reference_audio":"ref.wav","reference_text":"hello","language":"zh","prompt_language":"en"}"#,
        ).unwrap();
        for (request, expected) in [
            (None, Language::English),
            (Some(""), Language::English),
            (Some(" auto "), Language::Auto),
            (Some("zh"), Language::Chinese),
        ] {
            let resolved = resolve_synthesis(
                Some("demo"),
                Some("en"),
                None,
                None,
                SynthesisOverrides {
                    prompt_language: request.map(str::to_string),
                    ..Default::default()
                },
                temp.path(),
                temp.path(),
                RequestPathPolicy::default(),
            )
            .unwrap();
            assert_eq!(resolved.options.language, Language::English);
            assert_eq!(resolved.options.prompt_language, expected);
        }
    }

    #[test]
    fn invalid_reference_language_is_rejected_before_file_access() {
        let temp = tempfile::tempdir().unwrap();
        let error = resolve_synthesis(
            None,
            None,
            None,
            None,
            SynthesisOverrides {
                prompt_language: Some("xx".into()),
                ..Default::default()
            },
            temp.path(),
            temp.path(),
            RequestPathPolicy::default(),
        )
        .err()
        .unwrap();
        assert_eq!(error, "unsupported prompt_language: xx");
    }

    #[test]
    fn resolves_voice_profile_defaults() {
        let temp = tempfile::tempdir().unwrap();
        let voice_dir = temp.path().join("mao");
        std::fs::create_dir(&voice_dir).unwrap();
        write_ref_audio(&voice_dir.join("ref.wav"));
        std::fs::write(
            voice_dir.join("voice.json"),
            r#"{
                "reference_audio":"ref.wav",
                "reference_text":"会战兵力是八十万对六十万，优势在我",
                "language":"zh",
                "top_k":11,
                "top_p":0.8,
                "temperature":0.6,
                "speed":1.2,
                "max_tokens":123,
                "repetition_penalty":1.2
            }"#,
        )
        .unwrap();

        let resolved = resolve_synthesis(
            Some("mao"),
            None,
            None,
            None,
            SynthesisOverrides::default(),
            temp.path(),
            temp.path(),
            RequestPathPolicy::default(),
        )
        .unwrap();

        assert_eq!(resolved.voice.as_deref(), Some("mao"));
        assert_eq!(resolved.mode, "auto");
        assert_eq!(
            resolved.refer_path,
            voice_dir.join("ref.wav").to_string_lossy()
        );
        assert_eq!(resolved.prompt_text, "会战兵力是八十万对六十万，优势在我");
        assert_eq!(resolved.options.top_k, 11);
        assert!((resolved.options.top_p - 0.8).abs() < 0.001);
        assert!((resolved.options.temperature - 0.6).abs() < 0.001);
        assert!((resolved.options.speed - 1.2).abs() < 0.001);
        assert_eq!(resolved.options.max_tokens, 123);
        assert!((resolved.options.repetition_penalty - 1.2).abs() < 0.001);
    }

    #[test]
    fn request_values_override_voice_defaults() {
        let temp = tempfile::tempdir().unwrap();
        let voice_dir = temp.path().join("mao");
        std::fs::create_dir(&voice_dir).unwrap();
        let request_ref = temp.path().join("request.wav");
        write_ref_audio(&request_ref);
        std::fs::write(
            voice_dir.join("voice.json"),
            r#"{
                "reference_audio":"voice.wav",
                "reference_text":"voice prompt",
                "language":"zh",
                "top_k":11,
                "top_p":0.8,
                "temperature":0.6,
                "speed":1.2
            }"#,
        )
        .unwrap();

        let resolved = resolve_synthesis(
            Some("mao"),
            Some("en"),
            Some(request_ref.to_string_lossy().into_owned()),
            Some("request prompt".to_string()),
            SynthesisOverrides {
                seed: Some(0),
                prompt_language: Some("zh".into()),
                top_k: Some(33),
                top_p: Some(0.7),
                temperature: Some(0.5),
                speed: Some(0.9),
                max_tokens: Some(321),
                repetition_penalty: Some(1.1),
                split_sentences: Some(false),
                split_method: Some("cut5".to_string()),
                min_sentence_chars: Some(5),
                sentence_gap_ms: Some(300),
                sentence_fade_ms: Some(0),
                sv_embedding: None,
            },
            temp.path(),
            temp.path(),
            RequestPathPolicy::default(),
        )
        .unwrap();

        assert_eq!(resolved.language, Language::English);
        assert_eq!(
            resolved.refer_path,
            request_ref.canonicalize().unwrap().to_string_lossy()
        );
        assert_eq!(resolved.prompt_text, "request prompt");
        assert_eq!(resolved.options.seed, Some(0));
        assert_eq!(resolved.options.prompt_language, Language::Chinese);
        assert_eq!(resolved.options.top_k, 33);
        assert!((resolved.options.top_p - 0.7).abs() < 0.001);
        assert!((resolved.options.temperature - 0.5).abs() < 0.001);
        assert!((resolved.options.speed - 0.9).abs() < 0.001);
        assert_eq!(resolved.options.max_tokens, 321);
        assert!((resolved.options.repetition_penalty - 1.1).abs() < 0.001);
        assert!(!resolved.split_sentences);
        assert_eq!(resolved.split_method, SplitMethod::Cut5);
        assert_eq!(resolved.min_sentence_chars, 5);
        assert_eq!(resolved.sentence_gap_ms, 300);
        assert_eq!(resolved.sentence_fade_ms, 0);
    }

    #[test]
    fn resolves_sv_embedding_from_request() {
        let temp = tempfile::tempdir().unwrap();
        let ref_path = temp.path().join("ref.wav");
        let sv_path = temp.path().join("sv.safetensors");
        write_ref_audio(&ref_path);
        std::fs::write(
            &sv_path,
            b"not a real safetensors; resolver only checks existence",
        )
        .unwrap();

        let resolved = resolve_synthesis(
            None,
            Some("zh"),
            Some(ref_path.to_string_lossy().into_owned()),
            Some("prompt".to_string()),
            SynthesisOverrides {
                sv_embedding: Some(sv_path.to_string_lossy().into_owned()),
                ..Default::default()
            },
            temp.path(),
            temp.path(),
            RequestPathPolicy::default(),
        )
        .unwrap();

        let canonical_sv = sv_path.canonicalize().unwrap();
        assert_eq!(
            resolved.options.sv_embedding.as_deref(),
            Some(canonical_sv.as_path())
        );
    }

    #[test]
    fn resolves_voice_inference_mode() {
        let temp = tempfile::tempdir().unwrap();
        let voice_dir = temp.path().join("fast");
        std::fs::create_dir(&voice_dir).unwrap();
        write_ref_audio(&voice_dir.join("ref.wav"));
        std::fs::write(
            voice_dir.join("voice.json"),
            r#"{
                "reference_audio":"ref.wav",
                "reference_text":"prompt",
                "mode":"cuda-graph"
            }"#,
        )
        .unwrap();

        let resolved = resolve_synthesis(
            Some("fast"),
            None,
            None,
            None,
            SynthesisOverrides::default(),
            temp.path(),
            temp.path(),
            RequestPathPolicy::default(),
        )
        .unwrap();

        assert_eq!(resolved.mode, "cuda-graph");
    }

    #[test]
    fn resolves_voice_model_paths_from_models_dir() {
        let temp = tempfile::tempdir().unwrap();
        let models_dir = temp.path().join("models");
        std::fs::create_dir_all(models_dir.join("custom")).unwrap();
        std::fs::write(models_dir.join("custom/gpt.safetensors"), b"gpt").unwrap();
        std::fs::write(models_dir.join("custom/sovits.safetensors"), b"sovits").unwrap();
        let voice_dir = temp.path().join("voices").join("custom");
        std::fs::create_dir_all(&voice_dir).unwrap();
        write_ref_audio(&voice_dir.join("ref.wav"));
        std::fs::write(
            voice_dir.join("voice.json"),
            r#"{
                "reference_audio":"ref.wav",
                "reference_text":"prompt",
                "gpt_model":"custom/gpt.safetensors",
                "sovits_model":"custom/sovits.safetensors"
            }"#,
        )
        .unwrap();

        let resolved = resolve_synthesis(
            Some("custom"),
            None,
            None,
            None,
            SynthesisOverrides::default(),
            &temp.path().join("voices"),
            &models_dir,
            RequestPathPolicy::default(),
        )
        .unwrap();

        assert_eq!(
            resolved.models.gpt,
            Some(models_dir.join("custom/gpt.safetensors"))
        );
        assert_eq!(
            resolved.models.sovits,
            Some(models_dir.join("custom/sovits.safetensors"))
        );
    }

    #[test]
    fn rejects_missing_voice_model_path() {
        let temp = tempfile::tempdir().unwrap();
        let voice_dir = temp.path().join("voices").join("custom");
        std::fs::create_dir_all(&voice_dir).unwrap();
        write_ref_audio(&voice_dir.join("ref.wav"));
        std::fs::write(
            voice_dir.join("voice.json"),
            r#"{
                "reference_audio":"ref.wav",
                "reference_text":"prompt",
                "gpt_model":"missing/gpt.safetensors"
            }"#,
        )
        .unwrap();

        let error = resolve_synthesis(
            Some("custom"),
            None,
            None,
            None,
            SynthesisOverrides::default(),
            &temp.path().join("voices"),
            &temp.path().join("models"),
            RequestPathPolicy::default(),
        )
        .err()
        .expect("missing voice model should fail");

        assert!(error.contains("GPT model not found"));
    }

    #[test]
    fn rejects_requests_without_reference_data() {
        let error = resolve_synthesis(
            None,
            Some("zh"),
            None,
            None,
            SynthesisOverrides::default(),
            Path::new("voices"),
            Path::new("models"),
            RequestPathPolicy::default(),
        )
        .err()
        .expect("missing reference data should fail");

        assert!(error.contains("reference audio is required"));
    }

    #[test]
    fn rejects_missing_reference_audio_path() {
        let error = resolve_synthesis(
            None,
            Some("zh"),
            Some("missing.wav".to_string()),
            Some("prompt".to_string()),
            SynthesisOverrides::default(),
            Path::new("voices"),
            Path::new("models"),
            RequestPathPolicy::default(),
        )
        .err()
        .expect("missing reference path should fail");

        assert!(error.contains("reference audio not found"));
    }

    #[test]
    fn rejects_invalid_request_split_method() {
        let temp = tempfile::tempdir().unwrap();
        let ref_path = temp.path().join("ref.wav");
        write_ref_audio(&ref_path);

        let error = resolve_synthesis(
            None,
            Some("zh"),
            Some(ref_path.to_string_lossy().into_owned()),
            Some("prompt".to_string()),
            SynthesisOverrides {
                split_method: Some("comma".to_string()),
                ..Default::default()
            },
            temp.path(),
            temp.path(),
            RequestPathPolicy::default(),
        )
        .err()
        .expect("invalid split method should fail");

        assert!(error.contains("unsupported split_method"));
    }

    #[test]
    fn rejects_external_request_paths_by_default() {
        let temp = tempfile::tempdir().unwrap();
        let voices_dir = temp.path().join("voices");
        std::fs::create_dir(&voices_dir).unwrap();
        let outside = temp.path().join("outside.wav");
        write_ref_audio(&outside);

        let error = resolve_synthesis(
            None,
            Some("zh"),
            Some(outside.to_string_lossy().into_owned()),
            Some("prompt".to_string()),
            SynthesisOverrides::default(),
            &voices_dir,
            temp.path(),
            RequestPathPolicy::default(),
        )
        .err()
        .expect("external reference path should fail");

        assert!(error.contains("must be inside"));
    }

    #[test]
    fn allows_external_request_paths_when_enabled() {
        let temp = tempfile::tempdir().unwrap();
        let voices_dir = temp.path().join("voices");
        std::fs::create_dir(&voices_dir).unwrap();
        let outside = temp.path().join("outside.wav");
        write_ref_audio(&outside);

        let resolved = resolve_synthesis(
            None,
            Some("zh"),
            Some(outside.to_string_lossy().into_owned()),
            Some("prompt".to_string()),
            SynthesisOverrides::default(),
            &voices_dir,
            temp.path(),
            RequestPathPolicy {
                allow_external_reference_paths: true,
            },
        )
        .unwrap();

        assert_eq!(
            resolved.refer_path,
            outside.canonicalize().unwrap().to_string_lossy()
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_voice_roots_preserve_canonical_paths_and_reject_escapes() {
        let temp = tempfile::tempdir().unwrap();
        let voices = temp.path().join("voices");
        let alias = temp.path().join("voices-alias");
        std::fs::create_dir(&voices).unwrap();
        std::os::unix::fs::symlink(&voices, &alias).unwrap();
        let reference = voices.join("ref.wav");
        let outside = temp.path().join("outside.wav");
        write_ref_audio(&reference);
        write_ref_audio(&outside);
        std::os::unix::fs::symlink(&outside, voices.join("escape.wav")).unwrap();
        for (name, allowed) in [("ref.wav", true), ("escape.wav", false)] {
            let result = resolve_synthesis(
                None,
                Some("zh"),
                Some(alias.join(name).to_string_lossy().into_owned()),
                Some("prompt".into()),
                SynthesisOverrides::default(),
                &alias,
                temp.path(),
                RequestPathPolicy::default(),
            );
            if allowed {
                assert_eq!(
                    result.unwrap().refer_path,
                    reference.canonicalize().unwrap().to_string_lossy()
                );
            } else {
                assert!(result.err().unwrap().contains("must be inside"));
            }
        }
    }

    #[test]
    fn validate_text_rejects_empty_or_whitespace_input() {
        assert!(validate_text("", "text").is_err());
        assert!(validate_text("   ", "input").is_err());
        assert!(validate_text("hello", "text").is_ok());
    }
}
