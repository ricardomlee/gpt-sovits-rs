//! HTTP request models and request-to-inference resolution.

use gpt_sovits_rs::voice::{InferenceOptionOverrides, LoadedVoiceProfile, VoiceDefaults};
use gpt_sovits_rs::{InferenceOptions, Language, SplitMethod};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize)]
pub(super) struct TtsRequest {
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
    #[allow(dead_code)]
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

#[derive(Default)]
pub(super) struct SynthesisOverrides {
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
}

impl SynthesisOverrides {
    pub(super) fn from_tts(req: &TtsRequest) -> Self {
        Self {
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
        }
    }

    pub(super) fn from_openai(req: &OpenAiSpeechRequest) -> Self {
        Self {
            speed: req.speed,
            max_tokens: req.max_tokens,
            repetition_penalty: req.repetition_penalty,
            split_sentences: req.split_sentences,
            split_method: req.split_method.clone(),
            min_sentence_chars: req.min_sentence_chars,
            sentence_gap_ms: req.sentence_gap_ms,
            sentence_fade_ms: req.sentence_fade_ms,
            ..Default::default()
        }
    }

    pub(super) fn from_batch(req: &TtsBatchRequest) -> Self {
        Self {
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

    let refer_path = refer_wav_path
        .or_else(|| {
            voice
                .as_ref()
                .and_then(|v| v.reference_audio_path())
                .map(|p| p.to_string_lossy().into_owned())
        })
        .ok_or_else(|| {
            "reference audio is required; select a configured voice or pass refer_wav_path"
                .to_string()
        })?;
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
    let split_method = match overrides.split_method.as_deref().map(str::trim) {
        Some("") | None => defaults.split_method,
        Some(method) => SplitMethod::parse(method)
            .ok_or_else(|| format!("unsupported split_method: {method}"))?,
    };

    Ok(ResolvedSynthesis {
        voice: voice.map(|v| v.name),
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
    })
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
        )
        .unwrap();

        assert_eq!(resolved.language, Language::Chinese);
        assert_eq!(resolved.refer_path, ref_path.to_string_lossy());
        assert_eq!(resolved.prompt_text, "prompt");
        assert_eq!(resolved.options.top_k, 20);
        assert!((resolved.options.top_p - 0.9).abs() < 0.001);
        assert!((resolved.options.temperature - 0.7).abs() < 0.001);
        assert!((resolved.options.speed - 1.1).abs() < 0.001);
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
            },
            temp.path(),
        )
        .unwrap();

        assert_eq!(resolved.language, Language::English);
        assert_eq!(resolved.refer_path, request_ref.to_string_lossy());
        assert_eq!(resolved.prompt_text, "request prompt");
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
        )
        .unwrap();

        assert_eq!(resolved.mode, "cuda-graph");
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
        )
        .err()
        .expect("invalid split method should fail");

        assert!(error.contains("unsupported split_method"));
    }

    #[test]
    fn validate_text_rejects_empty_or_whitespace_input() {
        assert!(validate_text("", "text").is_err());
        assert!(validate_text("   ", "input").is_err());
        assert!(validate_text("hello", "text").is_ok());
    }
}
