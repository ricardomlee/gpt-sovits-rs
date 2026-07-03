//! Voice profile loading and defaults.

use crate::{InferenceOptions, Language, SplitMethod};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize, Default)]
pub struct VoiceProfile {
    pub reference_audio: Option<String>,
    pub reference_text: Option<String>,
    pub language: Option<String>,
    pub mode: Option<String>,
    pub split_sentences: Option<bool>,
    pub split_method: Option<String>,
    pub min_sentence_chars: Option<usize>,
    pub sentence_gap_ms: Option<u32>,
    pub sentence_fade_ms: Option<u32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub temperature: Option<f32>,
    pub speed: Option<f32>,
    pub max_tokens: Option<usize>,
    pub repetition_penalty: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct LoadedVoiceProfile {
    pub name: String,
    pub dir: PathBuf,
    pub profile: VoiceProfile,
}

impl LoadedVoiceProfile {
    pub fn load(name: &str, voices_dir: &Path) -> Result<Self, String> {
        let dir = voices_dir.join(name);
        let path = dir.join("voice.json");
        let data = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read voice profile {:?}: {}", path, e))?;
        let profile: VoiceProfile = serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse voice profile {:?}: {}", path, e))?;
        if let Some(mode) = profile.mode.as_deref() {
            validate_mode(mode)?;
        }
        if let Some(language) = profile.language.as_deref() {
            if Language::parse(language).is_none() {
                return Err(format!(
                    "Invalid language '{}' in voice profile {:?}; expected zh, en, ja, ko, yue, or auto",
                    language, path
                ));
            }
        }
        if let Some(method) = profile.split_method.as_deref() {
            if SplitMethod::parse(method).is_none() {
                return Err(format!(
                    "Invalid split_method '{}' in voice profile {:?}; expected sentence or cut5",
                    method, path
                ));
            }
        }
        Ok(Self {
            name: name.to_string(),
            dir,
            profile,
        })
    }

    pub fn resolve_path(&self, path: &str) -> PathBuf {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            path
        } else {
            self.dir.join(path)
        }
    }

    pub fn reference_audio_path(&self) -> Option<PathBuf> {
        self.profile
            .reference_audio
            .as_deref()
            .map(|path| self.resolve_path(path))
    }

    pub fn reference_text(&self) -> Option<&str> {
        self.profile.reference_text.as_deref()
    }
}

pub fn load_optional_voice_profile(
    voice_name: Option<&str>,
    voices_dir: &Path,
) -> Result<Option<LoadedVoiceProfile>, String> {
    match voice_name {
        Some(name) => LoadedVoiceProfile::load(name, voices_dir).map(Some),
        None => Ok(None),
    }
}

pub fn list_voice_profiles(voices_dir: &Path) -> Result<Vec<String>, String> {
    if !voices_dir.exists() {
        return Ok(Vec::new());
    }
    let entries = std::fs::read_dir(voices_dir)
        .map_err(|e| format!("Failed to read voices directory {:?}: {}", voices_dir, e))?;
    let mut voices = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read voices directory entry: {}", e))?;
        let path = entry.path();
        if path.is_dir() && path.join("voice.json").is_file() {
            if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                voices.push(name.to_string());
            }
        }
    }
    voices.sort();
    Ok(voices)
}

pub fn validate_mode(mode: &str) -> Result<(), String> {
    match mode {
        "auto" | "plain" | "kv" | "cuda-graph" => Ok(()),
        _ => Err(format!(
            "Invalid voice profile mode '{}'; expected auto, plain, kv, or cuda-graph",
            mode
        )),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoiceDefaults {
    pub language: String,
    pub mode: String,
    pub split_sentences: bool,
    pub split_method: SplitMethod,
    pub min_sentence_chars: usize,
    pub sentence_gap_ms: u32,
    pub sentence_fade_ms: u32,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub speed: f32,
    pub max_tokens: usize,
    pub repetition_penalty: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct InferenceOptionOverrides {
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub temperature: Option<f32>,
    pub speed: Option<f32>,
    pub max_tokens: Option<usize>,
    pub repetition_penalty: Option<f32>,
}

impl VoiceDefaults {
    pub fn from_profile(profile: Option<&VoiceProfile>) -> Self {
        Self {
            language: profile
                .and_then(|p| p.language.clone())
                .unwrap_or_else(|| "zh".to_string()),
            mode: profile
                .and_then(|p| p.mode.clone())
                .unwrap_or_else(|| "auto".to_string()),
            split_sentences: profile.and_then(|p| p.split_sentences).unwrap_or(true),
            split_method: profile
                .and_then(|p| p.split_method.as_deref())
                .and_then(SplitMethod::parse)
                .unwrap_or_default(),
            min_sentence_chars: profile.and_then(|p| p.min_sentence_chars).unwrap_or(12),
            sentence_gap_ms: profile.and_then(|p| p.sentence_gap_ms).unwrap_or(120),
            sentence_fade_ms: profile.and_then(|p| p.sentence_fade_ms).unwrap_or(8),
            top_k: profile.and_then(|p| p.top_k).unwrap_or(15),
            top_p: profile.and_then(|p| p.top_p).unwrap_or(0.95),
            temperature: profile.and_then(|p| p.temperature).unwrap_or(0.8),
            speed: profile.and_then(|p| p.speed).unwrap_or(1.0),
            max_tokens: profile.and_then(|p| p.max_tokens).unwrap_or(500),
            repetition_penalty: profile.and_then(|p| p.repetition_penalty).unwrap_or(1.35),
        }
    }

    pub fn to_inference_options(
        &self,
        language: Language,
        overrides: InferenceOptionOverrides,
    ) -> InferenceOptions {
        InferenceOptions::builder()
            .top_k(overrides.top_k.unwrap_or(self.top_k))
            .top_p(overrides.top_p.unwrap_or(self.top_p))
            .temperature(overrides.temperature.unwrap_or(self.temperature))
            .speed(overrides.speed.unwrap_or(self.speed))
            .language(language)
            .max_tokens(overrides.max_tokens.unwrap_or(self.max_tokens))
            .repetition_penalty(
                overrides
                    .repetition_penalty
                    .unwrap_or(self.repetition_penalty),
            )
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_fill_missing_values() {
        let defaults = VoiceDefaults::from_profile(None);
        assert_eq!(defaults.language, "zh");
        assert_eq!(defaults.mode, "auto");
        assert!(defaults.split_sentences);
        assert_eq!(defaults.split_method, SplitMethod::Sentence);
        assert_eq!(defaults.min_sentence_chars, 12);
        assert_eq!(defaults.sentence_gap_ms, 120);
        assert_eq!(defaults.sentence_fade_ms, 8);
        assert_eq!(defaults.top_k, 15);
        assert_eq!(defaults.top_p, 0.95);
        assert_eq!(defaults.temperature, 0.8);
        assert_eq!(defaults.max_tokens, 500);
    }

    #[test]
    fn profile_overrides_defaults() {
        let profile = VoiceProfile {
            language: Some("en".to_string()),
            mode: Some("kv".to_string()),
            split_sentences: Some(true),
            top_p: Some(0.8),
            max_tokens: Some(128),
            ..Default::default()
        };
        let defaults = VoiceDefaults::from_profile(Some(&profile));
        assert_eq!(defaults.language, "en");
        assert_eq!(defaults.mode, "kv");
        assert!(defaults.split_sentences);
        assert_eq!(defaults.top_p, 0.8);
        assert_eq!(defaults.max_tokens, 128);
    }

    #[test]
    fn builds_inference_options_with_overrides() {
        let profile = VoiceProfile {
            top_k: Some(11),
            top_p: Some(0.8),
            temperature: Some(0.6),
            speed: Some(1.1),
            max_tokens: Some(123),
            repetition_penalty: Some(1.2),
            ..Default::default()
        };
        let defaults = VoiceDefaults::from_profile(Some(&profile));
        let options = defaults.to_inference_options(
            Language::Chinese,
            InferenceOptionOverrides {
                top_k: Some(22),
                temperature: Some(0.5),
                ..Default::default()
            },
        );

        assert_eq!(options.top_k, 22);
        assert!((options.top_p - 0.8).abs() < 0.001);
        assert!((options.temperature - 0.5).abs() < 0.001);
        assert!((options.speed - 1.1).abs() < 0.001);
        assert_eq!(options.max_tokens, 123);
        assert!((options.repetition_penalty - 1.2).abs() < 0.001);
    }

    #[test]
    fn rejects_invalid_mode() {
        assert!(validate_mode("cuda-graph").is_ok());
        assert!(validate_mode("fast").is_err());
    }

    #[test]
    fn rejects_invalid_profile_language() {
        let temp = tempfile::tempdir().unwrap();
        let voice_dir = temp.path().join("test");
        std::fs::create_dir(&voice_dir).unwrap();
        std::fs::write(voice_dir.join("voice.json"), r#"{"language":"xx"}"#).unwrap();

        let error = LoadedVoiceProfile::load("test", temp.path()).unwrap_err();
        assert!(error.contains("Invalid language 'xx'"));
    }

    #[test]
    fn rejects_invalid_split_method() {
        let temp = tempfile::tempdir().unwrap();
        let voice_dir = temp.path().join("test");
        std::fs::create_dir(&voice_dir).unwrap();
        std::fs::write(
            voice_dir.join("voice.json"),
            r#"{"split_method":"comma"}"#,
        )
        .unwrap();

        let error = LoadedVoiceProfile::load("test", temp.path()).unwrap_err();
        assert!(error.contains("expected sentence or cut5"));
    }

    #[test]
    fn resolves_relative_reference_audio_from_voice_dir() {
        let loaded = LoadedVoiceProfile {
            name: "test".to_string(),
            dir: PathBuf::from("/tmp/voices/test"),
            profile: VoiceProfile {
                reference_audio: Some("ref.wav".to_string()),
                ..Default::default()
            },
        };
        assert_eq!(
            loaded.reference_audio_path().unwrap(),
            PathBuf::from("/tmp/voices/test/ref.wav")
        );
    }

    #[test]
    fn loads_profile_from_disk() {
        let temp = tempfile::tempdir().unwrap();
        let voice_dir = temp.path().join("mao");
        std::fs::create_dir(&voice_dir).unwrap();
        std::fs::write(
            voice_dir.join("voice.json"),
            r#"{"reference_audio":"ref.wav","reference_text":"hello","mode":"kv"}"#,
        )
        .unwrap();

        let loaded = LoadedVoiceProfile::load("mao", temp.path()).unwrap();
        assert_eq!(loaded.name, "mao");
        assert_eq!(loaded.reference_text(), Some("hello"));
        assert_eq!(
            loaded.reference_audio_path().unwrap(),
            voice_dir.join("ref.wav")
        );
        assert_eq!(loaded.profile.mode.as_deref(), Some("kv"));
    }

    #[test]
    fn lists_only_directories_with_voice_json() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("b_voice")).unwrap();
        std::fs::write(temp.path().join("b_voice").join("voice.json"), "{}").unwrap();
        std::fs::create_dir(temp.path().join("a_voice")).unwrap();
        std::fs::write(temp.path().join("a_voice").join("voice.json"), "{}").unwrap();
        std::fs::create_dir(temp.path().join("missing_config")).unwrap();
        std::fs::write(temp.path().join("file.txt"), "").unwrap();

        let voices = list_voice_profiles(temp.path()).unwrap();
        assert_eq!(voices, vec!["a_voice", "b_voice"]);
    }

    #[test]
    fn missing_voice_dir_lists_empty() {
        let temp = tempfile::tempdir().unwrap();
        let voices = list_voice_profiles(&temp.path().join("missing")).unwrap();
        assert!(voices.is_empty());
    }
}
