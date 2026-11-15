//! Inference option defaults and builder.

use crate::{Error, Language, Result};
use std::path::PathBuf;

/// Inference options
#[derive(Debug, Clone)]
pub struct InferenceOptions {
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub speed: f32,
    pub language: Language,
    /// Language of the reference transcript, detected independently by default.
    pub prompt_language: Language,
    pub max_tokens: usize,
    pub repetition_penalty: f32,
    pub sv_embedding: Option<PathBuf>,
    /// Request-local sampling seed. None preserves random synthesis.
    pub seed: Option<u64>,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self {
            top_k: 15,
            top_p: 0.95,
            temperature: 0.8,
            speed: 1.0,
            language: Language::Chinese,
            prompt_language: Language::Auto,
            max_tokens: 500,
            repetition_penalty: 1.35,
            sv_embedding: None,
            seed: None,
        }
    }
}

impl InferenceOptions {
    pub fn builder() -> InferenceOptionsBuilder {
        InferenceOptionsBuilder::default()
    }

    /// Check sampling domains and bound per-chunk generation and duration expansion.
    pub fn validate(&self) -> Result<()> {
        if !(1..=2048).contains(&self.max_tokens) {
            return Err(Error::ConfigError("max_tokens must be in 1..=2048".into()));
        }
        if !(0.25..=4.0).contains(&self.speed) {
            return Err(Error::ConfigError(
                "speed must be finite and in 0.25..=4.0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err(Error::ConfigError(
                "top_p must be finite and in 0..=1".into(),
            ));
        }
        for (name, value) in [
            ("temperature", self.temperature),
            ("repetition_penalty", self.repetition_penalty),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::ConfigError(format!(
                    "{name} must be finite and positive"
                )));
            }
        }
        Ok(())
    }
}

/// Bound silence allocation and fades for buffered and streaming synthesis.
pub fn validate_sentence_timing(gap_ms: u32, fade_ms: u32) -> Result<()> {
    if gap_ms > 10_000 {
        return Err(Error::ConfigError(
            "sentence_gap_ms must be in 0..=10000".into(),
        ));
    }
    if fade_ms > 1000 {
        return Err(Error::ConfigError(
            "sentence_fade_ms must be in 0..=1000".into(),
        ));
    }
    Ok(())
}

#[derive(Default)]
pub struct InferenceOptionsBuilder {
    top_k: Option<usize>,
    top_p: Option<f32>,
    temperature: Option<f32>,
    speed: Option<f32>,
    language: Option<Language>,
    prompt_language: Option<Language>,
    max_tokens: Option<usize>,
    repetition_penalty: Option<f32>,
    sv_embedding: Option<PathBuf>,
    seed: Option<u64>,
}

impl InferenceOptionsBuilder {
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

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
    pub fn prompt_language(mut self, lang: Language) -> Self {
        self.prompt_language = Some(lang);
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
    pub fn sv_embedding<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.sv_embedding = Some(path.into());
        self
    }

    pub fn build(self) -> InferenceOptions {
        InferenceOptions {
            top_k: self.top_k.unwrap_or(15),
            top_p: self.top_p.unwrap_or(0.95),
            temperature: self.temperature.unwrap_or(0.8),
            speed: self.speed.unwrap_or(1.0),
            language: self.language.unwrap_or(Language::Chinese),
            prompt_language: self.prompt_language.unwrap_or(Language::Auto),
            max_tokens: self.max_tokens.unwrap_or(500),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.35),
            sv_embedding: self.sv_embedding,
            seed: self.seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_numeric_boundaries_without_changing_valid_settings() {
        for (speed, top_p, max_tokens) in [(0.25, 0.0, 1), (1.0, 0.95, 500), (4.0, 1.0, 2048)] {
            InferenceOptions {
                speed,
                top_p,
                max_tokens,
                ..Default::default()
            }
            .validate()
            .unwrap();
        }
        for (gap, fade) in [(0, 0), (120, 8), (10_000, 1000)] {
            validate_sentence_timing(gap, fade).unwrap();
        }
        for (gap, fade) in [(10_001, 8), (u32::MAX, 8), (120, 1001)] {
            assert!(validate_sentence_timing(gap, fade).is_err());
        }
    }

    #[test]
    fn rejects_nonfinite_numeric_settings_from_library_and_cli_callers() {
        for name in ["speed", "top_p", "temperature", "repetition_penalty"] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut options = InferenceOptions::default();
                match name {
                    "speed" => options.speed = value,
                    "top_p" => options.top_p = value,
                    "temperature" => options.temperature = value,
                    _ => options.repetition_penalty = value,
                }
                let error = options.validate().unwrap_err();
                assert!(matches!(error, Error::ConfigError(_)));
                assert!(error.to_string().contains(name), "{error}");
            }
        }
    }

    #[test]
    fn builder_overrides_only_selected_fields() {
        let options = InferenceOptions::builder()
            .seed(u64::MAX)
            .top_k(7)
            .temperature(0.6)
            .language(Language::English)
            .prompt_language(Language::Chinese)
            .max_tokens(42)
            .build();

        assert_eq!(options.top_k, 7);
        assert_eq!(options.seed, Some(u64::MAX));
        assert_eq!(InferenceOptions::default().seed, None);
        assert_eq!(InferenceOptions::builder().build().seed, None);
        assert_eq!(options.top_p, 0.95);
        assert!((options.temperature - 0.6).abs() < f32::EPSILON);
        assert!((options.speed - 1.0).abs() < f32::EPSILON);
        assert_eq!(options.language, Language::English);
        assert_eq!(options.prompt_language, Language::Chinese);
        assert_eq!(InferenceOptions::default().prompt_language, Language::Auto);
        assert_eq!(
            InferenceOptions::builder().build().prompt_language,
            Language::Auto
        );
        assert_eq!(options.max_tokens, 42);
        assert!((options.repetition_penalty - 1.35).abs() < f32::EPSILON);
    }
}
