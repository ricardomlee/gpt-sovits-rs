//! Inference option defaults and builder.

use crate::Language;
use std::path::PathBuf;

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
    pub sv_embedding: Option<PathBuf>,
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
            sv_embedding: None,
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
    sv_embedding: Option<PathBuf>,
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
            max_tokens: self.max_tokens.unwrap_or(500),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.35),
            sv_embedding: self.sv_embedding,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_overrides_only_selected_fields() {
        let options = InferenceOptions::builder()
            .top_k(7)
            .temperature(0.6)
            .language(Language::English)
            .max_tokens(42)
            .build();

        assert_eq!(options.top_k, 7);
        assert_eq!(options.top_p, 0.95);
        assert!((options.temperature - 0.6).abs() < f32::EPSILON);
        assert!((options.speed - 1.0).abs() < f32::EPSILON);
        assert_eq!(options.language, Language::English);
        assert_eq!(options.max_tokens, 42);
        assert!((options.repetition_penalty - 1.35).abs() < f32::EPSILON);
    }
}
