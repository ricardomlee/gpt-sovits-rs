//! GPT-SoVITS Rust Implementation
//!
//! A high-performance text-to-speech inference engine implemented in Rust.
//!
//! # Example
//!
//! ```rust,no_run
//! use gpt_sovits_rs::{Pipeline, Config, InferenceOptions};
//!
//! let config = Config::default();
//! let mut pipeline = Pipeline::new(config).unwrap();
//!
//! pipeline.load_gpt("models/gpt-model.safetensors").unwrap();
//! pipeline.load_sovits("models/sovits-model.safetensors").unwrap();
//!
//! let options = InferenceOptions::default();
//! let audio = pipeline.inference(
//!     "你好，这是测试文本",
//!     "ref.wav",
//!     "参考文本",
//!     &options
//! ).unwrap();
//!
//! audio.save("output.wav").unwrap();
//! ```

pub mod audio_checks;
pub mod config;
pub mod inference;
#[cfg(feature = "mkl")]
mod mkl_compat;
pub mod model_paths;
pub mod models;
pub mod text_frontend;
pub mod utils;
pub mod voice;

// Re-export main types
pub use config::Config;
pub use inference::{split_sentences, split_sentences_for_language, InferenceOptions, Pipeline};
pub use utils::AudioBuffer;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Supported languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Language {
    #[default]
    Chinese,
    English,
    Japanese,
    Korean,
    Cantonese,
    Auto,
}

impl Language {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "zh" | "zh-cn" | "chinese" | "中文" => Some(Language::Chinese),
            "en" | "english" | "英文" => Some(Language::English),
            "ja" | "japanese" | "日文" => Some(Language::Japanese),
            "ko" | "korean" | "韩文" => Some(Language::Korean),
            "yue" | "cantonese" | "粤语" => Some(Language::Cantonese),
            "auto" | "多语种混合" => Some(Language::Auto),
            _ => None,
        }
    }
}

/// Error types for the library
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Inference failed: {0}")]
    InferenceError(String),

    #[error("Text processing failed: {0}")]
    TextError(String),

    #[error("Audio I/O failed: {0}")]
    AudioError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
