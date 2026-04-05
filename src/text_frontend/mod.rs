//! Text Frontend Module
//!
//! Handles text normalization, language detection, and grapheme-to-phoneme conversion

pub mod g2p;
pub mod lang_detect;
pub mod normalizer;
pub mod symbols;

pub use g2p::G2PConverter;
pub use lang_detect::LanguageDetector;
pub use normalizer::TextNormalizer;
pub use symbols::SymbolTable;

use crate::{Language, Result};

/// Text frontend processor
#[derive(Debug)]
pub struct TextFrontend {
    normalizer: TextNormalizer,
    language_detector: LanguageDetector,
    g2p_converter: G2PConverter,
    symbol_table: SymbolTable,
}

impl TextFrontend {
    /// Create a new text frontend processor
    pub fn new() -> Result<Self> {
        Ok(Self {
            normalizer: TextNormalizer::new(),
            language_detector: LanguageDetector::new()?,
            g2p_converter: G2PConverter::new()?,
            symbol_table: SymbolTable::new(),
        })
    }

    /// Process text and return phoneme sequence
    pub fn process(&self, text: &str, language: Language) -> Result<Vec<usize>> {
        // Step 1: Normalize text
        let normalized = self.normalizer.normalize(text)?;

        // Step 2: Detect language if auto
        let detected_lang = if language == Language::Auto {
            self.language_detector.detect(&normalized)?
        } else {
            language
        };

        // Step 3: Convert to phonemes
        let phonemes = self.g2p_converter.convert(&normalized, detected_lang)?;

        // Step 4: Convert to symbol IDs
        let ids = self.symbol_table.encode(&phonemes)?;

        Ok(ids)
    }

    /// Get phoneme string from text
    pub fn get_phonemes(&self, text: &str, language: Language) -> Result<String> {
        let normalized = self.normalizer.normalize(text)?;
        let lang = if language == Language::Auto {
            self.language_detector.detect(&normalized)?
        } else {
            language
        };
        self.g2p_converter.convert(&normalized, lang)
    }
}

impl Default for TextFrontend {
    fn default() -> Self {
        Self::new().expect("Failed to initialize TextFrontend")
    }
}
