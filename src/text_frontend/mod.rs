//! Text Frontend Module
//!
//! Handles text normalization, language detection, and grapheme-to-phoneme conversion

pub mod g2p;
pub mod lang_detect;
pub mod normalizer;
pub mod symbols;
pub mod tone_sandhi;

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

    /// Process text and return (phoneme_ids, word2ph) for proper BERT alignment.
    ///
    /// word2ph[i] = number of phonemes for BERT content token i.
    /// Only valid for Chinese text; for other languages returns empty word2ph (falls back to nearest-neighbor).
    pub fn process_with_word2ph(
        &self,
        text: &str,
        language: Language,
    ) -> Result<(Vec<usize>, Vec<usize>)> {
        let normalized = self.normalizer.normalize(text)?;
        let detected_lang = if language == Language::Auto {
            self.language_detector.detect(&normalized)?
        } else {
            language
        };

        match detected_lang {
            Language::Chinese => {
                let (phonemes, word2ph) = self
                    .g2p_converter
                    .convert_chinese_with_word2ph(&normalized)?;
                let ids = self.symbol_table.encode(&phonemes)?;
                Ok((ids, word2ph))
            }
            _ => {
                // For non-Chinese, fall back to standard processing with empty word2ph
                let phonemes = self.g2p_converter.convert(&normalized, detected_lang)?;
                let ids = self.symbol_table.encode(&phonemes)?;
                Ok((ids, Vec::new()))
            }
        }
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
