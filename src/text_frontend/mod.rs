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
        let (normalized, detected_lang) = self.normalize_for_model(text, language)?;

        let phonemes = self.g2p_converter.convert(&normalized, detected_lang)?;
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
        let (ids, word2ph, _) = self.process_with_word2ph_and_text(text, language)?;
        Ok((ids, word2ph))
    }

    /// Process text for inference and return the exact normalized text used by G2P.
    ///
    /// BERT must consume this same text so its content-token positions match word2ph.
    pub fn process_with_word2ph_and_text(
        &self,
        text: &str,
        language: Language,
    ) -> Result<(Vec<usize>, Vec<usize>, String)> {
        let (normalized, detected_lang) = self.normalize_for_model(text, language)?;
        match detected_lang {
            Language::Chinese => {
                let (phonemes, word2ph) = self
                    .g2p_converter
                    .convert_chinese_with_word2ph(&normalized)?;
                let ids = self.symbol_table.encode(&phonemes)?;
                Ok((ids, word2ph, normalized))
            }
            _ => {
                // For non-Chinese, fall back to standard processing with empty word2ph
                let phonemes = self.g2p_converter.convert(&normalized, detected_lang)?;
                let ids = self.symbol_table.encode(&phonemes)?;
                Ok((ids, Vec::new(), normalized))
            }
        }
    }

    /// Get phoneme string from text
    pub fn get_phonemes(&self, text: &str, language: Language) -> Result<String> {
        let (normalized, lang) = self.normalize_for_model(text, language)?;
        self.g2p_converter.convert(&normalized, lang)
    }

    fn normalize_for_model(&self, text: &str, language: Language) -> Result<(String, Language)> {
        let normalized = self.normalizer.normalize(text)?;
        let lang = if language == Language::Auto {
            self.language_detector.detect(&normalized)?
        } else {
            language
        };
        let normalized = if lang == Language::Chinese {
            self.normalizer.normalize_chinese_model_text(&normalized)
        } else {
            normalized
        };
        Ok((normalized, lang))
    }
}

impl Default for TextFrontend {
    fn default() -> Self {
        Self::new().expect("Failed to initialize TextFrontend")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chinese_alignment_uses_normalized_punctuation() {
        let frontend = TextFrontend::new().unwrap();
        let text = "中道崩殂，今天下三分。";
        let (ids, word2ph, normalized) = frontend
            .process_with_word2ph_and_text(text, Language::Chinese)
            .unwrap();

        assert_eq!(normalized, "中道崩殂,今天下三分.");
        assert_eq!(word2ph.len(), normalized.chars().count());
        assert_eq!(word2ph[4], 1);
        assert_eq!(word2ph.last(), Some(&1));
        assert_eq!(word2ph.iter().sum::<usize>(), ids.len());
    }
}
