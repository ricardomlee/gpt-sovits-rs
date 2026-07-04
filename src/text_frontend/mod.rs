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
use g2p::{parse_pinyin_override, PinyinOverride};

#[derive(Debug, Clone)]
struct RawPinyinOverride {
    clean_index: usize,
    ch: char,
    pinyin: PinyinOverride,
}

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
        let (normalized, detected_lang, pinyin_overrides) =
            self.normalize_for_model(text, language)?;

        let phonemes = if detected_lang == Language::Chinese && !pinyin_overrides.is_empty() {
            self.g2p_converter
                .convert_chinese_with_word2ph_and_overrides(&normalized, &pinyin_overrides)?
                .0
        } else {
            self.g2p_converter.convert(&normalized, detected_lang)?
        };
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
        let (normalized, detected_lang, pinyin_overrides) =
            self.normalize_for_model(text, language)?;
        match detected_lang {
            Language::Chinese => {
                let (phonemes, word2ph) = if pinyin_overrides.is_empty() {
                    self.g2p_converter
                        .convert_chinese_with_word2ph(&normalized)?
                } else {
                    self.g2p_converter
                        .convert_chinese_with_word2ph_and_overrides(
                            &normalized,
                            &pinyin_overrides,
                        )?
                };
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
        let (normalized, lang, pinyin_overrides) = self.normalize_for_model(text, language)?;
        if lang == Language::Chinese && !pinyin_overrides.is_empty() {
            Ok(self
                .g2p_converter
                .convert_chinese_with_word2ph_and_overrides(&normalized, &pinyin_overrides)?
                .0)
        } else {
            self.g2p_converter.convert(&normalized, lang)
        }
    }

    fn normalize_for_model(
        &self,
        text: &str,
        language: Language,
    ) -> Result<(String, Language, Vec<Option<PinyinOverride>>)> {
        let (clean_text, raw_overrides) = parse_pinyin_overrides(text)?;
        let normalized = self.normalizer.normalize(&clean_text)?;
        let lang = if language == Language::Auto {
            self.language_detector.detect(&normalized)?
        } else {
            language
        };
        let (normalized, pinyin_overrides) = if lang == Language::Chinese {
            let normalized = self.normalizer.normalize_chinese_model_text(&normalized);
            let overrides = remap_pinyin_overrides(&clean_text, &normalized, &raw_overrides)?;
            (normalized, overrides)
        } else {
            (normalized, Vec::new())
        };
        Ok((normalized, lang, pinyin_overrides))
    }
}

fn parse_pinyin_overrides(text: &str) -> Result<(String, Vec<RawPinyinOverride>)> {
    let chars: Vec<char> = text.chars().collect();
    let mut clean = String::new();
    let mut overrides = Vec::new();
    let mut clean_index = 0usize;
    let mut i = 0usize;

    while i < chars.len() {
        let ch = chars[i];
        clean.push(ch);
        if i + 1 < chars.len() && chars[i + 1] == '[' {
            if let Some(close_offset) = chars[i + 2..]
                .iter()
                .position(|candidate| *candidate == ']')
            {
                let close = i + 2 + close_offset;
                let marker: String = chars[i + 2..close].iter().collect();
                if is_pinyin_annotation_candidate(&marker) {
                    if !is_cjk(ch) {
                        return Err(crate::Error::TextError(format!(
                            "pinyin annotation [{marker}] must follow a Chinese character"
                        )));
                    }
                    overrides.push(RawPinyinOverride {
                        clean_index,
                        ch,
                        pinyin: parse_pinyin_override(&marker)?,
                    });
                    i = close + 1;
                    clean_index += 1;
                    continue;
                }
            }
        }
        clean_index += 1;
        i += 1;
    }

    Ok((clean, overrides))
}

fn is_pinyin_annotation_candidate(marker: &str) -> bool {
    marker.chars().last().is_some_and(|ch| ch.is_ascii_digit())
}

fn is_cjk(ch: char) -> bool {
    matches!(ch, '\u{4E00}'..='\u{9FFF}')
}

fn remap_pinyin_overrides(
    clean_text: &str,
    normalized_text: &str,
    raw_overrides: &[RawPinyinOverride],
) -> Result<Vec<Option<PinyinOverride>>> {
    if raw_overrides.is_empty() {
        return Ok(Vec::new());
    }

    let clean_chars: Vec<char> = clean_text.chars().collect();
    let normalized_chars: Vec<char> = normalized_text.chars().collect();
    let mut overrides = vec![None; normalized_chars.len()];

    for raw in raw_overrides {
        let normalized_index = if clean_chars.len() == normalized_chars.len()
            && normalized_chars.get(raw.clean_index) == Some(&raw.ch)
        {
            raw.clean_index
        } else {
            find_matching_occurrence(&clean_chars, &normalized_chars, raw)?
        };
        overrides[normalized_index] = Some(raw.pinyin.clone());
    }

    Ok(overrides)
}

fn find_matching_occurrence(
    clean_chars: &[char],
    normalized_chars: &[char],
    raw: &RawPinyinOverride,
) -> Result<usize> {
    let occurrence = clean_chars
        .iter()
        .take(raw.clean_index + 1)
        .filter(|ch| **ch == raw.ch)
        .count();
    normalized_chars
        .iter()
        .enumerate()
        .filter(|(_, ch)| **ch == raw.ch)
        .nth(occurrence.saturating_sub(1))
        .map(|(index, _)| index)
        .ok_or_else(|| {
            crate::Error::TextError(format!(
                "could not map pinyin annotation for '{}' after text normalization",
                raw.ch
            ))
        })
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

    #[test]
    fn pinyin_annotations_override_chinese_g2p_and_are_removed_from_model_text() {
        let frontend = TextFrontend::new().unwrap();

        let phones = frontend
            .get_phonemes("好[hao4]学", Language::Chinese)
            .unwrap();
        let (ids, word2ph, normalized) = frontend
            .process_with_word2ph_and_text("好[hao4]学", Language::Chinese)
            .unwrap();

        assert_eq!(normalized, "好学");
        assert!(phones.starts_with("h ao4"), "{phones}");
        assert_eq!(word2ph.len(), normalized.chars().count());
        assert_eq!(word2ph.iter().sum::<usize>(), ids.len());
    }

    #[test]
    fn pinyin_annotations_support_vowel_v_and_light_tone() {
        let frontend = TextFrontend::new().unwrap();

        let phones = frontend
            .get_phonemes("女[nv3]的[de5]", Language::Chinese)
            .unwrap();

        assert!(phones.contains("n v3"), "{phones}");
        assert!(phones.contains("d e5"), "{phones}");
    }

    #[test]
    fn pinyin_annotations_reject_invalid_tones() {
        let frontend = TextFrontend::new().unwrap();

        let err = frontend
            .get_phonemes("好[hao9]", Language::Chinese)
            .unwrap_err()
            .to_string();

        assert!(err.contains("unsupported tone"));
    }
}
