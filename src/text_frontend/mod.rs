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
        let lang = if language == Language::Auto {
            // Numeric literals and punctuation do not identify a language. Detect
            // the source script before expansion; number-only input defaults to Chinese.
            let script: String = clean_text.chars().filter(|ch| ch.is_alphabetic()).collect();
            self.language_detector.detect(&script)?
        } else {
            language
        };
        let mut normalized = self
            .normalizer
            .normalize_mapped(&clean_text, lang == Language::Chinese);
        let (normalized, pinyin_overrides) = if lang == Language::Chinese {
            normalized.normalize_chinese_model_text();
            let (normalized, source_indices) = normalized.into_parts();
            let overrides = remap_pinyin_overrides(&source_indices, &raw_overrides)?;
            (normalized, overrides)
        } else {
            (normalized.into_parts().0, Vec::new())
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
    source_indices: &[Option<usize>],
    raw_overrides: &[RawPinyinOverride],
) -> Result<Vec<Option<PinyinOverride>>> {
    if raw_overrides.is_empty() {
        return Ok(Vec::new());
    }

    let mut overrides = vec![None; source_indices.len()];

    for raw in raw_overrides {
        let normalized_index = source_indices
            .iter()
            .position(|source| *source == Some(raw.clean_index))
            .ok_or_else(|| {
                crate::Error::TextError(format!(
                    "could not map pinyin annotation for '{}' after text normalization",
                    raw.ch
                ))
            })?;
        overrides[normalized_index] = Some(raw.pinyin.clone());
    }

    Ok(overrides)
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

    #[test]
    fn numeric_expansion_keeps_annotations_on_original_characters() {
        let frontend = TextFrontend::new().unwrap();
        for (input, expected, annotated_index, tone) in [
            ("1一[yi1]", "一一", 1, 1),
            ("50%分[fen4]成", "百分之五十分成", 5, 4),
            ("0.5点[dian3]", "零点五点", 3, 3),
            ("100百[bai2]", "一百百", 2, 2),
        ] {
            let (normalized, _, overrides) = frontend
                .normalize_for_model(input, Language::Chinese)
                .unwrap();
            assert_eq!(normalized, expected, "{input}");
            assert_eq!(overrides.iter().flatten().count(), 1, "{input}");
            assert_eq!(overrides[annotated_index].as_ref().unwrap().tone, tone);
        }
    }

    #[test]
    fn numeric_expansion_preserves_bert_phone_alignment() {
        let frontend = TextFrontend::new().unwrap();
        let (ids, word2ph, normalized) = frontend
            .process_with_word2ph_and_text("50%分[fen4]成，数值3.14。", Language::Chinese)
            .unwrap();
        assert_eq!(normalized, "百分之五十分成,数值三点一四.");
        assert_eq!(word2ph.len(), normalized.chars().count());
        assert_eq!(word2ph.iter().sum::<usize>(), ids.len());
        assert!(word2ph.iter().all(|count| *count > 0));
    }

    #[test]
    fn annotation_positions_survive_whitespace_abbreviations_and_replacements() {
        let frontend = TextFrontend::new().unwrap();
        let (normalized, _, overrides) = frontend
            .normalize_for_model(
                "  嗯[en2]... Mr. 好[hao4]  50%分[fen4] 等[deng3]. 等[deng4]。",
                Language::Chinese,
            )
            .unwrap();
        assert_eq!(normalized, "恩… Mister  好 百分之五十分 等等 等.");
        let annotated: Vec<_> = normalized
            .chars()
            .zip(overrides)
            .filter_map(|(ch, pinyin)| pinyin.map(|p| (ch, p.base, p.tone)))
            .collect();
        assert_eq!(
            annotated,
            vec![
                ('恩', "en".into(), 2),
                ('好', "hao".into(), 4),
                ('分', "fen".into(), 4),
                ('等', "deng".into(), 3),
                ('等', "deng".into(), 4),
            ]
        );
    }

    #[test]
    fn numeric_annotations_reach_the_intended_phoneme_span() {
        let frontend = TextFrontend::new().unwrap();
        let (ids, word2ph, _) = frontend
            .process_with_word2ph_and_text("50%分[fen4]成", Language::Chinese)
            .unwrap();
        let start = word2ph[..5].iter().sum::<usize>();
        let expected = frontend.process("分[fen4]", Language::Chinese).unwrap();
        assert_eq!(&ids[start..start + word2ph[5]], &expected);
        let end_of_first_fen = word2ph[..2].iter().sum::<usize>();
        let expected = frontend.process("分[fen1]", Language::Chinese).unwrap();
        assert_eq!(&ids[word2ph[0]..end_of_first_fen], &expected);
    }

    #[test]
    fn language_is_resolved_before_chinese_number_expansion() {
        let frontend = TextFrontend::new().unwrap();
        for (input, requested, expected_lang, expected) in [
            ("50%", Language::English, Language::English, "50%"),
            (
                "value 3.14",
                Language::Auto,
                Language::English,
                "value 3.14",
            ),
            (
                "音量50%",
                Language::Auto,
                Language::Chinese,
                "音量百分之五十",
            ),
            ("50%", Language::Auto, Language::Chinese, "百分之五十"),
        ] {
            let (normalized, lang, _) = frontend.normalize_for_model(input, requested).unwrap();
            assert_eq!(lang, expected_lang, "{input}");
            assert_eq!(normalized, expected, "{input}");
        }
    }
}
