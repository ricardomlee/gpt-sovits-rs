//! Language Detection using Unicode character ranges

use crate::{Language, Result};

/// Language detector using Unicode character range analysis
#[derive(Debug)]
pub struct LanguageDetector {
    // Language detection thresholds
    chinese_threshold: usize,
    confidence_threshold: f32,
}

impl LanguageDetector {
    pub fn new() -> Result<Self> {
        Ok(Self {
            chinese_threshold: 1,
            confidence_threshold: 0.5,
        })
    }

    /// Detect language from text using Unicode character ranges
    pub fn detect(&self, text: &str) -> Result<Language> {
        let mut chinese_count = 0;
        let mut japanese_hiragana = 0;
        let mut japanese_katakana = 0;
        let mut korean_count = 0;
        let mut latin_count = 0;
        let mut other_count = 0;

        for ch in text.chars() {
            match ch {
                // CJK Unified Ideographs (Chinese, Japanese Kanji)
                '\u{4e00}'..='\u{9fff}' => chinese_count += 1,
                // Hiragana (Japanese only)
                '\u{3040}'..='\u{309f}' => japanese_hiragana += 1,
                // Katakana (Japanese only)
                '\u{30a0}'..='\u{30ff}' => japanese_katakana += 1,
                // Hangul Syllables (Korean only)
                '\u{ac00}'..='\u{d7af}' => korean_count += 1,
                // Hangul Jamo (Korean only)
                '\u{1100}'..='\u{11ff}' => korean_count += 1,
                // Basic Latin and Latin-1 Supplement
                '\u{0000}'..='\u{00ff}' => latin_count += 1,
                // Other characters
                _ => other_count += 1,
            }
        }

        let total = chinese_count + japanese_hiragana + japanese_katakana + korean_count + latin_count + other_count;
        if total == 0 {
            return Ok(Language::Chinese); // Default
        }

        // Detection logic with priority and confidence
        let japanese_total = japanese_hiragana + japanese_katakana;

        // Japanese has highest priority if hiragana/katakana present
        if japanese_total > 0 {
            let ratio = japanese_total as f32 / total as f32;
            if ratio >= self.confidence_threshold {
                return Ok(Language::Japanese);
            }
        }

        // Korean detection (Hangul is unique)
        if korean_count > 0 {
            let ratio = korean_count as f32 / total as f32;
            if ratio >= self.confidence_threshold {
                return Ok(Language::Korean);
            }
        }

        // Chinese detection (CJK characters)
        if chinese_count >= self.chinese_threshold {
            let ratio = chinese_count as f32 / total as f32;
            if ratio >= self.confidence_threshold {
                return Ok(Language::Chinese);
            }
        }

        // Latin script defaults to English
        if latin_count > 0 {
            let ratio = latin_count as f32 / total as f32;
            if ratio >= self.confidence_threshold {
                return Ok(Language::English);
            }
        }

        // Fallback based on dominant script
        let counts = [
            (chinese_count, Language::Chinese),
            (japanese_total, Language::Japanese),
            (korean_count, Language::Korean),
            (latin_count, Language::English),
        ];

        counts.iter()
            .max_by_key(|(count, _)| *count)
            .map(|(_, lang)| *lang)
            .or(Some(Language::Chinese))
            .ok_or_else(|| crate::Error::TextError("Failed to detect language".to_string()))
    }

    /// Detect language with confidence score
    pub fn detect_with_confidence(&self, text: &str) -> Result<(Language, f32)> {
        let mut chinese_count = 0;
        let mut japanese_hiragana = 0;
        let mut japanese_katakana = 0;
        let mut korean_count = 0;
        let mut latin_count = 0;
        let mut other_count = 0;

        for ch in text.chars() {
            match ch {
                '\u{4e00}'..='\u{9fff}' => chinese_count += 1,
                '\u{3040}'..='\u{309f}' => japanese_hiragana += 1,
                '\u{30a0}'..='\u{30ff}' => japanese_katakana += 1,
                '\u{ac00}'..='\u{d7af}' => korean_count += 1,
                '\u{1100}'..='\u{11ff}' => korean_count += 1,
                '\u{0000}'..='\u{00ff}' => latin_count += 1,
                _ => other_count += 1,
            }
        }

        let total = chinese_count + japanese_hiragana + japanese_katakana + korean_count + latin_count + other_count;
        if total == 0 {
            return Ok((Language::Chinese, 1.0));
        }

        let japanese_total = japanese_hiragana + japanese_katakana;
        let max_count = *[chinese_count, japanese_total, korean_count, latin_count].iter().max().unwrap_or(&0);
        let confidence = max_count as f32 / total as f32;

        let lang = if japanese_total > 0 && japanese_total == max_count {
            Language::Japanese
        } else if korean_count > 0 && korean_count == max_count {
            Language::Korean
        } else if chinese_count > 0 && chinese_count == max_count {
            Language::Chinese
        } else {
            Language::English
        };

        Ok((lang, confidence))
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new().expect("Failed to create LanguageDetector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_chinese() {
        let detector = LanguageDetector::new().unwrap();
        let lang = detector.detect("你好世界");
        assert_eq!(lang.unwrap(), Language::Chinese);
    }

    #[test]
    fn test_detect_english() {
        let detector = LanguageDetector::new().unwrap();
        let lang = detector.detect("Hello World");
        assert_eq!(lang.unwrap(), Language::English);
    }

    #[test]
    fn test_detect_japanese() {
        let detector = LanguageDetector::new().unwrap();
        let lang = detector.detect("こんにちは").unwrap();
        assert_eq!(lang, Language::Japanese);
    }

    #[test]
    fn test_detect_korean() {
        let detector = LanguageDetector::new().unwrap();
        let lang = detector.detect("안녕하세요");
        assert_eq!(lang.unwrap(), Language::Korean);
    }

    #[test]
    fn test_detect_mixed_zh_en() {
        let detector = LanguageDetector::new().unwrap();
        let lang = detector.detect("Hello 你好 World");
        // Mixed text: 2 Chinese chars vs 12 Latin chars
        // Latin count is higher, so it may be detected as English
        // The detector uses max count heuristic
        let result = lang.unwrap();
        // Accept either Chinese or English for mixed input
        assert!(result == Language::Chinese || result == Language::English);
    }

    #[test]
    fn test_detect_with_confidence() {
        let detector = LanguageDetector::new().unwrap();
        let (lang, confidence) = detector.detect_with_confidence("你好世界").unwrap();
        assert_eq!(lang, Language::Chinese);
        assert!(confidence > 0.5);
    }
}
