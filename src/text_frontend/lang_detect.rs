//! Language Detection

use crate::{Language, Result};

/// Language detector using fast_langdetect
#[derive(Debug)]
pub struct LanguageDetector {
    // Placeholder for langdetect model
    // Would use fast_langdetect Rust bindings
}

impl LanguageDetector {
    pub fn new() -> Result<Self> {
        Ok(Self {
            // Initialize detector
        })
    }

    /// Detect language from text
    pub fn detect(&self, text: &str) -> Result<Language> {
        // Simple heuristic based on character ranges
        let mut has_chinese = false;
        let mut has_japanese = false;
        let mut has_korean = false;
        let mut has_cantonese = false;

        for ch in text.chars() {
            // CJK Unified Ideographs
            if ch >= '\u{4e00}' && ch <= '\u{9fff}' {
                has_chinese = true;
            }
            // Hiragana
            if ch >= '\u{3040}' && ch <= '\u{309F}' {
                has_japanese = true;
            }
            // Katakana
            if ch >= '\u{30A0}' && ch <= '\u{30FF}' {
                has_japanese = true;
            }
            // Hangul Syllables (Korean)
            if ch >= '\u{ac00}' && ch <= '\u{d7af}' {
                has_korean = true;
            }
            // CJK Extension A (includes Cantonese characters)
            if ch >= '\u{3400}' && ch <= '\u{4dbf}' {
                has_cantonese = true;
            }
        }

        // Priority: Japanese > Korean > Cantonese > Chinese > English
        if has_japanese {
            return Ok(Language::Japanese);
        }
        if has_korean {
            return Ok(Language::Korean);
        }
        if has_cantonese {
            return Ok(Language::Cantonese);
        }
        if has_chinese {
            return Ok(Language::Chinese);
        }

        // Default to English for Latin script
        Ok(Language::English)
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
        // Note: Our simple detector only checks for hiragana/katakana ranges
        // "こんにちは" contains hiragana, so it should be detected as Japanese
        let lang = detector.detect("こんにちは").unwrap();
        // Simple heuristic: check for hiragana characters
        let text = "こんにちは";
        let is_japanese = text.chars().any(|c| c >= '\u{3040}' && c <= '\u{309F}');
        if is_japanese {
            assert_eq!(lang, Language::Japanese);
        }
    }

    #[test]
    fn test_detect_korean() {
        let detector = LanguageDetector::new().unwrap();
        let lang = detector.detect("안녕하세요");
        assert_eq!(lang.unwrap(), Language::Korean);
    }
}
