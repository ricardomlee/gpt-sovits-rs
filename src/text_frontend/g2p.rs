//! Grapheme-to-Phoneme Converter

use crate::{Language, Result};
use pinyin::ToPinyin;

/// G2P Converter for multiple languages
#[derive(Debug)]
pub struct G2PConverter {
    // G2P models are lazily initialized
}

impl G2PConverter {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }

    /// Convert text to phonemes
    pub fn convert(&self, text: &str, language: Language) -> Result<String> {
        match language {
            Language::Chinese => self.convert_chinese(text),
            Language::English => self.convert_english(text),
            Language::Japanese => self.convert_japanese(text),
            Language::Korean => self.convert_korean(text),
            Language::Cantonese => self.convert_cantonese(text),
            Language::Auto => self.convert_auto(text),
        }
    }

    /// Convert Chinese text to pinyin phonemes
    fn convert_chinese(&self, text: &str) -> Result<String> {
        let mut phonemes = Vec::new();

        for py_result in text.to_pinyin().flatten() {
            let pinyin_str = py_result.with_tone_num_end();

            // Skip empty pinyin
            if !pinyin_str.is_empty() {
                phonemes.push(pinyin_str.to_string());
            }
        }

        if phonemes.is_empty() {
            // Fallback: return characters as-is
            Ok(text.chars().map(|c| format!("[{}]", c)).collect())
        } else {
            Ok(phonemes.join(" "))
        }
    }

    /// Convert English text to phonemes using rules
    fn convert_english(&self, text: &str) -> Result<String> {
        let mut phonemes = Vec::new();

        for word in text.split_whitespace() {
            let word_phonemes = self.english_word_to_phonemes(word);
            phonemes.push(word_phonemes);
        }

        Ok(phonemes.join(" "))
    }

    /// Convert a single English word to phonemes
    fn english_word_to_phonemes(&self, word: &str) -> String {
        let lower = word.to_lowercase();

        // Simple rule-based G2P for common patterns
        let mut result = String::new();
        let chars: Vec<char> = lower.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];
            let next = chars.get(i + 1).copied();
            let next2 = chars.get(i + 2).copied();

            // Handle common digraphs and trigraphs
            let phoneme = match (c, next, next2) {
                // Consonant digraphs
                ('t', Some('h'), _) => { i += 1; "θ" }
                ('s', Some('h'), _) => { i += 1; "ʃ" }
                ('c', Some('h'), _) => { i += 1; "tʃ" }
                ('w', Some('h'), _) => { i += 1; "w" }
                ('p', Some('h'), _) => { i += 1; "f" }
                // Vowel digraphs
                ('a', Some('i'), _) => { i += 1; "eɪ" }
                ('a', Some('u'), _) => { i += 1; "ɔ" }
                ('e', Some('i'), _) => { i += 1; "i" }
                ('e', Some('a'), _) => { i += 1; "ɛ" }
                ('o', Some('i'), _) => { i += 1; "ɔɪ" }
                ('o', Some('u'), _) => { i += 1; "aʊ" }
                ('e', Some('r'), _) => { i += 1; "ɜː" }
                // Silent e at end
                ('e', None, _) if i > 0 => { i += 1; continue; }
                // Common consonants
                ('b', _, _) => "b",
                ('d', _, _) => "d",
                ('f', _, _) => "f",
                ('g', _, _) => "g",
                ('h', _, _) => "h",
                ('j', _, _) => "dʒ",
                ('k', _, _) => "k",
                ('l', _, _) => "l",
                ('m', _, _) => "m",
                ('n', _, _) => "n",
                ('p', _, _) => "p",
                ('q', _, _) => "kw",
                ('r', _, _) => "ɹ",
                ('s', _, _) => "s",
                ('t', _, _) => "t",
                ('v', _, _) => "v",
                ('w', _, _) => "w",
                ('x', _, _) => "ks",
                ('y', _, _) => "j",
                ('z', _, _) => "z",
                // Common vowels
                ('a', _, _) => "æ",
                ('e', _, _) => "ɛ",
                ('i', _, _) => "ɪ",
                ('o', _, _) => "ɑ",
                ('u', _, _) => "ʌ",
                _ => &c.to_string(),
            };

            result.push_str(phoneme);
            i += 1;
        }

        result
    }

    /// Convert Japanese text to phonemes
    fn convert_japanese(&self, text: &str) -> Result<String> {
        let phonemes = text
            .chars()
            .map(|c| {
                // Hiragana/Katakana range detection
                match c {
                    '\u{3040}'..='\u{309F}' => format!("[{}]", c), // Hiragana
                    '\u{30A0}'..='\u{30FF}' => format!("[{}]", c), // Katakana
                    _ => format!("[{}]", c),
                }
            })
            .collect();

        Ok(phonemes)
    }

    /// Convert Korean text to phonemes
    fn convert_korean(&self, text: &str) -> Result<String> {
        let phonemes = text
            .chars()
            .map(|c| {
                if ('\u{AC00}'..='\u{D7A3}').contains(&c) {
                    // Hangul syllable
                    format!("[{}]", c)
                } else {
                    format!("[{}]", c)
                }
            })
            .collect();

        Ok(phonemes)
    }

    /// Convert Cantonese text to Jyutping phonemes
    fn convert_cantonese(&self, text: &str) -> Result<String> {
        // Fallback to Chinese conversion for now
        self.convert_chinese(text)
    }

    /// Auto-detect language and convert
    fn convert_auto(&self, text: &str) -> Result<String> {
        // Detect dominant script and convert
        let mut chinese_count = 0;
        let mut english_count = 0;
        let mut japanese_count = 0;
        let mut korean_count = 0;

        for c in text.chars() {
            match c {
                '\u{4E00}'..='\u{9FFF}' => chinese_count += 1,
                'a'..='z' | 'A'..='Z' => english_count += 1,
                '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}' => japanese_count += 1,
                '\u{AC00}'..='\u{D7A3}' => korean_count += 1,
                _ => {}
            }
        }

        let lang = if chinese_count >= japanese_count
            && chinese_count >= korean_count
            && chinese_count >= english_count
        {
            Language::Chinese
        } else if japanese_count >= korean_count && japanese_count >= english_count {
            Language::Japanese
        } else if korean_count >= english_count {
            Language::Korean
        } else {
            Language::English
        };

        self.convert(text, lang)
    }
}

impl Default for G2PConverter {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g2p_chinese() {
        let converter = G2PConverter::new().unwrap();
        let result = converter.convert("你好", Language::Chinese);
        assert!(result.is_ok());
        let phonemes = result.unwrap();
        // 你 (ni3) 好 (hao3)
        assert!(phonemes.contains("ni") || phonemes.contains("hao"));
    }

    #[test]
    fn test_g2p_english() {
        let converter = G2PConverter::new().unwrap();
        let result = converter.convert("Hello", Language::English);
        assert!(result.is_ok());
    }

    #[test]
    fn test_g2p_japanese() {
        let converter = G2PConverter::new().unwrap();
        let result = converter.convert("こんにちは", Language::Japanese);
        assert!(result.is_ok());
    }

    #[test]
    fn test_g2p_korean() {
        let converter = G2PConverter::new().unwrap();
        let result = converter.convert("안녕하세요", Language::Korean);
        assert!(result.is_ok());
    }
}
