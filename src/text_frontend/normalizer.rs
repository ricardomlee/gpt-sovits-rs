//! Text Normalizer

use crate::Result;

/// Text normalizer for TTS input
#[derive(Debug)]
pub struct TextNormalizer {
    // Normalization rules
}

impl TextNormalizer {
    pub fn new() -> Self {
        Self {}
    }

    /// Normalize text for TTS
    pub fn normalize(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();

        // Normalize whitespace
        result = self.normalize_whitespace(&result);

        // Normalize punctuation
        result = self.normalize_punctuation(&result);

        // Expand numbers
        result = self.expand_numbers(&result);

        // Expand abbreviations
        result = self.expand_abbreviations(&result);

        Ok(result)
    }

    /// Normalize whitespace
    fn normalize_whitespace(&self, text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Normalize punctuation (convert to Chinese equivalents where appropriate)
    fn normalize_punctuation(&self, text: &str) -> String {
        text.replace('"', "\"")
            .replace('"', "\"")
            .replace('(', "（")
            .replace(')', "）")
            .replace('[', "【")
            .replace(']', "】")
            .replace(',', "，")
            .replace('.', ".")
            .replace('!', "!")
            .replace('?', "?")
            .replace(':', ":")
            .replace(';', ";")
    }

    /// Expand numbers to spoken form
    fn expand_numbers(&self, text: &str) -> String {
        let mut result = String::new();
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            if c.is_ascii_digit() {
                // Collect consecutive digits
                let mut num_str = c.to_string();
                while let Some(&next) = chars.peek() {
                    if next.is_ascii_digit() {
                        num_str.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }

                // Expand based on context
                let expanded = self.expand_number_sequence(&num_str);
                result.push_str(&expanded);
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Expand a sequence of digits to spoken form
    fn expand_number_sequence(&self, num: &str) -> String {
        // Handle special cases
        match num {
            "0" => return "零".to_string(),
            "1" => return "一".to_string(),
            "2" => return "二".to_string(),
            "3" => return "三".to_string(),
            "4" => return "四".to_string(),
            "5" => return "五".to_string(),
            "6" => return "六".to_string(),
            "7" => return "七".to_string(),
            "8" => return "八".to_string(),
            "9" => return "九".to_string(),
            "10" => return "十".to_string(),
            "100" => return "一百".to_string(),
            "1000" => return "一千".to_string(),
            _ => {}
        }

        // For longer numbers, expand digit by digit (common in TTS for phone numbers, etc.)
        // This is a simplification; proper number expansion would handle place values
        if num.len() <= 2 {
            // Small numbers: expand with place value
            self.expand_small_number(num)
        } else {
            // Longer numbers: expand digit by digit
            num.chars()
                .map(|c| self.digit_to_chinese(c))
                .collect::<Vec<_>>()
                .join(" ")
        }
    }

    /// Expand small numbers (1-99) with place values
    fn expand_small_number(&self, num: &str) -> String {
        let chars: Vec<char> = num.chars().collect();

        match chars.len() {
            1 => self.digit_to_chinese(chars[0]),
            2 => {
                let d0 = chars[0].to_digit(10).unwrap() as usize;
                let d1 = chars[1].to_digit(10).unwrap() as usize;
                if d0 == 1 {
                    format!("十{}", self.digit_to_chinese_char(d1))
                } else if d1 == 0 {
                    format!("{}十", self.digit_to_chinese_char(d0))
                } else {
                    format!(
                        "{}十{}",
                        self.digit_to_chinese_char(d0),
                        self.digit_to_chinese_char(d1)
                    )
                }
            }
            _ => num.to_string(),
        }
    }

    /// Convert a digit to Chinese character
    fn digit_to_chinese(&self, c: char) -> String {
        match c {
            '0' => "零".to_string(),
            '1' => "一".to_string(),
            '2' => "二".to_string(),
            '3' => "三".to_string(),
            '4' => "四".to_string(),
            '5' => "五".to_string(),
            '6' => "六".to_string(),
            '7' => "七".to_string(),
            '8' => "八".to_string(),
            '9' => "九".to_string(),
            _ => c.to_string(),
        }
    }

    /// Convert a digit (0-9) as usize to Chinese character
    fn digit_to_chinese_char(&self, d: usize) -> String {
        match d {
            0 => "零".to_string(),
            1 => "一".to_string(),
            2 => "二".to_string(),
            3 => "三".to_string(),
            4 => "四".to_string(),
            5 => "五".to_string(),
            6 => "六".to_string(),
            7 => "七".to_string(),
            8 => "八".to_string(),
            9 => "九".to_string(),
            _ => "?".to_string(),
        }
    }

    /// Expand common abbreviations
    fn expand_abbreviations(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Common English abbreviations
        result = result
            .replace("Mr.", "Mister ")
            .replace("Mrs.", "Missus ")
            .replace("Dr.", "Doctor ")
            .replace("Prof.", "Professor ")
            .replace("etc.", "et cetera ")
            .replace("e.g.", "for example ")
            .replace("i.e.", "that is ")
            .replace("vs.", "versus ");

        // Common Chinese abbreviations
        result = result
            .replace("等.", "等等")
            .replace("例.", "例如");

        result
    }
}

impl Default for TextNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_whitespace() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.normalize("hello    world").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_normalize_punctuation() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.normalize("hello (world)").unwrap();
        assert!(result.contains("（"));
    }

    #[test]
    fn test_expand_single_digit() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.normalize("我今年 5 岁").unwrap();
        assert!(result.contains("五"));
    }

    #[test]
    fn test_expand_two_digits() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.normalize("今年 25 年").unwrap();
        assert!(result.contains("二十") || result.contains("五"));
    }

    #[test]
    fn test_expand_abbreviations() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.normalize("Mr. Smith").unwrap();
        assert!(result.contains("Mister"));
    }
}
