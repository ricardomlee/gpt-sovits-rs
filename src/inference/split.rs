//! Text splitting policies for inference.

use crate::Language;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitMethod {
    #[default]
    Sentence,
    Cut5,
}

impl SplitMethod {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "sentence" => Some(Self::Sentence),
            "cut5" => Some(Self::Cut5),
            _ => None,
        }
    }
}

/// Split only at sentence-ending punctuation for stable production inference.
pub fn split_sentences(text: &str, min_chars: usize) -> Vec<String> {
    split_sentences_for_language(text, min_chars, Language::Chinese)
}

pub fn split_sentences_for_language(
    text: &str,
    min_chars: usize,
    language: Language,
) -> Vec<String> {
    split_text(text, min_chars, language, SplitMethod::Sentence)
}

/// Split using Python GPT-SoVITS `cut5`, including commas and semicolons.
pub fn split_cut5_for_language(text: &str, min_chars: usize, language: Language) -> Vec<String> {
    split_text(text, min_chars, language, SplitMethod::Cut5)
}

pub(super) fn split_text(
    text: &str,
    min_chars: usize,
    language: Language,
    method: SplitMethod,
) -> Vec<String> {
    const SENTENCE_DELIMITERS: &[char] = &['.', '?', '!', '。', '？', '！', '…', '\n'];
    const CUT5_DELIMITERS: &[char] = &[
        ',', '.', ';', '?', '!', '、', '，', '。', '？', '！', '；', '：', ':', '…', '\n',
    ];
    let delimiters = match method {
        SplitMethod::Sentence => SENTENCE_DELIMITERS,
        SplitMethod::Cut5 => CUT5_DELIMITERS,
    };

    let normalized = match method {
        SplitMethod::Sentence => text.to_string(),
        SplitMethod::Cut5 => text.replace("……", "。").replace("——", "，"),
    };
    let chars: Vec<char> = normalized.chars().collect();
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    for (index, &ch) in chars.iter().enumerate() {
        current.push(ch);
        let decimal_point = ch == '.'
            && index > 0
            && index + 1 < chars.len()
            && chars[index - 1].is_ascii_digit()
            && chars[index + 1].is_ascii_digit();
        if delimiters.contains(&ch) && !decimal_point && !current.trim().is_empty() {
            let trimmed = current.trim().to_string();
            if trimmed.chars().any(char::is_alphanumeric) {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let mut tail = current.trim().to_string();
    if tail.chars().any(char::is_alphanumeric) {
        if !tail
            .chars()
            .last()
            .is_some_and(|ch| delimiters.contains(&ch))
        {
            tail.push(if language == Language::English {
                '.'
            } else {
                '。'
            });
        }
        sentences.push(tail);
    }

    let mut merged: Vec<String> = Vec::new();
    let mut acc = String::new();
    for s in sentences {
        acc.push_str(&s);
        if acc.chars().count() >= min_chars {
            merged.push(acc.trim().to_string());
            acc.clear();
        }
    }
    if !acc.trim().is_empty() {
        if let Some(last) = merged.last_mut() {
            last.push_str(&acc);
        } else {
            merged.push(acc.trim().to_string());
        }
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cut5_keeps_decimal_numbers_together() {
        let chunks = split_cut5_for_language("value is 3.14, then stop", 1, Language::English);

        assert_eq!(chunks, vec!["value is 3.14,", "then stop."]);
    }

    #[test]
    fn sentence_split_adds_language_appropriate_tail_punctuation() {
        assert_eq!(
            split_sentences_for_language("hello world", 1, Language::English),
            vec!["hello world."]
        );
        assert_eq!(
            split_sentences_for_language("hello world", 1, Language::Chinese),
            vec!["hello world。"]
        );
    }
}
