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
    const SENTENCE_SOFT_DELIMITERS: &[char] = &[',', ';', ':', '、', '，', '；', '：'];
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
    let soft_split_chars = soft_split_chars(language);
    let hard_split_chars = hard_split_chars(language);

    for (index, &ch) in chars.iter().enumerate() {
        current.push(ch);
        let decimal_point = ch == '.'
            && index > 0
            && index + 1 < chars.len()
            && chars[index - 1].is_ascii_digit()
            && chars[index + 1].is_ascii_digit();
        let current_chars = current.trim().chars().count();
        let sentence_boundary = delimiters.contains(&ch) && !decimal_point;
        let soft_boundary = method == SplitMethod::Sentence
            && SENTENCE_SOFT_DELIMITERS.contains(&ch)
            && !decimal_point
            && current_chars >= soft_split_chars;
        let hard_boundary = method == SplitMethod::Sentence && current_chars >= hard_split_chars;
        if (sentence_boundary || soft_boundary || hard_boundary) && !current.trim().is_empty() {
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
            let merged_chars = last.chars().count() + acc.trim().chars().count();
            if merged_chars <= hard_split_chars {
                last.push_str(&acc);
            } else {
                merged.push(acc.trim().to_string());
            }
        } else {
            merged.push(acc.trim().to_string());
        }
    }

    merged
}

fn soft_split_chars(language: Language) -> usize {
    match language {
        Language::English => 80,
        Language::Chinese
        | Language::Japanese
        | Language::Korean
        | Language::Cantonese
        | Language::Auto => 32,
    }
}

fn hard_split_chars(language: Language) -> usize {
    match language {
        Language::English => 140,
        Language::Chinese
        | Language::Japanese
        | Language::Korean
        | Language::Cantonese
        | Language::Auto => 60,
    }
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

    #[test]
    fn sentence_split_breaks_long_chinese_comma_clauses() {
        let text = "这是一个很长的中文段落，它中间只有逗号和顿号，却没有句号，所以不能整段送进模型，否则长文本合成时容易吞字或者变成哼唱。";

        let chunks = split_sentences_for_language(text, 12, Language::Chinese);

        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 80));
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn sentence_split_hard_wraps_long_text_without_punctuation() {
        let text = "这是一段完全没有任何标点符号的长文本它需要被安全切开否则会作为一个过长片段进入模型导致生成结果不稳定而且在真实的长段语音合成请求中这类输入并不少见";

        let chunks = split_sentences_for_language(text, 12, Language::Chinese);

        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 61));
        assert_eq!(chunks.concat(), format!("{text}。"));
    }
}
