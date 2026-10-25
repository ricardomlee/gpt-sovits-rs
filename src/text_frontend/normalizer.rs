//! Text normalization with source-character positions for pinyin annotations.

mod numbers;

use crate::Result;

/// Text normalizer for TTS input.
#[derive(Debug, Default)]
pub struct TextNormalizer {}

/// Inserted characters have no source position, so they cannot acquire an annotation.
pub(super) struct NormalizedText(Vec<(char, Option<usize>)>);

impl NormalizedText {
    fn new(text: &str) -> Self {
        Self(
            text.chars()
                .enumerate()
                .map(|(i, ch)| (ch, Some(i)))
                .collect(),
        )
    }

    pub(super) fn into_parts(self) -> (String, Vec<Option<usize>>) {
        self.0.into_iter().unzip()
    }

    fn replace(&mut self, from: &str, to: &str) {
        let pattern: Vec<char> = from.chars().collect();
        let mut result = Vec::with_capacity(self.0.len());
        let mut i = 0;
        while i < self.0.len() {
            if self.0[i..]
                .iter()
                .take(pattern.len())
                .map(|(ch, _)| *ch)
                .eq(pattern.iter().copied())
            {
                result.extend(
                    to.chars()
                        .enumerate()
                        .map(|(offset, ch)| (ch, if offset == 0 { self.0[i].1 } else { None })),
                );
                i += pattern.len();
            } else {
                result.push(self.0[i]);
                i += 1;
            }
        }
        self.0 = result;
    }

    pub(super) fn normalize_chinese_model_text(&mut self) {
        self.replace("...", "…");
        for (ch, _) in &mut self.0 {
            *ch = match *ch {
                '嗯' => '恩',
                '呣' => '母',
                '：' | '；' | '，' | '·' | '、' | '/' => ',',
                '。' | '$' | '\n' => '.',
                '！' => '!',
                '？' => '?',
                '—' => '-',
                '~' | '～' => '…',
                ch => ch,
            };
        }
    }
}

impl TextNormalizer {
    pub fn new() -> Self {
        Self {}
    }

    /// Normalize text, expanding numbers using Chinese reading rules.
    /// Use `TextFrontend` for language-aware normalization and G2P.
    pub fn normalize(&self, text: &str) -> Result<String> {
        Ok(self.normalize_mapped(text, true).into_parts().0)
    }

    /// Normalize Chinese punctuation to the phone symbols used by GPT-SoVITS.
    /// BERT must consume this same text to preserve alignment with G2P.
    pub fn normalize_chinese_model_text(&self, text: &str) -> String {
        let mut normalized = NormalizedText::new(text);
        normalized.normalize_chinese_model_text();
        normalized.into_parts().0
    }

    pub(super) fn normalize_mapped(&self, text: &str, chinese_numbers: bool) -> NormalizedText {
        let mut normalized = NormalizedText(Vec::with_capacity(text.chars().count()));
        let mut whitespace = None;
        for (index, ch) in text.chars().enumerate() {
            if ch.is_whitespace() {
                whitespace = Some(index);
                continue;
            }
            if let Some(source) = whitespace.take() {
                if !normalized.0.is_empty() {
                    normalized.0.push((' ', Some(source)));
                }
            }
            let ch = match ch {
                '“' | '”' => '"',
                '(' => '（',
                ')' => '）',
                '[' => '【',
                ']' => '】',
                ',' => '，',
                '０'..='９' if chinese_numbers => {
                    char::from(b'0' + (ch as u32 - '０' as u32) as u8)
                }
                '％' if chinese_numbers => '%',
                '．' if chinese_numbers => '.',
                '＋' if chinese_numbers => '+',
                '－' | '−' if chinese_numbers => '-',
                ch => ch,
            };
            normalized.0.push((ch, Some(index)));
        }

        if chinese_numbers {
            let chars: Vec<char> = normalized.0.iter().map(|(ch, _)| *ch).collect();
            let mut expanded = Vec::with_capacity(chars.len());
            let mut i = 0;
            while i < chars.len() {
                if let Some((end, spoken)) = numbers::expand_at(&chars, i) {
                    expanded.extend(spoken.chars().map(|ch| (ch, None)));
                    i = end;
                } else {
                    expanded.push(normalized.0[i]);
                    i += 1;
                }
            }
            normalized.0 = expanded;
        }

        for (from, to) in [
            ("Mr.", "Mister "),
            ("Mrs.", "Missus "),
            ("Dr.", "Doctor "),
            ("Prof.", "Professor "),
            ("etc.", "et cetera "),
            ("e.g.", "for example "),
            ("i.e.", "that is "),
            ("vs.", "versus "),
            ("等.", "等等"),
            ("例.", "例如"),
        ] {
            normalized.replace(from, to);
        }
        normalized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_whitespace() {
        let normalizer = TextNormalizer::new();
        assert_eq!(
            normalizer.normalize("hello    world").unwrap(),
            "hello world"
        );
    }

    #[test]
    fn test_normalize_punctuation() {
        let normalizer = TextNormalizer::new();
        assert_eq!(
            normalizer.normalize("hello (world)").unwrap(),
            "hello （world）"
        );
    }

    #[test]
    fn test_normalize_chinese_model_punctuation() {
        let normalizer = TextNormalizer::new();
        let normalized = normalizer.normalize("中道崩殂，今天下三分。").unwrap();
        assert_eq!(
            normalizer.normalize_chinese_model_text(&normalized),
            "中道崩殂,今天下三分."
        );
    }

    #[test]
    fn test_expand_single_digit() {
        let normalizer = TextNormalizer::new();
        assert_eq!(normalizer.normalize("我今年 5 岁").unwrap(), "我今年 五 岁");
    }

    #[test]
    fn test_expand_two_digits() {
        let normalizer = TextNormalizer::new();
        assert_eq!(
            normalizer.normalize("今年 25 年").unwrap(),
            "今年 二十五 年"
        );
    }

    #[test]
    fn spoken_percentages_and_decimals() {
        let normalizer = TextNormalizer::new();
        for (input, expected) in [
            ("音量50%，等待10分钟。", "音量百分之五十，等待十分钟。"),
            ("完成率0.05%", "完成率百分之零点零五"),
            ("比例125%", "比例百分之一百二十五"),
            ("数值3.140", "数值三点一四零"),
            ("数值1001.01", "数值一千零一点零一"),
            ("数值10001.10", "数值一万零一点一零"),
            ("温度-3.5度", "温度负三点五度"),
            ("增长+12.5%", "增长百分之正十二点五"),
            ("音量５０％", "音量百分之五十"),
            ("音量50 %", "音量百分之五十"),
            ("气温−１２．５度", "气温负十二点五度"),
        ] {
            assert_eq!(normalizer.normalize(input).unwrap(), expected, "{input}");
        }
    }

    #[test]
    fn number_expansion_preserves_identifiers_and_sentence_stops() {
        let normalizer = TextNormalizer::new();
        for (input, expected) in [
            ("编号01和00", "编号零一和零零"),
            ("3. 下一项", "三. 下一项"),
            ("3-5次", "三-五次"),
            ("版本1.2.3", "版本一.二.三"),
            ("地址127.0.0.1", "地址一 二 七.零.零.一"),
            ("2026年", "二 零 二 六年"),
        ] {
            assert_eq!(normalizer.normalize(input).unwrap(), expected, "{input}");
        }
    }

    #[test]
    fn test_expand_abbreviations() {
        let normalizer = TextNormalizer::new();
        assert_eq!(normalizer.normalize("Mr. Smith").unwrap(), "Mister  Smith");
    }
}
