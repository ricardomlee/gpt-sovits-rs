//! English numeric literals, without guessing dates, prices, or phone numbers.

const SMALL: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

fn identifier(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn digits(chars: &[char]) -> String {
    chars
        .iter()
        .map(|ch| SMALL[(*ch as u8 - b'0') as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

fn cardinal(chars: &[char]) -> String {
    const SCALES: [&str; 5] = ["", "thousand", "million", "billion", "trillion"];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    if chars.len() > 15 || chars.len() > 1 && chars[0] == '0' {
        return digits(chars);
    }
    let groups: Vec<_> = chars.rchunks(3).collect();
    let mut words = Vec::new();
    for (index, group) in groups.iter().enumerate().rev() {
        let value = group
            .iter()
            .fold(0usize, |n, ch| n * 10 + (*ch as u8 - b'0') as usize);
        if value == 0 {
            continue;
        }
        if value >= 100 {
            words.extend([SMALL[value / 100], "hundred"]);
        }
        let remainder = value % 100;
        if remainder >= 20 {
            words.push(TENS[remainder / 10]);
        }
        let units = if remainder >= 20 {
            remainder % 10
        } else {
            remainder
        };
        if units != 0 {
            words.push(SMALL[units]);
        }
        if index != 0 {
            words.push(SCALES[index]);
        }
    }
    if words.is_empty() {
        words.push("zero");
    }
    words.join(" ")
}

pub(super) fn expand_at(chars: &[char], start: usize) -> Option<(usize, String)> {
    let standalone = start == 0
        || !identifier(chars[start - 1]) && !matches!(chars[start - 1], '.' | '+' | '-' | '%');
    let sign = standalone && matches!(chars[start], '+' | '-');
    let number_start = start + usize::from(sign);
    let leading_point = standalone && chars.get(number_start) == Some(&'.');
    let first_digit = number_start + usize::from(leading_point);
    if !chars.get(first_digit).is_some_and(char::is_ascii_digit) {
        return None;
    }
    let mut end = first_digit;
    while chars.get(end).is_some_and(char::is_ascii_digit) {
        end += 1;
    }
    let mut integer = chars[first_digit..end].to_vec();
    if !leading_point && integer.len() <= 3 {
        while matches!(chars.get(end), Some(',' | '，')) {
            let group_start = end + 1;
            let mut group_end = group_start;
            while chars.get(group_end).is_some_and(char::is_ascii_digit) {
                group_end += 1;
            }
            if group_end - group_start != 3 {
                break;
            }
            integer.extend_from_slice(&chars[group_start..group_end]);
            end = group_end;
        }
    }
    let mut fractions = Vec::new();
    if leading_point {
        fractions.push(&chars[first_digit..end]);
        integer = vec!['0'];
    }
    while chars.get(end) == Some(&'.') && chars.get(end + 1).is_some_and(char::is_ascii_digit) {
        let fraction_start = end + 1;
        end = fraction_start;
        while chars.get(end).is_some_and(char::is_ascii_digit) {
            end += 1;
        }
        fractions.push(&chars[fraction_start..end]);
    }

    let suffix = match integer.last() {
        _ if integer.len() > 1 && integer[integer.len() - 2] == '1' => ['t', 'h'],
        Some('1') => ['s', 't'],
        Some('2') => ['n', 'd'],
        Some('3') => ['r', 'd'],
        _ => ['t', 'h'],
    };
    let ordinal = !sign
        && fractions.is_empty()
        && standalone
        && integer.len() <= 15
        && (integer.len() == 1 || integer[0] != '0')
        && chars
            .get(end..end + 2)
            .is_some_and(|chars| chars.iter().map(char::to_ascii_lowercase).eq(suffix))
        && !chars.get(end + 2).is_some_and(|ch| identifier(*ch));
    if ordinal {
        end += 2;
    }
    let as_digits = !ordinal
        && (start > 0 && identifier(chars[start - 1])
            || chars.get(end).is_some_and(|ch| identifier(*ch))
            || fractions.len() > 1);
    let mut spoken = if as_digits {
        digits(&integer)
    } else {
        cardinal(&integer)
    };
    if ordinal {
        let last = spoken
            .rsplit_once(' ')
            .map_or(0, |(prefix, _)| prefix.len() + 1);
        let ordinal_word = match &spoken[last..] {
            "one" => "first".to_string(),
            "two" => "second".to_string(),
            "three" => "third".to_string(),
            "five" => "fifth".to_string(),
            "eight" => "eighth".to_string(),
            "nine" => "ninth".to_string(),
            "twelve" => "twelfth".to_string(),
            word if word.ends_with('y') => format!("{}ieth", &word[..word.len() - 1]),
            word => format!("{word}th"),
        };
        spoken.replace_range(last.., &ordinal_word);
    }
    for fraction in fractions {
        spoken.push_str(" point ");
        spoken.push_str(&digits(fraction));
    }
    if sign {
        spoken.insert_str(
            0,
            if chars[start] == '-' {
                "minus "
            } else {
                "plus "
            },
        );
    }
    let mut percent_index = end;
    while chars
        .get(percent_index)
        .is_some_and(|ch| ch.is_whitespace())
    {
        percent_index += 1;
    }
    if chars.get(percent_index) == Some(&'%') {
        spoken.push_str(" percent");
        end = percent_index + 1;
    }
    Some((end, spoken))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Language, text_frontend::TextNormalizer};

    #[test]
    fn english_numeric_readings_and_boundaries() {
        for (input, expected) in [
            ("0 19 20 101", "zero nineteen twenty one hundred one"),
            ("1,234", "one thousand two hundred thirty four"),
            ("1000000000001", "one trillion one"),
            ("3.140", "three point one four zero"),
            ("-0.05%", "minus zero point zero five percent"),
            ("＋１２．５ ％", "plus twelve point five percent"),
            (".5 -.05", "zero point five minus zero point zero five"),
            ("007", "zero zero seven"),
            ("v12", "v one two"),
            ("id_12x", "id_ one two x"),
            ("v1.2.3", "v one point two point three"),
            ("127.0.0.1", "one two seven point zero point zero point one"),
            ("21st 12th 100th", "twenty first twelfth one hundredth"),
            ("21ST 12TH", "twenty first twelfth"),
            ("11th 13th 20th", "eleventh thirteenth twentieth"),
            ("code:12,next:34", "code: twelve ，next: thirty four"),
            ("3. Next", "three . Next"),
            ("3-5", "three - five"),
            ("1,2,3", "one ， two ， three"),
            ("1,000,001", "one million one"),
            ("+ - .", "+ - ."),
        ] {
            assert_eq!(
                TextNormalizer::new()
                    .normalize_mapped(input, Language::English)
                    .into_parts()
                    .0,
                expected,
                "{input}"
            );
        }
    }

    #[test]
    fn very_large_values_do_not_overflow() {
        let chars = "9".repeat(100).chars().collect::<Vec<_>>();
        assert_eq!(cardinal(&chars), vec!["nine"; 100].join(" "));
    }
}
