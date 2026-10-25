//! Chinese readings for unambiguous percentages and decimal literals.

const DIGITS: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];

fn digit(ch: char) -> char {
    DIGITS[(ch as u8 - b'0') as usize]
}

/// Returns the end character index and spoken form of a numeric token.
pub(super) fn expand_at(chars: &[char], start: usize) -> Option<(usize, String)> {
    let sign = matches!(chars[start], '+' | '-')
        && chars.get(start + 1).is_some_and(char::is_ascii_digit)
        && (start == 0
            || !chars[start - 1].is_ascii_alphanumeric()
                && !matches!(chars[start - 1], '.' | '+' | '-' | '%'));
    let number_start = start + usize::from(sign);
    if !chars.get(number_start).is_some_and(char::is_ascii_digit) {
        return None;
    }

    let mut end = number_start;
    while chars.get(end).is_some_and(char::is_ascii_digit) {
        end += 1;
    }
    let integer_end = end;
    let mut parts = vec![&chars[number_start..end]];
    while chars.get(end) == Some(&'.') && chars.get(end + 1).is_some_and(char::is_ascii_digit) {
        let fraction_start = end + 1;
        end = fraction_start;
        while chars.get(end).is_some_and(char::is_ascii_digit) {
            end += 1;
        }
        parts.push(&chars[fraction_start..end]);
    }

    // Dotted versions and IPv4 addresses are not decimal values.
    if parts.len() > 2 {
        let mut spoken = if sign {
            chars[start].to_string()
        } else {
            String::new()
        };
        spoken.push_str(
            &parts
                .iter()
                .map(|part| legacy_integer(part))
                .collect::<Vec<_>>()
                .join("."),
        );
        return Some((end, spoken));
    }

    let mut percent_index = end;
    while chars
        .get(percent_index)
        .is_some_and(|ch| ch.is_whitespace())
    {
        percent_index += 1;
    }
    let percent = chars.get(percent_index) == Some(&'%');
    let mut spoken = String::new();
    if percent {
        spoken.push_str("百分之");
        end = percent_index + 1;
    }
    if sign {
        spoken.push(if chars[start] == '-' { '负' } else { '正' });
    }
    let integer = &chars[number_start..integer_end];
    spoken.push_str(&if percent || parts.len() == 2 {
        cardinal(integer)
    } else {
        legacy_integer(integer)
    });
    if let Some(fraction) = parts.get(1) {
        spoken.push('点');
        spoken.extend(fraction.iter().copied().map(digit));
    }
    Some((end, spoken))
}

/// Keep existing standalone integer readings; dates and phone numbers need context.
fn legacy_integer(digits: &[char]) -> String {
    if digits.len() > 1 && digits[0] == '0' {
        return digits.iter().copied().map(digit).collect();
    }
    if digits.len() <= 2 || digits == ['1', '0', '0'] || digits == ['1', '0', '0', '0'] {
        return cardinal(digits);
    }
    digits
        .iter()
        .map(|ch| digit(*ch).to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn cardinal(digits: &[char]) -> String {
    const GROUP_UNITS: [&str; 4] = ["", "万", "亿", "万亿"];
    if digits.len() > 16 || digits.len() > 1 && digits[0] == '0' {
        return digits.iter().copied().map(digit).collect();
    }
    let groups: Vec<&[char]> = digits.rchunks(4).collect();
    let mut result = String::new();
    let mut skipped_group = false;
    for (index, group) in groups.iter().enumerate().rev() {
        let value = group
            .iter()
            .fold(0u16, |value, ch| value * 10 + (*ch as u8 - b'0') as u16);
        if value == 0 {
            skipped_group = true;
            continue;
        }
        if !result.is_empty() && (skipped_group || value < 1000) {
            result.push('零');
        }
        let mut pending_zero = false;
        for (i, ch) in group.iter().enumerate() {
            let power = group.len() - i - 1;
            if *ch == '0' {
                pending_zero = true;
                continue;
            }
            if pending_zero && !result.is_empty() && !result.ends_with('零') {
                result.push('零');
            }
            if !(result.is_empty() && power == 1 && *ch == '1') {
                result.push(digit(*ch));
            }
            result.push_str(["", "十", "百", "千"][power]);
            pending_zero = false;
        }
        result.push_str(GROUP_UNITS[index]);
        skipped_group = false;
    }
    if result.is_empty() {
        result.push('零');
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinal_place_values_and_zeros() {
        for (input, expected) in [
            ("0", "零"),
            ("10", "十"),
            ("11", "十一"),
            ("20", "二十"),
            ("101", "一百零一"),
            ("110", "一百一十"),
            ("1000", "一千"),
            ("1001", "一千零一"),
            ("1010", "一千零一十"),
            ("10000", "一万"),
            ("10001", "一万零一"),
            ("10010", "一万零一十"),
            ("10100", "一万零一百"),
            ("11000", "一万一千"),
            ("100000", "十万"),
            ("100000001", "一亿零一"),
            ("100010001", "一亿零一万零一"),
            ("100100001", "一亿零一十万零一"),
            ("1000000000000", "一万亿"),
            ("1000000000001", "一万亿零一"),
            ("00", "零零"),
            ("01", "零一"),
        ] {
            assert_eq!(
                cardinal(&input.chars().collect::<Vec<_>>()),
                expected,
                "{input}"
            );
        }
    }

    #[test]
    fn very_large_values_do_not_overflow() {
        let input = "9".repeat(100);
        assert_eq!(
            cardinal(&input.chars().collect::<Vec<_>>()),
            "九".repeat(100)
        );
    }
}
