//! Grapheme-to-Phoneme Converter
//!
//! For Chinese: converts characters to initials + finals with tones,
//! matching GPT-SoVITS Python text frontend (symbols_v2 format).

use crate::{Language, Result};
use crate::text_frontend::tone_sandhi::ToneSandhi;
use jieba_rs::Jieba;
use pinyin::ToPinyin;
use std::collections::HashMap;

/// Mapping from pinyin base (without tone) to (initial, final) pairs.
/// Matches Python's `text.chinese2.pinyin_to_symbol_map`.
fn pinyin_split() -> HashMap<&'static str, (&'static str, &'static str)> {
    let mut map = HashMap::new();
    // Single finals (no initial)
    map.insert("a", ("", "a"));
    map.insert("ai", ("", "a"));
    map.insert("an", ("", "an"));
    map.insert("ang", ("", "ang"));
    map.insert("ao", ("", "ao"));
    map.insert("e", ("", "e"));
    map.insert("ei", ("", "e"));
    map.insert("en", ("", "en"));
    map.insert("eng", ("", "eng"));
    map.insert("er", ("", "er"));
    map.insert("o", ("", "o"));
    map.insert("ou", ("", "o"));
    // b
    map.insert("ba", ("b", "a")); map.insert("bai", ("b", "ai"));
    map.insert("ban", ("b", "an")); map.insert("bang", ("b", "ang"));
    map.insert("bao", ("b", "ao")); map.insert("bei", ("b", "e"));
    map.insert("ben", ("b", "en")); map.insert("beng", ("b", "eng"));
    map.insert("bi", ("b", "i")); map.insert("bian", ("b", "ian"));
    map.insert("biao", ("b", "iao")); map.insert("bie", ("b", "ie"));
    map.insert("bin", ("b", "in")); map.insert("bing", ("b", "ing"));
    map.insert("bo", ("b", "o")); map.insert("bu", ("b", "u"));
    // c
    map.insert("ca", ("c", "a")); map.insert("cai", ("c", "ai"));
    map.insert("can", ("c", "an")); map.insert("cang", ("c", "ang"));
    map.insert("cao", ("c", "ao")); map.insert("ce", ("c", "e"));
    map.insert("cei", ("c", "e")); map.insert("cen", ("c", "en"));
    map.insert("ceng", ("c", "eng")); map.insert("ci", ("c", "ir"));
    map.insert("cong", ("c", "ong")); map.insert("cou", ("c", "ou"));
    map.insert("cu", ("c", "u")); map.insert("cuan", ("c", "uan"));
    map.insert("cui", ("c", "ui")); map.insert("cun", ("c", "un"));
    map.insert("cuo", ("c", "uo"));
    // d
    map.insert("da", ("d", "a")); map.insert("dai", ("d", "ai"));
    map.insert("dan", ("d", "an")); map.insert("dang", ("d", "ang"));
    map.insert("dao", ("d", "ao")); map.insert("de", ("d", "e"));
    map.insert("dei", ("d", "e")); map.insert("den", ("d", "en"));
    map.insert("deng", ("d", "eng")); map.insert("di", ("d", "i"));
    map.insert("dia", ("d", "ia")); map.insert("dian", ("d", "ian"));
    map.insert("diao", ("d", "iao")); map.insert("die", ("d", "ie"));
    map.insert("ding", ("d", "ing")); map.insert("diu", ("d", "iu"));
    map.insert("dong", ("d", "ong")); map.insert("dou", ("d", "ou"));
    map.insert("du", ("d", "u")); map.insert("duan", ("d", "uan"));
    map.insert("dui", ("d", "ui")); map.insert("dun", ("d", "un"));
    map.insert("duo", ("d", "uo"));
    // f
    map.insert("fa", ("f", "a")); map.insert("fan", ("f", "an"));
    map.insert("fang", ("f", "ang")); map.insert("fei", ("f", "e"));
    map.insert("fen", ("f", "en")); map.insert("feng", ("f", "eng"));
    map.insert("fo", ("f", "o")); map.insert("fou", ("f", "ou"));
    map.insert("fu", ("f", "u"));
    // g
    map.insert("ga", ("g", "a")); map.insert("gai", ("g", "ai"));
    map.insert("gan", ("g", "an")); map.insert("gang", ("g", "ang"));
    map.insert("gao", ("g", "ao")); map.insert("ge", ("g", "e"));
    map.insert("gei", ("g", "e")); map.insert("gen", ("g", "en"));
    map.insert("geng", ("g", "eng")); map.insert("gong", ("g", "ong"));
    map.insert("gou", ("g", "ou")); map.insert("gu", ("g", "u"));
    map.insert("gua", ("g", "ua")); map.insert("guai", ("g", "uai"));
    map.insert("guan", ("g", "uan")); map.insert("guang", ("g", "uang"));
    map.insert("gui", ("g", "ui")); map.insert("gun", ("g", "un"));
    map.insert("guo", ("g", "uo"));
    // h
    map.insert("ha", ("h", "a")); map.insert("hai", ("h", "ai"));
    map.insert("han", ("h", "an")); map.insert("hang", ("h", "ang"));
    map.insert("hao", ("h", "ao")); map.insert("he", ("h", "e"));
    map.insert("hei", ("h", "e")); map.insert("hen", ("h", "en"));
    map.insert("heng", ("h", "eng")); map.insert("hong", ("h", "ong"));
    map.insert("hou", ("h", "ou")); map.insert("hu", ("h", "u"));
    map.insert("hua", ("h", "ua")); map.insert("huai", ("h", "uai"));
    map.insert("huan", ("h", "uan")); map.insert("huang", ("h", "uang"));
    map.insert("hui", ("h", "ui")); map.insert("hun", ("h", "un"));
    map.insert("huo", ("h", "uo"));
    // j
    map.insert("ji", ("j", "i")); map.insert("jia", ("j", "ia"));
    map.insert("jian", ("j", "ian")); map.insert("jiang", ("j", "iang"));
    map.insert("jiao", ("j", "iao")); map.insert("jie", ("j", "ie"));
    map.insert("jin", ("j", "in")); map.insert("jing", ("j", "ing"));
    map.insert("jiong", ("j", "iong")); map.insert("jiu", ("j", "iu"));
    map.insert("ju", ("j", "v")); map.insert("juan", ("j", "van"));
    map.insert("jue", ("j", "ve")); map.insert("jun", ("j", "vn"));
    // k
    map.insert("ka", ("k", "a")); map.insert("kai", ("k", "ai"));
    map.insert("kan", ("k", "an")); map.insert("kang", ("k", "ang"));
    map.insert("kao", ("k", "ao")); map.insert("ke", ("k", "e"));
    map.insert("kei", ("k", "e")); map.insert("ken", ("k", "en"));
    map.insert("keng", ("k", "eng")); map.insert("kong", ("k", "ong"));
    map.insert("kou", ("k", "ou")); map.insert("ku", ("k", "u"));
    map.insert("kua", ("k", "ua")); map.insert("kuai", ("k", "uai"));
    map.insert("kuan", ("k", "uan")); map.insert("kuang", ("k", "uang"));
    map.insert("kui", ("k", "ui")); map.insert("kun", ("k", "un"));
    map.insert("kuo", ("k", "uo"));
    // l
    map.insert("la", ("l", "a")); map.insert("lai", ("l", "ai"));
    map.insert("lan", ("l", "an")); map.insert("lang", ("l", "ang"));
    map.insert("lao", ("l", "ao")); map.insert("le", ("l", "e"));
    map.insert("lei", ("l", "e")); map.insert("leng", ("l", "eng"));
    map.insert("li", ("l", "i")); map.insert("lia", ("l", "ia"));
    map.insert("lian", ("l", "ian")); map.insert("liang", ("l", "iang"));
    map.insert("liao", ("l", "iao")); map.insert("lie", ("l", "ie"));
    map.insert("lin", ("l", "in")); map.insert("ling", ("l", "ing"));
    map.insert("liu", ("l", "iu")); map.insert("long", ("l", "ong"));
    map.insert("lou", ("l", "ou")); map.insert("lu", ("l", "u"));
    map.insert("lv", ("l", "v")); map.insert("luan", ("l", "uan"));
    map.insert("lve", ("l", "ve")); map.insert("lue", ("l", "ve"));
    map.insert("lun", ("l", "un")); map.insert("luo", ("l", "uo"));
    // m
    map.insert("ma", ("m", "a")); map.insert("mai", ("m", "ai"));
    map.insert("man", ("m", "an")); map.insert("mang", ("m", "ang"));
    map.insert("mao", ("m", "ao")); map.insert("me", ("m", "e"));
    map.insert("mei", ("m", "e")); map.insert("men", ("m", "en"));
    map.insert("meng", ("m", "eng")); map.insert("mi", ("m", "i"));
    map.insert("mian", ("m", "ian")); map.insert("miao", ("m", "iao"));
    map.insert("mie", ("m", "ie")); map.insert("min", ("m", "in"));
    map.insert("ming", ("m", "ing")); map.insert("miu", ("m", "iu"));
    map.insert("mo", ("m", "o")); map.insert("mou", ("m", "ou"));
    map.insert("mu", ("m", "u"));
    // n
    map.insert("na", ("n", "a")); map.insert("nai", ("n", "ai"));
    map.insert("nan", ("n", "an")); map.insert("nang", ("n", "ang"));
    map.insert("nao", ("n", "ao")); map.insert("ne", ("n", "e"));
    map.insert("nei", ("n", "e")); map.insert("nen", ("n", "en"));
    map.insert("neng", ("n", "eng")); map.insert("ni", ("n", "i"));
    map.insert("nian", ("n", "ian")); map.insert("niang", ("n", "iang"));
    map.insert("niao", ("n", "iao")); map.insert("nie", ("n", "ie"));
    map.insert("nin", ("n", "in")); map.insert("ning", ("n", "ing"));
    map.insert("niu", ("n", "iu")); map.insert("nong", ("n", "ong"));
    map.insert("nou", ("n", "ou")); map.insert("nu", ("n", "u"));
    map.insert("nv", ("n", "v")); map.insert("nuan", ("n", "uan"));
    map.insert("nve", ("n", "ve")); map.insert("nue", ("n", "ve"));
    map.insert("nun", ("n", "un")); map.insert("nuo", ("n", "uo"));
    // p
    map.insert("pa", ("p", "a")); map.insert("pai", ("p", "ai"));
    map.insert("pan", ("p", "an")); map.insert("pang", ("p", "ang"));
    map.insert("pao", ("p", "ao")); map.insert("pei", ("p", "e"));
    map.insert("pen", ("p", "en")); map.insert("peng", ("p", "eng"));
    map.insert("pi", ("p", "i")); map.insert("pian", ("p", "ian"));
    map.insert("piao", ("p", "iao")); map.insert("pie", ("p", "ie"));
    map.insert("pin", ("p", "in")); map.insert("ping", ("p", "ing"));
    map.insert("po", ("p", "o")); map.insert("pou", ("p", "ou"));
    map.insert("pu", ("p", "u"));
    // q
    map.insert("qi", ("q", "i")); map.insert("qia", ("q", "ia"));
    map.insert("qian", ("q", "ian")); map.insert("qiang", ("q", "iang"));
    map.insert("qiao", ("q", "iao")); map.insert("qie", ("q", "ie"));
    map.insert("qin", ("q", "in")); map.insert("qing", ("q", "ing"));
    map.insert("qiong", ("q", "iong")); map.insert("qiu", ("q", "iu"));
    map.insert("qu", ("q", "v")); map.insert("quan", ("q", "van"));
    map.insert("que", ("q", "ve")); map.insert("qun", ("q", "vn"));
    // r
    map.insert("ran", ("r", "an")); map.insert("rang", ("r", "ang"));
    map.insert("rao", ("r", "ao")); map.insert("re", ("r", "e"));
    map.insert("ren", ("r", "en")); map.insert("reng", ("r", "eng"));
    map.insert("ri", ("r", "ir")); map.insert("rong", ("r", "ong"));
    map.insert("rou", ("r", "ou")); map.insert("ru", ("r", "u"));
    map.insert("ruan", ("r", "uan")); map.insert("rui", ("r", "ui"));
    map.insert("run", ("r", "un")); map.insert("ruo", ("r", "uo"));
    // s
    map.insert("sa", ("s", "a")); map.insert("sai", ("s", "ai"));
    map.insert("san", ("s", "an")); map.insert("sang", ("s", "ang"));
    map.insert("sao", ("s", "ao")); map.insert("se", ("s", "e"));
    map.insert("sei", ("s", "e")); map.insert("sen", ("s", "en"));
    map.insert("seng", ("s", "eng")); map.insert("si", ("s", "ir"));
    map.insert("song", ("s", "ong")); map.insert("sou", ("s", "ou"));
    map.insert("su", ("s", "u")); map.insert("suan", ("s", "uan"));
    map.insert("sui", ("s", "ui")); map.insert("sun", ("s", "un"));
    map.insert("suo", ("s", "uo"));
    // t
    map.insert("ta", ("t", "a")); map.insert("tai", ("t", "ai"));
    map.insert("tan", ("t", "an")); map.insert("tang", ("t", "ang"));
    map.insert("tao", ("t", "ao")); map.insert("te", ("t", "e"));
    map.insert("tei", ("t", "e")); map.insert("teng", ("t", "eng"));
    map.insert("ti", ("t", "i")); map.insert("tian", ("t", "ian"));
    map.insert("tiao", ("t", "iao")); map.insert("tie", ("t", "ie"));
    map.insert("ting", ("t", "ing")); map.insert("tong", ("t", "ong"));
    map.insert("tou", ("t", "ou")); map.insert("tu", ("t", "u"));
    map.insert("tuan", ("t", "uan")); map.insert("tui", ("t", "ui"));
    map.insert("tun", ("t", "un")); map.insert("tuo", ("t", "uo"));
    // w
    map.insert("wa", ("w", "a")); map.insert("wai", ("w", "ai"));
    map.insert("wan", ("w", "an")); map.insert("wang", ("w", "ang"));
    map.insert("wei", ("w", "e")); map.insert("wen", ("w", "en"));
    map.insert("weng", ("w", "eng")); map.insert("wo", ("w", "o"));
    map.insert("wu", ("w", "u"));
    // x
    map.insert("xi", ("x", "i")); map.insert("xia", ("x", "ia"));
    map.insert("xian", ("x", "ian")); map.insert("xiang", ("x", "iang"));
    map.insert("xiao", ("x", "iao")); map.insert("xie", ("x", "ie"));
    map.insert("xin", ("x", "in")); map.insert("xing", ("x", "ing"));
    map.insert("xiong", ("x", "iong")); map.insert("xiu", ("x", "iu"));
    map.insert("xu", ("x", "v")); map.insert("xuan", ("x", "van"));
    map.insert("xue", ("x", "ve")); map.insert("xun", ("x", "vn"));
    // y
    map.insert("ya", ("y", "a")); map.insert("yan", ("y", "an"));
    map.insert("yang", ("y", "ang")); map.insert("yao", ("y", "ao"));
    map.insert("ye", ("y", "e")); map.insert("yi", ("y", "i"));
    map.insert("yin", ("y", "in")); map.insert("ying", ("y", "ing"));
    map.insert("yong", ("y", "ong")); map.insert("you", ("y", "ou"));
    map.insert("yu", ("y", "v")); map.insert("yuan", ("y", "van"));
    map.insert("yue", ("y", "ve")); map.insert("yun", ("y", "vn"));
    // z
    map.insert("za", ("z", "a")); map.insert("zai", ("z", "ai"));
    map.insert("zan", ("z", "an")); map.insert("zang", ("z", "ang"));
    map.insert("zao", ("z", "ao")); map.insert("ze", ("z", "e"));
    map.insert("zei", ("z", "e")); map.insert("zen", ("z", "en"));
    map.insert("zeng", ("z", "eng")); map.insert("zi", ("z", "ir"));
    map.insert("zong", ("z", "ong")); map.insert("zou", ("z", "ou"));
    map.insert("zu", ("z", "u")); map.insert("zuan", ("z", "uan"));
    map.insert("zui", ("z", "ui")); map.insert("zun", ("z", "un"));
    map.insert("zuo", ("z", "uo"));
    // zh
    map.insert("zha", ("zh", "a")); map.insert("zhai", ("zh", "ai"));
    map.insert("zhan", ("zh", "an")); map.insert("zhang", ("zh", "ang"));
    map.insert("zhao", ("zh", "ao")); map.insert("zhe", ("zh", "e"));
    map.insert("zhei", ("zh", "e")); map.insert("zhen", ("zh", "en"));
    map.insert("zheng", ("zh", "eng")); map.insert("zhi", ("zh", "ir"));
    map.insert("zhong", ("zh", "ong")); map.insert("zhou", ("zh", "ou"));
    map.insert("zhu", ("zh", "u")); map.insert("zhua", ("zh", "ua"));
    map.insert("zhuai", ("zh", "uai")); map.insert("zhuan", ("zh", "uan"));
    map.insert("zhuang", ("zh", "uang")); map.insert("zhui", ("zh", "ui"));
    map.insert("zhun", ("zh", "un")); map.insert("zhuo", ("zh", "uo"));
    // ch
    map.insert("cha", ("ch", "a")); map.insert("chai", ("ch", "ai"));
    map.insert("chan", ("ch", "an")); map.insert("chang", ("ch", "ang"));
    map.insert("chao", ("ch", "ao")); map.insert("che", ("ch", "e"));
    map.insert("chen", ("ch", "en")); map.insert("cheng", ("ch", "eng"));
    map.insert("chi", ("ch", "ir")); map.insert("chong", ("ch", "ong"));
    map.insert("chou", ("ch", "ou")); map.insert("chu", ("ch", "u"));
    map.insert("chua", ("ch", "ua")); map.insert("chuai", ("ch", "uai"));
    map.insert("chuan", ("ch", "uan")); map.insert("chuang", ("ch", "uang"));
    map.insert("chui", ("ch", "ui")); map.insert("chun", ("ch", "un"));
    map.insert("chuo", ("ch", "uo"));
    // sh
    map.insert("sha", ("sh", "a")); map.insert("shai", ("sh", "ai"));
    map.insert("shan", ("sh", "an")); map.insert("shang", ("sh", "ang"));
    map.insert("shao", ("sh", "ao")); map.insert("she", ("sh", "e"));
    map.insert("shei", ("sh", "e")); map.insert("shen", ("sh", "en"));
    map.insert("sheng", ("sh", "eng")); map.insert("shi", ("sh", "ir"));
    map.insert("shou", ("sh", "ou")); map.insert("shu", ("sh", "u"));
    map.insert("shua", ("sh", "ua")); map.insert("shuai", ("sh", "uai"));
    map.insert("shuan", ("sh", "uan")); map.insert("shuang", ("sh", "uang"));
    map.insert("shui", ("sh", "ui")); map.insert("shun", ("sh", "un"));
    map.insert("shuo", ("sh", "uo"));

    map
}

/// CMUdict data embedded at compile time (123 K entries, ARPABET format).
const CMUDICT_TXT: &str = include_str!("cmudict.txt");

/// G2P Converter for multiple languages
pub struct G2PConverter {
    pinyin_map: HashMap<&'static str, (&'static str, &'static str)>,
    jieba: Jieba,
    tone_sandhi: ToneSandhi,
    /// CMU Pronouncing Dictionary: UPPERCASE_WORD → space-separated ARPABET phonemes
    cmudict: HashMap<String, String>,
}

impl std::fmt::Debug for G2PConverter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("G2PConverter")
            .field("cmudict_entries", &self.cmudict.len())
            .finish()
    }
}

impl G2PConverter {
    pub fn new() -> Result<Self> {
        let cmudict = Self::load_cmudict();
        Ok(Self {
            pinyin_map: pinyin_split(),
            jieba: Jieba::new(),
            tone_sandhi: ToneSandhi::new(),
            cmudict,
        })
    }

    fn load_cmudict() -> HashMap<String, String> {
        let mut map = HashMap::with_capacity(125_000);
        for line in CMUDICT_TXT.lines() {
            if let Some((word, phones)) = line.split_once('\t') {
                map.insert(word.to_string(), phones.to_string());
            }
        }
        map
    }

    /// Look up a word in CMUdict, trying common normalizations.
    /// Returns space-separated ARPABET phonemes, or None if not found.
    fn cmudict_lookup(&self, word: &str) -> Option<&str> {
        let upper = word.to_uppercase();
        // Direct lookup
        if let Some(ph) = self.cmudict.get(&upper) {
            return Some(ph);
        }
        // Strip trailing punctuation
        let stripped = upper.trim_end_matches(|c: char| !c.is_alphabetic());
        if stripped != upper {
            if let Some(ph) = self.cmudict.get(stripped) {
                return Some(ph);
            }
        }
        // Strip 's possessive
        if let Some(base) = upper.strip_suffix("'S") {
            if let Some(ph) = self.cmudict.get(base) {
                return Some(ph);
            }
        }
        None
    }

    /// Spell out an unknown word letter-by-letter using ARPABET.
    /// Each letter maps to its English name pronunciation.
    fn spell_arpabet(word: &str) -> String {
        const LETTER_PHONES: [(&str, &str); 26] = [
            ("A", "EY1"), ("B", "B IY1"), ("C", "S IY1"), ("D", "D IY1"),
            ("E", "IY1"), ("F", "EH1 F"), ("G", "JH IY1"), ("H", "EY1 CH"),
            ("I", "AY1"), ("J", "JH EY1"), ("K", "K EY1"), ("L", "EH1 L"),
            ("M", "EH1 M"), ("N", "EH1 N"), ("O", "OW1"), ("P", "P IY1"),
            ("Q", "K Y UW1"), ("R", "AA1 R"), ("S", "EH1 S"), ("T", "T IY1"),
            ("U", "Y UW1"), ("V", "V IY1"), ("W", "D AH1 B AH0 L Y UW1"),
            ("X", "EH1 K S"), ("Y", "W AY1"), ("Z", "Z IY1"),
        ];
        word.to_uppercase()
            .chars()
            .filter_map(|c| {
                let idx = (c as u8).wrapping_sub(b'A') as usize;
                LETTER_PHONES.get(idx).map(|(_, ph)| *ph)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Convert Chinese text to phonemes with per-character phoneme counts (word2ph).
    ///
    /// Uses jieba word segmentation + full ToneSandhi rules (不/一/轻声/三声连读),
    /// matching the Python GPT-SoVITS text frontend.
    ///
    /// Returns (phoneme_string, word2ph) where word2ph[i] = number of phonemes for BERT
    /// content token i (after CLS/SEP removal). Includes ALL characters (Chinese and
    /// punctuation/spaces), with 0 for characters that produce no phonemes.
    pub fn convert_chinese_with_word2ph(&self, text: &str) -> Result<(String, Vec<usize>)> {
        // Step 1: jieba POS tagging
        let tags = self.jieba.tag(text, true);
        let seg: Vec<(String, String)> = tags
            .iter()
            .map(|t| (t.word.to_string(), t.tag.to_string()))
            .collect();

        // Step 2: pre-merge for cross-word sandhi contexts
        let merged = self.tone_sandhi.pre_merge_for_modify(seg);

        // Step 3: process each merged word
        let mut phonemes: Vec<String> = Vec::new();
        let mut word2ph: Vec<usize> = Vec::new();

        for (word, pos) in &merged {
            // English words (jieba POS "eng" or all ASCII letters): use English G2P.
            // In mixed Chinese-English text, English words would otherwise produce no
            // phonemes because to_pinyin() returns None for ASCII characters.
            let is_english = pos == "eng"
                || (!word.is_empty() && word.chars().all(|c| c.is_ascii_alphabetic() || c == '\''));

            if is_english {
                let eng_ph_str = self.english_word_to_phonemes(word);
                let eng_phonemes: Vec<String> = eng_ph_str
                    .split_whitespace()
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();
                let n_chars = word.chars().count().max(1);
                let n_ph = eng_phonemes.len();

                if n_ph == 0 {
                    // No phonemes produced (e.g. single punctuation): all chars contribute 0
                    word2ph.extend(std::iter::repeat(0).take(n_chars));
                } else {
                    for ph in eng_phonemes {
                        phonemes.push(ph);
                    }
                    // Distribute phoneme count across characters.
                    // First char carries the bulk; BERT alignment handles any length mismatch.
                    let per_char = n_ph / n_chars;
                    let remainder = n_ph % n_chars;
                    for i in 0..n_chars {
                        word2ph.push(per_char + if i < remainder { 1 } else { 0 });
                    }
                }
                continue;
            }

            // Chinese: Collect (base_pinyin, raw_tone) for each character in this word
            let char_data: Vec<Option<(String, u32)>> = word.as_str()
                .to_pinyin()
                .map(|opt| match opt {
                    None => None,
                    Some(py) => {
                        let s = py.with_tone_num_end().to_string();
                        if s.is_empty() {
                            return None;
                        }
                        if let Some(last) = s.chars().last() {
                            if last.is_ascii_digit() {
                                let base = s[..s.len() - 1].to_string();
                                let tone = last.to_digit(10).unwrap_or(5) as u32;
                                Some((base, tone))
                            } else {
                                Some((s, 5u32))
                            }
                        } else {
                            None
                        }
                    }
                })
                .collect();

            // Build tones vec (0 for non-Chinese chars) and apply sandhi
            let mut tones: Vec<u32> = char_data
                .iter()
                .map(|opt| opt.as_ref().map(|(_, t)| *t).unwrap_or(0))
                .collect();
            self.tone_sandhi.modified_tone(word, pos, &mut tones, &self.jieba);

            // Convert each char to phonemes and record word2ph
            for (i, opt) in char_data.iter().enumerate() {
                match opt {
                    None => word2ph.push(0),
                    Some((base, _)) => {
                        let tone = tones[i];
                        let before = phonemes.len();
                        if let Some(&(initial, final_base)) = self.pinyin_map.get(base.as_str()) {
                            let final_str = format!("{}{}", final_base, tone);
                            if !initial.is_empty() {
                                phonemes.push(initial.to_string());
                            }
                            phonemes.push(final_str);
                        } else {
                            phonemes.push(format!("{}{}", base, tone));
                        }
                        word2ph.push(phonemes.len() - before);
                    }
                }
            }
        }

        let phoneme_str = if phonemes.is_empty() {
            text.chars().map(|c| format!("[{}]", c)).collect()
        } else {
            phonemes.join(" ")
        };

        Ok((phoneme_str, word2ph))
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

    fn convert_chinese(&self, text: &str) -> Result<String> {
        let (phonemes, _) = self.convert_chinese_with_word2ph(text)?;
        Ok(phonemes)
    }

    /// Convert English text to ARPABET phonemes via CMUdict lookup.
    /// Words not found in CMUdict are spelled letter-by-letter.
    fn convert_english(&self, text: &str) -> Result<String> {
        let phonemes: Vec<String> = text
            .split_whitespace()
            .map(|word| self.english_word_to_phonemes(word))
            .collect();
        Ok(phonemes.join(" "))
    }

    /// Convert one English word to space-separated ARPABET phonemes.
    ///
    /// Strategy (in order):
    /// 1. CMUdict lookup (covers ~124K words)
    /// 2. Try hyphen-split parts individually (e.g. "state-of-the-art")
    /// 3. Spell letter-by-letter as fallback for abbreviations / proper nouns
    fn english_word_to_phonemes(&self, word: &str) -> String {
        // Strip surrounding punctuation for lookup, but remember to not lose it
        let alpha_word: String = word.chars().filter(|c| c.is_alphabetic() || *c == '\'').collect();
        if alpha_word.is_empty() {
            return String::new();
        }

        // Direct CMUdict lookup
        if let Some(phones) = self.cmudict_lookup(&alpha_word) {
            return phones.to_string();
        }

        // Hyphenated compound: look up each part
        if alpha_word.contains('-') {
            let parts: Vec<String> = alpha_word
                .split('-')
                .map(|part| {
                    self.cmudict_lookup(part)
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| Self::spell_arpabet(part))
                })
                .collect();
            if !parts.is_empty() {
                return parts.join(" ");
            }
        }

        // OOV fallback: spell letter-by-letter
        Self::spell_arpabet(&alpha_word)
    }

    /// Convert Japanese text to phonemes
    fn convert_japanese(&self, text: &str) -> Result<String> {
        let phonemes = text
            .chars()
            .map(|c| format!("[{}]", c))
            .collect();

        Ok(phonemes)
    }

    /// Convert Korean text to phonemes
    fn convert_korean(&self, text: &str) -> Result<String> {
        let phonemes = text
            .chars()
            .map(|c| format!("[{}]", c))
            .collect();

        Ok(phonemes)
    }

    /// Convert Cantonese to Jyutping phonemes
    fn convert_cantonese(&self, text: &str) -> Result<String> {
        self.convert_chinese(text)
    }

    /// Auto-detect language and convert
    fn convert_auto(&self, text: &str) -> Result<String> {
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
    fn test_g2p_chinese_initials_finals() {
        let converter = G2PConverter::new().unwrap();
        let result = converter.convert("你好", Language::Chinese).unwrap();
        // Tone sandhi: 你(3) + 好(3) → ni2 hao3
        assert!(result.contains("n "), "should have initial 'n'");
        assert!(result.contains("ao"), "should have final 'ao'");
        // Tone sandhi: ni3 + hao3 → ni2 + hao3
        assert_eq!(result, "n i2 h ao3", "third-tone sandhi should apply");
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

    #[test]
    fn test_bu_sandhi() {
        let converter = G2PConverter::new().unwrap();
        // 不怕: 不(4th-tone follows) → tone 2
        let result = converter.convert("不怕", Language::Chinese).unwrap();
        // 不 is bu4 normally; before 怕(pa4) → bu2. 怕 stays pa4.
        assert!(result.contains("u2"), "不 before 4th tone should become tone 2: got {}", result);
    }

    #[test]
    fn test_yi_sandhi() {
        let converter = G2PConverter::new().unwrap();
        // 一天: 一 before non-4th-tone → tone 4
        let result = converter.convert("一天", Language::Chinese).unwrap();
        assert!(result.contains("i4"), "一 before non-4th-tone should become tone 4: got {}", result);

        // 一段: 一 before 4th tone → tone 2
        let result2 = converter.convert("一段", Language::Chinese).unwrap();
        assert!(result2.contains("i2"), "一 before 4th-tone should become tone 2: got {}", result2);
    }

    #[test]
    fn test_neural_sandhi_word() {
        let converter = G2PConverter::new().unwrap();
        // 知识: in must_neural_tone_words → last char → tone 5
        let (phones, _) = converter.convert_chinese_with_word2ph("知识").unwrap();
        assert!(phones.contains("ir5"), "知识 last char should be tone 5: got {}", phones);
    }

    #[test]
    fn test_three_sandhi_three_char() {
        let converter = G2PConverter::new().unwrap();
        // 所有人: 所有(tone3+tone3) / 人(tone2) — sub1 all-tone-3 len==2 → first char → tone 2
        // 所=suo3, 有=you3, 人=ren2 — not all tone 3, else branch
        // split "所有人" → ["所有","人"] (2+1), t1=[3,3], t2=[2]
        // t1_all3=true, len==2 → tones[0]=2
        let (phones, _) = converter.convert_chinese_with_word2ph("所有人").unwrap();
        // 所 should be tone 2
        assert!(phones.contains("uo2"), "所 should change to tone 2 in 所有人: got {}", phones);
    }
}
