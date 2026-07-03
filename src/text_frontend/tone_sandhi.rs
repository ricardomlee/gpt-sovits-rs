use jieba_rs::Jieba;
use pinyin::ToPinyin;
use std::collections::HashSet;

static MUST_NEURAL_WORDS: &[&str] = &[
    "麻烦", "麻利", "鸳鸯", "高粱", "骨头", "骆驼", "马虎", "首饰", "馒头", "馄饨", "风筝", "难为",
    "队伍", "阔气", "闺女", "门道", "锄头", "铺盖", "铃铛", "铁匠", "钥匙", "里脊", "里头", "部分",
    "那么", "道士", "造化", "迷糊", "连累", "这么", "这个", "运气", "过去", "软和", "转悠", "踏实",
    "跳蚤", "跟头", "趔趄", "财主", "豆腐", "讲究", "记性", "记号", "认识", "规矩", "见识", "裁缝",
    "补丁", "衣裳", "衣服", "衙门", "街坊", "行李", "行当", "蛤蟆", "蘑菇", "薄荷", "葫芦", "葡萄",
    "萝卜", "荸荠", "苗条", "苗头", "苍蝇", "芝麻", "舒服", "舒坦", "舌头", "自在", "膏药", "脾气",
    "脑袋", "脊梁", "能耐", "胳膊", "胭脂", "胡萝", "胡琴", "胡同", "聪明", "耽误", "耽搁", "耷拉",
    "耳朵", "老爷", "老实", "老婆", "老头", "老太", "翻腾", "罗嗦", "罐头", "编辑", "结实", "红火",
    "累赘", "糨糊", "糊涂", "精神", "粮食", "簸箕", "篱笆", "算计", "算盘", "答应", "笤帚", "笑语",
    "笑话", "窟窿", "窝囊", "窗户", "稳当", "稀罕", "称呼", "秧歌", "秀气", "秀才", "福气", "祖宗",
    "砚台", "码头", "石榴", "石头", "石匠", "知识", "眼睛", "眯缝", "眨巴", "眉毛", "相声", "盘算",
    "白净", "痢疾", "痛快", "疟疾", "疙瘩", "疏忽", "畜生", "生意", "甘蔗", "琵琶", "琢磨", "琉璃",
    "玻璃", "玫瑰", "玄乎", "狐狸", "状元", "特务", "牲口", "牙碜", "牌楼", "爽快", "爱人", "热闹",
    "烧饼", "烟筒", "烂糊", "点心", "炊帚", "灯笼", "火候", "漂亮", "滑溜", "溜达", "温和", "清楚",
    "消息", "浪头", "活泼", "比方", "正经", "欺负", "模糊", "槟榔", "棺材", "棒槌", "棉花", "核桃",
    "栅栏", "柴火", "架势", "枕头", "枇杷", "机灵", "本事", "木头", "木匠", "朋友", "月饼", "月亮",
    "暖和", "明白", "时候", "新鲜", "故事", "收拾", "收成", "提防", "挖苦", "挑剔", "指甲", "指头",
    "拾掇", "拳头", "拨弄", "招牌", "招呼", "抬举", "护士", "折腾", "扫帚", "打量", "打算", "打点",
    "打扮", "打听", "打发", "扎实", "扁担", "戒指", "懒得", "意识", "意思", "情形", "悟性", "怪物",
    "思量", "怎么", "念头", "念叨", "快活", "忙活", "志气", "心思", "得罪", "张罗", "弟兄", "开通",
    "应酬", "庄稼", "干事", "帮手", "帐篷", "希罕", "师父", "师傅", "巴结", "巴掌", "差事", "工夫",
    "岁数", "屁股", "尾巴", "少爷", "小气", "小伙", "将就", "对头", "对付", "寡妇", "家伙", "客气",
    "实在", "官司", "学问", "学生", "字号", "嫁妆", "媳妇", "媒人", "婆家", "娘家", "委屈", "姑娘",
    "姐夫", "妯娌", "妥当", "妖精", "奴才", "女婿", "头发", "太阳", "大爷", "大方", "大意", "大夫",
    "多少", "多么", "外甥", "壮实", "地道", "地方", "在乎", "困难", "嘴巴", "嘱咐", "嘟囔", "嘀咕",
    "喜欢", "喇嘛", "喇叭", "商量", "唾沫", "哑巴", "哈欠", "哆嗦", "咳嗽", "和尚", "告诉", "告示",
    "含糊", "吓唬", "后头", "名字", "名堂", "合同", "吆喝", "叫唤", "口袋", "厚道", "厉害", "千斤",
    "包袱", "包涵", "匀称", "勤快", "动静", "动弹", "功夫", "力气", "前头", "刺猬", "刺激", "别扭",
    "利落", "利索", "利害", "分析", "出息", "凑合", "凉快", "冷战", "冤枉", "冒失", "养活", "关系",
    "先生", "兄弟", "便宜", "使唤", "佩服", "作坊", "体面", "位置", "似的", "伙计", "休息", "什么",
    "人家", "亲戚", "亲家", "交情", "云彩", "事情", "买卖", "主意", "丫头", "丧气", "两口", "东西",
    "东家", "世故", "不由", "不在", "下水", "下巴", "上头", "上司", "丈夫", "丈人", "一辈", "那个",
    "菩萨", "父亲", "母亲", "咕噜", "邋遢", "费用", "冤家", "甜头", "介绍", "荒唐", "大人", "泥鳅",
    "幸福", "熟悉", "计划", "扑腾", "蜡烛", "姥爷", "照顾", "喉咙", "吉他", "弄堂", "蚂蚱", "凤凰",
    "拖沓", "寒碜", "糟蹋", "倒腾", "报复", "逻辑", "盘缠", "喽啰", "牢骚", "咖喱", "扫把", "惦记",
];

static MUST_NOT_NEURAL_WORDS: &[&str] = &[
    "男子",
    "女子",
    "分子",
    "原子",
    "量子",
    "莲子",
    "石子",
    "瓜子",
    "电子",
    "人人",
    "虎虎",
    "幺幺",
    "干嘛",
    "学子",
    "哈哈",
    "数数",
    "袅袅",
    "局地",
    "以下",
    "娃哈哈",
    "花花草草",
    "留得",
    "耕地",
    "想想",
    "熙熙",
    "攘攘",
    "卵子",
    "死死",
    "冉冉",
    "恳恳",
    "佼佼",
    "吵吵",
    "打打",
    "考考",
    "整整",
    "莘莘",
    "落地",
    "算子",
    "家家户户",
    "青青",
];

pub struct ToneSandhi {
    must_neural: HashSet<&'static str>,
    must_not_neural: HashSet<&'static str>,
}

impl ToneSandhi {
    pub fn new() -> Self {
        Self {
            must_neural: MUST_NEURAL_WORDS.iter().cloned().collect(),
            must_not_neural: MUST_NOT_NEURAL_WORDS.iter().cloned().collect(),
        }
    }

    /// Get tones for all chars in word. Returns 0 for non-Chinese chars.
    fn word_tones(word: &str) -> Vec<u32> {
        word.to_pinyin()
            .map(|opt| match opt {
                None => 0,
                Some(py) => {
                    let s = py.with_tone_num_end();
                    s.chars().last().and_then(|c| c.to_digit(10)).unwrap_or(5)
                }
            })
            .collect()
    }

    /// True iff there is at least one Chinese char and all Chinese chars are tone 3.
    fn all_tone_three(tones: &[u32]) -> bool {
        let mut has_chinese = false;
        for &t in tones {
            if t > 0 {
                has_chinese = true;
                if t != 3 {
                    return false;
                }
            }
        }
        has_chinese
    }

    fn is_reduplication(word: &str) -> bool {
        let chars: Vec<char> = word.chars().collect();
        chars.len() == 2 && chars[0] == chars[1]
    }

    fn last_n_chars(word: &str, n: usize) -> String {
        word.chars()
            .rev()
            .take(n)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Split word into two sub-words using jieba cut_for_search, matching Python's _split_word.
    fn split_word(word: &str, jieba: &Jieba) -> (String, String) {
        let sub_words = jieba.cut_for_search(word, true);
        if sub_words.is_empty() {
            return (word.to_string(), String::new());
        }
        let mut sorted = sub_words.clone();
        sorted.sort_by_key(|s| s.chars().count());
        let first_sub = sorted[0];

        let word_chars: Vec<char> = word.chars().collect();
        let first_sub_chars: Vec<char> = first_sub.chars().collect();
        let first_sub_len = first_sub_chars.len();

        let first_begin_char = word_chars
            .windows(first_sub_len)
            .position(|w| w == first_sub_chars.as_slice())
            .unwrap_or(0);

        if first_begin_char == 0 {
            let second: String = word_chars[first_sub_len..].iter().collect();
            (first_sub.to_string(), second)
        } else {
            let second_end = word_chars.len().saturating_sub(first_sub_len);
            let second: String = word_chars[..second_end].iter().collect();
            (second, first_sub.to_string())
        }
    }

    /// "不" sandhi rules.
    fn bu_sandhi(word: &str, tones: &mut [u32]) {
        let chars: Vec<char> = word.chars().collect();
        // V不V pattern: middle "不" → neutral tone 5
        if chars.len() == 3 && chars.get(1) == Some(&'不') && tones.len() > 1 {
            tones[1] = 5;
            return;
        }
        // "不" before tone 4 → tone 2
        for (i, &c) in chars.iter().enumerate() {
            if c == '不' && i + 1 < tones.len() && tones[i + 1] == 4 {
                tones[i] = 2;
            }
        }
    }

    /// "一" sandhi rules.
    fn yi_sandhi(word: &str, tones: &mut [u32]) {
        let chars: Vec<char> = word.chars().collect();
        if !chars.contains(&'一') {
            return;
        }
        // Number sequences: keep original tone
        if chars
            .iter()
            .filter(|&&c| c != '一')
            .all(|c| c.is_ascii_digit())
        {
            return;
        }
        // V一V: e.g. 看一看 → "一" → neutral
        if chars.len() == 3
            && chars.get(1) == Some(&'一')
            && chars[0] == chars[2]
            && tones.len() >= 2
        {
            tones[1] = 5;
            return;
        }
        // 第一: "一" → tone 1
        if word.starts_with("第一") && tones.len() >= 2 {
            tones[1] = 1;
            return;
        }
        let punc = "：，；。？！\u{201C}\u{201D}\u{2018}\u{2019}':,;.?!";
        for (i, &c) in chars.iter().enumerate() {
            if c == '一' && i + 1 < chars.len() && i < tones.len() {
                let next_char = chars[i + 1];
                if i + 1 < tones.len() && tones[i + 1] == 4 {
                    tones[i] = 2;
                } else if !punc.contains(next_char) {
                    tones[i] = 4;
                }
            }
        }
    }

    /// Neutral tone (轻声) sandhi rules.
    fn neural_sandhi(&self, word: &str, pos: &str, tones: &mut [u32], jieba: &Jieba) {
        let chars: Vec<char> = word.chars().collect();
        let n = tones.len();
        if n == 0 {
            return;
        }

        let pos0 = pos.chars().next().unwrap_or(' ');
        let modal_particles = "吧呢哈啊呐噻嘛吖嗨哦哒额滴哩哟喽啰耶喔诶";

        // Reduplication: consecutive same chars in n/v/a POS → second → neutral
        for j in 1..chars.len() {
            if j < n
                && chars[j] == chars[j - 1]
                && "nva".contains(pos0)
                && !self.must_not_neural.contains(word)
            {
                tones[j] = 5;
            }
        }

        let last_char = chars.last().copied();
        let second_to_last = if chars.len() >= 2 {
            chars.get(chars.len() - 2).copied()
        } else {
            None
        };
        let ge_char_idx: Option<usize> = chars.iter().position(|&c| c == '个');

        if last_char
            .map(|c| modal_particles.contains(c) || "的地得".contains(c))
            .unwrap_or(false)
        {
            tones[n - 1] = 5;
        } else if chars.len() == 1
            && "了着过".contains(chars[0])
            && matches!(pos, "ul" | "uz" | "ug")
        {
            tones[0] = 5;
        } else if chars.len() > 1
            && ((last_char.map(|c| "们子".contains(c)).unwrap_or(false)
                && matches!(pos, "r" | "n")
                && !self.must_not_neural.contains(word))
                || (last_char.map(|c| "上下里".contains(c)).unwrap_or(false)
                    && "slf".contains(pos0))
                || (last_char.map(|c| "来去".contains(c)).unwrap_or(false)
                    && second_to_last
                        .map(|c| "上下进出回过起开".contains(c))
                        .unwrap_or(false)))
        {
            tones[n - 1] = 5;
        } else if let Some(gi) = ge_char_idx {
            let do_neutral = word == "个"
                || (gi >= 1 && {
                    let prev = chars[gi - 1];
                    prev.is_ascii_digit() || "几有两半多各整每做是".contains(prev)
                });
            if do_neutral && gi < n {
                tones[gi] = 5;
            }
        } else {
            let last2 = Self::last_n_chars(word, 2);
            if self.must_neural.contains(word) || self.must_neural.contains(last2.as_str()) {
                tones[n - 1] = 5;
            }
        }

        // Sub-word check (always runs, matches Python's unconditional block)
        let (sub1, sub2) = Self::split_word(word, jieba);
        let sub1_len = sub1.chars().count();
        let sub2_len = sub2.chars().count();

        if !sub1.is_empty() && sub1_len > 0 && sub1_len <= n {
            let last2_sub1 = Self::last_n_chars(&sub1, 2);
            if self.must_neural.contains(sub1.as_str())
                || self.must_neural.contains(last2_sub1.as_str())
            {
                tones[sub1_len - 1] = 5;
            }
        }
        if !sub2.is_empty() && sub2_len > 0 && sub1_len + sub2_len <= n {
            let last2_sub2 = Self::last_n_chars(&sub2, 2);
            if self.must_neural.contains(sub2.as_str())
                || self.must_neural.contains(last2_sub2.as_str())
            {
                tones[sub1_len + sub2_len - 1] = 5;
            }
        }
    }

    /// Third-tone (三声) sandhi rules.
    fn three_sandhi(word: &str, tones: &mut [u32], jieba: &Jieba) {
        let word_len = word.chars().count();
        let n = tones.len();
        if n == 0 {
            return;
        }

        if word_len == 2 && Self::all_tone_three(tones) {
            tones[0] = 2;
        } else if word_len == 3 {
            let (sub1, _sub2) = Self::split_word(word, jieba);
            let sub1_len = sub1.chars().count();

            if Self::all_tone_three(tones) {
                // All 3 chars are tone 3
                if sub1_len == 2 && n >= 2 {
                    // 2+1 split: first two → tone 2 (e.g. 蒙古/包)
                    tones[0] = 2;
                    tones[1] = 2;
                } else if sub1_len == 1 && n >= 2 {
                    // 1+2 split: second → tone 2 (e.g. 纸/老虎)
                    tones[1] = 2;
                }
            } else if sub1_len > 0 && sub1_len < n {
                // Mixed tones: process per sub-word
                let t1_all3 = Self::all_tone_three(&tones[..sub1_len]) && sub1_len == 2;
                let t1_last_tone = tones[sub1_len - 1]; // read before any modification
                let t2_first_tone = tones[sub1_len];
                let t2_all3 = Self::all_tone_three(&tones[sub1_len..]);

                // 所有/人: sub1 all tone 3 and len==2 → sub1[0] → tone 2
                if t1_all3 {
                    tones[0] = 2;
                }
                // 好/喜欢: sub2[0]==3, sub1.last==3, not all sub2 tone 3 → sub1.last → tone 2
                if !t2_all3 && t2_first_tone == 3 && t1_last_tone == 3 {
                    tones[sub1_len - 1] = 2;
                }
            }
        } else if word_len == 4 && n >= 4 {
            // Idiom: split into two 2-char groups
            let (t1, t2) = tones.split_at_mut(2);
            if Self::all_tone_three(t1) {
                t1[0] = 2;
            }
            if Self::all_tone_three(t2) {
                t2[0] = 2;
            }
        }
    }

    /// Apply all sandhi rules to a word's tones in-place.
    pub fn modified_tone(&self, word: &str, pos: &str, tones: &mut [u32], jieba: &Jieba) {
        Self::bu_sandhi(word, tones);
        Self::yi_sandhi(word, tones);
        self.neural_sandhi(word, pos, tones, jieba);
        Self::three_sandhi(word, tones, jieba);
    }

    // ---- Pre-merge functions ----

    /// Merge "不" with the following word.
    fn merge_bu(seg: Vec<(String, String)>) -> Vec<(String, String)> {
        let mut new_seg: Vec<(String, String)> = Vec::new();
        let mut last_word = String::new();
        for (word, pos) in seg {
            let word = if last_word == "不" {
                format!("不{}", word)
            } else {
                word
            };
            if word != "不" {
                new_seg.push((word.clone(), pos));
            }
            last_word = word;
        }
        if last_word == "不" {
            new_seg.push(("不".to_string(), "d".to_string()));
        }
        new_seg
    }

    /// Merge V一V patterns and "一" with following words.
    fn merge_yi(seg: Vec<(String, String)>) -> Vec<(String, String)> {
        // Function 1: V一V (e.g. 看一看)
        let orig = seg;
        let orig_len = orig.len();
        let mut new_seg: Vec<(String, String)> = Vec::new();
        let mut i = 0;

        while i < orig_len {
            let (ref word, ref pos) = orig[i];
            let mut merged = false;

            if i >= 1 && word == "一" && i + 1 < orig_len {
                let last = new_seg
                    .last()
                    .cloned()
                    .unwrap_or_else(|| orig[i - 1].clone());
                let next = &orig[i + 1];
                if last.0 == next.0 && last.1 == "v" && next.1 == "v" {
                    let combined = format!("{}一{}", last.0, next.0);
                    if let Some(entry) = new_seg.last_mut() {
                        entry.0 = combined;
                    }
                    i += 2;
                    merged = true;
                }
            }
            if !merged {
                new_seg.push((word.clone(), pos.clone()));
                i += 1;
            }
        }

        // Function 2: merge "一" with following word
        let seg = new_seg;
        let mut new_seg: Vec<(String, String)> = Vec::new();
        for (word, pos) in seg {
            if let Some(last) = new_seg.last_mut() {
                if last.0 == "一" {
                    last.0 = format!("一{}", word);
                    continue;
                }
            }
            new_seg.push((word, pos));
        }
        new_seg
    }

    /// Merge consecutive identical words (e.g. 奶+奶 → 奶奶).
    fn merge_reduplication(seg: Vec<(String, String)>) -> Vec<(String, String)> {
        let mut new_seg: Vec<(String, String)> = Vec::new();
        for (word, pos) in seg {
            if let Some(last) = new_seg.last_mut() {
                if last.0 == word {
                    last.0 = format!("{}{}", last.0, word);
                    continue;
                }
            }
            new_seg.push((word, pos));
        }
        new_seg
    }

    /// Merge adjacent all-tone-3 words where combined length ≤ 3.
    fn merge_continuous_three_tones(seg: Vec<(String, String)>) -> Vec<(String, String)> {
        let tones_list: Vec<Vec<u32>> = seg.iter().map(|(w, _)| Self::word_tones(w)).collect();
        let mut new_seg: Vec<(String, String)> = Vec::new();
        let mut merge_last = vec![false; seg.len()];

        for i in 0..seg.len() {
            let (ref word, ref pos) = seg[i];
            if i >= 1
                && Self::all_tone_three(&tones_list[i - 1])
                && Self::all_tone_three(&tones_list[i])
                && !merge_last[i - 1]
            {
                let prev_len = seg[i - 1].0.chars().count();
                let curr_len = word.chars().count();
                if !Self::is_reduplication(&seg[i - 1].0) && prev_len + curr_len <= 3 {
                    if let Some(last) = new_seg.last_mut() {
                        last.0 = format!("{}{}", last.0, word);
                        merge_last[i] = true;
                        continue;
                    }
                }
            }
            new_seg.push((word.clone(), pos.clone()));
        }
        new_seg
    }

    /// Merge adjacent words where the boundary chars (last of prev, first of curr) are both tone 3.
    fn merge_continuous_three_tones_2(seg: Vec<(String, String)>) -> Vec<(String, String)> {
        let tones_list: Vec<Vec<u32>> = seg.iter().map(|(w, _)| Self::word_tones(w)).collect();
        let mut new_seg: Vec<(String, String)> = Vec::new();
        let mut merge_last = vec![false; seg.len()];

        for i in 0..seg.len() {
            let (ref word, ref pos) = seg[i];
            let prev_last_tone = if i >= 1 {
                tones_list[i - 1]
                    .iter()
                    .rfind(|&&t| t > 0)
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            };
            let curr_first_tone = tones_list[i].iter().find(|&&t| t > 0).copied().unwrap_or(0);

            if i >= 1 && prev_last_tone == 3 && curr_first_tone == 3 && !merge_last[i - 1] {
                let prev_len = seg[i - 1].0.chars().count();
                let curr_len = word.chars().count();
                if !Self::is_reduplication(&seg[i - 1].0) && prev_len + curr_len <= 3 {
                    if let Some(last) = new_seg.last_mut() {
                        last.0 = format!("{}{}", last.0, word);
                        merge_last[i] = true;
                        continue;
                    }
                }
            }
            new_seg.push((word.clone(), pos.clone()));
        }
        new_seg
    }

    /// Merge "儿" with the preceding word (erhua).
    fn merge_er(seg: Vec<(String, String)>) -> Vec<(String, String)> {
        let mut new_seg: Vec<(String, String)> = Vec::new();
        for (i, (word, pos)) in seg.iter().enumerate() {
            if i >= 1 && word == "儿" && !new_seg.is_empty() && seg[i - 1].0 != "#" {
                if let Some(last) = new_seg.last_mut() {
                    last.0 = format!("{}儿", last.0);
                    continue;
                }
            }
            new_seg.push((word.clone(), pos.clone()));
        }
        new_seg
    }

    /// Run all pre-merge passes in the same order as Python.
    pub fn pre_merge_for_modify(&self, seg: Vec<(String, String)>) -> Vec<(String, String)> {
        let seg = Self::merge_bu(seg);
        let seg = Self::merge_yi(seg);
        let seg = Self::merge_reduplication(seg);
        let seg = Self::merge_continuous_three_tones(seg);
        let seg = Self::merge_continuous_three_tones_2(seg);
        Self::merge_er(seg)
    }
}

impl Default for ToneSandhi {
    fn default() -> Self {
        Self::new()
    }
}
