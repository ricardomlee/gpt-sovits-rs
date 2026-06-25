# GPT-SoVITS-RS 技术实现说明

## 推理流程（与 Python 原版对齐）

```
ref_text ──→ G2P ──→ ref_phoneme_ids
                          │
target_text → G2P ──→ target_phoneme_ids
                          │
                     拼接 phoneme_ids = [ref | target]
                          │
BERT(ref_text)   ──→ project(1024→512) ──→ word2ph 对齐 ──→ ref_bert_aligned
BERT(target_text)──→ project(1024→512) ──→ word2ph 对齐 ──→ target_bert_aligned
                                                                    │
                                              combined_bert = cat([ref, target], dim=1)
                                                                    │
ref_audio ──→ HuBERT(soxr 16kHz) ──→ VQ 量化 ──→ prompt_tokens
                   │
                   └──→ mel 频谱 ──→ enc_q ──→ 音色条件

GPT(phoneme_ids, prompt_tokens, combined_bert) ──→ semantic_tokens

SoVITS(semantic_tokens, target_phoneme_ids, ref_mel) ──→ 波形
```

## 关键实现细节

### ref_text 拼接（最关键）

Python 原版在调用 GPT 前拼接参考文本和目标文本的音素序列和 BERT 特征：
```python
phones = phones1 + phones2          # ref_phones + target_phones
bert = torch.cat([bert1, bert2], 1) # 分别对齐后拼接
```

Rust 实现位于 `src/inference/mod.rs::inference()`。缺少这一步时，GPT 只生成约 11–14 个 token（~0.5 秒静音）。

### HuBERT 重采样与静音填充

使用 `soxr = "0.6"`（libsoxr HQ），与 librosa 默认 soxr_hq 完全一致：
- 输入任意采样率 → 16kHz
- 重采样后追加 9600 个零样本（0.6s 静音，匹配 Python 预处理）
- **无论输入是否已为 16kHz，静音填充均须执行**（`src/models/hubert.rs::load_audio()`）

> **根因说明**：Python 在调用 HuBERT 前无条件追加 9600 样本，不区分是否重采样。若 16kHz 音频跳过此步骤，VQ prompt tokens 会变短，导致 GPT 把参考音频末尾内容（如"而中道崩殂"）当作目标内容生成，即输出音频前缀混入参考音频语义。

实现位于 `src/models/hubert.rs`。VQ prompt tokens 与 Python 计算结果 **20/20 一致**（包含 32kHz 和 16kHz 输入）。

### BERT 对齐（project_and_align_bert）

纯 Candle BERT 输出 `[1, seq+2, 1024]`（含 CLS/SEP）：
1. 去除 CLS（第 0 位）和 SEP（最后一位）
2. 线性投影 1024 → 512（`bert_proj` 权重）
3. 按 `word2ph` 展开到音素级（每个汉字对应若干音素）

实现位于 `src/models/gpt.rs::project_and_align_bert()`。

### GPT 生成——两种模式

**`generate_with_prompts_aligned_bert()`（标准模式）**
- 每步对完整序列 `[text_emb + prompt_emb + generated_emb]` 做 forward
- 时间复杂度 O(n²)

**`generate_with_prompts_aligned_bert_kv_cache()`（KV cache 模式）**
- Prefill：对 `[text_emb + prompt_emb]` 做一次完整 forward，填充 KV cache
- 自回归：每步只处理 1 个 token，时间复杂度 O(n)
- mask 策略：prefill 用 hybrid mask（文本双向 + 音频因果），自回归步 mask=None（新 token 看所有缓存）
- **embedding 查表**：`lookup_tokens()` 使用 `embedding.embedding(&ids_flat)` 纯 GPU 执行（`index_select`），无 D2H transfer；ids 须先 flatten 为 1D 再传入（Candle 0.10 要求）
- **D2H transfer 控制**：`sample_token()` 返回 `(sampled_token, argmax)`，两个值从同一次 `to_vec1()` 得到，避免为 argmax EOS 检查再做一次重复 transfer

### SoVITS enc_p——Post-norm 顺序

`src/models/sovits_encp.rs` 中 attention 和 FFN 均使用 **post-norm**（与 Python 一致）：
```
x = LayerNorm(x + Attn(x))   # 先残差加，后 norm
x = LayerNorm(x + FFN(x))
```
Pre-norm（先 norm 再 attention）是常见误写，会导致音质明显下降。

### SoVITS Flow——Fused Gate

`src/models/sovits_flow.rs` 中 `fused_add_tanh_sigmoid_multiply` 的正确实现：
```
in_act = a + b
result = tanh(in_act[:n]) * sigmoid(in_act[n:])
```
错误写法 `tanh(a)*sigmoid(a) + tanh(b)*sigmoid(b)` 与此数学上不等价，会破坏 Flow 解码。

### EOS 停止条件（与 Python 匹配）

```
step < 11:  logits 中屏蔽 EOS（audio_vocab_size-1），保证最短生成长度
step >= 11: argmax(logits) == EOS  OR  sampled == EOS → 停止
```

### 中文变调规则（`src/text_frontend/tone_sandhi.rs`）

**三声连读（`three_sandhi`）**：按 jieba 分词结构感知词边界
- 2字词全三声：前字变二声（你好 → ni2 hao3）
- 3字词全三声：按 1+2 或 2+1 分组，各组首字变二声（蒙古包 2+1 → 2 2 3；纸老虎 1+2 → 3 2 3）
- 4字成语：拆成两个 2字组分别处理
- 混合声调：局部相邻三声对也处理

**"不"变调（`bu_sandhi`）**：
- V不V 模式（看不看）→ 中间"不"变轻声
- 不 + 四声字 → "不"变二声（不对 → bu2 dui4）

**"一"变调（`yi_sandhi`）**：
- V一V 模式（看一看）→ "一"变轻声
- 第一 → "一"保持一声
- 一 + 四声 → "一"变二声；一 + 其他 → "一"变四声
- 纯数字序列中保留原声调

**轻声（`neural_sandhi`）**：叠词（名词/动词/形容词）、语气词（吧/呢/啊…）、的/地/得、了/着/过、们/子、方位词（上/下/里）、"来/去"趋向补语、量词"个"等

## 数值对齐状态

| 步骤 | Python 参考 | Rust 状态 |
|------|------------|-----------|
| G2P 音素 | `symbols_v2.json` 732 符号 | 完全一致 |
| BERT 特征 | Candle chinese-roberta-wwm-ext-large | 数值一致 |
| HuBERT 重采样 | librosa (soxr HQ) | VQ tokens 20/20（32kHz 和 16kHz 输入均一致） |
| VQ prompt tokens | Python softmax + argmin | 129/129 一致 |
| GPT 生成 token 数 | 83–123 个 | 同范围（受采样随机性影响） |
| SoVITS 音频 RMS | ~0.06–0.08 | 同范围 |
| 输出前缀 | 仅目标文本语义 | 已修复（静音填充确保 VQ token 覆盖完整参考音频） |

## 依赖说明

| crate | 版本 | 用途 |
|-------|------|------|
| `candle-core` | 0.10 | Tensor + CUDA |
| `soxr` | 0.6 | 音频重采样（需要系统包 `libsoxr-dev`） |
| `hound` | 3.5 | WAV I/O |
| `jieba-rs` | 0.6 | 中文分词 |
| `tokenizers` | 0.22 | BERT tokenizer |
| `safetensors` | 0.4 | 权重加载 |

## 已知限制

- 三声链超过4字时未进一步细化（但词边界内已结构感知处理）
