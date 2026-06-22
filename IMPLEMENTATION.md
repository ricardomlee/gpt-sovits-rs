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

### HuBERT 重采样

使用 `soxr = "0.6"`（libsoxr HQ），与 librosa 默认 soxr_hq 完全一致：
- 输入任意采样率 → 16kHz
- 重采样后追加 9600 个零样本（0.6s 静音，匹配 Python 预处理）
- VQ prompt tokens 与 Python 计算结果 **20/20 一致**

实现位于 `src/models/hubert.rs::resample_sinc()`。

### BERT 对齐（project_and_align_bert）

ONNX BERT 输出 `[1, seq+2, 1024]`（含 CLS/SEP）：
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

### EOS 停止条件（与 Python 匹配）

```
step < 11:  logits 中屏蔽 EOS（audio_vocab_size-1），保证最短生成长度
step >= 11: argmax(logits) == EOS  OR  sampled == EOS → 停止
```

### 中文三声连读变调

相邻两个三声（tone 3）的前一个变为二声（tone 2），位于 `src/text_frontend/g2p.rs`：
```
你好 (ni3 + hao3) → (ni2 + hao3)
```

## 数值对齐状态

| 步骤 | Python 参考 | Rust 状态 |
|------|------------|-----------|
| G2P 音素 | `symbols_v2.json` 732 符号 | 完全一致 |
| BERT 特征 | ONNX 同模型 | 数值一致 |
| HuBERT 重采样 | librosa (soxr HQ) | VQ tokens 20/20 |
| VQ prompt tokens | Python softmax + argmin | 129/129 一致 |
| GPT 生成 token 数 | 83–123 个 | 同范围（受采样随机性影响） |
| SoVITS 音频 RMS | ~0.06–0.08 | 同范围 |

## 依赖说明

| crate | 版本 | 用途 |
|-------|------|------|
| `candle-core` | 0.10 | Tensor + CUDA |
| `ort` | 2.0-rc.12 | ONNX Runtime（BERT / HuBERT） |
| `soxr` | 0.6 | 音频重采样（需要系统包 `libsoxr-dev`） |
| `hound` | 3.5 | WAV I/O |
| `jieba-rs` | 0.6 | 中文分词 |
| `tokenizers` | 0.22 | BERT tokenizer |
| `safetensors` | 0.4 | 权重加载 |

## 已知限制

- 变调规则只实现了相邻两个三声（3+3 → 2+3），未处理"不/一"变调和连续三声链
- SoVITS decode 时参考音频的 mel 频谱会影响解码前几帧，可能听到参考音频末尾内容
