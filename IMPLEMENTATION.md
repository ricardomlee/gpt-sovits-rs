# GPT-SoVITS Rust 实现说明

本文描述 1.2.2 的推理实现。
发布范围见 [文档入口](docs/README.md)，操作步骤见 [API](docs/API.md) 和 [模型指南](docs/MODELS.md)。

## 推理流程

```text
参考文本 / 目标文本
  -> 读音标注解析、语言判断、数字规范化
  -> G2P + word2ph                         -> 音素
  -> BERT -> 投影 1024 到 512 -> 音素对齐   -> BERT 条件

参考音频 -> 16 kHz + 尾部静音 -> HuBERT -> VQ -> prompt semantic tokens
参考音频 -> SoVITS 采样率频谱 -> ref_enc ----> speaker conditioning
匹配的 SV embedding（v2Pro）-> sv_emb ------> speaker conditioning

参考及目标音素 / BERT 条件 + prompt tokens
  -> GPT 自回归生成目标 semantic tokens
  -> SoVITS enc_p + flow + decoder（结合 speaker conditioning）
  -> 波形；可逐段合成后加间隔、淡入淡出并拼接
```

参考音色通过 `ref_enc` 编码，`enc_q` 用于训练时的 posterior encoder。
代码中的 `ref_mel` 名称沿用历史，实际输入是 STFT 幅度频谱。

## 文本和参考特征

[speaker.rs](src/inference/speaker.rs) 负责参考缓存和目标特征准备。
参考/目标的音素与对齐后的 BERT 特征分别拼接后送给 GPT。
G2P 与 BERT 使用同一份规范化文本，数字展开和读音覆盖通过原文索引映射保持一致。
1.2.2 的数字处理规则见 [API](docs/API.md)。

[HuBERT](src/models/hubert.rs) 使用 soxr 重采样到 16 kHz，并追加 9600 个零样本。
已经是 16 kHz 的输入也追加静音。另一条参考路径在
[ref_audio.rs](src/inference/ref_audio.rs) 计算 SoVITS 频谱（FFT 2048、hop 640）。
当前参考编码器使用前 704 个频率 bin。

[GPT](src/models/gpt.rs) 对 BERT 特征去除 CLS/SEP，投影到 512 维并根据 `word2ph` 展开。
非中文及无法直接对齐的情形使用回退处理。

每个 Pipeline 最多缓存 16 组参考特征，键包含参考音频路径、参考文本和 SV 路径。
原地修改文件后，需要更换路径、清理缓存或重启来刷新特征。

## GPT 解码

| 模式 | 行为 |
| --- | --- |
| `plain` | 每一步重新计算完整前缀，主要用于对照 |
| `kv` | Prefill 后缓存历史 K/V，每步只输入新 token；attention 仍需访问历史缓存 |
| `cuda-graph` | 静态 KV 加 CUDA Graph，减少 decode 的主机调度开销 |
| `auto` | CUDA F32 默认 Graph，其他设备默认 KV |

Graph 前 32 步使用 eager/static KV，短生成可能在 capture 前结束。首次 replay 做结果检查，
出现不适用情形时有 KV 回退保护。每次生成分别创建 Graph。
`GPT_SOVITS_DISABLE_CUDA_GRAPH=1` 让自动选择退回 KV；旧的 experimental 开关已删除。

Prefill 使用文本双向、音频因果的 hybrid mask；单步 decode 可看到所有有效历史缓存。
采样保留最短生成长度的 EOS 屏蔽，以及 argmax 或采样结果为 EOS 时停止的判断。
提前结束或漏读的样本通过分段日志、ASR 和试听排查。

## SoVITS 和 v2Pro

[sovits.rs](src/models/sovits.rs) 根据权重是否包含 `sv_emb.*` 等层加载 v2Pro 条件分支。
SV 接受 `[20480]` 或 `[1, 20480]`，应与参考音频匹配。
缺少 SV 时使用零向量。性能和发布验收使用与参考音频匹配的 SV。
Rust 转换器可转换已有 embedding，不负责从参考录音计算新的 SV。

EncP 的 attention/FFN 使用 post-norm：`LayerNorm(x + block(x))`。
Flow 的 gate 先做 `a + b` 再分通道执行 tanh/sigmoid。
声码器使用项目内的 SoVITS decoder。BigVGAN 尚未接入主推理路径，v3/v4 尚未支持。

[Config](src/config/mod.rs) 当前把 GPT 与 SoVITS 保持在 F32；`--half` 为兼容选项。
CUDA 不可用时普通配置可能回落 CPU。发布 GPU 基准会显式拒绝这种回落。
依赖版本以 [Cargo.toml](Cargo.toml) 和 [Cargo.lock](Cargo.lock) 为准。

## 服务和验收边界

HTTP 服务共享 BERT/HuBERT，按 GPT/SoVITS 模型对维护有界 LRU（默认两个 Pipeline），
并通过一个推理槽串行处理请求，其余请求排队。
`/tts/stream` 接收完整文本，逐段输出音频；协议见 [API](docs/API.md)。

模型无关测试覆盖文本、配置、缓存、请求和报告处理。真实模型验证需要显式执行
[发布验收](docs/RELEASE_ACCEPTANCE.md) 与 [GPU 基准](docs/PERFORMANCE.md)。
