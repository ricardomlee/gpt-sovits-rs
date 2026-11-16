# GPT-SoVITS Rust 实现说明

本文说明 Rust 推理链路和模块职责。
操作步骤见 [API](docs/API.md) 和 [模型指南](docs/MODELS.md)，代码目录与测试见
[开发指南](docs/DEVELOPMENT.md)。

## 推理流程

```mermaid
flowchart TD
    Text["参考文本 / 目标文本"] --> Frontend["文本前端 + BERT：音素及对齐特征"]
    Frontend --> GPT["GPT：生成目标 semantic tokens"]
    Ref["参考音频"] --> Hubert["16 kHz + 尾部静音 / HuBERT / VQ"]
    Hubert -->|"参考 semantic tokens"| GPT
    Ref --> RefEnc["SoVITS 采样率频谱 / ref_enc"]
    RefEnc --> Speaker["Speaker conditioning"]
    SV["匹配的 SV embedding / v2Pro"] --> Speaker
    GPT --> SoVITS["SoVITS：enc_p / flow / decoder"]
    Frontend -->|"目标音素"| SoVITS
    Speaker --> SoVITS
    SoVITS --> Audio["波形 / 分段拼接"]
```

参考音色通过 `ref_enc` 编码，`enc_q` 用于训练时的 posterior encoder。
代码中的 `ref_mel` 名称沿用历史，实际输入是 STFT 幅度频谱。

## 文本和参考特征

[speaker.rs](src/inference/speaker.rs) 负责参考缓存和目标特征准备。
文本前端解析注音、判断语言、展开数字，再生成音素和 `word2ph`；
BERT 特征按相同文本对齐到音素位置。
参考/目标的音素与对齐后的 BERT 特征分别拼接后送给 GPT。
G2P 与 BERT 使用同一份规范化文本，数字展开和读音覆盖通过原文索引映射保持一致。
数字与注音规则见 [Agent 接入](docs/AGENT_INTEGRATION.md#读音和数字)。

[HuBERT](src/models/hubert.rs) 使用 soxr 重采样到 16 kHz，并追加 9600 个零样本。
已经是 16 kHz 的输入也追加静音。另一条参考路径在
[ref_audio.rs](src/inference/ref_audio.rs) 计算 SoVITS 频谱（FFT 2048、hop 640）。
当前参考编码器使用前 704 个频率 bin。

[GPT 条件准备](src/models/gpt/conditioning.rs) 对 BERT 特征去除 CLS/SEP，投影到 512 维并根据 `word2ph` 展开。
非中文及无法直接对齐的情形使用回退处理。

每个 Pipeline 最多缓存 16 组参考特征，键包含参考音频路径、参考文本、参考语言和 SV 路径。
参考语言独立于目标语言；只切换待朗读文字的语言可以复用参考特征。
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

各后端共用[输入准备](src/models/gpt/conditioning.rs)：Prefill 使用文本双向、音频因果的
hybrid mask；单步 decode 可看到所有有效历史缓存。
[DecoderSampler](src/models/gpt/sampling.rs) 统一前 11 步的 EOS 屏蔽，以及
argmax 或采样结果为 EOS 时停止的判断，并保持停止前的 RNG 消耗顺序。
提前结束或漏读的样本通过分段日志、ASR 和试听排查。

`GPTModel` 的公开类型和方法位置保持不变；内部模块职责与独立测试入口见
[GPT Core](docs/DEVELOPMENT.md#gpt-core)。历史无参考 semantic tokens 的生成器保留在
`legacy.rs`，Pipeline 进入该兼容路径时记录警告。

## SoVITS 和 v2Pro

[sovits.rs](src/models/sovits.rs) 根据权重是否包含 `sv_emb.*` 等层加载 v2Pro 条件分支。
SV 接受 `[20480]` 或 `[1, 20480]`，应与参考音频匹配。
缺少 SV 时使用零向量。性能和发布验收使用与参考音频匹配的 SV。
Rust 转换器可转换已有 embedding，不负责从参考录音计算新的 SV。

EncP 的 attention/FFN 使用 post-norm：`LayerNorm(x + block(x))`。
Flow 的 gate 先做 `a + b` 再分通道执行 tanh/sigmoid。
普通 Conv1d 权重由共享加载器原样缓存，保留权重幅度和零值通道；`weight_g` / `weight_v`
形式仍使用 weight normalization。Flow 投影与训练用 EncQ 的普通卷积共用该加载器。
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
