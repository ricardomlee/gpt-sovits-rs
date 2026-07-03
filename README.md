# GPT-SoVITS-RS

GPT-SoVITS 的 Rust 推理实现。目标是把训练好的 GPT-SoVITS 模型放到本地服务里跑，少依赖 Python 环境，方便接入自己的助手或脚本。

原版 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 仍然更适合训练、微调和调参。本项目只做推理侧：模型加载、文本前端、GPT 生成、SoVITS 解码、CLI、HTTP 服务和部署模板。

## GPT-SoVITS 工作原理

GPT-SoVITS 是少样本语音克隆 TTS。给它一段参考音频和对应文字，再输入目标文本，模型会生成接近参考音色的语音。流程可以分成两步：

**第一步：GPT 生成语义 token**

语义 token 是离散声学单元，由 HuBERT 特征经过 VQ 量化得到。GPT 会把参考音频的语义 token 当作提示，再为目标文本生成新的语义 token 序列。

**第二步：SoVITS 解码波形**

SoVITS 使用语义 token 和参考音频的 mel 频谱，通过 Flow 模型和 HiFi-GAN 解码器生成波形。

```
【输入】
  参考音频 ──→ HuBERT ──→ VQ 量化 ──→ 参考语义 token（音色提示）
  参考文本 ──┐
             ├──→ G2P ──→ BERT ──→ 拼接 BERT 特征
  目标文本 ──┘                           │
                                         ↓
                              GPT 自回归生成
                      [参考 token | 目标语义 token 序列]
                                         │
  参考音频 ──→ mel 频谱 ──────────────────┤（音色条件）
                                         ↓
                              SoVITS 解码器
                         Flow + HiFi-GAN → 波形

【输出】目标语音（说话人音色 = 参考音频，内容 = 目标文本）
```

> 注意：Python 原版在调用 GPT 前会拼接参考文本和目标文本的音素，BERT 特征也要分别提取后拼接。本项目保留了这一步；缺少它时，GPT 通常只生成十几个 token，输出接近 0.5 秒静音。

## 与原版对比

| | GPT-SoVITS (Python) | GPT-SoVITS-RS (本项目) |
|---|---|---|
| 定位 | 训练 + 微调 + 推理 | 推理 |
| 部署 | Python 环境 + PyTorch | 单一二进制文件 |
| GPU 支持 | PyTorch CUDA | Candle CUDA |
| BERT/HuBERT | Python + PyTorch | Candle |
| 重采样 | librosa / soxr | libsoxr |
| API | Gradio Web UI | CLI / Rust 库 / HTTP |
| KV Cache | 动态 KV + SDPA | 动态 KV；另有静态 KV 和实验性 CUDA Graph |

## 特性

- soxr HQ 重采样和静音填充，VQ prompt tokens 已和 Python 参考结果对齐。
- 支持 `ref_text`：参考文本音素拼接、BERT 对齐拼接都在推理路径里。
- KV cache：prefill 阶段处理 text+prompt token，自回归阶段按单 token 解码。RTX 4060 Ti 上，长文本基准约 3.2x。
- CUDA Graph：`inference_cuda_graph()` 可以捕获 GPT 解码步骤，减少 CPU kernel launch 开销。
- 自定义 CUDA kernel：[cuda-oxide 实验记录](docs/CUDA_OXIDE_EXPERIMENT.md)；目前只做隔离验证，不进入默认推理路径。
- GPU 路径：embedding 查表在 GPU 上做；每步采样只保留必要的 D2H transfer。
- 中文文本前端：三声连读、"不/一"变调、轻声规则。
- 部署：编译后可用单一二进制运行，不需要 Python/PyTorch 推理环境。
- 语言：中文、英文、日文、韩文、粤语。
- HTTP API：`/tts`、`/tts/stream`、`/tts/batch`，另有 OpenAI-compatible `/v1/audio/speech` adapter。

## 快速开始

### 方式一：Docker

模型、voice profile 和输出目录都通过 volume 挂载，升级镜像不会覆盖用户数据。先复制环境模板：

```bash
cp .env.example .env
```

目录约定：

```text
models/   # 模型权重
voices/   # voice profiles + 参考音频
outputs/  # 可选输出目录
```

**CPU / NAS 版本**

```bash
docker compose -f compose.cpu.yml up -d --build
```

**CUDA GPU 版本**

> GPT 自回归生成 + SoVITS 解码 + BERT/HuBERT 特征提取全程使用 Candle CUDA，无 ONNX Runtime 依赖。

```bash
docker compose -f compose.cuda.yml up -d --build
```

启动后检查服务和可用角色：

```bash
curl http://localhost:9880/health
curl http://localhost:9880/voices
```

助手可以按 voice 调用：

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"mao","text":"人民，只有人民，才是创造世界历史的动力。"}' \
  --output output.wav
```

更多部署说明见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

### 方式二：从源码构建

**前置要求**

- Rust 1.75+（从 [rustup.rs](https://rustup.rs) 安装）
- libsoxr：`sudo apt install libsoxr-dev`
- CUDA Toolkit 12.x（可选，GPU 加速）

```bash
git clone https://github.com/ricardomlee/gpt-sovits-rs.git
cd gpt-sovits-rs

# x86_64 CPU 构建（推荐，使用 MKL）
cargo build --release --features mkl

# 通用 CPU 构建（ARM、非 x86 NAS）
cargo build --release

# CUDA GPU 构建
cargo build --release --features cuda

# HTTP API
cargo build --release --features http-api,mkl
cargo build --release --features http-api,cuda
```

MKL 会静态链接进二进制，运行时不用安装 Intel 工具包。它只适合 x86_64；ARM 设备使用通用 CPU 构建。本项目在 i5-12490F 上的短句测试中，MKL 和 CPU 卷积快路径将纯推理耗时从约 4.73 秒降到 2.8 秒；6.96 秒长句的 RTF 约为 1.29。

### 准备模型

从 [HuggingFace](https://huggingface.co/lj1995/GPT-SoVITS) 下载预训练模型，使用 `convert_sovits_weights.py` 转换为 safetensors 格式后放入 `models/` 目录：

```
models/
├── gpt-model.safetensors      # GPT 模型
├── sovits-model.safetensors   # SoVITS 模型
├── bert/
│   ├── bert.safetensors       # chinese-roberta-wwm-ext-large
│   └── tokenizer.json         # BERT tokenizer（与 safetensors 同目录）
└── hubert/
    └── hubert.safetensors     # HuBERT/Wav2Vec2 特征提取
```

GPT 和 SoVITS 需要分别转换：

```bash
python3 convert_gpt_weights.py /path/to/s1bert25hz-*.ckpt models/gpt-model.safetensors
python3 convert_sovits_weights.py /path/to/s2G*.pth models/sovits-model.safetensors
```

BERT 权重需使用 Hugging Face safetensors 格式，并把对应的 `tokenizer.json` 放在同一目录；若同目录缺失，程序会回退查找 `models/bert/tokenizer.json`。HuBERT 也使用 safetensors，不再加载 `models/onnx/*.onnx`。

### 运行推理

程序会从 `models/` 自动找到上面的四个模型。设备也会自动选择，CUDA 可用时用 CUDA，否则用 CPU。

```bash
cargo run --release --features cuda --bin gpt-sovits -- \
    --text "你好，世界！" \
    --reference-audio ref.wav \
    --reference-text "参考音频对应的文字"
```

默认输出为 `output.wav`。模型不在 `models/` 时，可以用 `--models-dir` 指定整个目录，也可以用 `--gpt-model` 等参数单独覆盖。

长文本建议开启分句合成。它会复用同一参考音频特征，再逐句拼接；目前比一次性生成整段更稳：

```bash
cargo run --release --features cuda --bin gpt-sovits -- \
    --device cuda --mode auto --split-sentences \
    --split-method sentence --min-sentence-chars 12 \
    --sentence-gap-ms 120 --sentence-fade-ms 8 \
    --max-tokens 500 --repetition-penalty 1.35 \
    --text "第一句话。第二句话。第三句话。" \
    --reference-audio ref.wav \
    --reference-text "参考音频对应的文字" \
    --output output_long.wav
```

`--mode` 可选 `auto`、`plain`、`kv`、`cuda-graph`，默认是 `auto`：CUDA F32 使用 CUDA Graph，CPU/MPS 使用动态 KV。Graph 会校验第一次 launch，发现结果偏离时从已校验的 KV 状态继续。需要临时关闭 Graph 时设置 `GPT_SOVITS_DISABLE_CUDA_GRAPH=1`，或显式传 `--mode kv`。文本默认只在完整句末分段，避免逗号处出现生硬长停顿；`--split-method cut5` 可复现 Python 的分段方式，`--no-split-sentences` 可完全关闭分段。CLI 日志会输出 `profile mode=... target=... ref=... target_bert=... gpt=... sovits=... total=...`，便于看时间花在哪里。

> BigVGAN 当前仍是实验加载入口，主推理路径使用 SoVITS 权重内置 decoder；普通 mel-to-waveform BigVGAN 不能直接替换 SoVITS latent decoder。

### Voice Profile

常用音色可以放在 `voices/<name>/voice.json`。CLI 传 `--voice <name>` 后，会读取参考音频、参考文本、语言、分句和采样默认值。配置里的相对路径按该 voice 目录解析：

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频对应的文字",
  "language": "zh",
  "mode": "auto",
  "split_sentences": true,
  "split_method": "sentence",
  "top_k": 15,
  "top_p": 0.95,
  "temperature": 0.8,
  "max_tokens": 500,
  "repetition_penalty": 1.35
}
```

```bash
cargo run --release --features cuda --bin gpt-sovits -- \
    --voice mao \
    --text "人民，只有人民，才是创造世界历史的动力。"
```

命令行参数会覆盖 voice profile 中的默认值。长期产品目标见 [docs/PRODUCT_GOAL.md](docs/PRODUCT_GOAL.md)，当前性能基线见 [docs/PERFORMANCE.md](docs/PERFORMANCE.md)。

语速在 SoVITS latent 上调整，与 Python `speed_factor` 一致；不会通过重采样最终 WAV 改变音高。分段和韵律对齐说明见 [docs/PROSODY.md](docs/PROSODY.md)。

查看已有音色不需要加载模型：

```bash
cargo run --release --bin gpt-sovits -- --list-voices
```

### 自动质量 Smoke Test

`quality_smoke` 会按 voice profile 生成一组固定句子，保存 WAV，并输出 `report.json`。它检查时长、RMS、削波比例、静音比例、DC offset 和 NaN/Inf；发现明显坏样本时返回非零退出码：

```bash
cargo run --release --features cuda --example quality_smoke -- \
    --voice mao \
    --output-dir quality_outputs/mao
```

## Rust API

```rust
use gpt_sovits_rs::{Pipeline, Config, InferenceOptions, Language};

let mut pipeline = Pipeline::new(Config::builder().with_device("cuda").build())?;
pipeline.load_gpt("models/gpt-model.safetensors")?;
pipeline.load_sovits("models/sovits-model.safetensors")?;
pipeline.load_bert("models/bert/bert.safetensors")?;
pipeline.load_hubert("models/hubert/hubert.safetensors")?;

let options = InferenceOptions::builder()
    .top_k(15).top_p(0.95).temperature(0.8)
    .language(Language::Chinese)
    .max_tokens(500)
    .build();

// 标准推理
let audio = pipeline.inference(
    "你好，这是合成语音",
    "ref.wav",
    "参考音频对应的文字",  // ref_text 对音质影响极大，不可省略
    &options,
)?;

// KV Cache 版本（更长文本时更快）
let audio = pipeline.inference_kv_cache(
    "你好，这是合成语音",
    "ref.wav",
    "参考音频对应的文字",
    &options,
)?;

// CUDA Graph 版本（生产高频调用，需要 cuda feature）
let audio = pipeline.inference_cuda_graph(
    "你好，这是合成语音",
    "ref.wav",
    "参考音频对应的文字",
    &options,
)?;

audio.save("output.wav")?;
```

### 推理模式

| | `inference()` | `inference_kv_cache()` | `inference_cuda_graph()` |
|---|---|---|---|
| GPT 策略 | 每步重算全序列 O(n²) | prefill + 单 token 解码 O(n) | 静态 KV；长输出延迟捕获 CUDA Graph |
| 适用场景 | 对照和调试 | CPU/MPS 或手动指定 | CUDA F32 默认 |
| 要求 | 无 | 无 | `cuda` feature |
| 音质 | 相同 | 相同 | 相同 |

RTX 4060 Ti 实测如下。测试加载 BERT、HuBERT 和 semantic tokenizer，先预热说话人缓存，使用 `top_k=1`、`max_tokens=300`，每种模式预热一次后取两次平均值。

| 文本 | 音频时长 | plain | 动态 KV | CUDA Graph | KV 加速 | Graph 相对 KV |
|---|---:|---:|---:|---:|---:|---:|
| 短 | 0.92s | 0.30s | 0.23s | 0.21s | 1.30x | 1.10x |
| 中 | 7.00s | 2.74s | 1.17s | 0.84s | 2.34x | 1.40x |
| 长 | 12.00s | 7.33s | 2.12s | 1.37s | 3.46x | 1.54x |

Graph 会先用 static KV 生成 32 个 token。短句在此之前结束，不需要 capture；长句则用后续 token 摊薄初始化成本。BERT 在三种模式下均保持开启，benchmark 还会逐样本检查 KV 与 Graph 的最终音频。复现命令：

```bash
cargo bench --features cuda --bench kv_cache_bench
```

## HTTP API

```bash
cargo run --release --features "cuda,http-api" --bin gpt-sovits -- \
    --http --port 9880
```

服务启动后提供这些端点：

**`GET /voices`**：列出 `voices/` 下可用的 voice profile

```bash
curl http://localhost:9880/voices
# {"voices":["mao","character_a"]}
```

**`POST /tts`**：单条文本，返回完整 WAV 文件

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "你好世界",
    "text_language": "zh",
    "refer_wav_path": "ref.wav",
    "prompt_text": "参考音频对应的文字"
  }' --output output.wav
```

使用 voice profile 时，助手只需要传 `voice` 和 `text`：

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"mao","text":"人民，只有人民，才是创造世界历史的动力。"}' \
  --output output.wav
```

`/tts` 响应会带上元信息 header，方便客户端记录和排障。例如
`X-TTS-Voice`、`X-TTS-Language`、`X-TTS-Text-Chars`、
`X-TTS-Duration-S`、`X-TTS-Sample-Rate` 和 `X-TTS-Channels`。

**`POST /tts/stream`**：单条文本，逐句流式返回 WAV（低延迟，可边下边播）

```bash
curl -X POST http://localhost:9880/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好世界","refer_wav_path":"ref.wav","prompt_text":"参考文字"}' \
  --output stream.wav
```

**`POST /tts/batch`**：多条文本，说话人特征只计算一次，结果以 NDJSON 流返回（每条完成即输出一行）

```bash
curl -X POST http://localhost:9880/tts/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": ["第一句话", "第二句话", "第三句话"],
    "refer_wav_path": "ref.wav",
    "prompt_text": "参考音频对应的文字"
  }' | while IFS= read -r line; do
    echo "$line" | python3 -c "
import sys,json,base64
d=json.load(sys.stdin)
open(f'out_{d[\"index\"]}.wav','wb').write(base64.b64decode(d['wav_base64']))
print(f'[{d[\"index\"]}] {d[\"duration_s\"]:.2f}s  {d[\"inference_ms\"]}ms')
"
done
```

`/tts/batch` 每行 JSON 也会包含 `voice`、`language` 和 `text_chars`，方便把输出音频和原始请求对应起来。

**`POST /v1/audio/speech`**：OpenAI-compatible TTS adapter，用于接入 agent 框架和助手客户端

```bash
curl -X POST http://localhost:9880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-sovits",
    "voice": "mao",
    "input": "人民，只有人民，才是创造世界历史的动力。",
    "response_format": "wav"
  }' --output output.wav
```

这个接口目前支持 `response_format: "wav"` 和 `"pcm"`。`voice` 会映射到本地
`voices/<name>/voice.json`。如果 agent 框架支持 OpenAI-compatible speech endpoint，可以把
base URL 设为 `http://localhost:9880/v1`。

OpenClaw 有自己的 TTS provider 层；接入时优先走 OpenAI-compatible provider，并强制
`response_format` 为 `wav` 或 `pcm`。如果它的 OpenAI provider 不方便覆盖输出格式，也可以先走
OpenClaw 的 Local CLI provider。更详细的接入策略见
[docs/AGENT_INTEGRATION.md](docs/AGENT_INTEGRATION.md)。

批量响应每行为：`{"index":0,"wav_base64":"...","sample_rate":32000,"duration_s":1.5,"inference_ms":820}`

## 项目结构

```
src/
├── inference/mod.rs        # 推理管线（ref_text 拼接、BERT 对齐、KV cache 调度）
├── models/
│   ├── gpt.rs              # GPT 自回归生成（prefill KV cache + CUDA graph）
│   ├── hubert.rs           # HuBERT，纯 Candle Wav2Vec2 + soxr 重采样
│   ├── bert.rs             # BERT，纯 Candle chinese-roberta-wwm-ext-large
│   ├── semantic_tokenizer.rs # VQ 语义 token 提取
│   ├── sovits.rs           # SoVITS 主模型
│   ├── sovits_decoder.rs   # HiFi-GAN 解码器
│   ├── sovits_flow.rs      # Flow（残差耦合层）
│   ├── transformer.rs      # Multi-head attention + KV cache
│   └── ...
├── text_frontend/
│   ├── g2p.rs              # G2P（中文拼音 + 三声连读变调）
│   └── symbols_v2.json     # GPT-SoVITS v2 符号表（732 符号）
└── utils/
    ├── kv_cache.rs         # KvCache / KvCacheManager
    └── audio_features.rs   # STFT / mel 频谱提取
```

## 开发与验证

```bash
# 端到端快速测试（需要 CUDA + 模型文件）
cargo run --features cuda --example e2e_quick

# GPU KV Cache 基准对比
cargo bench --features cuda --bench kv_cache_bench

# CPU Conv1d / im2col 基准对比
cargo bench --features mkl --bench conv1d_cpu_bench

# 全流程时间分析
cargo run --features cuda --example profile_kv_cache

# 单元 + 集成测试
cargo test
```

SoVITS 当前固定使用 F32。全模型 FP16 会在 CUDA 上生成静音，`--half` 暂时只保留兼容性并自动回退到 F32。

### 快速增量构建（开发用）

`Cargo.toml` 内置 `dev-gpu` profile（`opt-level=2`，`codegen-units=16`，关闭 LTO）。配合 mold linker + sccache，本地增量重编译可以从约 58s 降到约 11s：

```bash
# 安装 mold 和 sccache（一次性）
sudo apt install mold
cargo install sccache

# 开发时使用 dev-gpu profile
cargo build --profile dev-gpu --features cuda
cargo run --profile dev-gpu --features cuda --bin gpt-sovits -- --text "你好" ...
```

### 中间张量调试

设置 `SOVITS_DEBUG=1` 后，程序会在当前目录生成各阶段中间张量文件（`sovits_debug_*.txt`），用于和 Python 实现对比：

```bash
SOVITS_DEBUG=1 cargo run --profile dev-gpu --features cuda --bin gpt-sovits -- ...
# 生成: sovits_debug_ge.txt, sovits_debug_encp_m.txt, sovits_debug_flow_z.txt, 等
```

## 依赖说明

| 依赖 | 用途 |
|------|------|
| `candle-core` / `candle-nn` | Tensor 运算 + CUDA 后端（GPT、SoVITS、BERT、HuBERT 全部 Candle）|
| `candle-transformers` | BERT 模型结构（`bert::BertModel`）|
| `soxr` | 音频重采样（libsoxr HQ，对齐 librosa 输出） |
| `hound` | WAV 读写 |
| `jieba-rs` | 中文分词（G2P 前处理） |
| `tokenizers` | HuggingFace tokenizer（BERT 分词） |

## 许可证

MIT License

## 致谢

- 原始项目 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) by RVC-Boss
- [Candle](https://github.com/huggingface/candle) by Hugging Face
