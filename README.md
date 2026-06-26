# GPT-SoVITS-RS

**高性能 GPT-SoVITS 推理引擎** — 用 Rust 从零重写，专注推理性能与轻量化部署。

原版 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 侧重训练、微调和测试；本项目聚焦**推理侧**，目标是生产环境下的最快 TTS 推理和最小部署体积。

## GPT-SoVITS 工作原理

GPT-SoVITS 是一个**少样本语音克隆 TTS** 系统，只需 5–10 秒参考音频即可克隆说话人音色，核心思路分两步：

**第一步：用 GPT 把文字转换为"语义 token 序列"**

语义 token 是一种离散的声学单元（通过 HuBERT → VQ 量化得到），既携带文字内容，也携带说话节奏和音色。GPT 以参考音频的语义 token 作为"音色提示"，自回归生成目标文本对应的语义 token 序列。

**第二步：用 SoVITS 把语义 token 解码为波形**

SoVITS（基于 VITS 变体）以语义 token 为内容输入、以参考音频的 mel 频谱为音色输入，通过 Flow 模型 + HiFi-GAN 解码器合成波形。

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

> **关键细节**：Python 原版在调用 GPT 前会将参考文本音素和目标文本音素拼接，BERT 特征也分别提取后拼接。本项目完整复现了这一步，否则 GPT 只会生成 10 余个 token（约 0.5 秒静音）。

## 与原版对比

| | GPT-SoVITS (Python) | GPT-SoVITS-RS (本项目) |
|---|---|---|
| **定位** | 训练 + 微调 + 推理 | **纯推理引擎** |
| **部署** | Python 环境 + PyTorch | **单一二进制文件** |
| **GPU 支持** | PyTorch CUDA | **Candle CUDA** |
| **BERT/HuBERT** | Python + PyTorch | **纯 Candle（无 ONNX Runtime）** |
| **重采样** | librosa / soxr | **libsoxr**（VQ token 与 Python 完全一致）|
| **API** | Gradio Web UI | **CLI / Rust 库 / HTTP (WAV 流)** |
| **KV Cache** | 无 | **prefill + 单 token 解码**（GPU 场景加速） |

## 特性

- **推理精度**：soxr HQ 重采样 + 静音填充，VQ prompt tokens 与 Python 100% 一致（20/20）
- **ref_text 支持**：参考文本音素拼接、BERT 对齐拼接，完整复现 Python 推理路径
- **KV Cache**：prefill 阶段一次性处理所有 text+prompt token，自回归阶段单 token 解码；RTX 4060 Ti 实测长文本加速 **3.2x**
- **CUDA Graph**：`inference_cuda_graph()` 将 GPT 解码步骤捕获为 CUDA graph，消除 CPU kernel launch overhead，适合生产高频调用
- **GPU 优化**：embedding 查表全程在 GPU 执行（`index_select`），每生成步仅一次 D2H transfer（采样必须）
- **变调规则**：中文三声连读（2/3/4字词结构感知）、"不/一"变调、轻声规则
- **轻量部署**：编译为单一二进制，无需 Python/PyTorch 环境
- **多语言**：中文、英文、日文、韩文、粤语
- **HTTP API**：单条 WAV 流（`/tts/stream`）+ 批量 NDJSON 流（`/tts/batch`）

## 快速开始

### 方式一：Docker（推荐）

模型文件通过 `-v` 挂载到容器，无需重新构建镜像即可更换模型。

**CPU 版本**

```bash
docker build -t gpt-sovits-rs:cpu .

docker run --rm \
  -v /path/to/models:/app/models \
  -v /path/to/audio:/audio \
  gpt-sovits-rs:cpu \
    --device cpu \
    --gpt-model /app/models/gpt-model.safetensors \
    --sovits-model /app/models/sovits-model.safetensors \
    --bert-model /app/models/bert/bert.safetensors \
    --hubert-model /app/models/hubert/hubert.safetensors \
    --text "你好，世界！" \
    --reference-audio /audio/ref.wav \
    --reference-text "参考音频对应的文字" \
    --output /audio/output.wav
```

**CUDA GPU 版本**

> GPT 自回归生成 + SoVITS 解码 + BERT/HuBERT 特征提取全程使用 Candle CUDA，无 ONNX Runtime 依赖。

```bash
docker build -t gpt-sovits-rs:cuda -f Dockerfile.cuda .

docker run --rm --gpus all \
  -v /path/to/models:/app/models \
  -v /path/to/audio:/audio \
  gpt-sovits-rs:cuda \
    --device cuda \
    --gpt-model /app/models/gpt-model.safetensors \
    --sovits-model /app/models/sovits-model.safetensors \
    --bert-model /app/models/bert/bert.safetensors \
    --hubert-model /app/models/hubert/hubert.safetensors \
    --text "你好，世界！" \
    --reference-audio /audio/ref.wav \
    --reference-text "参考音频对应的文字" \
    --output /audio/output.wav
```

**HTTP API 服务**

```bash
docker run -d --gpus all \
  -p 9880:9880 \
  -v /path/to/models:/app/models \
  -v /path/to/audio:/audio \
  --name gpt-sovits \
  gpt-sovits-rs:cuda \
    --http --port 9880 --device cuda \
    --gpt-model /app/models/gpt-model.safetensors \
    --sovits-model /app/models/sovits-model.safetensors \
    --bert-model /app/models/bert/bert.safetensors \
    --hubert-model /app/models/hubert/hubert.safetensors

curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好世界","text_language":"zh","refer_wav_path":"/audio/ref.wav","prompt_text":"参考文本"}' \
  --output output.wav
```

### 方式二：从源码构建

**前置要求**

- Rust 1.75+（从 [rustup.rs](https://rustup.rs) 安装）
- libsoxr：`sudo apt install libsoxr-dev`
- CUDA Toolkit 12.x（可选，GPU 加速）

```bash
git clone https://github.com/ricardomlee/gpt-sovits-rs.git
cd gpt-sovits-rs

# CPU 构建
cargo build --release

# CUDA GPU 构建
cargo build --release --features cuda

# HTTP API（任意组合 cuda）
cargo build --release --features http-api
cargo build --release --features http-api,cuda
```

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

```bash
cargo run --release --features cuda --bin gpt-sovits -- \
    --device cuda \
    --gpt-model models/gpt-model.safetensors \
    --sovits-model models/sovits-model.safetensors \
    --bert-model models/bert/bert.safetensors \
    --hubert-model models/hubert/hubert.safetensors \
    --text "你好，世界！" \
    --reference-audio ref.wav \
    --reference-text "参考音频对应的文字" \
    --output output.wav
```

长文本建议开启分句合成，复用同一参考音频特征并逐句拼接，稳定性通常比一次性生成整段更好：

```bash
cargo run --release --features cuda --bin gpt-sovits -- \
    --device cuda --mode cuda-graph --split-sentences \
    --min-sentence-chars 12 --sentence-gap-ms 120 --sentence-fade-ms 8 \
    --max-tokens 500 --repetition-penalty 1.35 \
    --gpt-model models/gpt-model.safetensors \
    --sovits-model models/sovits-model.safetensors \
    --bert-model models/bert/bert.safetensors \
    --hubert-model models/hubert/hubert.safetensors \
    --text "第一句话。第二句话。第三句话。" \
    --reference-audio ref.wav \
    --reference-text "参考音频对应的文字" \
    --output output_long.wav
```

`--mode` 可选 `plain`、`kv`、`cuda-graph`。CLI 日志会输出 `profile mode=... target=... ref=... target_bert=... gpt=... sovits=... total=...`，用于定位瓶颈。

> BigVGAN 当前仍是实验加载入口，主推理路径使用 SoVITS 权重内置 decoder；普通 mel-to-waveform BigVGAN 不能直接替换 SoVITS latent decoder。

### Voice Profile

可以把常用角色音色保存为 `voices/<name>/voice.json`，CLI 通过 `--voice <name>` 自动读取参考音频、参考文本、语言、分句和采样默认值。配置里的相对路径按该 voice 目录解析：

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频对应的文字",
  "language": "zh",
  "mode": "cuda-graph",
  "split_sentences": true,
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
    --text "人民，只有人民，才是创造世界历史的动力。" \
    --gpt-model models/gpt-model.safetensors \
    --sovits-model models/sovits-model.safetensors \
    --bert-model models/bert/bert.safetensors \
    --hubert-model models/hubert/hubert.safetensors \
    --output output.wav
```

命令行参数会覆盖 voice profile 中的默认值。长期产品目标见 [docs/PRODUCT_GOAL.md](docs/PRODUCT_GOAL.md)。

### 自动质量 Smoke Test

`quality_smoke` 会按 voice profile 生成一组固定句子，保存 WAV，并输出 `report.json`。它会自动检查时长、RMS、削波比例、静音比例、DC offset 和 NaN/Inf；发现明显坏样本时返回非零退出码：

```bash
cargo run --release --features cuda --example quality_smoke -- \
    --voice mao \
    --gpt-model models/gpt-model.safetensors \
    --sovits-model models/sovits-model.safetensors \
    --bert-model models/bert/bert.safetensors \
    --hubert-model models/hubert/hubert.safetensors \
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
    .top_k(15).top_p(1.0).temperature(1.0)
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

### 推理模式对比

| | `inference()` | `inference_kv_cache()` | `inference_cuda_graph()` |
|---|---|---|---|
| **GPT 策略** | 每步重算全序列 O(n²) | prefill + 单 token 解码 O(n) | prefill + CUDA graph 单步 |
| **适用场景** | 短文本、调试 | 长文本 | 生产高频、延迟敏感 |
| **要求** | — | — | `cuda` feature |
| **音质** | 相同 | 相同 | 相同 |

RTX 4060 Ti 实测（`cargo bench --features cuda --bench kv_cache_bench`）：

| 文本长度 | plain | kv cache | 加速比 |
|----------|-------|----------|--------|
| 短（4 字） | 3.95s | 2.00s | **1.97x** |
| 中（28 字）| 16.94s | 7.16s | **2.37x** |
| 长（43 字）| 41.86s | 13.08s | **3.20x** |

加速比随文本长度增长，符合 O(n²) vs O(n) 的理论预期。

## HTTP API

```bash
cargo run --release --features "cuda,http-api" --bin gpt-sovits -- \
    --http --port 9880 --device cuda \
    --gpt-model models/gpt-model.safetensors \
    --sovits-model models/sovits-model.safetensors \
    --bert-model models/bert/bert.safetensors \
    --hubert-model models/hubert/hubert.safetensors
```

服务启动后提供三个端点：

**`POST /tts`** — 单条文本，返回完整 WAV 文件

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

**`POST /tts/stream`** — 单条文本，逐句流式返回 WAV（低延迟，可边下边播）

```bash
curl -X POST http://localhost:9880/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好世界","refer_wav_path":"ref.wav","prompt_text":"参考文字"}' \
  --output stream.wav
```

**`POST /tts/batch`** — 多条文本，说话人特征只计算一次，结果以 NDJSON 流返回（每条完成即输出一行）

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

批量响应每行为：`{"index":0,"wav_base64":"...","sample_rate":32000,"duration_s":1.5,"inference_ms":820}`

## 项目结构

```
src/
├── inference/mod.rs        # 推理管线（ref_text 拼接、BERT 对齐、KV cache 调度）
├── models/
│   ├── gpt.rs              # GPT 自回归生成（prefill KV cache + CUDA graph）
│   ├── hubert.rs           # HuBERT — 纯 Candle Wav2Vec2 + soxr 重采样
│   ├── bert.rs             # BERT — 纯 Candle chinese-roberta-wwm-ext-large
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

# FP16 / FP32 推理基准对比
cargo bench --features cuda --bench fp16_vs_fp32_bench

# 全流程时间分析
cargo run --features cuda --example profile_kv_cache

# 单元 + 集成测试
cargo test
```

### 快速增量构建（开发用）

`Cargo.toml` 内置 `dev-gpu` profile（`opt-level=2`，`codegen-units=16`，关闭 LTO），配合 mold linker + sccache 可将增量重编译从 58s 压到约 11s：

```bash
# 安装 mold 和 sccache（一次性）
sudo apt install mold
cargo install sccache

# 开发时使用 dev-gpu profile
cargo build --profile dev-gpu --features cuda
cargo run --profile dev-gpu --features cuda --bin gpt-sovits -- --text "你好" ...
```

### 中间张量调试

设置环境变量 `SOVITS_DEBUG=1` 可在当前目录生成各阶段中间张量文件（`sovits_debug_*.txt`），用于与 Python 实现对比验证：

```bash
SOVITS_DEBUG=1 cargo run --profile dev-gpu --features cuda --bin gpt-sovits -- ...
# 生成: sovits_debug_ge.txt, sovits_debug_encp_m.txt, sovits_debug_flow_z.txt, 等
```

## 依赖说明

| 依赖 | 用途 |
|------|------|
| `candle-core` / `candle-nn` | Tensor 运算 + CUDA 后端（GPT、SoVITS、BERT、HuBERT 全部 Candle）|
| `candle-transformers` | BERT 模型结构（`bert::BertModel`）|
| `soxr` | 音频重采样（libsoxr HQ，与 librosa 输出完全一致） |
| `hound` | WAV 读写 |
| `jieba-rs` | 中文分词（G2P 前处理） |
| `tokenizers` | HuggingFace tokenizer（BERT 分词） |

## 许可证

MIT License

## 致谢

- 原始项目 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) by RVC-Boss
- [Candle](https://github.com/huggingface/candle) by Hugging Face
