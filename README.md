# GPT-SoVITS-RS

**高性能 GPT-SoVITS 推理引擎** — 用 Rust 从零重写，专注推理性能与轻量化部署。

原版 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 侧重训练、微调和测试；本项目聚焦**推理侧**，目标是生产环境下的最快 TTS 推理和最小部署体积。

## 与原版对比

| | GPT-SoVITS (Python) | GPT-SoVITS-RS (本项目) |
|---|---|---|
| **定位** | 训练 + 微调 + 推理 | **纯推理引擎** |
| **部署** | Python 环境 + 多个依赖包 | **单一二进制文件** (~15MB) |
| **GPU 支持** | PyTorch CUDA | **Candle CUDA** (更轻量) |
| **启动时间** | 数秒 (Python + 模型加载) | **亚秒级** |
| **内存占用** | 高 (PyTorch runtime) | **低** (无 runtime 开销) |
| **API** | Gradio Web UI | **CLI / Rust 库 / HTTP (WAV 流)** |
| **推理加速** | 无 | **KV Cache (18x CPU 加速)** |
| **容器化** | 复杂 (Python 环境) | **多阶段 Docker (最小镜像)** |

## 特性

- 🚀 **极致推理性能**: 纯 Rust + Candle ML 框架，KV Cache 优化 (CPU 18x / GPU 1.65x 加速)
- 📦 **零依赖部署**: 编译为单一静态二进制，无需 Python/PyTorch 环境
- 🐳 **生产级容器**: 多阶段构建，CPU/CUDA 双镜像
- 🔌 **多种接入方式**: CLI 工具、Rust 库 API、HTTP 服务器 (直接返回 WAV 音频流)
- ✅ **数值精确**: 所有模块已与 Python 原版逐层对齐验证 (误差 < 1e-7)
- 🌍 **多语言**: 支持中文、英文、日文、韩文、粤语
- 🎯 **推理增强**: Repetition penalty 减少生成重复，语义 Tokenizer 提升音色还原
- 🔤 **完整 G2P**: 中文拼音完整映射，匹配 Python v2 符号表 (732 符号)

## 架构

```
输入文本 → G2P (中文拼音/英文) → BERT (ONNX) → GPT (KV Cache + Repetition Penalty)
                                          ↓                    ↓
                                    Hubert (ONNX)        语义 Tokenizer
                                          ↓                    ↓
                                  Semantic Tokens → SoVITS → BigVGAN → WAV
                                                         ↓
                                            enc_p/enc_q + Flow + Decoder
```

## 数值验证

所有核心模块已与 Python 原版进行逐层数值对比，确保精度对齐：

| 模块 | 验证方式 | 精度 |
|------|---------|------|
| GPT | 逐层注意力 + 输出 logits | 2.59e-6 |
| enc_p (SSL 编码器) | 逐层输出 | < 1e-7 |
| enc_q (参考编码器) | MRTE + SSL 子步骤 | < 1e-7 |
| SoVITS 解码器 | 逐层中间输出 (conv_pre, ups, resblocks, post_conv) | < 1e-7 |
| 音频输出 | RMS / 波形逐点对比 | RMS 一致, 最大差 1e-5 |

**关键修复：**
- ResBlock1 累加器：使用累积变量而非原始输入做 LeakyReLU
- conv_pre 前缺少 LeakyReLU(0.1)
- 去除了多余的 logs_p clamp
- dilated conv 参数（padding, dilation）对齐

## 快速开始

### 前置要求

- Rust 1.75+ (从 [rustup.rs](https://rustup.rs) 安装)
- CUDA Toolkit 12.x (可选，用于 GPU 加速)

### 构建

```bash
git clone https://github.com/ricardomlee/gpt-sovits-rs.git
cd gpt-sovits-rs

# CPU 版本
cargo build --release

# CUDA GPU 版本
cargo build --release --features cuda
```

### 准备模型

从 [HuggingFace](https://huggingface.co/lj1995/GPT-SoVITS) 下载预训练模型，转换为 safetensors 格式后放入 `models/` 目录：

```
models/
├── gpt-model.safetensors      # GPT 模型 (~148MB)
├── sovits-model.safetensors   # SoVITS 模型 (~101MB)
├── bigvgan.safetensors        # BigVGAN 声码器 (~430MB)
├── prompt_tokens.npy          # 预提取的 prompt tokens
└── onnx/
    ├── bert.onnx + .data      # BERT 特征模型
    └── hubert.onnx + .data    # Hubert 音频特征模型
```

### 运行推理

```bash
cargo run --release -- \
    --gpt-model models/gpt-model.safetensors \
    --sovits-model models/sovits-model.safetensors \
    --bigvgan-model models/bigvgan.safetensors \
    --text "你好，世界！" \
    --reference-audio ref.wav \
    --reference-text "参考文本" \
    --output output.wav
```

## Rust API 使用

```rust
use gpt_sovits_rs::{Pipeline, Config, InferenceOptions, Language};

let config = Config::builder()
    .with_device("cuda")
    .with_half_precision(false)
    .build();

let mut pipeline = Pipeline::new(config)?;

// 加载模型
pipeline.load_gpt("models/gpt-model.safetensors")?;
pipeline.load_sovits("models/sovits-model.safetensors")?;
pipeline.load_bigvgan("models/bigvgan.safetensors")?;

// 运行推理
let options = InferenceOptions::builder()
    .top_k(15)
    .top_p(0.95)
    .temperature(0.8)
    .speed(1.0)
    .language(Language::Chinese)
    .build();

let audio = pipeline.inference(
    "你好，这是测试文本",
    "ref.wav",
    "参考文本",
    &options,
)?;

// 保存到文件
audio.save("output.wav")?;

// 或获取 WAV 字节流 (适用于 HTTP API)
let wav_bytes: Vec<u8> = audio.to_wav_bytes()?;
```

## 命令行选项

```
Usage: gpt-sovits [OPTIONS] --text <TEXT> --output <OUTPUT>

Options:
      --gpt-model <PATH>       GPT 模型文件路径
      --sovits-model <PATH>    SoVITS 模型文件路径
      --bigvgan-model <PATH>   BigVGAN 模型文件路径
      --text <TEXT>            输入文本
      --reference-audio <PATH> 参考音频路径
      --reference-text <TEXT>  参考音频文本
      --language <LANG>        语言 (zh/en/ja/ko/yue)
      --top-k <N>              Top-k 采样 (默认：15)
      --top-p <P>              Top-p 采样 (默认：0.95)
      --temperature <T>        采样温度 (默认：0.8)
      --output <PATH>          输出 WAV 文件路径
  -h, --help                   打印帮助
  -V, --version                打印版本
```

### HTTP API 模式

启动 HTTP 服务器，直接返回 WAV 音频流：

```bash
cargo run --release --features "cuda,http-api" -- \
    --http --port 9880 \
    --gpt-model models/gpt-model.safetensors \
    --sovits-model models/sovits-model.safetensors \
    --bigvgan-model models/bigvgan.safetensors
```

**请求示例**：

```bash
# TTS 推理 → 直接返回 WAV 文件
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "你好世界", "text_language": "zh", "refer_wav_path": "ref.wav", "prompt_text": "参考文本"}' \
  --output tts_output.wav

# 健康检查
curl http://localhost:9880/health

# 切换参考音频
curl -X POST http://localhost:9880/change_refer \
  -H 'Content-Type: application/json' \
  -d '{"refer_wav_path": "new_ref.wav", "prompt_text": "新参考文本"}'
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/tts` | POST | TTS 推理，返回 `audio/wav` |
| `/change_refer` | POST | 切换参考音频 |
| `/control` | POST | 服务控制 (reload/unload) |

### Docker 部署

```bash
# CPU 版本
docker build -t gpt-sovits-rs .

# CUDA GPU 版本
docker build -f Dockerfile.cuda -t gpt-sovits-rs:cuda .

# 运行
docker run --gpus all -p 9880:9880 gpt-sovits-rs:cuda \
    --http --gpt-model /app/models/gpt-model.safetensors \
    --sovits-model /app/models/sovits-model.safetensors \
    --bigvgan-model /app/models/bigvgan.safetensors
```

## 性能基准测试

### KV Cache 优化

GPT 自回归生成默认启用 KV Cache 优化，避免重复计算之前 token 的 K/V 张量。

**基准测试结果** (500 tokens):

| 设备 | 配置 | 时间 | 加速比 |
|------|------|------|--------|
| CPU | 无 KV Cache | 368.82s | 1.0x |
| CPU | 启用 KV Cache | 20.48s | **18.0x** |
| GPU (RTX 4060 Ti) | 无 KV Cache | 13.23s | 1.0x |
| GPU (RTX 4060 Ti) | 启用 KV Cache | 8.01s | **1.65x** |

**原理**:
```
传统方法：O(n²) - 每个新 token 重新计算所有 K/V
KV Cache: O(n)  - 缓存 K/V，只计算新 token 的 K/V
```

**运行基准测试**:
```bash
# GPU 对比测试 (需要 CUDA)
cargo run --release --features cuda --example benchmark_gpu_kv_cache

# CPU 对比测试
cargo run --release --example benchmark_kv_cache
```

### 全流程性能分析 (KV Cache)

使用 profiler 分析推理流程各阶段耗时：

```bash
# 需要 CUDA
cargo run --release --features cuda --example profile_kv_cache
```

**实测结果** (RTX 4060 Ti, ~7s 音频):

| 阶段 | 时间 | 占比 |
|------|------|------|
| GPT (KV Cache) | ~500ms | 50% |
| SoVITS | ~300ms | 30% |
| BigVGAN | ~200ms | 20% |
| **总计** | **~1s** | 100% |

**输出音频**: ~7s @ 32kHz
**实时率 (RTF)**: < 1 (实时以上)

## 项目结构

```
gpt-sovits-rs/
├── Cargo.toml
├── Dockerfile                 # CPU 多阶段构建
├── Dockerfile.cuda            # CUDA GPU 多阶段构建
├── .github/workflows/ci.yml   # CI: check + test + Docker
├── src/
│   ├── main.rs                # CLI + HTTP API 入口
│   ├── lib.rs                 # 库入口
│   ├── config/
│   │   └── mod.rs             # 配置管理 (设备/精度/版本)
│   ├── text_frontend/
│   │   ├── mod.rs             # 文本处理
│   │   ├── normalizer.rs      # 文本规范化
│   │   ├── lang_detect.rs     # 语言检测
│   │   ├── g2p.rs             # G2P (中文拼音完整映射)
│   │   ├── symbols.rs         # 732 符号表 (v2, JSON 加载)
│   │   └── symbols_v2.json    # GPT-SoVITS v2 符号表数据
│   ├── models/
│   │   ├── mod.rs
│   │   ├── bert.rs            # BERT ONNX 特征提取
│   │   ├── hubert.rs          # Hubert ONNX 特征提取
│   │   ├── semantic_tokenizer.rs # 语义 token 提取
│   │   ├── gpt.rs             # GPT 自回归生成 (KV Cache + 重复惩罚)
│   │   ├── transformer.rs     # Multi-head attention + SwiGLU
│   │   ├── sovits.rs          # SoVITS 主模型
│   │   ├── sovits_decoder.rs  # HiFi-GAN 解码器
│   │   ├── sovits_encp.rs     # enc_p 文本编码器
│   │   ├── sovits_encq.rs     # enc_q 参考编码器
│   │   ├── sovits_flow.rs     # Flow 模块 (残差耦合)
│   │   ├── sovits_ref_enc.rs  # MelStyleEncoder
│   │   ├── bigvgan.rs         # BigVGAN 声码器 (SnakeBeta + AMP)
│   │   └── mrte.rs            # 多参考音色编码器
│   ├── inference/
│   │   └── mod.rs             # 推理管线编排
│   └── utils/
│       ├── mod.rs
│       ├── audio.rs           # 音频 I/O (WAV 读写/内存编码)
│       ├── audio_features.rs  # STFT/mel 频谱提取
│       ├── kv_cache.rs        # KV Cache 优化
│       └── weights.rs         # safetensors 权重加载
├── examples/
│   ├── benchmark_kv_cache.rs      # KV Cache 基准测试
│   ├── benchmark_gpu_kv_cache.rs  # GPU KV Cache 基准测试
│   ├── profile_pipeline.rs        # 全流程性能分析
│   ├── verify_gpt.rs              # GPT 数值验证
│   ├── check_phonemes.rs          # G2P 音素检查
│   └── e2e_quick.rs               # 快速端到端测试
├── tests/
│   └── integration_tests.rs       # 集成测试 (45 tests)
└── models/                        # 模型文件 (gitignore)
```

## 支持的模型

| 模型 | 格式 | 状态 | 精度 |
|------|------|------|------|
| GPT v1/v2/v3 | `.ckpt` → `.safetensors` | ✅ 已实现 | 已验证 ✓ |
| SoVITS v1/v2/v3 | `.pth` → `.safetensors` | ✅ 已实现 | 已验证 ✓ |
| BigVGAN v2 | `.pt` → `.safetensors` | ✅ 已实现 | 已验证 ✓ |
| BERT (RoBERTa) | ONNX | ✅ 已实现 | 已验证 ✓ |
| Hubert | ONNX | ✅ 已实现 | 已验证 ✓ |

## 开发

### 运行测试

```bash
cargo test
```

### CUDA 支持

```bash
export CUDA_HOME=/usr/local/cuda
cargo build --release --features cuda
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- 原始项目 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) by RVC-Boss
- [Candle](https://github.com/huggingface/candle) by Hugging Face
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) by NVIDIA

## 贡献

欢迎贡献！请阅读我们的贡献指南。
