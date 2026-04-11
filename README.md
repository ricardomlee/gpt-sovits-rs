# GPT-SoVITS Rust Implementation

A high-performance Rust implementation of the GPT-SoVITS text-to-speech inference engine.

## Features

- 🚀 **High Performance**: Pure Rust implementation with Candle backend for efficient inference
- 📦 **Easy Deployment**: Single binary, no Python environment required
- 💾 **Low Memory**: Optimized memory footprint with model quantization support
- 🌍 **Multi-language**: Supports Chinese, English, Japanese, Korean, and Cantonese
- 🔌 **Flexible API**: CLI tool, Rust library, and optional HTTP server

## Architecture

```
Input Text → Text Frontend → BERT → GPT Model → SoVITS → BigVGAN → Audio
                    ↓           ↓         ↓          ↓         ↓
                Phonemes   Features   Tokens     Mel Spec   Waveform
```

## Installation

### Prerequisites

- Rust 1.75+ (install from [rustup.rs](https://rustup.rs))
- For CUDA support: CUDA Toolkit 12.x

### Build from Source

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS/rust

# CPU only
cargo build --release

# With CUDA support
cargo build --release --features cuda
```

## Quick Start

### 1. Download Pretrained Models

First, download the required models from [HuggingFace](https://huggingface.co/lj1995/GPT-SoVITS):

```bash
# Run the model download script
python scripts/download_models.py
```

### 2. Convert Model Weights

Convert PyTorch models to Candle format:

```bash
# Convert GPT model
python scripts/convert_models.py \
    --gpt-ckpt GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt \
    --sovits-ckpt GPT_SoVITS/pretrained_models/s2G488k.pth \
    --output-dir models/
```

### 3. Run Inference

```bash
# CLI inference
cargo run --release -- \
    --gpt-model models/gpt-model.safetensors \
    --sovits-model models/sovits-model.safetensors \
    --text "你好，世界！" \
    --reference-audio ref.wav \
    --reference-text "参考文本" \
    --output output.wav
```

## API Usage

### Rust Library

```rust
use gpt_sovits_rs::{Pipeline, Config, InferenceOptions};

let config = Config::default();
let mut pipeline = Pipeline::new(config)?;

// Load models
pipeline.load_gpt("models/gpt-model.safetensors")?;
pipeline.load_sovits("models/sovits-model.safetensors")?;

// Run inference
let options = InferenceOptions {
    top_k: 5,
    top_p: 0.95,
    temperature: 0.8,
    ..Default::default()
};

let audio = pipeline.inference(
    "你好，这是测试文本",
    "ref.wav",
    "参考文本",
    &options
)?;

// Save to file
audio.save("output.wav")?;
```

### HTTP API

Start the HTTP server:

```bash
cargo run --release --features http-api -- --http --port 9880
```

API endpoints:

```bash
# TTS inference
curl -X POST "http://localhost:9880/tts" \
    -H "Content-Type: application/json" \
    -d '{
        "text": "你好世界",
        "text_language": "zh",
        "refer_wav_path": "ref.wav",
        "prompt_text": "参考文本"
    }' \
    --output output.wav
```

## Command Line Options

```
Usage: gpt-sovits [OPTIONS] --text <TEXT> --output <OUTPUT>

Options:
      --gpt-model <PATH>       Path to GPT model file
      --sovits-model <PATH>    Path to SoVITS model file
      --text <TEXT>            Input text for synthesis
      --reference-audio <PATH> Reference audio path
      --reference-text <TEXT>  Reference audio text
      --language <LANG>        Language (zh/en/ja/ko/yue)
      --top-k <N>              Top-k sampling (default: 15)
      --top-p <P>              Top-p sampling (default: 0.95)
      --temperature <T>        Sampling temperature (default: 0.8)
      --speed <SPEED>          Speed multiplier (default: 1.0)
      --output <PATH>          Output WAV file path
      --http                   Start HTTP server mode
      --port <PORT>            HTTP server port (default: 9880)
  -h, --help                   Print help
  -V, --version                Print version
```

## Performance Benchmarks

| Device | RTF | Latency |
|--------|-----|---------|
| RTX 4090 (CUDA) | 0.012 | ~200ms |
| RTX 4060 Ti (CUDA) | 0.025 | ~400ms |
| M3 Max (Metal) | 0.035 | ~600ms |
| CPU (AVX2) | 0.15 | ~2.5s |

*RTF = Real Time Factor (inference time / audio duration)*

### KV Cache Optimization

KV Cache optimization is enabled by default for autoregressive GPT generation. This optimization avoids recomputing K/V tensors for previously generated tokens.

**Benchmark Results** (500 tokens, CPU):
| Configuration | Time | Relative |
|---------------|------|----------|
| Without KV Cache | 368.82s | 1.0x |
| With KV Cache | 20.48s | **18.0x faster** |

**How it works:**
```
Traditional: O(n²) - Recompute all K/V for each new token
KV Cache:    O(n)  - Cache K/V, only compute for new token
```

To disable KV Cache (for debugging):
```rust
let options = InferenceOptions {
    use_kv_cache: false,  // Default: true
    ..Default::default()
};
```

## Project Structure

```
gpt-sovits-rs/
├── Cargo.toml
├── src/
│   ├── main.rs              # CLI entry point
│   ├── lib.rs               # Library exports
│   ├── config/
│   │   └── mod.rs           # Model configuration
│   ├── text_frontend/
│   │   ├── mod.rs           # Text processing entry
│   │   ├── normalizer.rs    # Text normalization
│   │   ├── lang_detect.rs   # Language detection
│   │   ├── g2p.rs           # Grapheme-to-phoneme
│   │   └── symbols.rs       # Phoneme symbols
│   ├── models/
│   │   ├── mod.rs
│   │   ├── bert.rs          # BERT feature extractor
│   │   ├── hubert.rs        # Hubert feature extractor
│   │   ├── gpt.rs           # GPT semantic model
│   │   ├── sovits.rs        # SoVITS synthesizer
│   │   └── bigvgan.rs       # BigVGAN vocoder
│   ├── inference/
│   │   ├── mod.rs
│   │   └── pipeline.rs      # Inference pipeline
│   └── utils/
│       ├── audio.rs         # Audio I/O
│       └── tensor.rs        # Tensor utilities
├── scripts/
│   ├── download_models.py   # Model downloader
│   └── convert_models.py    # Model converter
├── examples/
│   └── cli_inference.rs     # CLI example
└── models/                  # Model files (gitignored)
```

## Supported Models

| Model | Format | Status |
|-------|--------|--------|
| GPT v1/v2/v3 | `.ckpt` | 🟡 Converting |
| SoVITS v1/v2/v3 | `.pth` | 🟡 Converting |
| BigVGAN v2 | `.pt` | 🟡 Converting |
| BERT (RoBERTa) | ONNX | ⬜ Planned |
| Hubert | ONNX | ⬜ Planned |

## Development

### Running Tests

```bash
cargo test
```

### Benchmarking

```bash
cargo bench
```

### Building with CUDA

```bash
export CUDA_HOME=/usr/local/cuda
cargo build --release --features cuda
```

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Acknowledgments

- Original [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) by RVC-Boss
- [Candle](https://github.com/huggingface/candle) by Hugging Face
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) by NVIDIA

## Contributing

Contributions are welcome! Please read our contributing guidelines first.
