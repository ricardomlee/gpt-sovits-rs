# Development

This document collects maintainer notes that should not crowd the README homepage.

## Project Layout

```text
src/
├── cli.rs                    # CLI argument parsing and command dispatch
├── doctor.rs                 # Setup diagnostics for --doctor
├── server.rs                 # HTTP routes and handler orchestration
├── server/
│   ├── audio.rs              # HTTP audio byte helpers
│   ├── request.rs            # Request structs and synthesis resolution
│   └── response.rs           # Response/header helpers
├── inference/
│   ├── mod.rs                # Pipeline orchestration and decode mode dispatch
│   ├── options.rs            # Inference defaults and builder
│   ├── split.rs              # Text splitting policies
│   ├── speaker.rs            # Reference speaker cache and target feature preparation
│   └── ref_audio.rs          # Reference WAV mel extraction
├── models/
│   ├── gpt.rs                # GPT autoregressive generation, KV cache, CUDA Graph
│   ├── hubert.rs             # HuBERT/Wav2Vec2 feature extraction
│   ├── bert.rs               # Chinese RoBERTa
│   ├── semantic_tokenizer.rs # VQ semantic token extraction
│   ├── sovits.rs             # SoVITS main model
│   ├── sovits_decoder.rs     # HiFi-GAN style decoder
│   ├── sovits_flow.rs        # Residual coupling flow
│   └── ...
├── text_frontend/
│   ├── g2p.rs                # G2P and Chinese tone sandhi
│   └── symbols_v2.json       # GPT-SoVITS v2 symbol table
└── utils/
    ├── kv_cache.rs
    └── audio_features.rs
```

## Fast Local Checks

Run these before every commit:

```bash
cargo fmt --check
cargo test --quiet
cargo test --quiet --features http-api
cargo clippy --all-targets --features http-api -- -D warnings
cargo check --bin gpt-sovits-convert
```

The default Rust test suite must not require model files. Tests that load real models should be
gated by environment variables so CI and casual contributors can run the suite quickly.

## Real Model Smoke Test

Run this before publishing a release when local models and at least one voice profile are available:

```bash
GPT_SOVITS_RUN_MODEL_SMOKE=1 \
GPT_SOVITS_MODELS_DIR=models \
GPT_SOVITS_VOICES_DIR=voices \
GPT_SOVITS_SMOKE_VOICE=demo \
GPT_SOVITS_SMOKE_DEVICE=auto \
cargo test --test model_smoke -- --nocapture
```

Optional overrides:

```bash
GPT_SOVITS_GPT_MODEL=/path/to/gpt-model.safetensors
GPT_SOVITS_SOVITS_MODEL=/path/to/sovits-model.safetensors
GPT_SOVITS_BERT_MODEL=/path/to/bert.safetensors
GPT_SOVITS_HUBERT_MODEL=/path/to/hubert.safetensors
GPT_SOVITS_SMOKE_TEXT="你好，这是发布前的真实模型冒烟测试。"
GPT_SOVITS_SMOKE_MAX_TOKENS=80
```

The smoke test honors GPT/SoVITS and SV bindings from the selected voice profile, applies its text
splitting policy, and validates the generated waveform with the shared audio-quality thresholds. It
is intentionally skipped unless `GPT_SOVITS_RUN_MODEL_SMOKE=1`.

## End-to-End Smoke Tests

```bash
# Quick E2E test, requires CUDA and model files
cargo run --features cuda --example e2e_quick

# Quality smoke test through a voice profile
cargo run --release --features cuda --example quality_smoke -- \
  --voice demo \
  --output-dir quality_outputs/demo
```

`quality_smoke` writes WAV files and `report.json`, then checks duration, RMS, clipping ratio,
silence ratio, DC offset, and NaN/Inf.

## Docker Checks

Build the CPU image locally:

```bash
docker build -t gpt-sovits-rs:dev .
```

For CUDA image work, use the architecture-specific Dockerfile arguments already used by CI/release
workflows. Prefer testing the published CUDA image with `compose.cuda.yml` after release builds:

```bash
BERT_MODEL=/app/models/bert.safetensors \
HUBERT_MODEL=/app/models/hubert.safetensors \
CUDA_IMAGE_TAG=latest-cuda-sm89 \
docker compose -f compose.cuda.yml up -d

curl -f http://localhost:9880/health
curl -f http://localhost:9880/voices
docker compose -f compose.cuda.yml down
```

## Benchmarks

```bash
# GPT decode mode comparison
cargo bench --features cuda --bench kv_cache_bench

# CPU Conv1d / im2col comparison
cargo bench --features mkl --bench conv1d_cpu_bench

# Full pipeline profile example
cargo run --features cuda --example profile_kv_cache
```

## Fast Incremental CUDA Builds

`Cargo.toml` includes a `dev-gpu` profile with lower optimization cost for day-to-day CUDA work.

```bash
sudo apt install mold
cargo install sccache

cargo build --profile dev-gpu --features cuda
cargo run --profile dev-gpu --features cuda --bin gpt-sovits -- --text "你好" ...
```

## Debug Intermediate Tensors

Set `SOVITS_DEBUG=1` to write intermediate tensors into the current directory. This is useful for
comparing the Rust path against Python.

```bash
SOVITS_DEBUG=1 cargo run --profile dev-gpu --features cuda --bin gpt-sovits -- ...
```

Generated files include:

```text
sovits_debug_ge.txt
sovits_debug_encp_m.txt
sovits_debug_flow_z.txt
sovits_debug_audio.txt
```

## Precision Notes

SoVITS currently stays in F32. Full-model FP16 has produced silent audio on CUDA in this
implementation, so `--half` is kept as a compatibility option and falls back where needed.

GPT BF16/F16 experiments are documented in [PERFORMANCE.md](PERFORMANCE.md). BF16 saves memory but
has not improved speed on the current benchmark; F16 changes generation behavior.

## Dependency Roles

| Dependency | Purpose |
|---|---|
| `candle-core` / `candle-nn` | Tensor runtime and CUDA backend. |
| `candle-transformers` | Transformer building blocks used by BERT. |
| `soxr` | High-quality audio resampling. |
| `hound` | WAV I/O. |
| `jieba-rs` | Chinese segmentation before G2P. |
| `tokenizers` | Hugging Face tokenizer for BERT. |

## Release Checklist

Before tagging:

```bash
git status --short --branch
cargo run --quiet -- --version
cargo fmt --check
cargo test --quiet
cargo test --quiet --features http-api
cargo clippy --all-targets --features http-api -- -D warnings
cargo check --bin gpt-sovits-convert
```

Then:

1. Run the optional real model smoke test if model files are available.
2. Build release binaries or Docker images for the target platforms.
3. Run at least one CPU and one CUDA smoke test when CUDA artifacts changed.
4. Confirm model download/conversion instructions still match [MODELS.md](MODELS.md).
5. Update release notes under `docs/releases/`.
6. Push the release tag and watch the GitHub release workflow until all binary and container jobs finish successfully.
