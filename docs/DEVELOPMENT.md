# Development

This document collects details that should not crowd the README homepage.

## Project Layout

```text
src/
├── inference/mod.rs          # Inference pipeline, ref_text handling, BERT alignment, KV dispatch
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

## Checks

```bash
cargo fmt --check
cargo check
cargo test
python3 -m py_compile prepare_models.py convert_gpt_weights.py convert_sovits_weights.py
```

## End-to-End Smoke Tests

```bash
# Quick E2E test, requires CUDA and model files
cargo run --features cuda --example e2e_quick

# Quality smoke test through a voice profile
cargo run --release --features cuda --example quality_smoke -- \
  --voice demo \
  --output-dir quality_outputs/demo
```

`quality_smoke` writes WAV files and `report.json`, then checks duration, RMS, clipping ratio, silence ratio, DC offset, and NaN/Inf.

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

Set `SOVITS_DEBUG=1` to write intermediate tensors into the current directory. This is useful for comparing the Rust path against Python.

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

SoVITS currently stays in F32. Full-model FP16 has produced silent audio on CUDA in this implementation, so `--half` is kept as a compatibility option and falls back where needed.

GPT BF16/F16 experiments are documented in [PERFORMANCE.md](PERFORMANCE.md). BF16 saves memory but has not improved speed on the current benchmark; F16 changes generation behavior.

## Dependency Roles

| Dependency | Purpose |
|---|---|
| `candle-core` / `candle-nn` | Tensor runtime and CUDA backend. |
| `candle-transformers` | Transformer building blocks used by BERT. |
| `soxr` | High-quality audio resampling. |
| `hound` | WAV I/O. |
| `jieba-rs` | Chinese segmentation before G2P. |
| `tokenizers` | Hugging Face tokenizer for BERT. |

## Release Work

Before publishing:

1. Run the checks above.
2. Build release binaries or Docker images for the target platforms.
3. Run at least one CPU and one CUDA smoke test when CUDA artifacts changed.
4. Confirm model download/conversion instructions still match [MODELS.md](MODELS.md).
5. Update release notes under `docs/releases/`.
