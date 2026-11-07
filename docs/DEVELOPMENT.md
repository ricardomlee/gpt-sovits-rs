# Development

See [version scope](README.md) before comparing a source build with a release.
The root package contains the inference binary and the Rust weight converter.
The HTTP command-line client builds separately under `client/`.

## Build From Source

Use stable Rust and Cargo, a C/C++ build toolchain, pkg-config, and libsoxr headers.
HTTP builds also need CMake and libmp3lame. Cargo builds the bundled libopus source.
Codec roundtrip tests run on Linux and both macOS architectures in CI.
For a running v2/v2Pro service, install FFmpeg and jq and run
`bash examples/verify-http-formats.sh http://127.0.0.1:9880 demo /tmp/speech-formats`.
This maintainer check retains the requests, response headers, encoded files and
independently decoded WAVs. The service itself does not invoke FFmpeg.
CI uses Ubuntu 24.04 and the stable toolchain; the repository does not currently
declare a tested minimum Rust version.

```bash
sudo apt-get install -y build-essential pkg-config libsoxr-dev libmp3lame-dev cmake
cargo build --locked --release --features http-api
target/release/gpt-sovits --version
target/release/gpt-sovits-convert --version
```

On macOS, run `brew install libsoxr lame pkg-config cmake` and
`export LAME_LIB_DIR="$(brew --prefix lame)/lib"`. These builds are
CPU-only; this repository does not enable Candle's Metal backend. Linux ARM users
can build natively; the optional `mkl` feature is intended for x86_64 CPU builds.

CUDA builds additionally need the NVIDIA driver, CUDA Toolkit/nvcc, and a matching
GPU architecture. For an RTX 4060 Ti:

```bash
CUDA_COMPUTE_CAP=89 cargo build --locked --release --features cuda,http-api
```

Other GPUs need their own compute capability. The published CUDA runtime images
do not contain a source-build toolchain. Models and voices are prepared separately:
[models](MODELS.md), [deployment](DEPLOYMENT.md).

## Code Map

| Location | Responsibility |
| --- | --- |
| `src/cli.rs`, `src/config/`, `src/doctor.rs` | CLI dispatch, configuration, lightweight setup checks |
| `src/bin/gpt-sovits-convert.rs` | Rust checkpoint and embedding conversion |
| `client/` | Standalone HTTP CLI; independent of the inference engine and CUDA |
| `src/server.rs`, `src/server/` | Routes, handlers, lifecycle, request resolution, model registry and serialized inference state |
| `src/voice.rs`, `src/model_paths.rs` | Voice defaults, model bindings and discovery |
| `src/inference/` | Pipeline dispatch, reference cache, STFT, splitting and concatenation |
| `src/models/` | GPT/BERT/HuBERT/SoVITS, KV and CUDA Graph |
| `src/text_frontend/` | Pronunciation annotations, language detection, numeric normalization, G2P and tone sandhi |
| `src/audio_checks.rs`, `tests/` | Signal checks and model-free/opt-in integration tests |
| `examples/gpu_benchmark/`, `examples/speech_regression/` | Local GPU reports and offline HTTP speech regression |

The [implementation notes](../IMPLEMENTATION.md) explain the runtime paths and
known limits. Dependency versions live in Cargo.toml/Cargo.lock.

## Local Checks

```bash
cargo fmt --all -- --check
cargo test --locked
cargo test --locked --features http-api
cargo test --locked --example speech_regression
cargo test --locked --example gpu_benchmark
cargo clippy --locked --all-targets --features http-api -- -D warnings
cargo check --locked --bin gpt-sovits-convert
```

Default tests do not need models, a GPU, private audio, or network access.
Documentation tests check local Markdown link paths, JSON examples, Bash syntax
and documented CLI flags; the Rust API example is compiled as a no-run doctest.
These checks do not execute deployment snippets or verify external URLs.

## HTTP Client

The standalone client has its own manifest and lockfile, so installing it does not
build Candle or require libsoxr. Run its checks separately from the root package:

```bash
cargo fmt --manifest-path client/Cargo.toml -- --check
cargo clippy --locked --manifest-path client/Cargo.toml --all-targets -- -D warnings
cargo test --locked --manifest-path client/Cargo.toml
cargo build --locked --release --manifest-path client/Cargo.toml
client/target/release/gpt-sovits-client --version
```

Tests use a loopback mock server and synthetic WAVs to check request bodies,
stdin/stdout, HTTP errors, timeouts, and output-file preservation. CI runs these
without installing inference dependencies. Release jobs include the client in the
Linux/macOS binary archives. Client usage: [README](../client/README.md).

## Real Models and Performance

An opt-in model test honors the selected voice's model pair, SV, and splitting:

```bash
GPT_SOVITS_RUN_MODEL_SMOKE=1 \
GPT_SOVITS_MODELS_DIR=models \
GPT_SOVITS_VOICES_DIR=voices \
GPT_SOVITS_SMOKE_VOICE=demo \
GPT_SOVITS_SMOKE_DEVICE=cuda \
cargo test --locked --release --features cuda --test model_smoke -- --nocapture
```

`quality_smoke` additionally writes WAVs and a signal-quality report:

```bash
cargo run --locked --release --features cuda --example quality_smoke -- \
  --voice demo --output-dir quality_outputs/demo
```

Run the [local GPU benchmark](PERFORMANCE.md) **before every release**, and retain its
JSON/Markdown report plus the audio/ASR evidence. Check speech completeness with ASR
and listening. [Release acceptance](RELEASE_ACCEPTANCE.md) covers candidate images,
HTTP regression, ASR, and upgrade/rollback without touching the existing service.

The old `e2e_quick`, `profile_kv_cache`, and dtype examples have their own fixtures
and measurement boundaries described in the performance guide.

## Docker Checks

```bash
docker compose --env-file .env.example -f compose.cpu.yml config --quiet
docker compose --env-file .env.example -f compose.cuda.yml config --quiet
docker build -t gpt-sovits-rs:dev .
GPT_SOVITS_SMOKE_IMAGE=gpt-sovits-rs:dev \
  cargo test --locked --features http-api --test deployment_smoke -- --ignored --nocapture
```

The deployment smoke test uses an empty env file and no model mounts. It must reach
the expected missing-model diagnostic, catching unsupported Compose arguments and
missing shared libraries. Readiness and audio quality are checked in real-model acceptance.
For a CUDA image, also set `GPT_SOVITS_SMOKE_GPUS=all` on a GPU host.
CI retains v1.2.0 as a compatibility baseline.

## Incremental CUDA Work

`dev-gpu` uses lower optimization cost than `release`; label results separately.

```bash
cargo build --locked --profile dev-gpu --features cuda,http-api
cargo run --locked --profile dev-gpu --features cuda --bin gpt-sovits -- \
  --voice demo --text '你好，这是一次测试。' --output /tmp/demo.wav
```

Set `SOVITS_DEBUG=1` to dump intermediate tensors in the working directory, or
`GPT_SOVITS_SYNC_PROFILE=1` with CLI `--verbose` for synchronized stage analysis.
Disable both during latency measurements; debug output can also contain private data.
GPT/SoVITS stay F32. Half-precision and cuda-oxide experiments are archived in the performance guide.

## Before Tagging

Follow [release acceptance](RELEASE_ACCEPTANCE.md), including the local GPU report.
Check `cargo run --quiet --bin gpt-sovits -- --version` explicitly: the root package defines two binaries.
Update documentation with behavior changes and identify the version introducing each feature.
Do not change pinned first-run images until the new artifacts are published and verified.
