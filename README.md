# GPT-SoVITS-RS

<p align="center">
  <img src="assets/gpt-sovits-rs-logo.svg" alt="GPT-SoVITS-RS" width="880">
</p>

<p align="center">
  <b>Run GPT-SoVITS inference as a small Rust binary, CLI, or HTTP service.</b><br>
  No Python runtime or conversion scripts, simpler Docker deployment, and a stable API for agents and local tools.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#samples">Samples</a> ·
  <a href="#models">Models</a> ·
  <a href="#http-api">HTTP API</a> ·
  <a href="docs/DEPLOYMENT.md">Deployment</a>
</p>

## What Is This?

GPT-SoVITS-RS is a Rust inference implementation for trained GPT-SoVITS voices. It is meant for local assistants, scripts, NAS boxes, and Docker services where installing and operating a full Python/PyTorch stack is inconvenient.

Use the original [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) for training, fine-tuning, dataset preparation, and advanced experiments. Use this project when you already have compatible weights and want a deployable inference service.

Inference and checkpoint conversion run without Python. Use `gpt-sovits-convert` to convert raw PyTorch `.ckpt` / `.pth` / `.bin` / `.pt` files into runtime `safetensors`.

## Why Use It?

- **Deployable runtime:** release binary, Docker Compose, CLI, Rust API, and HTTP API.
- **Model conversion included:** GPT, SoVITS, Chinese RoBERTa, and Chinese HuBERT are converted to `safetensors`.
- **Voice profiles:** package reference audio, reference text, language, split settings, and sampling defaults into `voices/<name>/voice.json`.
- **Long-text friendly:** sentence splitting, reference feature caching, sentence gap/fade, streaming endpoint.
- **Agent friendly:** OpenAI-compatible `/v1/audio/speech` adapter.
- **GPU path:** Candle CUDA with KV cache and CUDA Graph for the GPT decode loop.

## Quick Start

### Docker

Docker Compose pulls prebuilt images from GHCR. Models, voices, and outputs stay in mounted folders.
The images do not contain model weights, but they do include both `gpt-sovits` and
`gpt-sovits-convert`.

```bash
git clone https://github.com/ricardomlee/gpt-sovits-rs.git
cd gpt-sovits-rs

mkdir -p models/bert models/hubert voices outputs
gpt-sovits-convert gpt /path/to/s1bert25hz.ckpt models/gpt-model.safetensors
gpt-sovits-convert sovits /path/to/s2G2333k.pth models/sovits-model.safetensors
gpt-sovits-convert bert /path/to/chinese-roberta-wwm-ext-large/pytorch_model.bin models/bert/bert.safetensors
cp /path/to/chinese-roberta-wwm-ext-large/tokenizer.json models/bert/tokenizer.json
gpt-sovits-convert hubert /path/to/chinese-hubert-base/pytorch_model.bin models/hubert/hubert.safetensors

cp .env.example .env
```

If you want a Docker-only conversion flow, run the converter from the image and mount your own
source-model directory plus a writable output directory:

```bash
docker run --rm \
  -v "$PWD/models:/models" \
  -v "/path/to/source-models:/source:ro" \
  --entrypoint gpt-sovits-convert \
  ghcr.io/ricardomlee/gpt-sovits-rs:latest \
  gpt /source/s1bert25hz.ckpt /models/gpt-model.safetensors
```

Create a voice profile from your own 3-10 second reference clip:

```bash
mkdir -p voices/demo
cp /path/to/reference.wav voices/demo/ref.wav
cat > voices/demo/voice.json <<'JSON'
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频里逐字对应的文字",
  "language": "zh",
  "mode": "auto",
  "split_sentences": true
}
JSON
```

CPU / NAS:

```bash
docker compose -f compose.cpu.yml up -d
```

CUDA:

```bash
docker compose -f compose.cuda.yml up -d
```

Check the server:

```bash
docker compose -f compose.cpu.yml ps   # or compose.cuda.yml
curl http://localhost:9880/health
curl http://localhost:9880/voices
```

Synthesize with a voice profile:

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"你好，这是 GPT-SoVITS-RS 的本地语音服务。"}' \
  --output output.wav
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for CUDA image tags, `.env` options, volumes, and production notes.

Check a local setup before running inference:

```bash
./gpt-sovits --doctor --voice demo
```

### Release Binary

Download a release package. It contains both the inference server and the Rust model converter:

```bash
./gpt-sovits --version
./gpt-sovits-convert --version
./gpt-sovits \
  --text "你好，这是一次语音合成测试。" \
  --reference-audio ref.wav \
  --reference-text "参考音频对应的文字" \
  --output output.wav
```

Linux CUDA users should prefer the CUDA Docker image, or build from source for the local compute capability.

### From Source

```bash
# x86_64 CPU, with MKL
cargo build --release --features mkl

# generic CPU, useful for ARM/NAS
cargo build --release

# CUDA
cargo build --release --features cuda

# HTTP API
cargo build --release --features http-api,mkl
cargo build --release --features http-api,cuda
```

Requirements: stable Rust, `libsoxr-dev`, and CUDA Toolkit 13.x for CUDA builds.

## Samples

GitHub README does not reliably render standalone WAV players inline. Keep sample audio under `examples/samples/` and link to it from the table below when the voice/audio is safe to publish.

| Voice | Text | Output | Notes |
|---|---|---|---|
| demo | 你好，这是 GPT-SoVITS-RS 的本地语音服务。 | pending | Add a small public-safe WAV sample before publishing a release showcase. |

Recommended sample format:

```text
examples/samples/
├── README.md
├── demo_zh.wav
└── demo_long_text.wav
```

If you want browser playback directly on GitHub, use a GitHub Pages demo page with HTML `<audio controls>`, or publish a short `.mp4` sample with waveform plus audio.

## Models

Models are not bundled with the binary or Docker images. Download official GPT-SoVITS v2
checkpoints, or use your own trained checkpoints, then convert them with the Rust converter:

```bash
mkdir -p models/bert models/hubert
gpt-sovits-convert gpt /path/to/s1bert25hz.ckpt models/gpt-model.safetensors
gpt-sovits-convert sovits /path/to/s2G2333k.pth models/sovits-model.safetensors
gpt-sovits-convert bert /path/to/chinese-roberta-wwm-ext-large/pytorch_model.bin models/bert/bert.safetensors
cp /path/to/chinese-roberta-wwm-ext-large/tokenizer.json models/bert/tokenizer.json
gpt-sovits-convert hubert /path/to/chinese-hubert-base/pytorch_model.bin models/hubert/hubert.safetensors
```

Expected layout:

```text
models/
├── gpt-model.safetensors
├── sovits-model.safetensors
├── bert/
│   ├── bert.safetensors
│   └── tokenizer.json
└── hubert/
    └── hubert.safetensors
```

Custom GPT-SoVITS v2 or v2Pro weights:

```bash
gpt-sovits-convert gpt /path/to/custom-gpt.ckpt models/gpt-model.safetensors
gpt-sovits-convert sovits /path/to/custom-sovits.pth models/sovits-model.safetensors
```

For v2Pro voices, convert the optional speaker-verification embedding generated by GPT-SoVITS preprocessing and pass it with the voice:

```bash
gpt-sovits-convert sv /path/to/logs/voice_v2pro/7-sv_cn/ref.wav.pt voices/demo/ref_sv.safetensors
```

Read [docs/MODELS.md](docs/MODELS.md) for the full model layout and v2Pro notes.

## Voice Profiles

Create `voices/<name>/voice.json`:

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频对应的文字",
  "sv_embedding": "ref_sv.safetensors",
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

Then call it by name:

```bash
./gpt-sovits --voice demo --text "这句话会使用 demo 音色。"
```

Reference audio still matters. A 3-10 second clean clip with exactly matching text usually works best.

## HTTP API

Start the service:

```bash
cargo run --release --features "cuda,http-api" --bin gpt-sovits -- \
  --http --port 9880
```

Core endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | health check |
| `GET /voices` | list local voice profiles |
| `POST /tts` | synthesize one WAV |
| `POST /tts/stream` | stream sentence chunks |
| `POST /tts/batch` | synthesize multiple texts as NDJSON |
| `POST /v1/audio/speech` | OpenAI-compatible speech adapter |

Minimal request:

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"你好世界。"}' \
  --output output.wav
```

Full API examples live in [docs/API.md](docs/API.md).

For real-time assistants, send short completed speech chunks to `/tts` as the LLM produces sentences; use `/tts/stream` when the full text is already known. See [docs/AGENT_INTEGRATION.md](docs/AGENT_INTEGRATION.md).

## Troubleshooting

Run doctor first:

```bash
./gpt-sovits --doctor --voice demo
```

It checks the model layout, safetensors headers, BERT tokenizer, requested device, voice profile, reference audio, and reference text without loading the full model stack.

## Performance

The project is optimized for practical local inference and deployment simplicity, not for matching every Python fast-path benchmark. Current highlights:

- CPU x86_64 builds can use MKL and a faster Conv1d path.
- CUDA builds use KV cache and CUDA Graph for GPT decode.
- Speaker/reference features are cached across repeated calls.
- Long text is split and concatenated for stability; batch-parallel long-text synthesis is not the current product focus.

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for measured numbers and profiling commands.

## Documentation

| Topic | Link |
|---|---|
| Model download and conversion | [docs/MODELS.md](docs/MODELS.md) |
| Docker, binary, and server deployment | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| HTTP and Rust API details | [docs/API.md](docs/API.md) |
| Performance baseline | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| Agent integration | [docs/AGENT_INTEGRATION.md](docs/AGENT_INTEGRATION.md) |
| Development and debugging | [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) |
| Prosody and speed | [docs/PROSODY.md](docs/PROSODY.md) |
| Product goal | [docs/PRODUCT_GOAL.md](docs/PRODUCT_GOAL.md) |

## License

MIT License. Model weights come from their respective publishers and keep their own licenses and usage restrictions.

## Credits

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) by RVC-Boss
- [Candle](https://github.com/huggingface/candle) by Hugging Face
