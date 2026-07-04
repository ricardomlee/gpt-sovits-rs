# API

GPT-SoVITS-RS exposes three integration surfaces:

- CLI: one-shot synthesis and inspection.
- Rust API: embed the inference pipeline in another Rust program.
- HTTP API: run a local TTS service.

## CLI

```bash
./gpt-sovits \
  --text "你好，这是一次语音合成测试。" \
  --reference-audio ref.wav \
  --reference-text "参考音频对应的文字" \
  --output output.wav
```

Useful options:

| Option | Purpose |
|---|---|
| `--doctor` | Check model files, voice profile, reference audio/text, and device setup without running inference. |
| `--voice <name>` | Load `voices/<name>/voice.json`. |
| `--models-dir <dir>` | Search a custom model directory. |
| `--device auto|cuda|cpu|mps` | Select runtime device. |
| `--mode auto|plain|kv|cuda-graph` | Select GPT decode path. |
| `--split-sentences` | Split long text and concatenate chunks. |
| `--split-method sentence|cut5` | Use smooth sentence splitting or Python-compatible punctuation splitting. |
| `--max-tokens <n>` | Raise this for long sentences. |
| `--list-voices` | List voice profiles without loading models. |
| `--inspect <model.safetensors>` | Print model tensor names and shapes. |

Run setup diagnostics:

```bash
./gpt-sovits --doctor --voice demo
```

`doctor` is intentionally lightweight: it validates file layout and safetensors headers, but does not load all model tensors or run inference.

The CLI logs a profile line:

```text
profile mode=... target=... ref=... target_bert=... gpt=... sovits=... total=...
```

Use it to see where inference time is going.

## Rust API

```rust
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};

let mut pipeline = Pipeline::new(Config::builder().with_device("cuda").build())?;
pipeline.load_gpt("models/gpt-model.safetensors")?;
pipeline.load_sovits("models/sovits-model.safetensors")?;
pipeline.load_bert("models/bert/bert.safetensors")?;
pipeline.load_hubert("models/hubert/hubert.safetensors")?;

let options = InferenceOptions::builder()
    .top_k(15)
    .top_p(0.95)
    .temperature(0.8)
    .language(Language::Chinese)
    .max_tokens(500)
    .build();

let audio = pipeline.inference_kv_cache(
    "你好，这是合成语音。",
    "ref.wav",
    "参考音频对应的文字",
    &options,
)?;

audio.save("output.wav")?;
```

Decode methods:

| Method | GPT strategy | Typical use |
|---|---|---|
| `inference()` | Recompute sequence every step | Debugging and parity checks. |
| `inference_kv_cache()` | Prefill + single-token KV decode | CPU, MPS, and explicit KV mode. |
| `inference_cuda_graph()` | KV decode with CUDA Graph replay | CUDA F32 production path. |

All three methods use the same SoVITS synthesis path, so audio quality should match for the same generated semantic tokens.

## HTTP Server

```bash
cargo run --release --features "cuda,http-api" --bin gpt-sovits -- \
  --http --port 9880
```

### `GET /health`

Health check.

```bash
curl http://localhost:9880/health
```

### `GET /voices`

List voice profiles under `voices/`.

```bash
curl http://localhost:9880/voices
```

Example response:

```json
{"voices":["demo","character_a"]}
```

### `POST /tts`

Synthesize one WAV.

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "你好世界。",
    "text_language": "zh",
    "refer_wav_path": "ref.wav",
    "prompt_text": "参考音频对应的文字"
  }' \
  --output output.wav
```

With a voice profile:

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"你好世界。"}' \
  --output output.wav
```

The response includes metadata headers such as:

```text
X-TTS-Voice
X-TTS-Language
X-TTS-Text-Chars
X-TTS-Duration-S
X-TTS-Sample-Rate
X-TTS-Channels
```

### `POST /tts/stream`

Synthesize one request as sentence chunks and stream the WAV response.

```bash
curl -X POST http://localhost:9880/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"第一句话。第二句话。第三句话。"}' \
  --output stream.wav
```

### `POST /tts/batch`

Synthesize multiple texts. Speaker/reference features are computed once, and results are returned as NDJSON.

```bash
curl -X POST http://localhost:9880/tts/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "voice": "demo",
    "texts": ["第一句话", "第二句话", "第三句话"]
  }' | while IFS= read -r line; do
    echo "$line" | python3 -c "
import sys,json,base64
d=json.load(sys.stdin)
open(f'out_{d[\"index\"]}.wav','wb').write(base64.b64decode(d['wav_base64']))
print(f'[{d[\"index\"]}] {d[\"duration_s\"]:.2f}s  {d[\"inference_ms\"]}ms')
"
done
```

Each line includes fields like:

```json
{"index":0,"wav_base64":"...","sample_rate":32000,"duration_s":1.5,"inference_ms":820}
```

### `POST /v1/audio/speech`

OpenAI-compatible speech adapter for agent frameworks and clients that already know the OpenAI audio API shape.

```bash
curl -X POST http://localhost:9880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-sovits",
    "voice": "demo",
    "input": "你好，这是本地语音服务。",
    "response_format": "wav"
  }' \
  --output output.wav
```

Supported `response_format` values:

- `wav`
- `pcm`

Set the client base URL to `http://localhost:9880/v1` when using compatible agent frameworks.
