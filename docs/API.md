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
| `--sv-embedding <path>` | Optional v2Pro speaker-verification embedding safetensors. |
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

`voices/<name>/voice.json` may also bind fine-tuned model weights. The HTTP server keeps the same
request shape (`voice` + `text`) and lazily loads the model pair for that voice:

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频对应的文字",
  "gpt_model": "character_a/gpt.safetensors",
  "sovits_model": "character_a/sovits.safetensors"
}
```

Model paths are relative to `--models-dir`; reference audio and SV embedding paths are relative to
the voice directory. Model pairs use a bounded LRU cache (two entries by default), while BERT and
HuBERT are shared between pipelines. Change the limit with `--max-cached-pipelines`.

### `POST /tts`

Synthesize one WAV.

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "你好世界。",
    "text_language": "zh",
    "refer_wav_path": "voices/demo/ref.wav",
    "prompt_text": "参考音频对应的文字"
  }' \
  --output output.wav
```

Request-supplied `refer_wav_path` and `sv_embedding` files must be inside `--voices-dir` by default.
Locally managed `voice.json` paths are trusted. For a debugging setup that intentionally reads other
directories, start the server with `--allow-external-reference-paths`.

With a voice profile:

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"你好世界。"}' \
  --output output.wav
```

For agent integrations that stream text from an LLM, prefer sending one finished speech chunk per `/tts` request instead of waiting for a full paragraph:

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"先帝创业未半而中道崩殂。"}' \
  --output chunk_001.wav
```

Keep chunks short and speech-like. For Chinese, 10-25 spoken characters per chunk is usually safer than one long paragraph. The client should play returned chunks in order.

Common aliases are accepted for easier client integration:

| Canonical field | Accepted aliases |
| --- | --- |
| `text` | `input` |
| `text_language` | `language`, `lang`, `languageCode` |
| `refer_wav_path` | `reference_audio`, `referenceAudio`, `prompt_wav_path`, `promptWavPath` |
| `prompt_text` | `reference_text`, `referenceText` |
| `sv_embedding` | `svEmbedding`, `speakerEmbedding` |

Long text quality controls:

| Field | Purpose |
| --- | --- |
| `split_sentences` | Enable chunked synthesis. Defaults to the voice profile value, usually `true`. |
| `split_method` | `sentence` keeps sentence cadence and now protects very long comma clauses; `cut5` also splits on commas/semicolons like Python GPT-SoVITS. |
| `min_sentence_chars` | Merge very short chunks until this many characters are reached. |
| `sentence_gap_ms` | Silence between chunks. |
| `sentence_fade_ms` | Fade each chunk in/out before concatenation. |
| `max_tokens` | Maximum semantic tokens per chunk. Raise only when intentionally synthesizing longer chunks. |
| `repetition_penalty` | GPT repetition penalty. |

CamelCase aliases such as `splitMethod`, `minSentenceChars`, `maxTokens`, and `repetitionPenalty` are also accepted.

For long Chinese prose, prefer explicit chunking:

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{
    "voice": "demo",
    "text": "臣亮言：先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "split_method": "sentence",
    "min_sentence_chars": 12,
    "sentence_gap_ms": 120,
    "sentence_fade_ms": 8
  }' \
  --output prose.wav
```

Chinese pronunciation annotations can be embedded directly in `text` for polyphonic characters:

```json
{
  "voice": "demo",
  "text": "这个人很好[hao4]学，银行的行[hang2]长正在重[zhong4]新安排会议。"
}
```

The service removes the bracketed markers before BERT/G2P alignment and forces the marked character to use that pinyin. Use tone `1`-`5`; `5` is neutral tone.

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

Synthesize one already-complete text request as sentence chunks and stream one WAV response.

```bash
curl -X POST http://localhost:9880/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"第一句话。第二句话。第三句话。"}' \
  --output stream.wav
```

This endpoint is useful when the client already has the full text and wants lower first-byte latency than buffered `/tts`. It is not a token-streaming input endpoint for LLM output. When an agent is still generating text, split complete sentences on the client side and call `/tts` once per speech chunk.

On a single local service instance, concurrent TTS requests are queued behind one inference pipeline. Parallel calls mostly add scheduling complexity unless you run multiple workers or multiple GPUs.

### `POST /tts/batch`

Synthesize multiple texts. Speaker/reference features are computed once, and results are returned as NDJSON.

```bash
curl -X POST http://localhost:9880/tts/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "voice": "demo",
    "texts": ["第一句话", "第二句话", "第三句话"]
  }' | while IFS= read -r line; do
    index=$(echo "$line" | jq -r '.index')
    echo "$line" | jq -r '.wav_base64' | base64 -d > "out_${index}.wav"
    echo "$line" | jq -r '"[\(.index)] \(.duration_s)s  \(.inference_ms)ms"'
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

## Errors

HTTP API errors use one JSON shape:

```json
{"success":false,"error":"text must not be empty","message":"text must not be empty"}
```

Requests are rejected before inference when required text is empty, the voice profile cannot be loaded, the language is unsupported, reference audio is missing, or reference text is empty.
