# API

This page follows the current source branch. See [version scope](README.md#先选你的版本)
for differences from published v1.2.1. Numeric normalization from #10 is unreleased.

GPT-SoVITS-RS exposes three integration surfaces:

- [CLI](#cli): one-shot synthesis and inspection.
- [Rust API](#rust-api): embed the inference pipeline in another Rust program.
- [HTTP API](#http-server): call a running service from an application.

Examples use the deployment default `http://127.0.0.1:9880` and a configured `demo` voice.
For the [first-run demo](FIRST_RUN.zh-CN.md), use port **9881**. Its reference is English;
`text_language: "auto"` detects reference and target languages independently.
Create your own voice with the [deployment guide](DEPLOYMENT.md#准备目录和音色).

## CLI

```bash
./gpt-sovits \
  --text "你好，这是一次语音合成测试。" \
  --reference-audio ref.wav \
  --reference-text "参考音频对应的文字" \
  --output output.wav
```

Useful options for one-shot synthesis:

| Option | Purpose |
|---|---|
| `--doctor` | Check model files, voice profile, reference audio/text, and device setup without running inference. |
| `--voice <name>` | Load `voices/<name>/voice.json`. |
| `--voices-dir <dir>` | Directory containing voice profiles; defaults to `voices`. |
| `--models-dir <dir>` | Search a custom model directory. |
| `--language <language>` | Use the voice's language by default; `auto` detects reference and target languages independently. |
| `--device <device>` | `auto`, `cuda`, `cpu`, or `mps`; acceleration requires a build with that backend. Published macOS binaries are CPU-only. |
| `--mode <mode>` | `auto` (default), `plain`, `kv`, or `cuda-graph`. Auto selects CUDA Graph for CUDA F32 unless disabled, and KV otherwise. |
| `--split-sentences` | Split long text and concatenate chunks. |
| `--split-method <method>` | `sentence` (default) or `cut5` (also splits at commas/semicolons). |
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
These are per-chunk internal timings, not HTTP latency or fully synchronized GPU
kernel timings. See [measurement boundaries](PERFORMANCE.md#指标定义) before comparing results.

## Rust API

```rust,no_run
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = Pipeline::new(Config::builder().with_device("auto").build())?;
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

    let audio = pipeline.inference_with_mode(
        "auto",
        "你好，这是合成语音。",
        "ref.wav",
        "参考音频对应的文字",
        &options,
    )?;

    audio.save("output.wav")?;
    Ok(())
}
```

Decode methods:

| Method | GPT strategy | Typical use |
|---|---|---|
| `inference()` | Recompute sequence every step | Debugging and parity checks. |
| `inference_kv_cache()` | Prefill + single-token KV decode | CPU and explicit KV mode. |
| `inference_cuda_graph()` | KV decode with CUDA Graph replay | CUDA F32 production path. |

`inference_with_mode("auto", ...)` selects the device-appropriate path. These methods
synthesize one chunk; `inference_split_with_method()` provides splitting and concatenation.
For v2Pro, also set `options.sv_embedding` and use `preload_speaker_with_options()`
when preloading a reference. See the [implementation and cache notes](../IMPLEMENTATION.md).

All three methods use the same SoVITS synthesis path, so audio quality should match for the same generated semantic tokens.

## HTTP Server

Use an existing [deployment](DEPLOYMENT.md), or start a source build with prepared models:

```bash
cargo run --locked --release --features http-api --bin gpt-sovits -- \
  --http --models-dir models --voices-dir voices --port 9880
```

For a CUDA build, use `--features cuda,http-api` and add `--device cuda` to the server arguments.

Starting with 1.2.1, standalone binaries default to `127.0.0.1`.
For LAN access, explicitly set `--host 0.0.0.0`; the server has no authentication.
`--host <IP>` overrides `GPT_SOVITS_HOST`;
both accept IPv4 or unbracketed IPv6 literals (`::1`, `::`). Docker/Compose sets
the environment variable to `0.0.0.0` for port forwarding. Published v1.2.0
binaries always bind to `0.0.0.0` and do not accept `--host`.

| Endpoint | Response | Use |
| --- | --- | --- |
| [`GET /health`](#get-health) | Plain text `OK` | Readiness |
| [`GET /voices`](#get-voices) | JSON | Available voice IDs |
| [`GET /status`](#get-status) | JSON | Queue and model-cache state |
| [`POST /warmup`](#post-warmup) | JSON | Preload a voice |
| [`POST /tts`](#post-tts) | Complete WAV | One text request |
| [`POST /tts/stream`](#post-ttsstream) | Streaming WAV | Play a complete text request as chunks arrive |
| [`POST /tts/batch`](#post-ttsbatch) | NDJSON | Multiple texts, one result per item |
| [`POST /v1/audio/speech`](#post-v1audiospeech) | WAV or PCM | Compatible speech clients |

### `GET /health`

Health check.

```bash
curl --fail http://127.0.0.1:9880/health
```

### `GET /voices`

List voice profiles under `voices/`.

```bash
curl --fail http://127.0.0.1:9880/voices
```

Example response:

```json
{"voices":["demo","character_a"]}
```

### `GET /status`

Returns lightweight runtime status without loading another model:

```json
{"status":"ready","cached_pipelines":1,"pipeline_cache_capacity":2,"pipeline_cache_hits":4,"pipeline_cache_misses":1,"pipeline_evictions":0,"queued_requests":0,"busy":false,"gpu_inference_serialized":true,"max_text_chars":10000,"max_batch_items":64,"queue_timeout_seconds":120}
```

The counters are process-local and reset when the service restarts. `queued_requests` counts calls
waiting for the serialized inference slot; `busy` includes model loading, warmup, and synthesis.
Synthesis requests that cannot enter the inference slot within `--queue-timeout-secs` receive HTTP 503. The
timeout only covers queue waiting; model loading and inference are not cancelled midway.

### `POST /warmup`

Loads one voice's model pair and reference features before the first synthesis request:

```bash
curl --fail-with-body http://127.0.0.1:9880/warmup \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo"}'
```

Warming more voices than the configured cache capacity evicts the least recently used pipeline.
For automatic startup warmup, pass `--preload-voices demo` or set
`PRELOAD_VOICES=demo` in Compose; separate multiple voice IDs with commas. The server fails startup when a configured voice cannot be
loaded, so health checks never report ready with a broken preload configuration.

HTTP requests accept at most 10,000 Unicode characters per synthesis item and 64 items per batch by
default. Change these limits with `--max-text-chars` and `--max-batch-items`, or with
`MAX_TEXT_CHARS` and `MAX_BATCH_ITEMS` in Compose.

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

Synthesize one WAV using a configured voice:

```bash
curl --fail-with-body http://127.0.0.1:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"你好世界。","text_language":"auto"}' \
  --output output.wav
```

For debugging, supply reference files and their transcript directly:

```bash
curl --fail-with-body http://127.0.0.1:9880/tts \
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

HTTP synthesis settings use this precedence: request fields, then `voice.json`, then
built-in defaults. For LLM output, send completed sentences in order;
[agent integration](AGENT_INTEGRATION.md) covers chunking, playback and cancellation.

| Field | Default when absent from the voice | Purpose |
| --- | --- | --- |
| `text_language` | `zh` | `auto` detects each text's language; `zh` and `en` force Chinese or English |
| `top_k` | `15` | GPT top-k sampling |
| `top_p` | `0.95` | GPT nucleus sampling |
| `temperature` | `0.8` | GPT sampling temperature |
| `speed` | `1.0` | Speech-speed multiplier |

The voice field for language is `language`; the request field is `text_language`.
The legacy `prompt_language` field is accepted but ignored. Use `auto` when reference
and target languages differ. Set the decode `mode` in `voice.json`.

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
| `split_method` | `sentence` splits at sentence ends and bounds long clauses; `cut5` also splits at commas/semicolons. |
| `min_sentence_chars` | Merge very short chunks until this many characters are reached. |
| `sentence_gap_ms` | Silence between chunks. |
| `sentence_fade_ms` | Fade each chunk in/out before concatenation. |
| `max_tokens` | Maximum semantic tokens per chunk. Raise only when intentionally synthesizing longer chunks. |
| `repetition_penalty` | GPT repetition penalty. |

CamelCase aliases such as `splitMethod`, `minSentenceChars`, `maxTokens`, and `repetitionPenalty` are also accepted.

For long Chinese prose, prefer explicit chunking:

```bash
curl --fail-with-body http://127.0.0.1:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{
    "voice": "demo",
    "text": "臣亮言：先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "auto",
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
  "text": "这个人很好[hao4]学，银行的行[hang2]长正在重[chong2]新安排会议。",
  "text_language": "auto"
}
```

The service removes the bracketed markers before BERT/G2P alignment and forces the marked character to use that pinyin. Use tone `1`-`5`; `5` is neutral tone.

On unreleased master, Chinese numbers such as `50%`, `3.140`, and `-3.5` are expanded
before G2P. Fullwidth digits work too; annotations remain bound to the original
characters. Published v1.2.1 does not include this fix. Dates, currency, phone numbers,
and English numbers still need caller-supplied spoken forms when exact reading matters.

The response includes metadata headers such as:

```text
X-TTS-Voice
X-TTS-Language
X-TTS-Text-Chars
X-TTS-Duration-S
X-TTS-Sample-Rate
X-TTS-Channels
```

`X-TTS-Language` reports the selected language mode, including the literal `auto`.

### `POST /tts/stream`

Synthesize one already-complete text request as sentence chunks and stream one WAV response.

```bash
curl --fail-with-body http://127.0.0.1:9880/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"第一句话。第二句话。第三句话。","text_language":"auto"}' \
  --output stream.wav
```

Playback can begin after the first synthesized chunk arrives. The request contains the complete
text; for LLM token streams, use the client-side queue described in [agent integration](AGENT_INTEGRATION.md).

The stream uses an unknown-length WAV header and PCM16 samples at the generated
sample rate (currently 32 kHz mono). It always splits the input, regardless of
`split_sentences`; `split_method` controls the boundaries. A synthesis error after
HTTP 200 terminates the body: HTTP status alone does not establish completeness.
Some file players require repaired RIFF/data lengths after saving the stream;
the regression capture tool writes both the raw response and a repaired WAV.
HTTP headers can arrive before any PCM, so curl's `time_starttransfer` is not audio TTFA.

### `POST /tts/batch`

Synthesize multiple texts sequentially, reusing reference features. Results arrive as NDJSON,
one line per item. The shell example requires Bash, curl 7.76+, jq and base64.

```bash
set -euo pipefail
curl --fail-with-body http://127.0.0.1:9880/tts/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "voice": "demo",
    "text_language": "auto",
    "texts": ["第一句话", "第二句话", "第三句话"]
  }' | while IFS= read -r line; do
    if ! echo "$line" | jq -e '.error == null and (.wav_base64 | type == "string")' > /dev/null; then
      echo "$line" >&2
      exit 1
    fi
    index=$(echo "$line" | jq -r '.index')
    echo "$line" | jq -r '.wav_base64' | base64 -d > "out_${index}.wav"
    echo "$line" | jq -r '"[\(.index)] \(.duration_s)s  \(.inference_ms)ms"'
done
```

Success and failure lines have these shapes; `error` is omitted on success and
`wav_base64` is omitted on failure:

```jsonl
{"index":0,"voice":"demo","language":"auto","text_chars":4,"wav_base64":"...","sample_rate":32000,"duration_s":1.5,"inference_ms":820}
{"index":1,"voice":"demo","language":"auto","text_chars":4,"error":"synthesis failed","sample_rate":0,"duration_s":0.0,"inference_ms":12}
```

### `POST /v1/audio/speech`

OpenAI-compatible speech adapter for agent frameworks and clients that already know the OpenAI audio API shape.

```bash
curl --fail-with-body http://127.0.0.1:9880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-sovits",
    "voice": "demo",
    "input": "你好，这是本地语音服务。",
    "text_language": "auto",
    "response_format": "wav"
  }' \
  --output output.wav
```

`input` and `voice` are required. `response_format` defaults to `wav`; `pcm` returns
raw signed PCM16 little-endian, currently 32 kHz mono. MP3 and Opus requests return 400.
`speed` overrides the voice's speed setting.

`model` and `instructions` are accepted for client compatibility and currently ignored.
The selected voice profile determines the GPT/SoVITS weights. To change pronunciation,
use annotations in `input`; `instructions` does not control emotion or delivery.

Set the client base URL to `http://127.0.0.1:9880/v1` when using compatible agent frameworks.

## Errors

Application validation and inference errors before streaming use this JSON shape:

```json
{"success":false,"error":"text must not be empty","message":"text must not be empty"}
```

Requests are rejected before inference when required text is empty, the voice profile cannot be loaded, the language is unsupported, reference audio is missing, or reference text is empty.

Malformed JSON/content-type errors may instead come directly from Axum. Synthesis queue
timeouts return 503; buffered inference failures return 500. `/warmup` failures return 400.
A batch may return
HTTP 200 with an individual line containing `error` and no `wav_base64`, so check
each line before decoding `wav_base64`. A broken WAV stream fails at the transport
level rather than appending a JSON error to its PCM bytes.
