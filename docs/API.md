# API

GPT-SoVITS-RS exposes three integration surfaces:

- [HTTP API](#http-server): call a running service from an application.
- [CLI](#cli): one-shot synthesis and inspection.
- [Rust API](#rust-api): embed the inference pipeline in another Rust program.

For shell scripts and agent command providers, the standalone
[HTTP client](../client/README.md) calls a running service without loading models locally.

Examples use the deployment default `http://127.0.0.1:9880` and a configured `demo` voice.
For the [first-run demo](FIRST_RUN.zh-CN.md), use port **9881**. Its reference is English;
`text_language: "auto"` detects the target language. Reference text has its own
`prompt_language`, which defaults to `auto` independently.
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
| `--language <language>` | Target text language; defaults to the voice's `language`. |
| `--prompt-language <language>` | Reference text language; defaults to the voice's `prompt_language`, or `auto`. |
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

`doctor` checks file layout and safetensors headers. Follow it with a real synthesis
request to verify model loading and audio output.

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

Choose the device before loading, through `Config` for a pipeline or a standalone
model's `load_with_device()` constructor. `Model::to_device()` accepts the current
`Model::device()` name as a case-insensitive no-op; other names return
`ConfigError`. Reload the model's weights to change devices.

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

Reference text is detected independently by default. Use
`.prompt_language(Language::Chinese)` to force a Chinese reference while keeping
`.language(Language::English)` for English output. Direct `InferenceOptions`
struct literals must include `prompt_language` or use `..Default::default()`.

All three methods use the same SoVITS synthesis path, so audio quality should match for the same generated semantic tokens.

For low-level, reference-conditioned `GPTModel` generation, empty phonemes or a
zero token budget return an empty sequence. Static KV and CUDA Graph execution
report insufficient or overflowing KV capacity as `ConfigError`. The higher-level
`Pipeline` continues to require `max_tokens` in `1..=2048`. On CPU, the Graph entry
delegates to dynamic KV decoding.

`GPTModel::generate()` and `generate_with_features()` retain the legacy generator
without reference semantic tokens. A pipeline that enters this compatibility
path logs a warning. The existing public method signatures are unchanged;
[GPT core maintenance](DEVELOPMENT.md#gpt-core) describes the private modules and tests.

## HTTP Server

Use an existing [deployment](DEPLOYMENT.md), or start a source build with prepared models:

```bash
cargo run --locked --release --features http-api --bin gpt-sovits -- \
  --http --models-dir models --voices-dir voices --port 9880
```

For a CUDA build, use `--features cuda,http-api` and add `--device cuda` to the server arguments.

Standalone binaries default to `127.0.0.1`.
For LAN access, explicitly set `--host 0.0.0.0`; the server has no authentication.
`--host <IP>` overrides `GPT_SOVITS_HOST`;
both accept IPv4 or unbracketed IPv6 literals (`::1`, `::`). Docker/Compose sets
the environment variable to `0.0.0.0` for port forwarding.

| Endpoint | Response | Use |
| --- | --- | --- |
| [`GET /health`](#get-health) | Plain text `OK` | Readiness |
| [`GET /voices`](#get-voices) | JSON | Available voice IDs |
| [`GET /status`](#get-status) | JSON | Queue and model-cache state |
| [`POST /warmup`](#post-warmup) | JSON | Preload a voice |
| [`POST /tts`](#post-tts) | Complete WAV | One text request |
| [`POST /tts/stream`](#post-ttsstream) | Streaming WAV | Play a complete text request as chunks arrive |
| [`POST /tts/batch`](#post-ttsbatch) | NDJSON | Multiple texts, one result per item |
| [`POST /v1/audio/speech`](#post-v1audiospeech) | WAV, PCM, MP3 or Ogg/Opus | Compatible speech clients |

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

Reference features are cached by audio path, reference text, reference language, and SV path.
Changing only the target language reuses the same reference features.
Rust `Pipeline` model loads and mutable model/frontend access invalidate this cache;
failed loads preserve the existing models and cache. Call `clear_speaker_cache()`
after replacing reference files in place. Preloading uses the same cache as inference.

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

Effective settings are validated after overrides are merged: `max_tokens` is 1-2048,
`speed` is 0.25-4.0, `top_p` is 0-1, and `temperature` and `repetition_penalty` must
be finite and positive. `sentence_gap_ms` is 0-10000 and `sentence_fade_ms` is 0-1000.
Invalid settings return HTTP 400 before entering the inference queue. CLI and
pipeline callers use the same limits; values are rejected, not clamped.

| Field | Default when absent from the voice | Purpose |
| --- | --- | --- |
| `text_language` | `zh` | Target text language; `auto` detects it, `zh` and `en` force Chinese or English |
| `prompt_language` | `auto` | Reference text language, independent of `text_language` |
| `top_k` | `15` | GPT top-k sampling |
| `top_p` | `0.95` | GPT nucleus sampling |
| `temperature` | `0.8` | GPT sampling temperature |
| `speed` | `1.0` | Speech-speed multiplier |
| `seed` | Random | Optional unsigned 64-bit integer controlling GPT sampling and SoVITS noise |

The voice field for target language is `language`; the request field is `text_language`.
Both voice profiles and requests use `prompt_language` for the reference transcript.
An explicit request overrides the corresponding voice default. To force the same
language for both texts, set both fields to that language.
For Chinese mixed with ASCII words or abbreviations, `auto` uses the mixed Chinese
G2P path even when Latin letters are the majority, so Chinese phones are retained.
Explicit target language selection affects only the target text.

For debugging, pass `"seed": 42` to `/tts`, `/tts/stream`, `/tts/batch`, or
`/v1/audio/speech`, or use CLI `--seed 42`. Rust callers can use
`InferenceOptions::builder().seed(42)`. Successful seeded responses include
`x-tts-seed`; an older service may silently ignore the request field.
Omitting the seed preserves random synthesis. Seeds are request-local, including
zero; voice profiles do not set a default seed. Batch items each start with the
same seed. Sentence chunks use `seed + chunk_index` modulo 2^64, in both buffered
and streaming synthesis. Keep models, reference audio, text, splitting, parameters,
decode mode, build and device fixed when replaying a result. Different hardware,
decode modes or dependency versions can produce different results for the same seed.
Compare decoded PCM rather than compressed-file bytes; containers such as Ogg
have independently generated identifiers. Seeded acoustic noise is generated on
the host and copied to the model device, so measure its cost separately from
normal unseeded inference.
`prompt_language` works on `/tts`, `/tts/stream`, `/tts/batch`, and `/v1/audio/speech`.
Older versions through 1.3.1 ignore it; use `text_language: "auto"` with those servers
when the reference and target languages differ. Set the decode `mode` in `voice.json`.

Common aliases are accepted for easier client integration:

| Canonical field | Accepted aliases |
| --- | --- |
| `text` | `input` |
| `text_language` | `language`, `lang`, `languageCode` |
| `refer_wav_path` | `reference_audio`, `referenceAudio`, `prompt_wav_path`, `promptWavPath` |
| `prompt_text` | `reference_text`, `referenceText` |
| `sv_embedding` | `svEmbedding`, `speakerEmbedding` |

Chunking and generation limits use the same request-over-voice precedence:

| Field | Default when absent from the voice | Purpose |
| --- | --- | --- |
| `split_sentences` | `true` | Enable chunked synthesis. |
| `split_method` | `sentence` | Split at sentence ends and bound long clauses; `cut5` also splits at commas/semicolons. |
| `min_sentence_chars` | `12` | Merge very short chunks until this many characters are reached. |
| `sentence_gap_ms` | `120` | Silence between chunks. |
| `sentence_fade_ms` | `8` | Fade each chunk in/out before concatenation. |
| `max_tokens` | `500` | Maximum semantic tokens per chunk. |
| `repetition_penalty` | `1.35` | GPT repetition penalty. |

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

Chinese numbers such as `50%`, `3.140`, and `-3.5` are expanded
before G2P. Fullwidth digits work too; annotations remain bound to the original
characters.

English text expands `123` to "one hundred twenty three", `12.5%` to
"twelve point five percent", and `21st` to "twenty first". Leading-zero codes and
digits attached to letters are read individually: `007` becomes "zero zero seven",
and `v12` becomes "v one two". Pure numeric input in `auto` defaults to Chinese;
set `text_language: "en"` for an English reading. Dates, times, prices, and phone
numbers still need caller-supplied spoken forms when exact reading matters.

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

`input` and `voice` are required. `response_format` defaults to `wav`:

| Format | Content-Type | Output |
| --- | --- | --- |
| `wav` | `audio/wav` | PCM16 WAV at the model's sample rate, currently 32 kHz mono |
| `pcm` | `application/octet-stream` | Raw signed PCM16 little-endian at the same rate |
| `mp3` | `audio/mpeg` | 64 kbit/s mono MP3, with encoder delay and padding metadata |
| `opus` | `audio/ogg` | Ogg/Opus, 48 kHz mono, 48 kbit/s target VBR |

MP3 and Opus responses contain complete files. Opus is resampled using libsoxr;
its pre-skip and final granule preserve the audio duration. `x-tts-sample-rate`
reports the encoded output rate. Unsupported formats return 400 before inference.
`/tts`, `/tts/stream` and `/tts/batch` continue to return WAV.
`speed` overrides the voice's speed setting.

`model` and `instructions` are accepted for client compatibility and currently ignored.
The selected voice profile determines the GPT/SoVITS weights. To change pronunciation,
use annotations in `input`; `instructions` does not control emotion or delivery.

Set the client base URL to `http://127.0.0.1:9880/v1` when using compatible agent frameworks.
For Hermes, configure its built-in OpenAI provider as shown in
[Agent integration](AGENT_INTEGRATION.md#hermes). The server performs audio encoding;
the binary archives and Docker images include the required libraries.

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

Configured BERT and HuBERT models must load successfully for CLI/server startup.
Feature extraction, device transfer, semantic encoding, and BERT alignment failures
stop inference with the failed stage in the error message. Library callers can
still omit these optional models explicitly.
BERT rejects inputs over 512 tokens (including special tokens) or token IDs outside
its vocabulary before GPU execution. Split long text into shorter chunks.

Reference WAVs support 8/16/24/32-bit PCM and 32-bit float. Truncated samples and
NaN/Inf values are rejected; each PCM width uses its own amplitude scale.
