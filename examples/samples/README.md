# Public Samples

## Sun Showcase

The README and listening page use the maintainer's local **Sun v2Pro + SV** voice,
generated on 2026-10-05. Only the synthesized outputs are included, not the
model pair, SV embedding, or reference recording. These samples are not the LJ Speech voice
used by the downloadable first-run demo below. Do not apply the public-domain
statement about LJ Speech to Sun's source material. These are synthetic samples,
not recordings or endorsements by the original speaker.

| File | Exact target text | Duration |
|---|---|---|
| [sun-greeting.wav](sun-greeting.wav) | 你好，欢迎回来。今天有什么想和我聊聊的吗？ | 3.28 s |
| [sun-zh.wav](sun-zh.wav) | 你的本地语音服务已经准备好了。接下来的语音会在这台电脑上生成。 | 4.84 s |

Both use CPU F32, the local Sun GPT/SoVITS v2Pro model pair and matching SV embedding,
`top_k=15`, `top_p=0.95`, `temperature=0.8`, `mode=auto` (KV on CPU),
`split_method=sentence`, a 120 ms gap, 8 ms fade, 500-token cap, and repetition
penalty 1.35. The Chinese reference is 4.98 seconds with a matching transcript.
Both use `language=zh`. The synthesized outputs have no external edits.
They use the same local inference binary identified in the baseline section below.

Local Qwen3-ASR-0.6B transcriptions match both target texts exactly. This checks
these two outputs, not voice similarity or future generation reliability.

Reproduction requires the same privately held model pair, reference, and SV
embedding; the project does not supply them. Sampling is stochastic, so identical
output is not guaranteed. With an authorized Sun profile containing these paths
and settings, the invocation is:

```bash
gpt-sovits --device cpu --models-dir /path/to/models --voices-dir /path/to/voices \
  --voice sun \
  --text "你好，欢迎回来。今天有什么想和我聊聊的吗？" \
  --output sun-greeting.wav
```

The first-run script deliberately continues to use freely downloadable standard
weights and an LJ Speech reference. It will not install Sun or produce this voice.

| Artifact | SHA-256 |
|---|---|
| Sun greeting WAV | `6158e1fa6db040daece33ec26ff39c2d89e07cceb58a493fe0f6245669797e81` |
| Sun service confirmation WAV | `8441746c064748554208f136dcffd8d6123eccbf8ae202617bcfac140afc302b` |

## Reproducible First-Run Baseline

These are real, unedited outputs from the Rust CPU inference path, generated on
2026-10-05 with the standard GPT-SoVITS v2 model pair. They are not private
fine-tuned voices, human recordings of the target text, or speed benchmarks.

| File | Target text | Duration |
|---|---|---|
| [local-assistant-en.wav](local-assistant-en.wav) | Your local voice service is ready. All speech is generated on your own computer. | 5.40 s |
| [local-assistant-zh.wav](local-assistant-zh.wav) | 你的本地语音服务已经准备好了。接下来的语音会在这台电脑上生成。 | 6.36 s |

Both are mono, 32 kHz, 16-bit PCM. The Chinese example uses the same English
reference (cross-language synthesis). These short examples do not establish
pronunciation accuracy, long-text completeness, or similarity for other voices.
Automated checks verify valid, non-silent audio, not what a listener hears.

### Reference and Rights

Reference: **LJ Speech 1.1, LJ001-0001**, recorded by Linda Johnson for LibriVox,
aligned by Keith Ito. The [dataset publisher](https://keithito.com/LJ-Speech-Dataset/)
identifies its recordings, transcripts, and annotations as public domain in the US.
Check the applicable rules in your jurisdiction. This is a clearly labeled
synthetic demonstration, not an endorsement or a statement by the original reader.

The [pinned reference copy](https://github.com/coqui-ai/TTS/blob/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/tests/data/ljspeech/wavs/LJ001-0001.wav)
is downloaded only when preparing the demo. No Coqui implementation code is
included. We distribute only the generated outputs here; no reference dataset,
private recordings, or model weights are committed.

Exact reference transcript:

> Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition

Model sources and SHA-256 checksums are in the
[download manifest](../first-run/downloads.txt). Models keep their own licenses;
conversion does not relicense them.

### Reproduce

Prepare the Docker demo using the [first-run guide](../../docs/FIRST_RUN.zh-CN.md).
From the prepared demo directory, generate the English sample:

```bash
docker compose run --rm --no-deps -v "$PWD:/output" tts \
  --device cpu --models-dir /app/models --voices-dir /app/voices \
  --voice demo --text "Your local voice service is ready. All speech is generated on your own computer." \
  --output /output/reproduced-en.wav
```

For the Chinese sample, keep the same voice and use `--language auto` so reference
and target text are detected separately:

```bash
docker compose run --rm --no-deps -v "$PWD:/output" tts \
  --device cpu --models-dir /app/models --voices-dir /app/voices \
  --voice demo --language auto \
  --text "你的本地语音服务已经准备好了。接下来的语音会在这台电脑上生成。" \
  --output /output/reproduced-zh.wav
```

Settings: CPU F32, greedy `top_k=1`, KV decoding through `mode=auto`, sentence
splitting, 120 ms gap, 8 ms fade, 500-token cap. No external audio post-processing.
The committed samples were produced using the existing local 1.2.0-versioned
binary (SHA-256 below), not inside Docker. Numerical differences across builds
and hardware mean reproduction is not a byte-for-byte promise.

| Artifact | SHA-256 |
|---|---|
| Local inference binary | `77073d50893d79fad326d0175de46c936d48fa3c0355d9f2f7af48a4d2c88399` |
| English WAV | `3ea40c1ff94b4b48ee84773660c068ce4cc4c08890b704c2be40994b34b4d2e5` |
| Chinese WAV | `86256f06ef8df2ca62738eb2ef05e3b3c07736a12ccae4d2bfe0c3f79d02639f` |

The **First-run acceptance** workflow separately runs the pinned published Docker
image from download through HTTP synthesis and uploads the resulting WAV for
inspection. Signal checks are not a substitute for listening.

## Browser Playback

The [static showcase](../../site/README.md) contains native browser audio players.
Build it with `bash site/build.sh /tmp/gpt-sovits-site`, then open that directory's
`index.html`. GitHub Pages deployment is handled separately from software releases.
