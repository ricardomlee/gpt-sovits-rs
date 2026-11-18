# SoVITS Wasm Lab

Render short Mandarin text to WAV on the browser's CPU. The Web Worker runs
the Rust text frontend, BERT, GPT and SoVITS, and keeps the models loaded for
subsequent sentences. The web server only serves static assets.

## Prepare a Voice

Set up a voice with a reference WAV and transcript, matching GPT and SoVITS
weights, BERT and HuBERT. See [model setup](../../docs/MODELS.md). Run the
reference exporter once from this experiment directory:

```bash
cargo run --locked --release --example native -- prepare-reference \
  /path/to/models /path/to/voices my-voice /path/to/reference.safetensors
```

The reference file contains HuBERT-derived semantic tokens, aligned BERT
features, reference STFT and the voice's SV embedding when configured. Treat
the file like the reference audio and keep it outside the web directory.

## Open the Lab

Install the Rust target and use `wasm-bindgen` CLI **0.2.130**, matching `Cargo.lock`.
Run from this experiment directory (adjust the artifact path for a custom
`CARGO_TARGET_DIR`):

```bash
rustup target add wasm32-unknown-unknown
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS='-C target-feature=+simd128' \
  cargo build --locked --release --target wasm32-unknown-unknown --lib
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/wasm_sovits.wasm
node serve.mjs
```

Open `http://127.0.0.1:9898/`. Select the SoVITS, GPT and BERT safetensors,
BERT's `tokenizer.json`, and the prepared reference. Enter a Mandarin sentence
of up to 80 characters. **Unload voice** releases the Worker's model memory.

The experiment uses CPU SIMD, seed 42 and the standard sampling defaults
with a 256-token cap. It has been exercised on desktop Chromium. Reference
preparation stays native; there is no HuBERT checkpoint in the browser.

## Measured on Chromium

On an i5-12490F with Chromium 147 and a local v2Pro voice:

| Measurement | Result |
| --- | ---: |
| Load three checkpoints and prepare the Worker | 4.4 s |
| Render 3.04 s of audio from Mandarin text | 9.5-9.7 s |
| Render 2.32 s of audio from a second sentence | 7.2 s |
| Peak Wasm linear memory for text synthesis | 2,064 MiB |

Both text outputs match the native CPU run's sample counts, with a maximum
difference of 2 PCM16 steps. Replaying the same text after another sentence
produces byte-identical WAV. These results describe the desktop prototype,
not real-time playback performance or mobile memory requirements.

## Acoustic Baseline

The native example also accepts `prepare TRACE_DIR REF_32K_MONO.wav SV|- CASE`
and `render WEIGHTS CASE WAV` for fixed-token comparisons. Obtain the trace by
running the single-utterance CLI with `SOVITS_DEBUG=1 --mode cuda-graph
--no-split-sentences` in a separate output directory. This path uses noise scale
0 to compare the acoustic model independently of GPT sampling.

For the same 258 MiB SoVITS checkpoint and fixed input, loading tensor ranges
instead of buffering the whole checkpoint reduced Wasm linear memory after
load from 633 to 380 MiB, and peak through render from 633 to 494 MiB. The WAV
was byte-identical. Linear memory excludes JavaScript and browser overhead.
