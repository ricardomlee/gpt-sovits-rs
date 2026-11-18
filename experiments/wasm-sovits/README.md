# SoVITS Wasm Lab

Run the existing SoVITS acoustic model on the browser's CPU in a Web Worker.
Give it local SoVITS weights and a prepared case containing semantic tokens,
phoneme IDs, reference STFT, and optionally a v2Pro SV embedding. The page
reads both files locally and renders a playable, downloadable WAV. It uses the
same Rust decoder as native inference, with noise scale 0 for the comparison.

## Prepare a Case

Use a real single-utterance trace from the native CLI. Run it in a separate
output directory: `SOVITS_DEBUG=1` writes intermediate tensors there. Use
`--mode cuda-graph --no-split-sentences` on either CPU or CUDA; CPU uses static
KV without GPU capture. For example, from the output directory:

```bash
SOVITS_DEBUG=1 gpt-sovits --device cpu --mode cuda-graph --no-split-sentences \
  --models-dir /path/to/models --voices-dir /path/to/voices --voice my-voice \
  --seed 42 --text 'Hello from the browser.' --output pipeline.wav
ffmpeg -i /path/to/ref.wav -ar 32000 -ac 1 -c:a pcm_f32le ref-32k.wav
```

From this experiment directory, with `CASE_DIR` pointing to that output
directory and `WEIGHTS`/`SV` pointing to the matching local files (`SV=-` when
there is no embedding):

```bash
cargo run --locked --release --example native -- prepare \
  "$CASE_DIR" "$CASE_DIR/ref-32k.wav" "$SV" "$CASE_DIR/case.safetensors"
RAYON_NUM_THREADS=1 cargo run --locked --release --example native -- render \
  "$WEIGHTS" "$CASE_DIR/case.safetensors" "$CASE_DIR/native.wav"
```

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

Open `http://127.0.0.1:9898/`, select the weights and prepared case, and render.
The loopback server serves only the page and compiled assets. Keep model,
reference, trace and case files outside `web/`.

## First Result

One Mandarin case with a local v2Pro voice, 76 semantic tokens and 3.04 seconds
of output. On an i5-12490F, Chromium 147 rendered it in 4.74-5.11 seconds across
three runs; native single-threaded render took 3.21 seconds. Surrounding file
selection and browser asset loading are outside the render times. The Wasm
binary is 2.2 MiB; its linear memory reached 623 MiB with 258 MiB of model
weights. This excludes JavaScript file buffers and the rest of browser memory.

The three browser WAVs were identical. Compared with the native WAV they had
the same sample count and a maximum difference of 2 PCM16 quantization steps.
The experiment covers one acoustic case on desktop Chromium; it does not run
GPT, BERT or reference audio preprocessing in the browser yet.
