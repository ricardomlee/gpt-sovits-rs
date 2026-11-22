# GitHub Pages

The [Chinese homepage](https://tts.ricli.tech/) and [English homepage](https://tts.ricli.tech/en/)
play the same three recordings: two Mandarin readings by Sun and the public English demo.
The Mandarin synthesis lab has a [Chinese interface](https://tts.ricli.tech/lab/) and an
[English interface](https://tts.ricli.tech/lab/en.html). Both run the same local WebAssembly worker.

`build.sh` packages only named public images, samples, page files and compiled
Wasm assets. The lab uses the same source as the
[standalone experiment](../experiments/wasm-sovits/README.md).

## Preview

With Rust, `wasm-bindgen` CLI 0.2.130 and Node installed, run from the repo root:

```bash
rustup target add wasm32-unknown-unknown
(
  cd experiments/wasm-sovits
  CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS='-C target-feature=+simd128' \
    cargo build --locked --release --target wasm32-unknown-unknown --lib
  wasm-bindgen --target web --no-typescript --out-dir web/pkg \
    target/wasm32-unknown-unknown/release/wasm_sovits.wasm
)
bash site/build.sh _site
node site/serve.cjs _site
```

Open `http://127.0.0.1:9898/gpt-sovits-rs/`. Use an empty output directory for
`build.sh`. The preview also exercises relative links under a project subpath.

To check playback and Worker/Wasm startup on the staged site, use an installed
Google Chrome (or set `CHROME_BIN` to your Chromium executable):

```bash
npm install --prefix /tmp/pages-test --no-save --ignore-scripts playwright-core@1.59.1
NODE_PATH=/tmp/pages-test/node_modules node site/check.cjs _site
```

The Pages workflow uses the runner's Chrome on the same directory it publishes.
An invalid checkpoint exercises the lab's Worker and error handling without model files.

## Publish

In repository **Settings > Pages > Build and deployment**, select **GitHub Actions**
once. The **Audio showcase** workflow validates and deploys relevant changes on `master`.
Pull requests build and test. Merges touching the site, lab, or Rust dependencies
trigger publication. The public site is [tts.ricli.tech](https://tts.ricli.tech/).

The hero and lab illustrations in `assets/pages-*.webp` are generated artwork.
Forks can update repository links and enable their own Pages site.
