# GitHub Pages

The homepage plays two reviewed Sun samples and links to the browser lab at
`lab/`. GitHub Pages serves the lab's WebAssembly, JavaScript and CSS. The lab
reads local model files and synthesizes in a browser Worker.

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
`build.sh`. The preview uses the same path prefix as the public site.

Chromium checks cover both sample players, images, downloads, responsive layout,
the artifact's file list, and real lab Worker/Wasm startup under the Pages prefix:

```bash
npm install --prefix /tmp/pages-test --no-save --ignore-scripts playwright@1.59.1
/tmp/pages-test/node_modules/.bin/playwright install --with-deps chromium
NODE_PATH=/tmp/pages-test/node_modules node site/check.cjs _site
```

The Pages workflow installs Playwright outside the repository and uploads
screenshots. Its lab check uses an invalid checkpoint to test startup and error
handling without private model files.

## Publish

In repository **Settings > Pages > Build and deployment**, select **GitHub Actions**
once. The **Audio showcase** workflow validates and deploys relevant changes on `master`.
Pull requests build and test. Merges touching the site, lab, or Rust dependencies
trigger publication. The live URL is:

`https://ricardomlee.github.io/gpt-sovits-rs/`

The hero and lab illustrations in `assets/pages-*.webp` are generated artwork.
Forks can update repository links and enable their own Pages site.
