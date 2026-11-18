#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:?Usage: bash site/build.sh OUTPUT_DIRECTORY [WASM_PKG_DIRECTORY]}
lab="$root/experiments/wasm-sovits/web"
pkg=${2:-"$lab/pkg"}
for file in wasm_sovits.js wasm_sovits_bg.wasm; do
  if [[ ! -s "$pkg/$file" ]]; then
    printf 'Missing browser artifact: %s. See site/README.md.\n' "$pkg/$file" >&2
    exit 1
  fi
done
if [[ -d "$out" && -n $(find "$out" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
  printf 'Refusing nonempty output directory: %s\n' "$out" >&2
  exit 1
fi
mkdir -p "$out/assets" "$out/audio" "$out/lab/pkg"
# Explicit allowlist: never publish repository contents, voices, or model directories.
cp "$root/site/index.html" "$root/site/styles.css" "$out/"
cp "$root/assets/pages-voice-console.webp" "$root/assets/pages-browser-lab.webp" "$out/assets/"
cp "$root/examples/samples/sun-greeting.wav" "$root/examples/samples/sun-zh.wav" "$out/audio/"
cp "$lab/index.html" "$lab/lab.css" "$lab/worker.js" "$lab/checkpoint.js" "$out/lab/"
cp "$pkg/wasm_sovits.js" "$pkg/wasm_sovits_bg.wasm" "$out/lab/pkg/"
touch "$out/.nojekyll"
