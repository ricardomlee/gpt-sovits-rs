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
mkdir -p "$out/assets" "$out/audio" "$out/en" "$out/lab/pkg"
# Explicit allowlist: never publish repository contents, voices, or model directories.
cp "$root/site/index.html" "$root/site/styles.css" "$out/"
cp "$root/site/en/index.html" "$out/en/"
cp "$root/assets/pages-voice-console.webp" "$root/assets/pages-browser-lab.webp" "$out/assets/"
cp "$root/examples/samples/"{sun-greeting,sun-zh,local-assistant-en}.wav "$out/audio/"
cp "$lab/"{index.html,en.html,lab.css,lab.js,worker.js,checkpoint.js} "$out/lab/"
cp "$pkg/wasm_sovits.js" "$pkg/wasm_sovits_bg.wasm" "$out/lab/pkg/"
touch "$out/.nojekyll"
