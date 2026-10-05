#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:?Usage: bash site/build.sh OUTPUT_DIRECTORY}
if [[ -d "$out" && -n $(find "$out" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
  printf 'Refusing nonempty output directory: %s\n' "$out" >&2
  exit 1
fi
mkdir -p "$out/assets" "$out/audio"
# Explicit allowlist: never publish repository contents, voices, or model directories.
cp "$root/site/index.html" "$root/site/styles.css" "$out/"
cp "$root/assets/gpt-sovits-rs-banner.png" "$out/assets/"
cp "$root/examples/samples/sun-greeting.wav" "$root/examples/samples/sun-zh.wav" "$out/audio/"
touch "$out/.nojekyll"
