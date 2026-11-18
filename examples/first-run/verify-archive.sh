#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  printf 'Usage: bash examples/first-run/verify-archive.sh LINUX_ARCHIVE PREPARED_DEMO NEW_DIRECTORY\n' >&2
  exit 2
fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
demo=$(cd -- "$2" && pwd)
mkdir -- "$3"
out=$(cd -- "$3" && pwd)
mkdir "$out/bundle"
tar -xzf "$1" -C "$out/bundle" --strip-components=1
image=$(docker buildx build --load --quiet -f "$here/Dockerfile.binary" "$out/bundle")

run=(docker run --rm --network none --user "$(id -u):$(id -g)" -v "$demo:/demo:ro" -v "$out:/out")
"${run[@]}" --entrypoint gpt-sovits-convert "$image" sovits \
  /demo/source/gsv-v2final-pretrained/s2G2333k.pth /out/sovits.safetensors
"${run[@]}" "$image" --device cpu --voice demo --models-dir /demo/models --voices-dir /demo/voices \
  --sovits-model /out/sovits.safetensors --text 'Hello, this is the packaged binary offline test.' --output /out/offline.wav
cd -- "$here/../.."
FIRST_RUN_WAV="$out/offline.wav" cargo test --locked --test first_run generated_http_audio_is_valid -- --ignored --nocapture
