#!/usr/bin/env bash
# Prepare the public v2 demo; never modifies an existing deployment.
set -euo pipefail

if [[ $# != 1 || ${1:-} == --help ]]; then
  printf 'Usage: bash examples/first-run/prepare.sh NEW_DIRECTORY\n'
  printf 'Downloads upstream models, verifies SHA-256, and converts with Docker (no Python).\n'
  printf 'Set DEMO_MODEL_SOURCE=modelscope to try ModelScope first.\n'
  printf 'Maintainers: DEMO_IMAGE selects a candidate; DEMO_PULL_POLICY=never uses a local image.\n'
  [[ $# == 1 && $1 == --help ]] && exit 0
  exit 2
fi
image=${DEMO_IMAGE:-ghcr.io/ricardomlee/gpt-sovits-rs:1.3.5}
pull_policy=${DEMO_PULL_POLICY:-always}
model_source=${DEMO_MODEL_SOURCE:-huggingface}
if [[ ! $image =~ ^[a-zA-Z0-9][a-zA-Z0-9._/:@-]*$ ]]; then
  printf 'Invalid DEMO_IMAGE. Use a Docker image reference without whitespace or shell syntax.\n' >&2
  exit 2
fi
case "$pull_policy" in
  always|never) ;;
  *) printf 'DEMO_PULL_POLICY must be always or never.\n' >&2; exit 2 ;;
esac
case "$model_source" in
  huggingface|modelscope) ;;
  *) printf 'DEMO_MODEL_SOURCE must be huggingface or modelscope.\n' >&2; exit 2 ;;
esac
for command in docker curl; do
  command -v "$command" >/dev/null || { printf 'Missing command: %s\n' "$command" >&2; exit 1; }
done
if command -v sha256sum >/dev/null; then
  checksum() { sha256sum "$1" | cut -d ' ' -f 1; }
elif command -v shasum >/dev/null; then
  checksum() { shasum -a 256 "$1" | cut -d ' ' -f 1; }
else
  printf 'Install sha256sum or shasum first.\n' >&2
  exit 1
fi
platform=$(docker info --format '{{.OSType}}/{{.Architecture}}')
case "$platform" in
  linux/x86_64|linux/amd64) ;;
  *) printf 'This demo requires a Linux amd64 Docker engine (found %s). See docs/DEPLOYMENT.md for other platforms.\n' "$platform" >&2; exit 1 ;;
esac
docker compose version >/dev/null
if [[ $pull_policy == never ]]; then
  docker image inspect "$image" >/dev/null || { printf 'Candidate image is not available locally: %s\n' "$image" >&2; exit 1; }
fi

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
destination=$1
mkdir -p -- "$destination"
destination=$(cd -- "$destination" && pwd)
marker="$destination/.gpt-sovits-demo"
if [[ ! -f "$marker" ]]; then
  if [[ -n $(find "$destination" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    printf 'Refusing nonempty directory: %s. Choose a new demo directory.\n' "$destination" >&2
    exit 1
  fi
  printf 'gpt-sovits-public-v2-demo\n' > "$marker"
elif [[ $(< "$marker") != gpt-sovits-public-v2-demo ]]; then
  printf 'Unrecognized demo marker: %s. Choose a new directory.\n' "$marker" >&2
  exit 1
fi

# Serialize preparation, and remove only our own lock on failure or interruption.
lock="$destination/.prepare-lock"
mkdir -- "$lock" || { printf 'Preparation already running (or stale lock): %s\n' "$lock" >&2; exit 1; }
trap 'rmdir -- "$lock"' EXIT
mkdir -p "$destination/source" "$destination/models/bert" "$destination/models/hubert" "$destination/voices/demo"

while read -r expected relative url mirror; do
  [[ -z "$expected" || "$expected" == \#* ]] && continue
  file="$destination/source/$relative"
  mkdir -p -- "$(dirname -- "$file")"
  if [[ -f "$file" ]]; then
    if [[ $(checksum "$file") != "$expected" ]]; then
      printf 'Checksum mismatch: %s. Remove this file and retry.\n' "$file" >&2
      exit 1
    fi
    printf 'Verified cached %s\n' "$relative"
    continue
  fi
  if [[ $model_source == modelscope && -n $mirror ]]; then
    urls=("$mirror" "$url")
  else
    urls=("$url" "$mirror")
  fi
  for source_url in "${urls[@]}"; do
    [[ -n "$source_url" ]] || continue
    printf 'Downloading %s from %s\n' "$relative" "$source_url"
    if curl --fail --location --retry 3 --connect-timeout 30 --max-time 1800 --output "$file.part" "$source_url"; then
      if [[ $(checksum "$file.part") == "$expected" ]]; then
        mv -- "$file.part" "$file"
        break
      fi
      printf 'Checksum mismatch: %s from %s.\n' "$file.part" "$source_url" >&2
    fi
  done
  if [[ ! -f "$file" ]]; then
    printf 'Could not obtain a verified copy of %s. Nothing will be converted.\n' "$relative" >&2
    exit 1
  fi
done < "$here/downloads.txt"

if [[ $pull_policy == always ]]; then docker pull "$image"; fi
convert() {
  docker run --rm --network none --user "$(id -u):$(id -g)" \
    --volume "$destination/source:/source:ro" --volume "$destination/models:/models" \
    --entrypoint gpt-sovits-convert "$image" "$1" "/source/$2" "/models/$3.tmp"
  mv -- "$destination/models/$3.tmp" "$destination/models/$3"
}
convert gpt gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt gpt-model.safetensors
convert sovits gsv-v2final-pretrained/s2G2333k.pth sovits-model.safetensors
convert bert chinese-roberta-wwm-ext-large/pytorch_model.bin bert/bert.safetensors
convert hubert chinese-hubert-base/pytorch_model.bin hubert/hubert.safetensors
cp "$destination/source/chinese-roberta-wwm-ext-large/tokenizer.json" "$destination/models/bert/tokenizer.json"
cp "$destination/source/reference.wav" "$destination/voices/demo/ref.wav"
cp "$here/voice.json" "$destination/voices/demo/voice.json"
cp "$here/compose.yml" "$destination/compose.yml"
cp "$here/request.json" "$destination/request.json"
compose_pull_policy=missing
if [[ $pull_policy == never ]]; then compose_pull_policy=never; fi
printf 'DEMO_IMAGE=%s\nDEMO_PULL_POLICY=%s\n' "$image" "$compose_pull_policy" > "$destination/.env"

docker run --rm --network none \
  --volume "$destination/models:/app/models:ro" --volume "$destination/voices:/app/voices:ro" \
  "$image" --doctor --device cpu --models-dir /app/models --voices-dir /app/voices --voice demo
printf '\nPrepared %s\nNext: cd into that directory and run docker compose up -d --wait --wait-timeout 900\n' "$destination"
