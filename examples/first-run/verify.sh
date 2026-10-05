#!/usr/bin/env bash
# Maintainer acceptance: own an isolated demo, capture evidence, always stop it.
set -euo pipefail
if [[ $# != 2 ]]; then
  printf 'Usage: bash examples/first-run/verify.sh NEW_DEMO_DIRECTORY NEW_ARTIFACT_DIRECTORY\n' >&2
  exit 2
fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
demo=$1
artifacts=$2
# Refuse reuse so an acceptance run cannot overwrite a deployment or old evidence.
for directory in "$demo" "$artifacts"; do
  if [[ -e "$directory" ]]; then
    printf 'Acceptance requires a new directory: %s\n' "$directory" >&2
    exit 1
  fi
done
mkdir -p -- "$artifacts"
artifacts=$(cd -- "$artifacts" && pwd)
bash "$here/prepare.sh" "$demo" > "$artifacts/prepare.log" 2>&1
demo=$(cd -- "$demo" && pwd)
project="gpt-sovits-accept-$$"
compose=(docker compose --project-name "$project" --env-file "$demo/.env" -f "$demo/compose.yml")
cleanup() {
  local result=$?
  trap - EXIT
  "${compose[@]}" logs --no-color > "$artifacts/service.log" 2>&1 || true
  "${compose[@]}" down || result=1
  exit "$result"
}
trap cleanup EXIT
"${compose[@]}" config --format json > "$artifacts/compose.json"
"${compose[@]}" up -d --wait --wait-timeout 900
container=$("${compose[@]}" ps -q tts)
docker inspect --format '{{.Image}}' "$container" > "$artifacts/image-id.txt"
docker exec "$container" gpt-sovits --version > "$artifacts/version.txt"
if [[ -n ${DEMO_EXPECTED_VERSION:-} ]]; then
  [[ $(< "$artifacts/version.txt") == "gpt-sovits $DEMO_EXPECTED_VERSION" ]] || {
    printf 'Unexpected binary version in the running image. See version.txt.\n' >&2
    exit 1
  }
fi
url="http://127.0.0.1:${DEMO_PORT:-9881}"
for route in health status voices; do
  curl --fail --silent --show-error --max-time 30 "$url/$route" > "$artifacts/$route.json"
done
curl --fail-with-body --silent --show-error --max-time 300 "$url/tts" \
  -H 'Content-Type: application/json' --data-binary "@$demo/request.json" \
  --dump-header "$artifacts/headers.txt" --output "$artifacts/first.wav"
cd -- "$root"
FIRST_RUN_WAV="$artifacts/first.wav" cargo test --locked --test first_run \
  generated_http_audio_is_valid -- --ignored --nocapture
printf 'Acceptance passed; evidence: %s\n' "$artifacts"
