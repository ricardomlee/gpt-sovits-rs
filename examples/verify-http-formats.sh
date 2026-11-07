#!/usr/bin/env bash
# Maintainer check: independent FFmpeg decoding of actual HTTP responses.
set -euo pipefail
if [[ $# != 3 ]]; then
  printf 'Usage: bash examples/verify-http-formats.sh URL VOICE NEW_EVIDENCE_DIRECTORY\n' >&2
  exit 2
fi
url=${1%/}
voice=$2
out=$3
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
for tool in curl jq ffmpeg ffprobe cargo; do
  command -v "$tool" > /dev/null
done
if [[ -e "$out" ]]; then
  printf 'Evidence directory already exists: %s\n' "$out" >&2
  exit 1
fi
mkdir -p -- "$out"
out=$(cd -- "$out" && pwd)
text=${SPEECH_TEST_TEXT:-Your local voice service is ready. All speech is generated on your own computer.}
for format in wav pcm mp3 opus; do
  jq -n --arg voice "$voice" --arg input "$text" --arg format "$format" \
    '{model:"tts-1",voice:$voice,input:$input,response_format:$format}' > "$out/$format.request.json"
  curl --fail-with-body --silent --show-error --max-time 300 "$url/v1/audio/speech" \
    -H 'Content-Type: application/json' --data-binary "@$out/$format.request.json" \
    --dump-header "$out/$format.headers" --output "$out/$format.audio"
  case "$format" in
    wav) mime=audio/wav; codec=pcm_s16le; rate=32000 ;;
    pcm) mime=application/octet-stream; codec=pcm_s16le; rate=32000 ;;
    mp3) mime=audio/mpeg; codec=mp3; rate=32000 ;;
    opus) mime=audio/ogg; codec=opus; rate=48000 ;;
  esac
  grep -Eiq "^content-type: $mime" "$out/$format.headers"
  grep -Eiq "^x-tts-response-format: $format" "$out/$format.headers"
  grep -Eiq "^x-tts-sample-rate: $rate" "$out/$format.headers"
  input=()
  if [[ $format == pcm ]]; then input=(-f s16le -ar "$rate" -ac 1); fi
  ffprobe -v error "${input[@]}" -show_entries stream=codec_name,sample_rate,channels \
    -of json "$out/$format.audio" > "$out/$format.probe.json"
  jq -e --arg codec "$codec" --arg rate "$rate" \
    '.streams | length == 1 and .[0].codec_name == $codec and .[0].sample_rate == $rate and .[0].channels == 1' \
    "$out/$format.probe.json" > /dev/null
  ffmpeg -v error -xerror "${input[@]}" -i "$out/$format.audio" \
    -c:a pcm_s16le "$out/$format.decoded.wav"
  (
    cd -- "$root"
    FIRST_RUN_WAV="$out/$format.decoded.wav" FIRST_RUN_SAMPLE_RATE="$rate" cargo test --locked --test first_run \
      generated_http_audio_is_valid -- --ignored --nocapture
  ) > "$out/$format.validation.log" 2>&1
done
jq -n --arg voice "$voice" \
  '{model:"tts-1",voice:$voice,input:"hello",response_format:"flac"}' > "$out/unsupported.request.json"
status=$(curl --silent --show-error --max-time 30 "$url/v1/audio/speech" \
  -H 'Content-Type: application/json' --data-binary "@$out/unsupported.request.json" \
  --output "$out/unsupported.response.json" --write-out '%{http_code}')
[[ $status == 400 ]]
jq -e '.success == false and (.error | contains("unsupported response_format"))' \
  "$out/unsupported.response.json" > /dev/null
printf 'All four HTTP formats decoded successfully; evidence: %s\n' "$out"
