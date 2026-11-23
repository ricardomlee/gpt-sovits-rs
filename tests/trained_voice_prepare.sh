#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
data="$scratch/data with spaces"
mkdir -- "$scratch/bin"
cat > "$scratch/bin/gpt-sovits-convert" <<'SH'
#!/usr/bin/env bash
set -eu
[[ $# == 4 && $2 == -- && -s $3 ]]
if [[ $1 == "${FAIL_CONVERSION:-}" ]]; then
  printf partial > "$4.tmp"
  exit 7
fi
printf '%s' "$1" > "$4"
SH
chmod +x "$scratch/bin/gpt-sovits-convert"
export PATH="$scratch/bin:$PATH"
unset FAIL_CONVERSION

for file in gpt.ckpt sovits.pth 'ref audio.wav' 'ref audio.wav.pt'; do
  printf fixture > "$scratch/$file"
done
text=$'A "quote" and a backslash \\.\nA second line.'
prepare() {
  local name=$1
  shift
  bash "$repo/examples/trained-voice/prepare.sh" "$data" "$name" \
    "$scratch/gpt.ckpt" "$scratch/sovits.pth" "$scratch/ref audio.wav" en "$text" "$@"
}

if FAIL_CONVERSION=sv prepare custom "$scratch/ref audio.wav.pt"; then
  printf 'Expected conversion failure\n' >&2
  exit 1
else
  [[ $? == 7 ]]
fi
[[ ! -e "$data/models/custom" && ! -e "$data/voices/custom" ]]
prepare custom "$scratch/ref audio.wav.pt"
profile="$data/voices/custom/voice.json"
jq -e --arg text "$text" '
  .reference_text == $text and .language == "en" and .prompt_language == "en"
  and .gpt_model == "custom/gpt.safetensors" and .sovits_model == "custom/sovits.safetensors"
  and .reference_audio == "ref.wav" and .sv_embedding == "ref_sv.safetensors"
' "$profile" >/dev/null
cmp -- "$scratch/ref audio.wav" "$data/voices/custom/ref.wav"
[[ $(<"$data/voices/custom/ref_sv.safetensors") == sv ]]

cp -- "$profile" "$scratch/profile.json"
if prepare custom; then
  printf 'Overwrote an existing voice\n' >&2
  exit 1
fi
cmp -- "$profile" "$scratch/profile.json"
prepare v2
jq -e 'has("sv_embedding") | not' "$data/voices/v2/voice.json" >/dev/null
