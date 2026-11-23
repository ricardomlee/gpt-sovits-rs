#!/usr/bin/env bash
# Assemble a new voice with the released converter.
set -euo pipefail

usage() {
  printf 'Usage: bash examples/trained-voice/prepare.sh DATA_DIR VOICE GPT_CKPT SOVITS_PTH REF_WAV LANG REF_TEXT [SV_PT]\n'
  printf 'LANG is zh, en, or auto. Pass the matching training SV file for a v2Pro voice.\n'
}
fail() { printf '%s\n' "$*" >&2; exit 1; }

if [[ $# == 1 && $1 == --help ]]; then usage; exit 0; fi
if [[ $# != 7 && $# != 8 ]]; then usage >&2; exit 2; fi
root=$1
name=$2
gpt=$3
sovits=$4
ref=$5
lang=$6
text=$7
sv=${8:-}

[[ -n $root ]] || fail 'DATA_DIR must not be empty.'
[[ $name =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$ ]] ||
  fail 'VOICE must be 1-64 ASCII letters, digits, dashes or underscores, starting with a letter or digit.'
case "$lang" in zh|en|auto) ;; *) fail 'LANG must be zh, en, or auto.' ;; esac
[[ $text =~ [^[:space:]] ]] || fail 'REF_TEXT must contain the reference transcript.'
for tool in gpt-sovits-convert jq; do
  command -v "$tool" >/dev/null || fail "Missing command: $tool"
done
for input in "$gpt" "$sovits" "$ref" ${sv:+"$sv"}; do
  [[ -f $input && -r $input && -s $input ]] || fail "Missing, unreadable, or empty input: $input"
done

models="$root/models/$name"
voice="$root/voices/$name"
for directory in "$models" "$voice"; do
  [[ ! -e $directory && ! -L $directory ]] || fail "Voice path already exists: $directory. Choose a new VOICE name."
done
mkdir -p -- "$root/models" "$root/voices"
made_models=
made_voice=
cleanup() {
  if [[ -n $made_voice && -f "$made_voice/voice.json" ]]; then return; fi
  if [[ -n $made_voice ]]; then
    rm -f -- "$made_voice"/{ref.wav,ref_sv.safetensors,ref_sv.safetensors.tmp,.voice.json.tmp}
    rmdir -- "$made_voice" || true
  fi
  if [[ -n $made_models ]]; then
    rm -f -- "$made_models"/{gpt.safetensors,gpt.safetensors.tmp,sovits.safetensors,sovits.safetensors.tmp}
    rmdir -- "$made_models" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -- "$models"
made_models=$models
mkdir -- "$voice"
made_voice=$voice

gpt-sovits-convert gpt -- "$gpt" "$models/gpt.safetensors"
gpt-sovits-convert sovits -- "$sovits" "$models/sovits.safetensors"
[[ -s "$models/gpt.safetensors" && -s "$models/sovits.safetensors" ]] || fail 'The converter did not write both models.'
cp -- "$ref" "$voice/ref.wav"
if [[ -n $sv ]]; then
  gpt-sovits-convert sv -- "$sv" "$voice/ref_sv.safetensors"
  [[ -s "$voice/ref_sv.safetensors" ]] || fail 'The converter did not write the SV embedding.'
fi
jq -n --arg name "$name" --arg lang "$lang" --arg text "$text" --arg sv "$sv" '
  {reference_audio: "ref.wav", reference_text: $text, language: $lang, prompt_language: $lang,
   gpt_model: ($name + "/gpt.safetensors"), sovits_model: ($name + "/sovits.safetensors")}
  + (if $sv == "" then {} else {sv_embedding: "ref_sv.safetensors"} end)
' > "$voice/.voice.json.tmp"
# The service only discovers a voice after the complete profile is in place.
mv -- "$voice/.voice.json.tmp" "$voice/voice.json"
printf 'Prepared %s\n' "$voice/voice.json"
