# Deployment

The intended deployment shape is a local HTTP TTS service with three mounted folders:

```text
models/   model weights
voices/   voice profiles and reference audio
outputs/  optional generated files and reports
```

Copy `.env.example` to `.env` and edit paths or model names if your layout differs.

For a starter voice profile, copy `voices.example/mao` to `voices/mao` and put the matching reference audio at `voices/mao/ref.wav`.

## CPU

```bash
cp .env.example .env
docker compose -f compose.cpu.yml up -d --build
curl http://localhost:9880/health
curl http://localhost:9880/voices
```

## CUDA

Set `CUDA_COMPUTE_CAP` in `.env` for your GPU, then start:

```bash
cp .env.example .env
docker compose -f compose.cuda.yml up -d --build
curl http://localhost:9880/health
curl http://localhost:9880/voices
```

## Voice Layout

Each voice lives under `voices/<name>/voice.json`. Paths inside `voice.json` are resolved relative to that voice directory.

```text
voices/
  mao/
    voice.json
    ref.wav
```

Example request for assistants:

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"mao","text":"人民，只有人民，才是创造世界历史的动力。"}' \
  --output output.wav
```

Legacy request fields remain available for debugging:

```json
{
  "text": "你好世界",
  "refer_wav_path": "/app/voices/mao/ref.wav",
  "prompt_text": "参考音频对应的文字",
  "text_language": "zh"
}
```
