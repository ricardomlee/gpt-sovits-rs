# Connect an agent to the GPT-SoVITS-RS speech API

[Chinese guide](AGENT_INTEGRATION.md)

**1. Check the existing service.** From the agent's environment, set `TTS_URL`
to the address it can reach. The deployment default is port `9880`; the
[first-run demo](../README.md#docker-service) uses `9881`. Inside a container,
`localhost` refers to that container.

```bash
TTS_URL=${TTS_URL:-http://127.0.0.1:9880}
curl -fsS --max-time 5 "$TTS_URL/health"
curl -fsS --max-time 5 "$TTS_URL/voices"
```

If you get `OK` and a list of voices, reuse that service. If the request fails,
check the address. [Choose a build](#choose-a-build) only for a new server.

**2. Test a voice.** Pick a name from the list and a sentence in that voice's
language. For the public English demo:

```bash
curl --fail-with-body --max-time 300 "$TTS_URL/tts" \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"Your local voice service is ready."}' \
  --output reply.wav
```

**3. Connect the app.** Play `reply.wav` and check the words. Use the same voice
with `POST /tts`, or set the base URL to `$TTS_URL/v1` for [Hermes](#hermes)
and other OpenAI-compatible clients.

## Hermes

In the shell where you tested `TTS_URL`, replace `demo` with the voice you heard:

```bash
hermes config set tts.openai.base_url "${TTS_URL:?}/v1"
hermes config set tts.openai.api_key local
hermes config set tts.openai.model tts-1
hermes config set tts.openai.voice demo
hermes config set tts.use_gateway false
hermes config set tts.provider openai
```

Call `text_to_speech` with a short sentence. Hermes uses HTTP directly. The
local API key satisfies the SDK; the server has no built-in authentication.

## Choose a build

Run `uname -sm` on the **TTS host**:

- Linux x86_64: start with the [CPU demo](../README.md#docker-service) or the [native CLI](../README.md#native-cli).
- Mac: use the [native CPU archive](https://github.com/ricardomlee/gpt-sovits-rs/releases/latest), `macos-aarch64` for Apple Silicon or `macos-x86_64` for Intel.
- ARM Linux: [build from source](DEVELOPMENT.md#build-from-source).

For NVIDIA on Linux or WSL2 x86_64, read the GPU's compute capability:

```bash
PATH="$PATH:/usr/lib/wsl/lib" nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

Match it to a published CUDA image for your release. A 4060 (8.9) uses `sm89`;
a 5060 (12.0) needs `sm120`, which 1.3.3 does not publish. If no tag matches,
use CPU on x86_64 or build for your GPU. The [deployment guide (in Chinese)](DEPLOYMENT.md#选择部署方式)
lists the image tags and Docker commands.

## Add a voice

1. Create `my-voice/` under the server's voices directory (the demo uses `~/gpt-sovits-demo/voices/`). Save a clear 3-10 second recording you have permission to use as `ref.wav` there.
2. Copy the [example voice.json](../examples/first-run/voice.json) into the same directory. Set `reference_text` to the exact words in the recording, `prompt_language` to its language, and `language` to the target language.
3. Check `/voices` again and test `my-voice` with a sentence in the target language.

Standard v2 reuses the loaded weights without a restart. For fine-tuned v2Pro
voices, see the [model and SV guide (in Chinese)](MODELS.md#v2pro-sv).

## Playback

Submit each completed sentence in order. The service can synthesize the next
sentence while the current one plays. On interruption, stop playback and discard
the old reply; synthesis already in progress runs to completion. For longer
prepared text, see the [streaming API](API.md#post-ttsstream).
