# GPT-SoVITS-RS

**English** | [简体中文](README.zh-CN.md)

<p align="center">
  <img src="assets/gpt-sovits-rs-logo.svg" alt="GPT-SoVITS-RS" width="640">
</p>

Use a short recording and its transcript to read new text in that voice.
GPT-SoVITS-RS runs locally in Rust with Candle. Conversion and inference need no Python runtime.

## Quick Start

| What do you want to do? | Where to start |
| --- | --- |
| Hear a voice in your browser | [Play the English and Mandarin samples](https://tts.ricli.tech/en/#samples). No installation or model downloads. To synthesize your own Mandarin text, open the [Wasm lab](https://tts.ricli.tech/lab/en.html) with your [prepared model files](experiments/wasm-sovits/README.md#prepare-a-voice). |
| Make one WAV on your computer | [Native CLI](#native-cli) for Linux x86_64 or macOS, or a [Windows CPU/CUDA package](docs/WINDOWS.md). See the [public demo model guide](docs/MODELS.md#准备公开-v2-演示模型). |
| Give an app a speech API | [Docker service](#docker-service). The Linux x86_64 CPU demo downloads public weights and includes an English voice. For other hardware, [choose a build](docs/AGENT_INTEGRATION.en.md#choose-a-build). |

Already have a running service? [Send it a sentence](#connect-your-app).

## Samples

| Voice | What you'll hear | Audio |
| --- | --- | --- |
| Public English demo (v2) | The rain has stopped. We can walk home by the river. | [Play WAV](examples/samples/local-assistant-en.wav) |
| Sun (private v2Pro voice) | 面包刚出炉，坐下来一起吃早餐吧。 | [Play WAV](examples/samples/sun-greeting.wav) |

[Compare both voices reading the same Chinese passage](examples/samples/README.md). The public demo installs the English voice; Sun is available to listen to on the website.

## Native CLI

Download the [v1.3.5 release](https://github.com/ricardomlee/gpt-sovits-rs/releases/tag/v1.3.5): Linux x86_64 (14.9 MiB), Apple silicon (9.0 MiB), or Intel Mac (9.8 MiB). These are CPU packages containing the CLI, converter and libraries; weights are separate. Keep the bundled libraries beside the executables as extracted, including the macOS dylibs.

```bash
pkg=gpt-sovits-1.3.5-linux-x86_64
curl -fLO "https://github.com/ricardomlee/gpt-sovits-rs/releases/download/v1.3.5/$pkg.tar.gz"
tar -xzf "$pkg.tar.gz"
```

On Mac, use `gpt-sovits-1.3.5-macos-aarch64` or `gpt-sovits-1.3.5-macos-x86_64` for `pkg`. The release includes `SHA256SUMS` to verify the download.

Point `data` at your `models/` and `voices/` parent directory and choose a voice. The example below reuses the public demo files. Starting without Docker? [Convert the weights](docs/MODELS.md) and [add your own voice](docs/AGENT_INTEGRATION.en.md#add-a-voice) first.

```bash
data="$HOME/gpt-sovits-demo"
"./$pkg/gpt-sovits" --device cpu --models-dir "$data/models" --voices-dir "$data/voices" \
  --voice demo --seed 42 --text "The rain has stopped. We can walk home by the river." --output first.wav
```

With the models on disk, the v1.3.4 CLI took 9.1 to 9.2 seconds to load them and write this 4.04-second clip on an i5-12490F (two runs, six CPU threads). Downloads and conversion are separate.

## Docker Service

The demo needs Linux x86_64, Docker Engine with Compose v2, Git, Bash, curl, SHA-256 tools, 6 GiB of free disk and 4 GiB of available memory. The [setup script](examples/first-run/prepare.sh) verifies about 1.1 GB of [model downloads](examples/first-run/models.toml). The CPU image downloads separately. Give the script a new or empty directory.

```bash
git clone https://github.com/ricardomlee/gpt-sovits-rs.git
cd gpt-sovits-rs
bash examples/first-run/prepare.sh "$HOME/gpt-sovits-demo"
cd "$HOME/gpt-sovits-demo"
docker compose up -d --wait --wait-timeout 900
export TTS_URL="http://127.0.0.1:${DEMO_PORT:-9881}"
curl --fail-with-body --max-time 300 "$TTS_URL/tts" \
  -H 'Content-Type: application/json' --data-binary @request.json --output first.wav
```

Play `first.wav`. Change `text` in `request.json` to another English sentence and repeat the curl command. The demo listens on **9881**; a regular deployment defaults to **9880**. Run `docker compose down` in the demo directory to stop it and keep the models.

## Connect Your App

Keep `TTS_URL` from the demo, or set it to the address your app can reach. Pick a voice returned by `/voices`. For the English demo:

```bash
TTS_URL=${TTS_URL:-http://127.0.0.1:9880}
curl -fsS "$TTS_URL/voices"
curl --fail-with-body --max-time 300 "$TTS_URL/tts" -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"The backup is complete."}' --output reply.wav
```

Use `POST /tts` for WAV, `/tts/stream` for sentence-by-sentence output, or the [OpenAI-compatible speech endpoint](docs/API.md#http-server) for clients such as [Hermes](docs/AGENT_INTEGRATION.en.md#hermes).

## Models and Speech Quality

| | Supported today |
| --- | --- |
| Voices | GPT-SoVITS v2 and v2Pro. Use a clear 3–10 second reference clip and its exact transcript. Full v2Pro conditioning also needs a matching SV embedding; the converter reads an existing one, not raw audio. |
| Text | Chinese, English and mixed text. Long passages can skip words. Play finished sentences in order and listen back to important output. |
| Inference | F32 with the built-in SoVITS decoder. `--half` is a compatibility flag; BigVGAN and v3/v4 are outside the inference path. |
| Hardware | CPU on Linux and macOS (no Metal in the release). Linux/WSL2 CUDA images: `sm75`, `sm86`, `sm89`, `sm90`. No prebuilt `sm120` image for RTX 50-series cards. |
| HTTP | No built-in authentication. Bind to localhost or use an authenticating reverse proxy. |

## Documentation

- Get started: [Docker walkthrough](docs/FIRST_RUN.zh-CN.md) [中文] · [Model conversion](docs/MODELS.md) [中文] · [Deployment](docs/DEPLOYMENT.md) [中文].
- Connect: [API](docs/API.md) [EN] · Agent guides: [English](docs/AGENT_INTEGRATION.en.md) / [中文](docs/AGENT_INTEGRATION.md).
- Go deeper: [Inference and source map](IMPLEMENTATION.md) [中文] · [Build and test](docs/DEVELOPMENT.md) [EN] · [Performance reports](docs/PERFORMANCE.md) [中文] · [Documentation index](docs/README.md) [中文].

For interrupted downloads, rerun setup to reuse verified files. For startup errors, run `docker compose logs --tail 100` in the demo directory. [Report where you got stuck](https://github.com/ricardomlee/gpt-sovits-rs/issues/new?template=first-run.yml) without private models, recordings or credentials.

## License and Credits

Code: [MIT](LICENSE). Models retain their publishers' licenses. Use recordings you have permission to use; see [sample provenance](examples/samples/README.md). Built on [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) and [Candle](https://github.com/huggingface/candle). For training, use upstream GPT-SoVITS.
