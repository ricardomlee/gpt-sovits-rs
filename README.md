# GPT-SoVITS-RS

<p align="center">
  <img src="assets/gpt-sovits-rs-logo.svg" alt="GPT-SoVITS-RS" width="880">
</p>

**Turn GPT-SoVITS voices into a local HTTP service, without a Python runtime.**

For people who already use GPT-SoVITS and want their assistant or application to
speak with a configured voice. Conversion and inference run in Rust. Training stays
in [upstream GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS).

[Try it](#quick-start) | [Listen](#samples) | [中文上手](docs/FIRST_RUN.zh-CN.md) |
[Use your models](docs/MODELS.md) | [API](docs/API.md)

## Samples

[Listen to Sun / download the generated WAV](examples/samples/sun-greeting.wav):

> 你好，欢迎回来。今天有什么想和我聊聊的吗？

Actual CPU inference with the maintainer's local **Sun v2Pro + SV** voice. Only generated
audio is shared, not the model, SV embedding, or reference. [Settings and provenance](examples/samples/README.md).
This short example is not a quality or speed benchmark; long text can still lose words.
There is also a [service confirmation sample](examples/samples/sun-zh.wav) and a
[browser listening page](site/README.md).
The downloadable first-run demo below uses standard v2 weights, not Sun.

## Quick Start

**Linux x86_64, Docker Engine + Compose v2, Bash, curl, and SHA-256 tools.**
Allow at least 6 GiB free disk and 4 GiB available memory. Downloads are about
1.1 GB plus the container; conversion and the first model load take time.
No Rust, Python, API key, or GPU is needed for this CPU demo.

The script downloads weights directly from their publisher, verifies checksums,
converts all four models inside the published **1.2.0** image, and checks the setup.
Read [what it downloads](examples/first-run/downloads.txt) and
[the script](examples/first-run/prepare.sh) before running it. Model licenses remain
separate from this project's MIT license.

```bash
git clone https://github.com/ricardomlee/gpt-sovits-rs.git
cd gpt-sovits-rs
bash examples/first-run/prepare.sh "$HOME/gpt-sovits-demo"
cd "$HOME/gpt-sovits-demo"
docker compose up -d --wait --wait-timeout 900
curl --fail-with-body --max-time 300 http://127.0.0.1:9881/tts \
  -H 'Content-Type: application/json' --data-binary @request.json \
  --output first.wav
```

Play `first.wav` with your usual audio player. The demo uses its own directory,
Compose project, and **localhost-only port 9881**, without changing a service on 9880.
Stop it with `docker compose down` in the demo directory; downloaded files stay there.
If preparation is interrupted, rerun the same command: verified downloads are reused.
Use a new directory, not an existing models or deployment directory.

Already have weights or use another platform?

| Your setup | Start here |
|---|---|
| Your own v2 / v2Pro voice | [Conversion and SV requirements](docs/MODELS.md), then [deployment](docs/DEPLOYMENT.md) |
| NVIDIA GPU | [CUDA images and architecture tags](docs/DEPLOYMENT.md#cuda) |
| Linux / macOS without Docker | [Download binaries](https://github.com/ricardomlee/gpt-sovits-rs/releases/latest), then [binary setup](docs/DEPLOYMENT.md#binary) |
| ARM Linux / NAS | [Build from source](docs/DEVELOPMENT.md); Docker images are currently amd64 |

## Connect Your App

Once a voice is configured, send only its name and the text:

```bash
curl --fail-with-body http://127.0.0.1:9881/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"The backup is complete."}' --output reply.wav
```

An [OpenAI-compatible speech endpoint](docs/API.md), sentence streaming, and multiple
voice profiles are also available. For an LLM assistant, submit completed short
sentences in order; [agent integration](docs/AGENT_INTEGRATION.md) covers playback
and errors. The service is not an authenticated public API.

## Before You Choose It

- Supports compatible **v2 and v2Pro** weights, not arbitrary newer GPT-SoVITS checkpoints.
- A clean 3-10 second reference clip and matching transcript are still required.
- v2Pro needs an SV embedding for the full voice-conditioning path. The released converter
  can convert an existing training embedding; it does not extract one from audio.
- Sentence splitting helps, but does **not** guarantee that every word will be spoken.
- No claim of being faster than Python. See [measured performance](docs/PERFORMANCE.md).
- Models are not bundled or rehosted. Only use voices and recordings you have permission to use.

## Help and Contribute

If the first request fails, check `docker compose logs --tail 100` in the demo directory.
[First-run troubleshooting](docs/FIRST_RUN.zh-CN.md) covers downloads, memory, ports,
and readiness. [Tell us where you got stuck, or what worked](https://github.com/ricardomlee/gpt-sovits-rs/issues/new?template=first-run.yml).
Do not attach private models, recordings, or credentials.

[Deployment](docs/DEPLOYMENT.md) | [API](docs/API.md) |
[Development](docs/DEVELOPMENT.md) | [Product goal](docs/PRODUCT_GOAL.md)

## License and Credits

Code: [MIT](LICENSE). Models keep their publishers' licenses.
The public sample has [separate provenance](examples/samples/README.md).
Built on [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) and
[Hugging Face Candle](https://github.com/huggingface/candle).
