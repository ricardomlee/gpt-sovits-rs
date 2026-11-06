# HTTP Command-Line Client

`gpt-sovits-client` sends text to a running GPT-SoVITS service and saves its WAV
response. Use it from a shell, a script, or an agent's command provider.
The service owns models, reference audio, SV embeddings, and inference settings.

This client is new in this branch. The v1.2.1 release archives do not include it.
It works with the v1.2.1 `/tts` endpoint; build it from source until the next release.

## Install

From the repository root, with stable Rust and Cargo installed:

```bash
cargo install --locked --path client
gpt-sovits-client --help
```

Cargo installs into `$CARGO_HOME/bin` (normally `~/.cargo/bin`); add it to `PATH`.
This standalone crate builds independently of the inference engine, libsoxr, and CUDA.

## Use

Choose a configured voice from `GET /voices`:

```bash
curl --fail-with-body http://127.0.0.1:9880/voices
gpt-sovits-client --voice demo --text 'Hello, how can I help?' -o reply.wav
printf '%s\n' 'Hello, how can I help?' | gpt-sovits-client --voice demo > reply.wav
```

`--voice` is required. `--url` defaults to `http://127.0.0.1:9880`; supply the service
base URL, including any reverse-proxy prefix. `--text` defaults to UTF-8 stdin,
and omitting `-o` / `--output` sends WAV bytes to stdout. `--timeout` defaults to
120 seconds for the HTTP request, including queueing and audio download.

Each invocation sends one `/tts` request containing `voice` and `text`. Language,
splitting, and sampling follow the voice profile and server defaults. The client
waits for the complete 32 kHz mono PCM16 WAV. It reports failures on stderr with a
nonzero exit code and does not retry automatically or follow redirects.

With `-o`, a temporary file is validated before replacing the destination; failed
requests preserve an existing file. Shell redirection (`> reply.wav`) truncates
the destination before the client starts, so use `-o` when replacing a recording.

Agent configuration and sentence-by-sentence playback: [Agent integration](../docs/AGENT_INTEGRATION.md).
Build and test commands: [Development](../docs/DEVELOPMENT.md#http-client).
