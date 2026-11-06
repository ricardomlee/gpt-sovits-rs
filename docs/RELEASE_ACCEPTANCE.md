# Release Acceptance

Test a candidate before tagging, without replacing an existing service. No private
weights, embeddings, reference recordings, or generated regression audio belong in Git.
The public first-run default stays on the last verified published image until the
new release exists; old-version compatibility is tested separately from the candidate.

## Candidate CPU Image

Requires Docker Compose, curl, jq, Rust, and the normal build dependencies. From the repo:

```bash
docker build -t gpt-sovits-rs:candidate .
export DEMO_EXPECTED_VERSION
DEMO_EXPECTED_VERSION=$(cargo metadata --no-deps --format-version 1 --locked | jq -r '.packages[] | select(.name == "gpt-sovits-rs") | .version')
DEMO_IMAGE=gpt-sovits-rs:candidate DEMO_PULL_POLICY=never \
  bash examples/first-run/verify.sh /tmp/tts-candidate-demo /tmp/tts-candidate-evidence
```

Both directories must be new. Set `DEMO_PORT` if 9881 is occupied. Preparation
verifies upstream downloads and converts all four models with the selected image.
The verifier uses a unique Compose project, checks readiness and the running
binary version, makes a real HTTP synthesis request, validates its WAV, and records
the image ID, configuration, headers, and logs. Its exit trap removes only its own
container/network, including on failures. Downloaded models stay in the demo directory.
Do not rerun preparation on a production voice directory.

The main CI's CPU-container job runs this against its newly built image. The
separate first-run workflow verifies published 1.2.1 and keeps 1.2.0 as a
compatibility baseline. Local GPU and speech checks follow below.

## Local GPU Performance

Before each tag, run the candidate's [GPU benchmark](PERFORMANCE.md) on the local
GPU with the same voice, matching SV, models and corpus as the previous baseline.
Build the candidate from the recorded commit.
Use `release`, warm up, repeat, and keep the automatically generated `report.json`,
`report.md`, logs, WAVs and ASR manifest outside Git. Record concurrent GPU workloads.

The report measures warm in-process synthesis. Compare per-case latency, audio duration
and RTF after checking speech completeness with ASR and listening. Keep models and
sampling settings fixed across versions. The tool rejects CPU fallback, missing v2Pro
SV, incomplete measurements and failed signal checks.

CI tests report calculations and failure handling; the local run supplies GPU measurements.
Leave GPU acceptance pending when hardware is unavailable. Review reports for private
data before publishing. Complete the candidate-image and HTTP checks as well.

## Speech Regression

Use the same hardware, model files, voice profile, SV embedding, corpus, and ASR
model snapshot for both versions. Record their hashes alongside the reports.
Run an isolated HTTP service for each version sequentially on the same spare port.
For CUDA, build the candidate with the repository Dockerfile and the correct
compute capability; test the actual candidate image, not an old locally built binary.

Build the Rust capture/comparison tool (curl 7.76+ is its HTTP transport):

```bash
cargo build --locked --example speech_regression
target/debug/examples/speech_regression capture \
  --url http://127.0.0.1:19881 --voice sun --label 'v1.2.1 baseline; image sha256:...' \
  --output-dir /tmp/tts-baseline --repeats 3
# Replace only the isolated service with the candidate, then:
target/debug/examples/speech_regression capture \
  --url http://127.0.0.1:19881 --voice sun --label 'candidate commit ...; image sha256:...' \
  --output-dir /tmp/tts-candidate --repeats 3
```

The [tracked corpus](../examples/speech_regression/corpus.json) covers short speech,
multiple sentences, digits, decimal/percentage boundaries, pronunciation annotations,
mixed text, a paragraph, streaming WAV, and the OpenAI-compatible speech endpoint.
An optional `--corpus` selects another JSON corpus. All attempts are retained;
there are no automatic retries, cherry-picking, or sampling overrides. Reference
and target languages are independently detected using `text_language=auto`.
Sampling remains stochastic, so repeats matter. Output directories cannot be reused.

Each run contains requests, raw HTTP responses, headers, curl timings, waveform
metrics, `capture.json`, and `asr.jsonl`. Streaming responses retain their raw
unknown-length WAV header; the ASR copy has only RIFF/data lengths repaired. PCM
samples are unchanged. Curl's first-byte time includes HTTP headers; measuring the
first playable audio requires timing PCM arrival in the client.

ASR is a separate optional tool. With a locally installed Audire and a pinned,
offline Qwen3-ASR model snapshot, evaluate every captured clip:

```bash
audire evaluate /tmp/tts-baseline/asr.jsonl --backend qwen3-asr \
  --model-dir /path/to/pinned-asr-model --device cpu --max-new-tokens 256 --json \
  > /tmp/tts-baseline/asr.json
audire evaluate /tmp/tts-candidate/asr.jsonl --backend qwen3-asr \
  --model-dir /path/to/pinned-asr-model --device cpu --max-new-tokens 256 --json \
  > /tmp/tts-candidate/asr.json
target/debug/examples/speech_regression compare \
  --baseline /tmp/tts-baseline --candidate /tmp/tts-candidate \
  --output /tmp/tts-comparison.json
```

Comparison accepts Audire schema version 1 and checks exact sample coverage,
references, requests, backend, metric, and normalization counts. Missing or failed
captures never count as passes. Character, word, and mixed-language error rates
(CER, WER, MER) are supported and identified per case in the comparison report.
It compares the mean error rate over repetitions
per case, then the mean across cases: defaults allow at most +0.05 overall and
+0.20 for any case. These are absolute error-rate differences, not relative percentages.
The overall mean is an equal-weight regression indicator, not a corpus-wide CER.
Choose tolerances **before** running with `--max-mean-increase` and `--max-case-increase`.
A measured regression returns a nonzero status and retains the comparison JSON.

Inspect the hypotheses and listen to suspect clips to distinguish synthesis errors
from transcription errors, especially for digits and letter sequences. Also listen
for voice similarity, naturalness and polyphonic pronunciation. Retain failed takes
and the original thresholds when investigating a regression.

## Upgrade and Rollback

In the isolated deployment, exercise `last published -> candidate -> last published`
(currently `1.2.1 -> candidate -> 1.2.1`) with identical
read-only model/voice mounts. At each stage check `/health`, `/status`, `/voices`,
and real audio. Record the image ID and version, then compare SHA-256 manifests of
models, the voice JSON, reference recording, and SV embedding before and after.
Use a distinct container name and loopback-only spare port. Never run the production
Compose file unchanged for this exercise: it has a fixed production container name.

The 1.2.1 standalone binary defaults to loopback. LAN callers must explicitly select
`--host 0.0.0.0` or `GPT_SOVITS_HOST=0.0.0.0`. Docker's environment already does this;
host port exposure is separate. The server still has no built-in authentication.

## Publish

1. Require candidate CI, a fresh local GPU Markdown/JSON report, and CUDA/SV/speech/rollback evidence before tagging.
2. Match the root and client package versions in both manifests and lockfiles, the tag, and release notes.
   Release preflight checks metadata before any binary/image publishing job starts.
3. Use absolute HTTPS links in release notes. Pin repository documentation links to
   the release tag, for example `https://github.com/ricardomlee/gpt-sovits-rs/blob/v1.2.1/docs/RELEASE_ACCEPTANCE.md`.
   The workflow copies the Markdown body verbatim; relative paths lose the source
   file's directory context on a Release page. After publication, open each link
   from the actual Release page, not just from the repository Markdown preview.
4. Watch all Linux/macOS and CPU/CUDA release jobs; verify all three packaged binaries and hashes.
5. After the versioned image exists, run the same verifier against it with pulling
   enabled. Then update the first-run defaults in `prepare.sh` and `compose.yml`,
   and the README/first-run guide. Include the new published version in the
   first-run matrix and keep the 1.2.0 compatibility baseline pinned.
6. Only then update a running personal deployment, retaining its old image tag for rollback.
