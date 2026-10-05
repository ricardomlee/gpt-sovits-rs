# Release Acceptance

Test a candidate before tagging, without replacing an existing service. No private
weights, embeddings, reference recordings, or generated regression audio belong in Git.
The public first-run default stays on the last verified published image until the
new release exists; old-version compatibility is tested separately from the candidate.

## Candidate CPU Image

Requires Docker Compose, curl, Rust, and the normal build dependencies. From the repo:

```bash
docker build -t gpt-sovits-rs:candidate .
DEMO_IMAGE=gpt-sovits-rs:candidate DEMO_PULL_POLICY=never DEMO_EXPECTED_VERSION=1.2.1 \
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
separate first-run workflow keeps verifying published 1.2.0. Neither substitutes
for GPU testing or proves semantic completeness from a WAV's shape alone.

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
  --url http://127.0.0.1:19881 --voice sun --label '1.2.0; image sha256:...' \
  --output-dir /tmp/tts-baseline --repeats 3
# Replace only the isolated service with the candidate, then:
target/debug/examples/speech_regression capture \
  --url http://127.0.0.1:19881 --voice sun --label '1.2.1 candidate; image sha256:...' \
  --output-dir /tmp/tts-candidate --repeats 3
```

Eight cases cover short speech, multiple sentences, digits, pronunciation annotations,
mixed text, a paragraph, streaming WAV, and the OpenAI-compatible speech endpoint.
An optional `--corpus` selects another JSON corpus. All attempts are retained;
there are no automatic retries, cherry-picking, or sampling overrides. Reference
and target languages are independently detected using `text_language=auto`.
Sampling remains stochastic, so repeats matter. Output directories cannot be reused.

Each run contains requests, raw HTTP responses, headers, curl timings, waveform
metrics, `capture.json`, and `asr.jsonl`. Streaming responses retain their raw
unknown-length WAV header; the ASR copy has only RIFF/data lengths repaired. PCM
samples are unchanged. Curl's first-byte time includes HTTP headers; it is **not**
claimed as time to the first audible sample.

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
captures never count as passes. It compares the mean error rate over repetitions
per case, then the mean across cases: defaults allow at most +0.05 overall and
+0.20 for any case. These are absolute error-rate differences, not relative percentages.
Choose tolerances **before** running with `--max-mean-increase` and `--max-case-increase`.
A measured regression returns a nonzero status and retains the comparison JSON.

Always inspect the hypotheses and listen to suspect clips. An ASR error is not
automatically a TTS error; digits and letter sequences may transcribe differently.
Passing relative thresholds does not mean either version is error-free, nor can
ASR reliably judge voice similarity, naturalness, or every polyphonic pronunciation.
Investigate failures without silently widening thresholds or replacing bad takes.

## Upgrade and Rollback

In the isolated deployment, exercise `1.2.0 -> candidate -> 1.2.0` with identical
read-only model/voice mounts. At each stage check `/health`, `/status`, `/voices`,
and real audio. Record the image ID and version, then compare SHA-256 manifests of
models, the voice JSON, reference recording, and SV embedding before and after.
Use a distinct container name and loopback-only spare port. Never run the production
Compose file unchanged for this exercise: it has a fixed production container name.

The 1.2.1 standalone binary defaults to loopback. LAN callers must explicitly select
`--host 0.0.0.0` or `GPT_SOVITS_HOST=0.0.0.0`. Docker's environment already does this;
host port exposure is separate. The server still has no built-in authentication.

## Publish

1. Require candidate CI and local CUDA/SV/regression/rollback evidence before tagging.
2. Match `Cargo.toml`, the package entry in `Cargo.lock`, the tag, and release notes.
   Release preflight checks metadata before any binary/image publishing job starts.
3. Watch all Linux/macOS and CPU/CUDA release jobs; verify packaged binaries and hashes.
4. After the versioned image exists, run the same verifier against it with pulling
   enabled. Then update the first-run defaults in `prepare.sh` and `compose.yml`,
   and the README/first-run guide. Keep the 1.2.0 compatibility workflow pinned.
5. Only then update a running personal deployment, retaining its old image tag for rollback.
