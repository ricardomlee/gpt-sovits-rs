# Product Goal: Personal GPT-SoVITS TTS

This project should become a local-first, lightweight TTS service built on GPT-SoVITS inference. It should be easy to run on a PC or NAS, simple to integrate with a personal assistant, and capable of speaking with selected character voices without per-sentence parameter tuning.

## End State

- One binary or Docker Compose service starts a local TTS server.
- Models and voices live in mounted folders, so upgrades do not overwrite user data.
- Assistant integrations call a stable API with `voice`, `text`, and optional `style`.
- Users manage voice profiles instead of raw reference audio and sampling parameters.
- The inference engine keeps evolving with the best practical CPU/GPU backends and algorithms.

## Product Principles

- **Local first**: no cloud dependency for normal synthesis.
- **Simple deployment**: PC and NAS users should not need Python or PyTorch for inference.
- **Voice profiles over knobs**: users choose a voice/style; the system chooses stable defaults.
- **Inference-only Rust core**: training and fine-tuning stay in the Python GPT-SoVITS ecosystem.
- **Measurable progress**: performance, stability, and quality changes should be benchmarkable.

## Architecture

1. **Inference Core**
   - Rust + Candle GPT-SoVITS inference.
   - CPU, CUDA, and future backend-specific optimizations.
   - KV cache, CUDA graph, sentence splitting, and speaker feature cache.

2. **Voice Profile Layer**
   - `voices/<name>/voice.json` stores reference audio, reference text, language, and defaults.
   - Relative paths are resolved from the voice profile directory.
   - CLI and HTTP APIs should accept `voice` instead of raw reference paths.

3. **Automatic Strategy Layer**
   - Choose split mode based on text length.
   - Choose sampling defaults from the voice profile and text shape.
   - Insert stable sentence gaps/fades.
   - Detect likely bad generations, such as hitting `max_tokens`, and retry with safer settings.

4. **Service Layer**
   - CLI remains the debug and batch interface.
   - HTTP exposes assistant-friendly endpoints.
   - Docker Compose handles deployment on PC/NAS.

## Voice Profile Format

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频对应的文字",
  "language": "zh",
  "mode": "cuda-graph",
  "split_sentences": true,
  "min_sentence_chars": 12,
  "sentence_gap_ms": 120,
  "sentence_fade_ms": 8,
  "top_k": 15,
  "top_p": 0.95,
  "temperature": 0.8,
  "speed": 1.0,
  "max_tokens": 500,
  "repetition_penalty": 1.35
}
```

Example layout:

```text
voices/
  mao/
    voice.json
    ref.wav
```

CLI usage:

```bash
cargo run --release --features cuda --bin gpt-sovits -- \
  --voice mao \
  --text "人民，只有人民，才是创造世界历史的动力。" \
  --gpt-model models/gpt-model.safetensors \
  --sovits-model models/sovits-model.safetensors \
  --bert-model models/bert/bert.safetensors \
  --hubert-model models/hubert/hubert.safetensors \
  --output output.wav
```

Command-line values override profile defaults.

## Roadmap

### Phase 1: Usable Local Voice Profiles

- Add CLI `--voice` support.
- Add profile-based reference audio/text/language/default sampling.
- Keep raw CLI arguments for debugging and overrides.
- Add documentation and a sample profile.

### Phase 2: Assistant-Friendly HTTP

- Add `voice` to `/tts`, `/tts/stream`, and `/tts/batch`.
- Load voice profiles at server start or lazily on first use.
- Preload/cache selected voice features.
- Return structured errors for missing voices and invalid profiles.

### Phase 3: Automatic Quality Strategy

- Centralize text-length and punctuation-based strategy selection.
- Add automatic long-text splitting.
- Add retry-on-truncation with safer sampling.
- Add quality smoke tests that generate a fixed sentence matrix and fail on objective audio defects.
- Track objective audio metrics before listening: duration, RMS, peak, clipping ratio, silence ratio, DC offset, and non-finite samples.

### Phase 4: Deployment Packaging

- Add Docker Compose templates for CPU and CUDA.
- Keep `models/`, `voices/`, and `outputs/` as mounted volumes.
- Document NAS deployment expectations and fallback CPU settings.

### Phase 5: Continuous Optimization

- Keep benchmark targets under `benches/`.
- Track RTF, latency, token count, and max-token hits by voice/text set.
- Continue GPT decode, SoVITS decode, feature extraction, and backend optimizations.
