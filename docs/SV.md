# Native v2Pro SV Features

Current source builds can compute v2Pro/v2ProPlus speaker-verification features
directly from reference WAV audio. Inference and checkpoint conversion remain
Rust-only. This feature is not in the published v1.2.0 images yet.

## Prepare the Encoder Once

Download `sv/pretrained_eres2netv2w24s4ep4.ckpt` from the
[official GPT-SoVITS model repository](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/sv),
or use the same file from an existing GPT-SoVITS installation. Model weights are
not included in this repository, releases, or container images.

```bash
gpt-sovits-convert sv-model \
  /path/to/pretrained_eres2netv2w24s4ep4.ckpt \
  models/sv/sv.safetensors
```

The converter validates the inference architecture and removes unused classifier
weights and training counters. The runtime evaluates ERes2NetV2 in F32 on CPU or
CUDA. It does not run a Python process or require ONNX Runtime.

The encoder is discovered at `models/sv/sv.safetensors` or `models/sv.safetensors`.
Use `--sv-model <PATH>` or `GPT_SOVITS_SV_MODEL=<PATH>` to override it. In Compose,
set `SV_MODEL` to a container path, or leave it empty for automatic discovery.

## Voice Configuration

For a v2Pro model pair, the usual profile is sufficient:

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频里逐字对应的文字",
  "gpt_model": "my-voice/gpt.safetensors",
  "sovits_model": "my-voice/sovits.safetensors",
  "language": "zh"
}
```

CLI, HTTP, warmup, and preloaded voices use the same extraction path. Reference
features are cached in the existing bounded per-pipeline cache. The encoder
weights are shared between HTTP voice pipelines. Restart the service or call
`Pipeline::clear_speaker_cache()` after replacing reference files in place.

For the Rust API, call `pipeline.load_sv(path)` along with the other model loaders.
Loading a new SV or SoVITS model clears that pipeline's cached reference features.

An explicit `sv_embedding` in a profile/request (or CLI `--sv-embedding`) still
takes precedence, and works without encoder weights. The existing converter
command remains available:

```bash
gpt-sovits-convert sv /path/to/reference.wav.pt voices/demo/ref_sv.safetensors
```

**Compatibility change:** v2Pro without either an encoder or an explicit
embedding now fails with setup instructions. It no longer silently substitutes
a zero vector. Existing profiles with a valid `sv_embedding` continue to work;
v2 models do not need an SV encoder. `--doctor --voice <NAME>` checks this setup.

## Numerical Contract

The reference is GPT-SoVITS commit `bf81cdb14a38b674b6e9996dabc97340bc9978d2`:

- [`sv.py`](https://github.com/RVC-Boss/GPT-SoVITS/blob/bf81cdb14a38b674b6e9996dabc97340bc9978d2/GPT_SoVITS/sv.py): ERes2NetV2 with baseWidth=24, scale=4, expansion=4.
- [`TTS._get_ref_spec`](https://github.com/RVC-Boss/GPT-SoVITS/blob/bf81cdb14a38b674b6e9996dabc97340bc9978d2/GPT_SoVITS/TTS_infer_pack/TTS.py): average channels, resample to the SoVITS rate, apply the upstream peak clamp, then resample to 16 kHz. No HuBERT tail padding or peak normalization is added.
- [`kaldi.py`](https://github.com/RVC-Boss/GPT-SoVITS/blob/bf81cdb14a38b674b6e9996dabc97340bc9978d2/GPT_SoVITS/eres2net/kaldi.py): 80-bin log-power fbank, 25 ms Povey windows, 10 ms shift, 512-point FFT, per-frame DC removal, pre-emphasis=0.97, dither=0, snip_edges, no CMVN.
- [`ERes2NetV2.forward3`](https://github.com/RVC-Boss/GPT-SoVITS/blob/bf81cdb14a38b674b6e9996dabc97340bc9978d2/GPT_SoVITS/eres2net/ERes2NetV2.py): `[1, 20480]` mean of fused intermediate features, **not** the 192-dimensional classification embedding. No L2 normalization is applied.

F32 implementations are compared numerically rather than bit-for-bit. Default
tests include an upstream synthetic fbank fixture, audio validation, resampling,
embedding precedence, and missing-encoder errors; no models or Python are needed.

For full parity, prepare an external upstream F32 safetensors fixture containing
`samples` (`[N]`, the prepared 16 kHz waveform), `fbank` (`[1,T,80]`), and
`embedding` (`[1,20480]`), then run:

```bash
cargo run --profile dev-gpu --features cuda --example sv_parity -- \
  --model models/sv/sv.safetensors \
  --baseline /path/to/upstream-reference.safetensors \
  --reference /path/to/reference.wav --device cuda
```

Use `--device cpu` for CPU verification. The check compares fbank, encoder output
on upstream fbank, the complete feature path, and WAV decoding/resampling
separately. It checks finite values, shapes, relative L2 error, and cosine
similarity. Real reference audio, embeddings, and model files stay local.

## Attribution

The ERes2NetV2/AFF topology follows Alibaba's
[3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker), as adapted by
GPT-SoVITS. Original copyright: 3D-Speaker, Apache License 2.0; the license is
included in `licenses/3D-Speaker-Apache-2.0.txt`. `src/models/sv/network.rs` is a
Rust/Candle implementation of that inference topology, without the training and
classification paths. The audio code implements the upstream torchaudio Hann
sinc resampling convention and fixed Kaldi fbank configuration.
