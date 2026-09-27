# SV Frontend Fixture

`synthetic_fbank.safetensors` contains no model weights or speech. Its input is
1,040 deterministic synthetic F32 samples:

```text
samples[i] = (((i * 17 + 23) % 101) - 50) / 512
```

The `fbank` tensor was produced by GPT-SoVITS commit
`bf81cdb14a38b674b6e9996dabc97340bc9978d2`, using its
`GPT_SoVITS/eres2net/kaldi.py` with `num_mel_bins=80`,
`sample_frequency=16000`, `dither=0`, and all other defaults.
Generation used PyTorch/torchaudio 2.4.1 on CPU in F32. Shape: `[1, 5, 80]`.

The Rust test reads these golden values without installing or invoking Python.
Real speech and full encoder parity fixtures are local-only; see `docs/SV.md`.
