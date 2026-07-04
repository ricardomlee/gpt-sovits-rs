# Samples

This folder is reserved for short public-safe audio samples used by the README.

Suggested format:

| File | Text | Notes |
|---|---|---|
| `demo_zh.wav` | `你好，这是 GPT-SoVITS-RS 的本地语音服务。` | Short CLI/API demo. |
| `demo_long_text.wav` | A 2-3 sentence paragraph. | Demonstrates sentence splitting. |

Keep samples small and easy to review:

- WAV, mono, 32 kHz.
- Prefer 3-12 seconds per sample.
- Use voices and reference audio that are safe to publish.
- Include the exact target text in this README.
- Do not commit private character voices or unlicensed reference audio.

`*.wav` is ignored by default in the repository. Add a reviewed public sample explicitly with:

```bash
git add -f examples/samples/demo_zh.wav
```
