# Public Samples

Four unedited recordings generated on 2026-11-22 with the official [v1.3.3 Linux x86_64 release](https://github.com/ricardomlee/gpt-sovits-rs/releases/tag/v1.3.3). All use CPU F32, KV decoding (`mode=auto`), seed 42 and mono 32 kHz PCM16. [Listen to Sun in your browser](https://tts.ricli.tech/#samples).

| Voice | Exact target text | WAV |
| --- | --- | --- |
| Sun · breakfast | 面包刚出炉，坐下来一起吃早餐吧。 | [2.56 s](sun-greeting.wav) |
| Sun · after the rain | 雨停了，河边的路灯亮了。我们慢慢走回家吧。 | [3.48 s](sun-zh.wav) |
| Public demo · same Chinese passage | 雨停了，河边的路灯亮了。我们慢慢走回家吧。 | [4.12 s](local-assistant-zh.wav) |
| Public demo · English | The rain has stopped. We can walk home by the river. | [4.12 s](local-assistant-en.wav) |

The two Chinese readings of the rain passage use different voices and references: Sun uses a Chinese recording; the public v2 demo uses an English one. Shorter sentence chunks were used for the cross-language demo so both sentences are heard. Local Qwen3-ASR-0.6B transcriptions match all four target texts ignoring punctuation and spacing. That check measures the words, not speaker similarity.

## Reproducible First-Run Baseline

The [first-run demo](../../docs/FIRST_RUN.zh-CN.md) downloads standard v2 weights and the public reference. Its [HTTP request](../first-run/request.json) produces the English line above. The released CPU container and the Linux CLI returned byte-identical English WAVs on the generation machine.

To run the CLI commands, put the unpacked `gpt-sovits` on your `PATH` (keep its libraries in place). Run from the demo directory containing `models/` and `voices/demo/`:

```bash
gpt-sovits --device cpu --voice demo --seed 42 \
  --text "The rain has stopped. We can walk home by the river." --output local-assistant-en.wav
gpt-sovits --device cpu --voice demo --seed 42 --language zh --prompt-language en \
  --min-sentence-chars 4 --text "雨停了，河边的路灯亮了。我们慢慢走回家吧。" --output local-assistant-zh.wav
```

The [demo profile](../first-run/voice.json) uses greedy GPT sampling (`top_k=1`). The Chinese command sets the target and reference languages separately. Both use sentence splitting, a 120 ms gap, 8 ms fade and a 500-token cap.

### Reference and Rights

The reference is **LJ Speech 1.1, LJ001-0001**, read by Linda Johnson for LibriVox and aligned by Keith Ito. The [dataset publisher](https://keithito.com/LJ-Speech-Dataset/) identifies its recordings, transcripts and annotations as public domain in the US; check the applicable rules in your jurisdiction. The generated samples are not recordings of the original reader or an endorsement.

Exact reference transcript:

> Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition

The [pinned reference copy](https://github.com/coqui-ai/TTS/blob/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/tests/data/ljspeech/wavs/LJ001-0001.wav) and models are downloaded and verified using the [SHA-256 manifest](../first-run/downloads.txt). No Coqui implementation code, reference dataset or weights are bundled here. Models retain their own licenses.

## Sun Showcase

Sun uses a private v2Pro GPT/SoVITS pair, a 4.98-second Chinese reference with its transcript, and the matching SV embedding. Only synthesized output is published. The LJ Speech terms above do not apply to Sun. These clips are not recordings of or endorsements by the original speaker.

With that authorized profile and its files available under `voices/sun/` and `models/`:

```bash
gpt-sovits --device cpu --voice sun --seed 42 \
  --text "面包刚出炉，坐下来一起吃早餐吧。" --output sun-greeting.wav
gpt-sovits --device cpu --voice sun --seed 42 \
  --text "雨停了，河边的路灯亮了。我们慢慢走回家吧。" --output sun-zh.wav
```

The Sun profile uses Chinese, `top_k=15`, `top_p=0.95`, temperature 0.8, repetition penalty 1.35 and the same sentence/gap/fade/token settings as the English demo. The public first-run script does not install this profile.

## Checksums

| Artifact | SHA-256 |
| --- | --- |
| v1.3.3 Linux x86_64 archive | `2c513c17a78a797dbdae3ebd34e650a6c1304144d8cb8d741040ecafb1550cca` |
| Inference executable | `6e1eb7610d613202c77573c1fc303fd8341fd1e799de4872d16d1e88a9b4bf37` |
| sun-greeting.wav | `dc1579a7ad62e9bacc737368f72705e26f16431731cd304567b569ca34c51e60` |
| sun-zh.wav | `3d71f7871d97d41215df5e2be5c651467e3f778cc67844326522a4385ccb20e4` |
| local-assistant-zh.wav | `e499bf34ea73b3626cb62f8fddd270886574f0350a853efa60d4951bdc89719a` |
| local-assistant-en.wav | `493e3ad383b1288e956a0fc1207234e1fde5c91d3b8181af26c4b31ec160975a` |

These hashes identify the published files. A different build or machine can produce different samples even with a fixed seed. The [release workflow](../../.github/workflows/release.yml) also saves an HTTP WAV from the CPU image in its `linux-acceptance` artifact.
