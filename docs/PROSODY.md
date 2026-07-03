# 韵律与分段

默认配置以听感稳定为先：

- 只在句号、问号、感叹号等完整句末切分，逗号不会拆成两次推理。
- 过短的句子向后合并，句间补 120ms 静音，并做 8ms 淡入淡出。
- 采样使用 `top_k=15`、`top_p=0.95`、`temperature=0.8`、`repetition_penalty=1.35`。
- `speed` 在 SoVITS EncP 的 latent 时间轴上做线性插值，不重采样最终音频。

Python 的 `cut5` 会在逗号、分号等位置切分，并在片段之间加入 300ms 静音。它适合兼容测试，但中文短句容易听起来断续，因此不再作为默认值。完整复现时使用 `--split-method cut5 --min-sentence-chars 5 --sentence-gap-ms 300 --sentence-fade-ms 0`，采样参数设为 `top_p=1`、`temperature=1`。

参考音频本身的语速和语气仍会影响结果。分段能提高长文本稳定性，但每段独立生成也可能带来轻微语气变化。
