# 韵律与分段

默认行为参考 Python GPT-SoVITS：

- 按 `cut5` 标点切分，保留句末标点，小于 5 个字符的片段向后合并。
- 每段之间加入 300ms 静音，不额外淡入淡出。
- 采样默认使用 `top_k=15`、`top_p=1`、`temperature=1`、`repetition_penalty=1.35`。
- `speed` 在 SoVITS EncP 的 latent 时间轴上做线性插值，不重采样最终音频。

这些值可以在 `voice.json` 里覆盖。参考音频本身的语速和语气仍会影响结果；分段能稳定长文本，但每段独立生成也可能带来轻微语气变化。
