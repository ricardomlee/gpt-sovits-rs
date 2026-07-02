# 推理性能

## 当前基线

测试设备为 RTX 4060 Ti，输入“你好世界。”，参考音频为 `mao.wav`，KV cache、F32、贪心采样。开启逐阶段同步分析后，热路径约 286 ms，生成 20 个 semantic token 和 0.8 秒音频。

| 阶段 | 时间 | 占比 |
| --- | ---: | ---: |
| GPT | 227 ms | 79% |
| SoVITS | 53 ms | 19% |
| 目标文本 BERT | 5 ms | 2% |
| 文本前端、参考缓存 | <1 ms | <1% |

SoVITS 内部：

| 阶段 | 时间 |
| --- | ---: |
| decoder | 34 ms |
| EncP | 9 ms |
| flow | 3 ms |
| 参考编码 | 2 ms |
| 采样及准备 | 2 ms |

decoder 的 34 ms 中，resblocks 约 24 ms，转置卷积上采样约 8 ms。

首次使用一个音色还要处理参考音频：HuBERT、semantic tokenizer、频谱和参考 BERT 合计约 60-80 ms。`preload_speaker()` 会缓存结果，后续每句不再重复计算。

## 优化顺序

1. GPT decode：继续减少每 token 的 kernel launch、临时 Tensor 和同步。CUDA Graph 需要先解决长输出稳定性。
2. SoVITS decoder：验证融合 LeakyReLU；继续分析 resblock Conv1d 和转置卷积。cuDNN 可作为可选后端，但不能成为轻量部署的硬依赖。
3. EncP 和 flow：当前合计约 12 ms，只有 profiler 证明可融合时再改。
4. 参考音频：保持预加载，优化冷启动和音色切换，不占用热路径预算。
5. BERT 和文本前端：当前收益空间很小，优先保证文本与韵律质量。

静态 KV 已做过无 CUDA Graph 对照，没有收益：短文本动态/静态为 0.31/0.33 秒，中等文本为 2.05/2.09 秒，长文本为 3.59/3.62 秒。改成只计算有效长度后，中长文本可快约 2-5%，但短文本仍变慢；生成 32 token 后再切换的自适应方案也没有稳定收益。因此默认继续使用动态 KV。

## 分析命令

普通日志只适合看整段时间。CUDA 异步执行时，使用下面的分析模式才能得到可信的 SoVITS 子阶段时间：

```bash
GPT_SOVITS_SYNC_PROFILE=1 RUST_LOG=gpt_sovits_rs=debug \
  target/release/gpt-sovits --device cuda --mode kv \
  --split-sentences --text "你好世界。" \
  --reference-audio mao.wav \
  --reference-text "会战兵力是八十万对六十万，优势在我。" \
  --output /tmp/profile.wav
```

`GPT_SOVITS_SYNC_PROFILE` 会在各阶段插入同步，只用于分析，不用于生产运行。kernel 汇总方法见 [cuda-oxide 实验记录](CUDA_OXIDE_EXPERIMENT.md)。

只采集模型已加载后的 KV 热路径：

```bash
GPT_SOVITS_CUDA_PROFILE=1 nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda --sample=none --cpuctxsw=none \
  --output=/tmp/gpt-sovits-hot \
  target/release/examples/e2e_quick
```

一次 21 token 的热路径采集包含约 23,500 次 kernel launch、37,000 次异步显存分配和 9,100 次 H2D。H2D 总量不到 1 MB，说明主要瓶颈是大量小操作的调度和临时 Tensor，不是传输带宽。

CUDA Graph 的 300-token 回归显示，capture 外的 graphable 路径与动态 KV 一致，但 stream capture 后第一个 logits 就会明显偏离。初步判断与 Candle 算子在 capture 内的临时显存分配有关。实验模式会用 eager logits 检查第一次 graph launch，发现偏离时从已校验的 KV 状态继续，不再把错误 token 交给 SoVITS，也不用从头重算。

可用真实模型、`mao.wav` 和 300 个 token 的确定性生成检查三条解码路径：

```bash
cargo run --release --features cuda --example compare_cuda_graph_tokens
```
