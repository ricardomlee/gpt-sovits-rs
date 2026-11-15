# 文本前端：共享词典与音素编码

## 条件

- 对照源码：`451647b79154d0e4c95b73d138e3466617faa00a`；候选源码：`f558790ff9489ea749d2893b9991bb678669073c`。
- Linux x86_64 / WSL2，Intel Core i5-12490F，Rust 1.95.0，默认 CPU features，release 优化。
- 两版使用同一份 [前端微基准](../../benches/inference_bench.rs)。各项预热 1 秒、测量 2 秒、10 个样本；表中为 Criterion 时间估计。

## 热前端

| 操作 | 对照 | 候选 |
| --- | --- | --- |
| 创建并释放前端，词典已初始化 | 94.763 ms | 64.267 us |
| 中文规范化、G2P、编码及 word2ph | 81.602 us | 16.564 us |
| 英文规范化、G2P、编码及 word2ph | 59.803 us | 4.526 us |
| 中英混读规范化、G2P、编码及 word2ph | 101.420 us | 24.727 us |

```bash
# 对照与候选使用相同基准文件，在同一 CARGO_TARGET_DIR 中依次运行。
cargo bench --locked --bench inference_bench -- text_frontend \
  --sample-size 10 --warm-up-time 1 --measurement-time 2 --save-baseline frontend-before
cargo bench --locked --bench inference_bench -- text_frontend \
  --sample-size 10 --warm-up-time 1 --measurement-time 2 --baseline frontend-before
```

## 新进程与多管线

每个独立进程创建并保留 N 个 CPU `Pipeline`，不加载模型权重。
创建耗时包含首次词典初始化；随后用首个管线的前端处理 10 段文本，管线一直保留到进程结束。
峰值 RSS 使用 `/usr/bin/time` 的 `%M` 读取，KiB 换算为 MiB。
每个 N 交替运行对照与候选 3 对，以下为各自中位数；操作系统文件缓存保留。

| 保留管线数 | 对照创建耗时 | 候选创建耗时 | 对照峰值 RSS | 候选峰值 RSS |
| --- | --- | --- | --- | --- |
| 1 | 88.392 ms | 82.843 ms | 76.402 MiB | 64.750 MiB |
| 4 | 376.224 ms | 80.671 ms | 261.324 MiB | 65.035 MiB |
| 8 | 711.933 ms | 82.188 ms | 507.879 MiB | 65.449 MiB |

中文分词器和英文词典由标准库 `LazyLock` 各初始化一次。
英文词条直接借用内嵌数据，避免逐词分配字符串。
后续前端仍独立创建拼音映射、变调规则和符号表。

## 一致性与范围

10 段文本覆盖中文变调、显式拼音、数字、日期、中英混读、英文连字符及未知词。
18 次进程测量中，规范化文本、音素 ID 和 word2ph 的序列化结果 SHA-256 均为：

```text
e8945dba8b5a5dc38b6a0e8d2258ba8df10d9be55dc278ae2d3664f02c0e99c9
```

回归测试检查并发初始化共享词典、全部 732 个符号 ID，以及新编码器与原有整表贪心匹配规则的一致性。
差分输入包括整个词表、无分隔符拼接、256 组固定 seed 随机组合和 Unicode 边界。

这些数据覆盖文本前端和空管线。神经模型加载、GPU 推理、HTTP 延迟及完整合成速度另行测量。
共享词典首次使用后常驻进程，直到退出才释放。
