# cuda-oxide 实验记录

## 目的

验证 [cuda-oxide](https://github.com/NVlabs/cuda-oxide) 能否用于本项目的自定义 CUDA 融合算子，同时不改变 Candle 的张量、显存和 CUDA context 管理。

这不是替换 Candle。合适的用法是：cuda-oxide 负责把少量性能关键的 Rust kernel 编译成 PTX，Candle 继续负责模型和运行时。

## 当前结果

测试环境：

- NVIDIA RTX 4060 Ti，计算能力 8.9
- CUDA 13.2，驱动 610.62
- cuda-oxide 0.2.1，固定提交 `ae41d1edce943d2f8c8f720e7024a9ef5e98ade9`
- `nightly-2026-04-03`
- LLVM 23

测试算子是 SoVITS 中可用的 fused LeakyReLU，输入为 1,048,576 个 F32 元素。20 次预热、500 次执行后：

| 路径 | 平均 kernel 时间 | 最大误差 |
| --- | ---: | ---: |
| cuda-oxide 自带 host runtime | 7.97 us | 0 |
| cuda-oxide + Candle，复用输出 | 11.83-16.59 us | 0 |
| cuda-oxide + Candle，每次分配输出 | 19.36-30.71 us | 0 |
| Candle 四算子 LeakyReLU | 76.57-91.80 us | 0 |

输入为 1,048,576 个 F32 元素。表中时间包含 host launch 和 Candle 输出分配开销，不是只看 GPU event 的 kernel 时间。每次分配输出的 cuda-oxide 路径更接近生产用法，相比当前由 `maximum/minimum/multiply/add` 组成的 Candle 实现快 3 倍以上。

已经验证：

- Rust kernel 可以生成适用于 `sm_89` 的 PTX。
- PTX 可以通过 Candle 使用的 cudarc context 动态加载。
- kernel 可以直接读写 Candle F32 Tensor 的显存，不需要设备间复制。
- 主 crate 仍使用 stable Rust；nightly 和 cuda-oxide 依赖只存在于独立实验目录。

## 复现

cuda-oxide 目前要求 Linux、CUDA Toolkit 12 以上、指定 nightly，以及带 `rust-src`、`rustc-dev` 的工具链。

```bash
rustup toolchain install nightly-2026-04-03 \
  --component rust-src,rustc-dev,rust-analyzer
cargo +nightly-2026-04-03 install \
  --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide --force

cd experiments/cuda-oxide
cargo oxide setup
cargo oxide doctor
cargo oxide run --arch sm_89
```

本机 LLVM 23 需要给当前版本的探测逻辑提供兼容名称：

```bash
mkdir -p /tmp/cuda-oxide-tools
ln -sf /usr/lib/llvm-23/bin/llc /tmp/cuda-oxide-tools/llc-22
PATH=/tmp/cuda-oxide-tools:$PATH \
CUDA_OXIDE_OPT=/usr/lib/llvm-23/bin/opt \
cargo oxide build --arch sm_89

cd ../..
cargo run --release --features cuda --example cuda_oxide_candle
```

生成的 `.ll` 和 `.ptx` 是本机实验产物，不提交到仓库。

## 判断

cuda-oxide 可以继续实验，但目前不应进入默认推理路径：

- 项目仍处于早期阶段，编译器和 API 变化快。
- 工具链要求比主项目严格，部署时直接编译不够轻量。
- 单次激活仍只有几十微秒，但 SoVITS 会反复调用。融合 LeakyReLU 有明确的局部收益，是否进入生产路径取决于 E2E 收益和部署成本。
- 真正有价值的是融合多个操作，减少中间 Tensor 和 kernel launch；需要先用 profiler 找到稳定热点。

下一次实验应选择一个实际占时明显、边界清楚的 SoVITS 操作链，并同时对比 Candle 基线、数值误差和端到端耗时。没有稳定收益时，不引入生产依赖。

## 端到端分析

Nsight Systems 对同一次短文本的 plain 和 KV 推理汇总显示：

- GPT FFN 用 `clamp(0, MAX)` 实现 ReLU，产生了约 1200 对 `maximum/minimum` kernel。
- 改用 Candle 原生 `relu()` 后，这部分变成单 kernel，约 1200 次合计 3.15 ms。
- 原实现同等调用量的 `maximum/minimum` 约需 16 ms，GPU 时间预计减少约 13 ms。
- WaveNet 门控的 `tanh/sigmoid` 各只有约 38 次、总计不足 0.1 ms，不值得优先融合。
- 剩余约 194 对 `maximum/minimum` 来自 SoVITS LeakyReLU，适合作为下一项 cuda-oxide 融合实验。

这些数字会随 GPT 生成 token 数变化，因此用于判断热点和 kernel 数量，不当作稳定的端到端基准。

采集命令：

```bash
cargo build --release --features cuda --example e2e_quick
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --output=/tmp/gpt-sovits-profile \
  target/release/examples/e2e_quick
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum \
  /tmp/gpt-sovits-profile.nsys-rep
```

下一步应把 cuda-oxide LeakyReLU 作为显式实验开关接入 SoVITS，使用预编译 PTX，并比较相同输入、相同随机种子下的 E2E 延迟和音频逐样本误差。默认路径暂时不依赖 PTX。
