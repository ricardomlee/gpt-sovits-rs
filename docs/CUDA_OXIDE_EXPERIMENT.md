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
| Candle cudarc context | 12.3-15.8 us | 0 |

Candle 路径四次测得 15.79、12.28、12.47 和 14.64 us。表中时间包含逐次 host launch 开销，不是只看 GPU event 的 kernel 时间，因此用于比较本项目实际调用方式更合适。

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
- 单个逐元素激活只有微秒级耗时，而且 Candle 接入路径在这个小算子上更慢，替换它本身不会改善端到端速度。
- 真正有价值的是融合多个操作，减少中间 Tensor 和 kernel launch；需要先用 profiler 找到稳定热点。

下一次实验应选择一个实际占时明显、边界清楚的 SoVITS 操作链，并同时对比 Candle 基线、数值误差和端到端耗时。没有稳定收益时，不引入生产依赖。
