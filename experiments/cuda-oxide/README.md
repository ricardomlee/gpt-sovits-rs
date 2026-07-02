# cuda-oxide 实验

这个目录用于验证 cuda-oxide 生成的 PTX，不参与主项目构建。当前实验是 SoVITS 使用的 fused LeakyReLU，包含数值检查和 kernel 平均耗时。

要求：

- Linux、NVIDIA GPU、CUDA Toolkit 12 以上
- `cargo-oxide` 0.2.1
- `nightly-2026-04-03`、`rust-src`、`rustc-dev`
- LLVM 21 以上，包含 `llc` 和 `opt`

运行：

```bash
cd experiments/cuda-oxide
cargo oxide doctor
cargo oxide run --arch sm_89
```

cuda-oxide 0.2.1 的 doctor 只查找 `llc-21` 或 `llc-22`。本机使用 LLVM 23 时，可以临时提供兼容名称：

```bash
mkdir -p /tmp/cuda-oxide-tools
ln -sf /usr/lib/llvm-23/bin/llc /tmp/cuda-oxide-tools/llc-22
PATH=/tmp/cuda-oxide-tools:$PATH \
CUDA_OXIDE_OPT=/usr/lib/llvm-23/bin/opt \
cargo oxide run --arch sm_89
```

生成 PTX 后，可以回到仓库根目录验证 Candle 集成：

```bash
cargo run --release --features cuda --example cuda_oxide_candle
```

实验背景、结果和结论见 [`../../docs/CUDA_OXIDE_EXPERIMENT.md`](../../docs/CUDA_OXIDE_EXPERIMENT.md)。
