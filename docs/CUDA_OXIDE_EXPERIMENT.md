# cuda-oxide 实验

最近复测：2026-11-16。实验使用 Rust 编写 F32 LeakyReLU kernel，生成 PTX 后
直接读写 Candle Tensor 的显存。生产推理路径保持原样。

## 上游更新

[CUDA Rust](https://github.com/NVIDIA/cuda-rust) 已合并 cuda-oxide 与 cutile-rs
及共享 host runtime。本轮固定提交 `348895e421a92d13a2c3a843ccb20ed1fb8b7f8b`：

- cargo-oxide / cuda-device / cuda-host 0.4.0，cuda-core 0.5.0。
- 工具链固定为 Rust 1.99.1，自动探测其 LLVM 23；旧的 `llc-22` 兼容名称已移除。
- `#[cuda_module]` 将设备产物嵌入实验二进制，并生成类型化的 kernel 启动方法。
- 编译器仍需要 `rustc-dev` 与内部编译器接口；`cargo oxide` 管理
  `RUSTC_BOOTSTRAP`。工具链声明和 Cargo 依赖限定在独立实验目录。

Candle 互操作示例只加载生成的 PTX，沿用 Candle 的 context 和 stream。
参考上游 [安装说明](https://github.com/NVIDIA/cuda-rust/blob/348895e421a92d13a2c3a843ccb20ed1fb8b7f8b/cuda-oxide/cuda-oxide-book/getting-started/installation.md)
及 [类型化启动接口](https://github.com/NVIDIA/cuda-rust/blob/348895e421a92d13a2c3a843ccb20ed1fb8b7f8b/cuda-oxide/README.md)。

## 本机结果

- RTX 4060 Ti，sm89，驱动 630.20，CUDA 13.2.51，WSL2。
- 主项目：Candle 0.11.1，Rust 1.95.0，release，F32，独立 CUDA stream。
- 实验编译器：Rust 1.99.1，自带 LLVM 23.1.1。
- 每组预热 20 次、计时 500 次，重复 3 轮。下表为三轮平均调用耗时的中位数，单位 us。
  包含主机启动开销和末尾同步；分配路径还包含输出分配及清零。

| 元素数 | slope | 融合，复用输出 | 融合，分配输出 | 项目当前实现 | Candle 自带实现 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 257 | 0.1 | 6.52 | 13.22 | 40.83 | 41.68 |
| 257 | 0.01 | 7.45 | 12.50 | 63.73 | 55.97 |
| 16,384 | 0.1 | 5.58 | 10.35 | 61.48 | 49.80 |
| 16,384 | 0.01 | 6.10 | 16.65 | 42.29 | 31.54 |
| 65,536 | 0.1 | 6.46 | 9.91 | 41.77 | 45.62 |
| 65,536 | 0.01 | 5.67 | 12.99 | 40.96 | 37.96 |
| 262,144 | 0.1 | 7.70 | 20.66 | 41.77 | 39.12 |
| 262,144 | 0.01 | 12.52 | 13.05 | 47.28 | 34.58 |
| 1,048,576 | 0.1 | 12.73 | 18.16 | 80.21 | 59.00 |
| 1,048,576 | 0.01 | 15.81 | 17.44 | 68.55 | 57.23 |

项目基线直接调用 `sovits_decoder::leaky_relu`，Candle 基线调用
`candle_nn::ops::leaky_relu`。30 组对照的最大误差均为 0。检查覆盖两个负斜率、
非整线程块长度及偏移 7 个元素的连续视图；错误检查会拒绝长度不符、NaN 和无穷值。
Compute Sanitizer memcheck 对独立 kernel 和 Candle 互操作均报告 0 errors。

测量存在主机调度波动；这些数值用于评估算子融合，不代表上游版本间的速度变化。

## 实际推理热点

在提交 `cd1549b` 的推理实现上采集 Sun v2Pro + SV：3 段文本、seeds 11/42、
每段预热一次，共 9 次合成。auto 模式实际使用 CUDA Graph。所有信号检查通过，
9 个 WAV 与依赖升级验收样本逐字节一致。

Nsight Systems 2025.6.3 开启 CUDA Graph node trace，采集包含模型及参考加载：

- 识别到 864 条连续的 `maximum → minimum → const_set → multiply → add`
  激活链，每次合成约 96 条；其 GPU kernel 时间合计 101.97 ms。
- 全部 GPU kernel 时间合计 2798.19 ms，上述激活链占 3.64%。
- 转置卷积与 im2col1d 合计 563.27 ms，约占 20.1%。

上述比例统计设备 kernel 时间。主机分配、launch、同步及 profiler 开销需在端到端
对照中另外衡量；本轮没有接入融合 kernel 的 TTS 候选版本。

结论：保留独立实验。融合算子有明确局部收益，下一步先做小范围端到端 A/B/A，
再决定是否承担预编译 PTX 的发布、架构覆盖和运行时维护成本。

## 复现

Linux 需要 CUDA Toolkit 13+、兼容驱动、Clang/libclang 21+。
在仓库根目录安装固定版本：

```bash
rustup toolchain install 1.99.1 --profile minimal \
  --component rust-src,rustc-dev,llvm-tools --no-self-update
cargo +1.99.1 install --locked \
  --git https://github.com/NVIDIA/cuda-rust.git \
  --rev 348895e421a92d13a2c3a843ccb20ed1fb8b7f8b cargo-oxide

cd experiments/cuda-oxide
cargo oxide doctor
cargo oxide run --arch sm_89

cd ../..
cargo run --locked --release --features cuda --example cuda_oxide_candle
cargo test --locked --release --features cuda --example cuda_oxide_candle
compute-sanitizer --tool memcheck --error-exitcode 1 \
  target/release/examples/cuda_oxide_candle
```

设备 `.ll`、`.ptx` 及其他生成物不提交；模型、WAV、原始 profile 和报告保留在本地
`outputs/cuda-oxide-20261116/`。当前发布基准见 [性能指南](PERFORMANCE.md)。
旧工具链和 2026 年 7 月数据见 [历史记录](https://github.com/ricardomlee/gpt-sovits-rs/blob/cd1549be6e51dd16294615ff35f349726d18b570/docs/CUDA_OXIDE_EXPERIMENT.md)。
