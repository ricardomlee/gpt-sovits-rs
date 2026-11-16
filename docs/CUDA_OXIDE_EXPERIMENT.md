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
- 主项目：Candle 0.11.1，Rust 1.95.0，release，F32；独立 CUDA stream，
  按生产配置关闭事件跟踪。
- 实验编译器：Rust 1.99.1，自带 LLVM 23.1.1。
- 每组预热 20 次、计时 500 次，重复 3 轮。下表为三轮平均调用耗时的中位数，单位 us。
  包含主机启动开销和末尾同步；分配路径还包含输出分配及清零。

| 元素数 | slope | 融合，复用输出 | 融合，分配输出 | 项目当前实现 | Candle 自带实现 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 257 | 0.1 | 5.36 | 11.65 | 49.08 | 25.83 |
| 257 | 0.01 | 6.39 | 10.79 | 31.07 | 25.84 |
| 16,384 | 0.1 | 5.56 | 10.98 | 37.63 | 39.48 |
| 16,384 | 0.01 | 5.12 | 9.51 | 30.46 | 27.49 |
| 65,536 | 0.1 | 5.63 | 11.14 | 32.66 | 26.98 |
| 65,536 | 0.01 | 5.42 | 10.92 | 35.33 | 37.10 |
| 262,144 | 0.1 | 7.37 | 15.07 | 52.98 | 40.68 |
| 262,144 | 0.01 | 9.68 | 15.80 | 50.27 | 35.05 |
| 1,048,576 | 0.1 | 12.07 | 17.45 | 73.57 | 62.89 |
| 1,048,576 | 0.01 | 15.32 | 20.72 | 77.83 | 57.11 |

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

上述比例统计设备 kernel 时间。完整合成的主机分配、launch 和同步成本通过下面的
独立计时比较，未在延迟测试中开启 profiler。

## 完整推理对照

在 `74242c9` 基础上，临时为现有 Candle `CustomOp1` 增加 CUDA 分支，嵌入上述
Rust kernel 的 PTX。保留 Candle 的张量、context、stream 和分配器，输出仍为新分配
的 Tensor。F32 CUDA 输入走实验分支，其他路径保持原样。

同一个 release 二进制按 A/B/A/B/A 顺序启动五个进程，A 使用原实现，B 使用融合
算子。固定 Sun v2Pro + SV、5 类文本和 10 个 seed；每类每轮预热 3 次、测量 10 次。
共 250 次计时与 75 次预热。计时覆盖完整合成至主机 AudioBuffer，模型与参考预加载
单列；HTTP 排队、编码和传输不在本轮比较范围。

每轮热推理耗时的中位数，单位 ms：

| 文本 | A1 | B1 融合 | A2 | B2 融合 | A3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 中文问候 | 466.40 | 446.19 | 464.46 | 455.03 | 454.86 |
| 多音字注音 | 424.39 | 402.02 | 424.50 | 416.98 | 426.21 |
| 中英混读 | 498.64 | 477.61 | 512.53 | 485.31 | 499.12 |
| 英文 | 749.40 | 725.96 | 757.23 | 742.93 | 731.19 |
| 中文段落 | 1973.06 | 1876.85 | 1942.40 | 1900.59 | 1932.58 |

每个 B 样本按相同文本和 seed，与前后 A 样本耗时的均值配对。合并所有配对样本，
总耗时降低 **2.85%**；B1 与 B2 分别降低 **3.81% / 1.88%**。五类文本在两轮 B 中的
配对节省中位数均为正。收益幅度随批次波动，不将这个点估计视为其他硬件的保证。

全部 325 段音频通过信号检查；相对 A1 的 260 次 WAV 对照全部逐字节一致。五轮各有
115 次实际 CUDA Graph 捕获并重放，未观察到回退。临时算子的两种 slope、非连续输入、
存储偏移和空 Tensor 检查通过，memcheck 为 0 errors。

实验二进制 SHA-256：`3b73ba2c90d0961e0b36acfe4a49d7ef25466aaa3e06b90898c9451240b62246`。
源代码补丁、PTX、语料、模型及设置哈希保留在本地报告中。临时推理接入和开关已撤回。

## 接入判断

架构改动小：复用现有 Candle 自定义算子即可，模型、流水线和 HTTP 接口均可保持
原样。使用预生成 PTX 时，用户无需安装 cuda-oxide，也无需增加另一套 GPU runtime。

长期维护仍需增加：

- 固定编译器工具链，并记录 Rust kernel 到 PTX 的生成来源。
- 验证 PTX 的目标架构和驱动兼容性；当前仅在 sm89 上实测。
- 维护 F32、布局和数值契约，以及不支持配置的回退行为。
- 在 Candle 或 kernel 编译器升级时重跑算子与完整推理回归。

**当前建议保留独立实验，不接入默认推理路径。** 约 3% 的完整合成收益尚不足以为
单个激活算子增加独立的生产构建与兼容维护链路。后续优先考虑由 Candle 提供融合
算子，或在能消除更多中间 Tensor 的操作链上重新评估 cuda-oxide。

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
