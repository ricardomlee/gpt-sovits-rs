# cuda-oxide 实验

独立的 F32 LeakyReLU kernel 实验，使用 cuda-oxide 0.4.0 生成 PTX 和嵌入式模块。
依赖固定于 NVIDIA/cuda-rust 提交 `348895e421a92d13a2c3a843ccb20ed1fb8b7f8b`，
工具链为 Rust 1.99.1。主项目的构建和推理路径保持原样。

安装固定版本的方法和环境要求见 [实验记录](../../docs/CUDA_OXIDE_EXPERIMENT.md#复现)。
在本目录运行：

```bash
cargo oxide doctor
cargo oxide run --arch sm_89
```

该程序验证嵌入式模块的类型化启动。回到仓库根目录，复用 Candle 的 context、
stream 和显存，对比融合算子、项目实现与 Candle 自带实现：

```bash
cargo run --locked --release --features cuda --example cuda_oxide_candle
cargo test --locked --release --features cuda --example cuda_oxide_candle
```

对照包含 5 种长度、两个 slope，各重复 3 轮；计时区间包含主机启动开销。
数值检查覆盖连续视图的存储偏移和不满线程块的尾部。
