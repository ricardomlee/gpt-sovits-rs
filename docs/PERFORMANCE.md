# 推理性能

本文描述 1.2.2 的推理路径和测量方法。版本差异见 [文档入口](README.md)。

## 当前推理路径

- GPT 和 SoVITS 使用 F32。`--half` 目前不会启用完整半精度推理。
- `--mode auto` 在 CUDA F32 上选择 `cuda-graph`，其他设备选择 `kv`。
  `GPT_SOVITS_DISABLE_CUDA_GRAPH=1` 可让 auto 选择 KV；不存在旧文档中的实验启用开关。
- Graph 的前 32 个生成步骤走 eager/static KV；较短生成可能不触发 capture。
  首次 replay 有校验和回退保护，Graph 在每次生成中创建。
- 服务复用常驻模型和参考特征缓存。多音色缓存容量、队列及预加载见 [HTTP API](API.md)。

## 发布前本地 GPU 报告

每次打 tag 前，对候选提交运行一次。需要 CUDA 工具链、模型和可用的本地音色；
模型转换与 SV 配置见 [模型指南](MODELS.md)。基准在独立进程中运行，现有容器保持原样。
尽量避开其他 GPU 工作；同时运行的服务即使空闲也会占显存，需在验收记录中说明。

```bash
cargo build --locked --release --features cuda --example gpu_benchmark
mkdir -p outputs/gpu-benchmark
target/release/examples/gpu_benchmark \
  --voice sun --voices-dir voices --models-dir models \
  --build-label 'release; cuda' \
  --output-dir "outputs/gpu-benchmark/$(date -u +%Y%m%dT%H%M%SZ)"
```

`sun` 是本机音色示例，不随仓库提供。换成自己的 voice ID，并为历次发布固定同一组模型、
参考音频、文本和 SV。目录必须是新的，父目录需存在。构建后立即运行，保留提交与工作区状态；
报告中的 Git、Rust 和 nvcc 信息来自运行时环境，`build-label` 由调用者填写实际构建配置。
可执行文件、Cargo.lock、输入文件和语料均记录 SHA-256。

默认使用 [语音回归语料](../examples/speech_regression/corpus.json)，每项预热 2 次、测量 5 次。
基准读取语料文本，在进程内按音色的分句、语言和采样设置合成，每次尝试均保留。
语料中的 HTTP endpoint/stream 设置由 [发布验收](RELEASE_ACCEPTANCE.md) 工具使用。
`--corpus` 可指定同格式语料，`--mode kv` 可单独测 KV；对照时其他设置必须相同。
缺少 BERT、HuBERT、v2Pro 的 SV、CUDA，或波形检查失败时返回非零状态。

输出包括：

- `report.json`：元数据、模型加载/参考预加载耗时、所有尝试及逐项 min/median/max、RTF。
- `report.md`：可直接阅读的汇总；失败或测量次数不全的项不产生性能摘要。
- `inference.log`：模型加载、生成和 Graph 回退等诊断日志。
- WAV 和 `asr.jsonl`：保留预热及正式样本，ASR 清单只包括正式测量样本。

`signal_checks_passed` 记录静音、削波、非有限值等波形检查结果。语音完整性通过固定 ASR
模型和试听检查，步骤见 [语音回归流程](RELEASE_ACCEPTANCE.md#speech-regression)。
模型、参考音频和私人生成内容留在本地；对外发布报告前审查文本、音色名和元数据。

## 指标定义

| 指标 | 测量边界 |
| --- | --- |
| 热推理耗时 | 模型/参考已加载，从调用 Pipeline 到拿到主机内存中的完整 AudioBuffer；含文本前端、分段和拼接，不含保存 WAV、质量检查、HTTP 排队 |
| 加载/预加载 | 单独记录对应函数的墙钟耗时，保留操作系统文件缓存 |
| RTF | 每次推理秒数 / 该次生成音频秒数，再对重复测量汇总；越低越快 |
| HTTP 总耗时 | 包括传输、排队以及可能的模型切换 |
| HTTP TTFB | curl 收到首字节（可能只有响应头）的时间；首段 PCM 到达时间需在客户端另行测量 |
| 阶段日志 | 默认记录 CUDA 异步调用的主机耗时；同步分析方法见下文 |

比较版本时固定 profile、模型、SV、语料、采样和设备。随机采样会改变 token 数及音频长度，
应结合输出完整性、音频时长和多次测量的时延、RTF 判断变化。

## 已记录的测量

[2026-11-06 Sun v2Pro + SV / RTX 4060 Ti](benchmarks/2026-11-06-sun-cuda.md)：
release、12 项、每项预热 2 次和测量 5 次。84 次波形检查通过，ASR 异常待试听确认。
报告包含逐项耗时、输入哈希和识别结果。

## 定位瓶颈

逐阶段分析会插入同步，不应混入正式耗时报告。CLI 会设置日志过滤器，调试日志需要 `--verbose`：

```bash
GPT_SOVITS_SYNC_PROFILE=1 target/release/gpt-sovits \
  --verbose --device cuda --mode kv --voice sun --voices-dir voices \
  --models-dir models --text '你好，这是一次推理分析。' --output /tmp/profile.wav
```

旧工具仍可用于局部实验，但有不同测量边界：

| 工具 | 用途与限制 |
| --- | --- |
| `kv_cache_bench` | 解码路径对照，使用自有贪心参数和参考音频配置，未加载 voice 的 SV |
| `gpt_dtype_bench` | GPT 阶段的精度实验，依赖本地固定模型和参考素材 |
| `inference_bench` | 合成张量微基准，记录主机调用耗时；CUDA 操作异步执行 |
| `profile_kv_cache` / `e2e_quick` | 开发调试例子，依赖各自硬编码素材 |
| `conv1d_cpu_bench` | CPU Conv1d 微基准 |

[2026 年 7 月记录](history/performance-2026-07.md)及 [cuda-oxide 实验](CUDA_OXIDE_EXPERIMENT.md)
保存了早期 KV、半精度和 Graph 实验的测量条件与结果。
