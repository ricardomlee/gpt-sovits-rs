# GPT-SoVITS-RS 文档索引

## 快速上手与模型准备

| 你要做什么 | 文档 | 语言 |
| --- | --- | --- |
| 试听、选用浏览器或原生程序 | [中文主页](../README.zh-CN.md) · [English README](../README.md) | [中/EN] |
| 没有模型，生成第一段语音 | [用 Docker 运行公开示例](FIRST_RUN.zh-CN.md)（本机端口 `9881`） | [中文] |
| 用自己的音色 | [添加参考录音和逐字稿](DEPLOYMENT.md#添加音色) · [接入自己训练的 v2 / v2Pro 权重](MODELS.md#自训练音色模型) | [中文] |
| 对照试听或复现样本 | [四段音频的文本、来源及参数](../examples/samples/README.md) | [EN] |
| 在浏览器里读自己写的中文 | [为 Wasm 准备参考特征](../experiments/wasm-sovits/README.md#prepare-a-voice) | [EN] |

## 部署与应用接入

| 你要做什么 | 文档 | 语言 |
| --- | --- | --- |
| 在新机器上启动服务 | [按主机选择 CPU、Mac 或 CUDA 版本](DEPLOYMENT.md#选择部署方式)（模板默认端口 `9880`） | [中文] |
| 调用已有服务、接入 Hermes | [中文接入指南](AGENT_INTEGRATION.md) · [English guide](AGENT_INTEGRATION.en.md) | [中/EN] |
| 查 HTTP、CLI、Rust 接口，调整分段和停顿 | [API 参考](API.md) · [分句参数](API.md#chunking) | [EN] |
| 从 shell 调用 HTTP 服务 | [命令行客户端](../client/README.md) | [EN] |

## 原理、性能与开发

| 你要做什么 | 文档 | 语言 |
| --- | --- | --- |
| 跟一次语音请求读源码 | [推理流程图和五步源码导读](../IMPLEMENTATION.md#推理流程) | [中文] |
| 从源码构建、运行测试 | [开发指南与模块地图](DEVELOPMENT.md) | [EN] |
| 测量推理速度、对照版本 | [基准方法与 GPU 报告](PERFORMANCE.md) · [2026 年 7 月历史测量](history/performance-2026-07.md) | [中文] |
| 了解 Rust CUDA 算子实验 | [cuda-oxide 实验与取舍](CUDA_OXIDE_EXPERIMENT.md) | [中文] |
| 验收发行包、检查语音回归 | [发布验收与回滚](RELEASE_ACCEPTANCE.md) | [EN] |

按版本查阅：

- Sun CUDA 基准 [中文]：[1.3.4](benchmarks/v1.3.4-sun-cuda.md)、[1.3.3](benchmarks/v1.3.3-sun-cuda.md)、[1.3.2](benchmarks/v1.3.2-sun-cuda.md)、[1.3.1](benchmarks/v1.3.1-sun-cuda.md)、[1.3.0](benchmarks/v1.3.0-sun-cuda.md)、[1.2.2](benchmarks/v1.2.2-sun-cuda.md)。
- 专项测量 [中文]：[转置卷积](benchmarks/2026-11-16-cuda-transposed-convolution.md)、[文本前端](benchmarks/2026-11-15-text-frontend.md)、[Sun 本地测量](benchmarks/2026-11-06-sun-cuda.md)、[通道归一化修复](benchmarks/v1.2.2-layer-norm-fix.md)。
- 发布说明 [EN]：[1.3.5](releases/v1.3.5.md)、[1.3.4](releases/v1.3.4.md)、[1.3.3](releases/v1.3.3.md)、[1.3.2](releases/v1.3.2.md)、[1.3.1](releases/v1.3.1.md)、[1.3.0](releases/v1.3.0.md)、[1.2.2](releases/v1.2.2.md)、[1.2.1](releases/v1.2.1.md)、[1.2.0](releases/v1.2.0.md)、[1.1.0](releases/v1.1.0.md)、[1.0.2](releases/v1.0.2.md)、[1.0.1](releases/v1.0.1.md)、[1.0.0](releases/v1.0.0.md)。

文档随代码维护。使用旧版本时，从[对应的标签](https://github.com/ricardomlee/gpt-sovits-rs/tags)查看当时的文档。
