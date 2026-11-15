# 文档

GPT-SoVITS-RS 将已有音色变成本地 HTTP 语音服务。先跑出声音，再配置自己的音色，
最后接入应用；模型和参考录音保存在服务所在机器。

## 开始使用

| 你要做什么 | 从这里开始 |
| --- | --- |
| 先听效果 | [在线试听](https://ricardomlee.github.io/gpt-sovits-rs/) |
| 没有模型，生成第一段语音 | [Docker 首次运行](FIRST_RUN.zh-CN.md) |
| 使用已有或微调的 v2 / v2Pro 模型 | [下载与转换模型](MODELS.md) |
| 长期运行 CPU / CUDA 服务，配置音色和升级 | [部署](DEPLOYMENT.md) |
| 接入 Hermes 或自己的 Agent | [Agent 接入](AGENT_INTEGRATION.md) |
| 查 HTTP 字段、CLI 选项或 Rust 调用 | [API](API.md) |
| 从 shell 调用已有服务 | [HTTP 命令行客户端](../client/README.md) |
| 调整分段、停顿和语速 | [韵律与分段](PROSODY.md) |

首次体验使用 **9881** 端口和 `demo` 音色；正式部署及 API 示例使用 **9880**。
接入时先查询 `GET /voices`，将示例中的音色名换成自己的配置。

## 开发与维护

| 主题 | 文档 |
| --- | --- |
| 推理流程图、模型条件和缓存职责 | [实现说明](../IMPLEMENTATION.md) |
| 编译、代码目录、测试和 CI | [开发](DEVELOPMENT.md) |
| 测量方法、GPU 自动报告和历史结果 | [性能](PERFORMANCE.md) |
| 发版验收、语音回归和回滚 | [发布验收](RELEASE_ACCEPTANCE.md) |
| 项目范围与工程原则 | [产品目标](PRODUCT_GOAL.md) |

文档随仓库代码维护。使用固定版本时，从[对应 tag](https://github.com/ricardomlee/gpt-sovits-rs/tags)
进入该版本的文档；变更记录见 [Releases](https://github.com/ricardomlee/gpt-sovits-rs/releases)。
[历史测量](history/performance-2026-07.md)和 [cuda-oxide 实验](CUDA_OXIDE_EXPERIMENT.md)
保留当时的环境与结论，当前测量入口在性能指南。
