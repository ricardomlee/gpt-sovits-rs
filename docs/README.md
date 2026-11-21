# 文档

GPT-SoVITS-RS 将已有音色变成本地 HTTP 语音服务。先跑出声音，再配置自己的音色，
最后接入应用；模型和参考录音保存在服务所在机器。

## 开始使用

| 你要做什么 | 从这里开始 |
| --- | --- |
| 先听效果 | [在线试听](https://ricardomlee.github.io/gpt-sovits-rs/) |
| 没有模型，生成第一段语音 | [Docker 首次运行](FIRST_RUN.zh-CN.md) |
| 用自己的录音换音色 | [添加音色](DEPLOYMENT.md#添加音色)；已有微调权重见[模型转换](MODELS.md) |
| 新机器上部署，选 CPU、Mac 或 CUDA 版本 | [按主机选包](DEPLOYMENT.md#选择部署方式) |
| 已有服务，接入 Hermes 或自己的 Agent | [查音色并试一句](AGENT_INTEGRATION.md) |
| 查 HTTP 字段、CLI 选项或 Rust 调用 | [API](API.md) |
| 从 shell 调用已有服务 | [HTTP 命令行客户端](../client/README.md) |
| 调整分段、停顿和语速 | [API 分段参数](API.md#chunking) |

首次体验使用 **9881** 端口和 `demo` 音色；正式部署及 API 示例使用 **9880**。
接入时先查询 `GET /voices`，将示例中的音色名换成自己的配置。

## 开发与维护

| 主题 | 文档 |
| --- | --- |
| 推理流程图、模型条件和缓存职责 | [实现说明](../IMPLEMENTATION.md) |
| 编译、代码目录、测试和 CI | [开发](DEVELOPMENT.md) |
| 测量方法、GPU 自动报告和历史结果 | [性能](PERFORMANCE.md) |
| 发版验收、语音回归和回滚 | [发布验收](RELEASE_ACCEPTANCE.md) |

文档随仓库代码维护。使用固定版本时，从[对应 tag](https://github.com/ricardomlee/gpt-sovits-rs/tags)
进入该版本的文档；变更记录见 [Releases](https://github.com/ricardomlee/gpt-sovits-rs/releases)。
[历史测量](history/performance-2026-07.md)保留早期数据与环境；
[cuda-oxide 实验](CUDA_OXIDE_EXPERIMENT.md)单独记录算子探索，当前发布测量入口在性能指南。
