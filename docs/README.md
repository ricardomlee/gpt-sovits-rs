# 文档导航与版本范围

核对日期：2026-11-06，运行时代码基线：[`ee4d719`](https://github.com/ricardomlee/gpt-sovits-rs/commit/ee4d719)。
文档描述所在分支的代码，下载版与主线的功能差异列在下表。

## 先选你的版本

| 范围 | 状态 |
| --- | --- |
| [已发布 v1.2.1](https://github.com/ricardomlee/gpt-sovits-rs/releases/tag/v1.2.1) | 首次体验脚本固定使用此版；支持 v2/v2Pro、SV 文件、HTTP、分句流式和多模型缓存 |
| 主线，尚未发布 | 在 v1.2.1 之上加入中文百分比、小数、全角数字归一化、注音位置修复和数字切句保护（[#10](https://github.com/ricardomlee/gpt-sovits-rs/pull/10)） |

主线的 Cargo 包版本暂时仍为 `1.2.1`，源码构建与正式发行物需要结合提交或镜像 digest 区分。
报告问题和性能时，同时提供源码提交、构建 features/profile 或镜像 digest。
下载旧版用户应阅读[该 tag 下的文档](https://github.com/ricardomlee/gpt-sovits-rs/tree/v1.2.1/docs)。

## 使用与维护

| 任务 | 文档 |
| --- | --- |
| 没有模型，先跑出声音 | [首次运行](FIRST_RUN.zh-CN.md) |
| 下载、转换自己的模型和 SV | [模型](MODELS.md) |
| 配置 Docker、二进制、音色和升级 | [部署](DEPLOYMENT.md) |
| HTTP、CLI、Rust 调用 | [API](API.md) |
| 接入助手、句子播放、读音控制 | [Agent 接入](AGENT_INTEGRATION.md)、[韵律与分段](PROSODY.md) |
| 理解当前推理和缓存 | [实现说明](../IMPLEMENTATION.md) |
| 构建、测试与代码目录 | [开发](DEVELOPMENT.md) |
| 性能测量、自动报告、旧实验 | [性能](PERFORMANCE.md) |
| 发版前验收与回滚 | [发布验收](RELEASE_ACCEPTANCE.md) |
| 项目范围与优先级 | [产品目标](PRODUCT_GOAL.md) |

`releases/` 保留各版本发布说明。
`history/` 和 [cuda-oxide 实验](CUDA_OXIDE_EXPERIMENT.md)保存历史测量及当时的环境。

## 维护约定

- 改 CLI、HTTP 字段、默认值或模型格式时，同一个 PR 更新对应文档和示例。
- 未发布变化明确标注；首次体验的固定镜像只在正式发布并验收后更新。
- 每次发布前在本地 GPU 上运行[基准报告](PERFORMANCE.md)，保存提交、构建、硬件、
  模型/音色配置、输入、预热、重复次数和原始数据，以此建立版本基线。
- 日常 CI 检查本地 Markdown 链接、JSON 示例、Bash 语法和 CLI 选项。
  语义正确性、外链状态和真实模型效果仍需人工核对及实际运行。
