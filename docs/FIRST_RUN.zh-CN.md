# 第一次运行

这条路径用于先试出第一段语音，不需要已有的模型、参考音频或 Python 环境。
它使用官方通用 v2 权重和公开的英文参考音频，不代表你的微调音色效果。
网页上的 Sun 试听仅提供生成结果；首次体验不会下载 Sun 模型、SV 文件或参考录音。
已有模型请直接看[模型转换](MODELS.md)和[部署](DEPLOYMENT.md)。

## 准备

- Linux x86_64、Docker Engine、Compose v2、Bash、curl，以及 sha256sum 或 shasum。
- 至少 6 GiB 可用磁盘、4 GiB 可用内存。模型源文件下载约 1.1 GB，另需下载镜像。
- 能访问 GitHub、GHCR、Hugging Face。运行前阅读[下载清单](../examples/first-run/downloads.txt)
  和[准备脚本](../examples/first-run/prepare.sh)。运行后会从上游下载，不会把模型放入本仓库。
- macOS 可用[原生二进制](DEPLOYMENT.md#binary)；ARM Linux 目前需要[源码构建](DEVELOPMENT.md)。
  这份 Docker 首次体验不把 amd64 模拟运行当成受支持的 ARM 路径。

## 跑出第一段音频

```bash
git clone https://github.com/ricardomlee/gpt-sovits-rs.git
cd gpt-sovits-rs
bash examples/first-run/prepare.sh "$HOME/gpt-sovits-demo"
cd "$HOME/gpt-sovits-demo"
docker compose up -d --wait --wait-timeout 900
curl --fail-with-body --max-time 300 http://127.0.0.1:9881/tts \
  -H 'Content-Type: application/json' --data-binary @request.json \
  --output first.wav
```

用本机播放器打开 `first.wav`。文本为：

> Your local voice service is ready. All speech is generated on your own computer.

准备脚本完成四个模型的校验与转换、tokenizer 复制、音色配置和 doctor 检查。
转换与推理都在固定版本 `1.2.0` 容器中运行，不需要安装宿主机转换器。
服务会预加载 demo 音色再就绪；首次启动慢不等于每次请求都要重载模型。
只有下载阶段需要外网，推理不调用云端服务。

使用独立目录、Compose 项目 `gpt-sovits-demo` 和仅本机开放的 `9881` 端口，
不会替换已有的 `9880` 服务。不要同时启动两份同名 demo 项目。
在 demo 目录运行 `docker compose down` 即可停止；模型和生成结果保留。

## 出错时

| 现象 | 处理 |
|---|---|
| Docker permission denied / daemon unavailable | 先让当前用户能正常运行 `docker info`，脚本会在下载前检查 |
| 下载失败 | 从仓库重新运行同一条 prepare 命令，已校验文件会复用，未完成的文件会重新下载 |
| Checksum mismatch | 不要跳过校验，移除报错指向的缓存文件后重试；仍失败请反馈 |
| Refusing nonempty directory | 选择一个新的 demo 目录，不要拿已有部署目录试用 |
| Preparation already running | 确认没有另一份脚本运行；若之前被强制终止，移除报错中的空 `.prepare-lock` 目录后重试 |
| 9881 被占用 | 启动前 `export DEMO_PORT=9882`，同时把 curl URL 改成 9882 |
| 启动超时 / 容器退出 | 在 demo 目录执行 `docker compose logs --tail 100`、`docker compose ps -a`；检查内存和 Docker 分配的资源 |
| curl 返回非 2xx | 先查日志；`--fail-with-body` 会报错，此时输出文件可能是 JSON 错误，不能当 WAV 播放 |
| 声音不理想 / 吞字 | 先试短句；参考音频、转写和模型会影响效果，分句并不能保证不漏字 |

prepare 可在专用 demo 目录重试，但会重新转换模型并恢复 demo 配置。
自定义音色请迁移到[正式部署流程](DEPLOYMENT.md)，不要修改 demo 后再次 prepare。

## 换成自己的声音

通用模型需要你有权使用的 3-10 秒参考录音及准确转写；微调模型还需要替换兼容的
GPT/SoVITS 权重。v2Pro 请同时准备训练预处理产生的 SV embedding，不能只换权重而漏掉它。
详情见[模型文档](MODELS.md)。本流程不会下载任何维护者的私人模型或角色声音。

## 接入和反馈

应用只需发送 `voice` 和 `text`，看[API](API.md)和[Agent 接入](AGENT_INTEGRATION.md)。
长回复建议按完整短句顺序提交、顺序播放；并行请求不会让单个推理设备自动并行。

[首次试用反馈](https://github.com/ricardomlee/gpt-sovits-rs/issues/new?template=first-run.yml)
只需系统、版本、走到哪一步和脱敏错误。成功跑通也欢迎记录；不要上传私人音频、模型或密钥。
