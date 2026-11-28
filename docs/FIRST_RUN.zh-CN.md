# 用 Docker 运行 GPT-SoVITS-RS，生成第一段语音

[中文主页](../README.zh-CN.md) · [英文上手指南](../README.md#docker-service) · [已有服务，直接接入](AGENT_INTEGRATION.md)

下面用 [v1.3.5](https://github.com/ricardomlee/gpt-sovits-rs/releases/tag/v1.3.5) 的 CPU 镜像启动本机语音服务，生成[这段英文示例音频](../examples/samples/local-assistant-en.wav)。

使用 Windows、Linux 或 macOS 原生程序时，可按[模型准备手册](MODELS.md#准备公开-v2-演示模型)生成同一套模型与音色。

## 准备

- Linux x86_64、Docker Engine、Compose v2、Git、Bash、curl，以及 sha256sum 或 shasum。
- 至少 6 GiB 可用磁盘、4 GiB 可用内存。模型源文件下载约 1.1 GB，另需下载镜像。
- 能访问 GitHub、GHCR、Hugging Face 或魔搭。[准备脚本](../examples/first-run/prepare.sh)根据同一份 [models.toml](../examples/first-run/models.toml) 准备模型，再写入 Compose 配置。

## 跑出第一段音频

先准备模型，启动服务。`~/gpt-sovits-demo` 用新目录。此示例监听本机 **9881**；[正式部署模板](DEPLOYMENT.md#选择部署方式)默认是 **9880**。后面的命令都沿用当前终端的 `TTS_URL`。

```bash
git clone https://github.com/ricardomlee/gpt-sovits-rs.git
cd gpt-sovits-rs
bash examples/first-run/prepare.sh "$HOME/gpt-sovits-demo"
cd "$HOME/gpt-sovits-demo"
docker compose up -d --wait --wait-timeout 900
export TTS_URL="http://127.0.0.1:${DEMO_PORT:-9881}"
```

需要优先使用魔搭时，在准备命令前加 `DEMO_MODEL_SOURCE=modelscope`。下载文件仍使用同一组 SHA-256。
支持 `prepare-demo` 的镜像会用包内的 Rust 转换器生成 `demo-data/`，脚本会把该路径写进
Compose 配置。使用 v1.3.5 镜像时，模型和音色目录直接位于演示目录下。

在同一终端把 `request.json` 发给服务，生成 `first.wav`：

```bash
curl --fail-with-body --max-time 300 "$TTS_URL/tts" \
  -H 'Content-Type: application/json' --data-binary @request.json \
  --output first.wav
```

用播放器打开 `first.wav`，听到的应当是：

> The rain has stopped. We can walk home by the river.

想让它读自己写的英文，把 `request.json` 里的 `text` 换掉。例如：

```json
{"voice":"demo","text":"The backup is complete."}
```

再执行上面的 `curl`，把输出改为 `--output reply.wav`，就能保留两段音频。

## 换成自己的声音

在 Compose 挂载的 `voices/` 下[添加音色](DEPLOYMENT.md#添加音色)，放入有权使用的录音和逐字稿。v1.3.5 镜像的默认路径是 `~/gpt-sovits-demo/voices/`。创建示例音色 `my-voice` 后，可以用同一个服务地址[让它说一句中文](DEPLOYMENT.md#验证服务)。

用完后，在 `~/gpt-sovits-demo` 运行 `docker compose down`。模型、音色和生成的音频都会保留。

## 出错时

| 现象 | 处理 |
|---|---|
| Docker permission denied / daemon unavailable | 先让当前用户能正常运行 `docker info`，脚本会在下载前检查 |
| 下载失败 | 从仓库重新运行同一条 prepare 命令，已校验文件会复用，未完成的文件会重新下载 |
| Checksum mismatch | 不要跳过校验，移除报错指向的缓存文件后重试；仍失败请反馈 |
| Refusing nonempty directory | 选择一个新的 demo 目录，不要拿已有部署目录试用 |
| Preparation already running | 确认没有另一份脚本运行；若之前被强制终止，移除报错中的空 `.prepare-lock` 目录后重试 |
| 9881 被占用 | 启动前 `export DEMO_PORT=9882`，上面的 `TTS_URL` 会使用同一端口 |
| 启动超时 / 容器退出 | 在 demo 目录执行 `docker compose logs --tail 100`、`docker compose ps -a`；检查内存和 Docker 分配的资源 |
| curl 返回非 2xx | 先查日志；`--fail-with-body` 会报错，此时输出文件可能是 JSON 错误，不能当 WAV 播放 |
| 声音不理想 / 吞字 | 先试短句，核对参考音频和转写；保留失败请求与音频以便排查 |

同一时间只运行一个 `gpt-sovits-demo` Compose 项目。重跑 prepare 会重新转换模型，并恢复 `request.json` 和 `voices/demo/` 中的示例文件。自己的音色用单独目录；
长期使用见[部署](DEPLOYMENT.md)，微调权重和 v2Pro 的参考 SV 见[模型转换](MODELS.md)。

[首次试用反馈](https://github.com/ricardomlee/gpt-sovits-rs/issues/new?template=first-run.yml)
只需系统、版本、走到哪一步和脱敏错误。成功跑通也欢迎记录；不要上传私人音频、模型或密钥。
