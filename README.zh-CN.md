# GPT-SoVITS-RS

[English](README.md) | **简体中文**

<p align="center">
  <img src="assets/gpt-sovits-rs-logo.svg" alt="GPT-SoVITS-RS" width="640">
</p>

给它一段录音和对应的文字，就能用这个音色朗读新句子。模型转换和语音合成都在本机运行，使用 Rust 和 Candle，无需安装 Python。

## 从哪里开始

| 想做什么 | 从这里开始 |
| --- | --- |
| 打开浏览器听效果 | [试听英文示例和 Sun 的中文录音](https://tts.ricli.tech/#samples)，不用安装或下载模型。想让它读自己写的中文，可带上[准备好的模型和参考特征](experiments/wasm-sovits/README.md#prepare-a-voice)进入 [Wasm 实验室](https://tts.ricli.tech/lab/)。 |
| 在电脑上生成一个 WAV | [原生命令行](#原生命令行)支持 Linux x86_64 和 macOS。准备好转换后的模型和参考音色即可运行，不用启动 Docker 或 HTTP 服务。 |
| 给应用接上语音 API | [Docker 服务](#docker-服务)提供 Linux x86_64 的 CPU 示例，自动准备公开模型和英文音色。其他机器先[按硬件选版本](docs/DEPLOYMENT.md#选择部署方式)。 |

已经有运行中的服务？[直接发一句话试试](#接入应用)。

## 试听

| 音色 | 朗读原文 | 音频 |
| --- | --- | --- |
| 公开英文示例（v2） | The rain has stopped. We can walk home by the river. | [播放 WAV](examples/samples/local-assistant-en.wav) |
| Sun（私人 v2Pro 音色） | 面包刚出炉，坐下来一起吃早餐吧。 | [播放 WAV](examples/samples/sun-greeting.wav) |

[用同一段中文对比两种音色](examples/samples/README.md)。Docker 示例装的是英文音色，Sun 可以在网页上试听。

## 原生命令行

从 [v1.3.3 发布页](https://github.com/ricardomlee/gpt-sovits-rs/releases/tag/v1.3.3) 下载 CPU 压缩包：Linux x86_64 为 14.9 MiB，苹果芯片 Mac 为 9.0 MiB，Intel Mac 为 9.8 MiB。包内有推理程序、转换器和配套库，模型单独准备。保留解压后的目录结构，包括 macOS 上与程序放在一起的 dylib。

```bash
pkg=gpt-sovits-1.3.3-linux-x86_64
curl -fLO "https://github.com/ricardomlee/gpt-sovits-rs/releases/download/v1.3.3/$pkg.tar.gz"
tar -xzf "$pkg.tar.gz"
```

苹果芯片把 `pkg` 改为 `gpt-sovits-1.3.3-macos-aarch64`，Intel Mac 改为 `gpt-sovits-1.3.3-macos-x86_64`。发布页的 `SHA256SUMS` 可用来校验下载。

把 `data` 设为含有 `models/` 和 `voices/` 的目录，音色名换成自己的。下面直接复用公开示例的文件。第一次使用又不想装 Docker，可以先[转换权重](docs/MODELS.md#准备官方-v2-模型)，再[添加音色](docs/DEPLOYMENT.md#添加音色)。

```bash
data="$HOME/gpt-sovits-demo"
"./$pkg/gpt-sovits" --device cpu --models-dir "$data/models" --voices-dir "$data/voices" \
  --voice demo --seed 42 --text "The rain has stopped. We can walk home by the river." --output first.wav
```

模型就绪后，v1.3.3 在 i5-12490F 上用 10.3 秒加载模型并写出了 4.12 秒的音频（六个 CPU 线程，单次实测）。首次下载和转换的时间另计。

## Docker 服务

在 Linux x86_64 上准备 Docker Engine、Compose v2、Git、Bash、curl 和 SHA-256 校验工具，留出 6 GiB 磁盘、4 GiB 可用内存。[准备脚本](examples/first-run/prepare.sh)会下载并校验约 1.1 GB 的[模型文件](examples/first-run/downloads.txt)，另外还要下载 CPU 镜像。首次运行请使用新的或空的目录。

```bash
git clone https://github.com/ricardomlee/gpt-sovits-rs.git
cd gpt-sovits-rs
bash examples/first-run/prepare.sh "$HOME/gpt-sovits-demo"
cd "$HOME/gpt-sovits-demo"
docker compose up -d --wait --wait-timeout 900
export TTS_URL="http://127.0.0.1:${DEMO_PORT:-9881}"
curl --fail-with-body --max-time 300 "$TTS_URL/tts" \
  -H 'Content-Type: application/json' --data-binary @request.json --output first.wav
```

打开 `first.wav` 试听。在 `request.json` 里把 `text` 换成另一句英文，再执行一次 curl 就能听到新句子。示例监听 **9881**；正式部署默认用 **9880**。在示例目录运行 `docker compose down` 可停止服务，模型会保留。

## 接入应用

沿用示例的 `TTS_URL`，或改成应用能访问的服务地址。先从 `/voices` 选一个音色。用英文 `demo` 试一句：

```bash
TTS_URL=${TTS_URL:-http://127.0.0.1:9880}
curl -fsS "$TTS_URL/voices"
curl --fail-with-body --max-time 300 "$TTS_URL/tts" -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"The backup is complete."}' --output reply.wav
```

`POST /tts` 返回 WAV，`/tts/stream` 可逐段接收音频。[Hermes](docs/AGENT_INTEGRATION.md#hermes) 等客户端也可以调用[兼容 OpenAI 的语音接口](docs/API.md#http-server)。

## 支持范围

| | 当前支持 |
| --- | --- |
| 音色 | GPT-SoVITS v2、v2Pro。准备 3 至 10 秒的清晰参考录音和逐字稿。v2Pro 完整音色还需要与录音匹配的 SV；转换器读取已有 embedding，不从录音提取。 |
| 文本 | 中文、英文和中英混读。长文可能漏字；建议按句生成，按顺序播放，重要内容请试听核对。 |
| 推理 | 使用 F32 和内置 SoVITS 解码器。`--half` 是兼容参数；主推理路径未接入 BigVGAN，不支持 v3/v4。 |
| 硬件 | Linux、macOS 的发行包使用 CPU，Mac 未启用 Metal。Linux/WSL2 的 CUDA 镜像提供 `sm75`、`sm86`、`sm89`、`sm90`，暂无 RTX 50 系列的 `sm120` 预编译镜像。 |
| HTTP | 无内置鉴权。绑定本机地址，或在服务前配置带鉴权的反向代理。 |

## 文档

- 上手：[Docker 首次运行](docs/FIRST_RUN.zh-CN.md) [中文] · [模型转换](docs/MODELS.md) [中文] · [部署](docs/DEPLOYMENT.md) [中文]。
- 接入：[API](docs/API.md) [EN] · 助手接入：[中文](docs/AGENT_INTEGRATION.md) / [English](docs/AGENT_INTEGRATION.en.md)。
- 深入：[推理流程与源码导读](IMPLEMENTATION.md) [中文] · [编译与测试](docs/DEVELOPMENT.md) [EN] · [性能报告](docs/PERFORMANCE.md) [中文] · [文档总索引](docs/README.md) [中文]。

下载中断可以重跑准备脚本，已校验的文件会复用。服务启动失败时，在示例目录执行 `docker compose logs --tail 100`，或查[首次运行排错表](docs/FIRST_RUN.zh-CN.md#出错时)。[反馈问题](https://github.com/ricardomlee/gpt-sovits-rs/issues/new?template=first-run.yml)时请勿上传私人模型、录音或密钥。

## 许可与致谢

源码使用 [MIT 许可证](LICENSE)，模型保留各自的许可证。请使用有权使用的录音；公开音频见[样本来源](examples/samples/README.md)。项目基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 和 [Candle](https://github.com/huggingface/candle)。训练音色请使用上游 GPT-SoVITS。
