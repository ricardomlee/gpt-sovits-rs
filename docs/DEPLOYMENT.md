# 部署 GPT-SoVITS-RS 本地语音服务

已有服务时，先[查音色并试一句](AGENT_INTEGRATION.md)。本页的部署模板默认用本机 **9880**；[首次运行示例](FIRST_RUN.zh-CN.md)默认用 **9881**。

## 选择部署方式

在**运行 TTS 的主机**上执行 `uname -sm`。有 NVIDIA 显卡时，再查计算能力（CC）；这条命令也适用于 WSL2：

```bash
PATH="$PATH:/usr/lib/wsl/lib" nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

| 主机 | 部署方式 |
| --- | --- |
| Linux x86_64，使用 CPU | [Docker 镜像](#cpu)或 [Linux 二进制包](#二进制包) |
| Linux 或 WSL2 x86_64，使用 NVIDIA 显卡 | [CUDA 镜像](#cuda)以 `1.3.5-cuda-` 开头，按计算能力选后缀：7.5 → `sm75`，8.6 → `sm86`，8.9 → `sm89`，9.0 → `sm90` |
| macOS | [原生 CPU 二进制包](#二进制包)：苹果芯片选 `macos-aarch64`，Intel 芯片选 `macos-x86_64` |
| Linux ARM64，包括 ARM 架构的 NAS | [从源码构建](DEVELOPMENT.md#build-from-source) |

例如，RTX 4060 和 4090 用 `1.3.5-cuda-sm89`。RTX 5060 和 5090 需要 `sm120`，1.3.5 没有这个镜像。
遇到未列出的架构，在 x86_64 上可先用 [CPU](#cpu)，或[为目标显卡构建 CUDA 镜像](#cuda)。
[NVIDIA 显卡计算能力表](https://developer.nvidia.com/cuda/gpus) · [下载发行包](https://github.com/ricardomlee/gpt-sovits-rs/releases/latest)

Linux x86_64 没有模型时，可从[公开 CPU 示例](FIRST_RUN.zh-CN.md)开始。已有权重时先准备下面的目录，再从仓库根目录运行 Compose。

## 准备目录和音色

按[模型文档](MODELS.md)取得转换器并准备权重。已有训练产出的 `.pth` 和 `.ckpt` 时，
可直接走[自训练音色流程](MODELS.md#自训练音色模型)。部署使用以下目录：

```text
models/   模型权重
voices/   音色配置和参考音频
outputs/  输出文件和测试报告，可选
```

首次使用 Compose 时复制 `.env.example`；已有 `.env` 的部署保留原配置：

```bash
cp .env.example .env
```

`.env` 由 Compose 读取，独立二进制使用命令行参数。个人部署先在 `.env` 中设置端口和
固定的 CPU 镜像版本；CUDA 镜像标签在下文按显卡选择：

```env
GPT_SOVITS_PORT=127.0.0.1:9880
GPT_SOVITS_IMAGE_TAG=1.3.5
```

路径按配置位置解析：

| 配置 | 路径所在位置 | 示例 |
| --- | --- | --- |
| `MODELS_DIR`、`VOICES_DIR`、`OUTPUTS_DIR` | 宿主机目录，挂载进容器 | `MODELS_DIR=/home/me/tts/models` |
| `GPT_MODEL`、`SOVITS_MODEL`、`BERT_MODEL`、`HUBERT_MODEL` | 容器内模型文件 | `GPT_MODEL=/app/models/gpt-model.safetensors` |
| `voice.json` 的 `reference_audio`、`sv_embedding` | 相对于该音色目录 | `ref.wav`、`ref_sv.safetensors` |
| `voice.json` 的 `gpt_model`、`sovits_model` | 相对于模型根目录 | `demo/gpt.safetensors` |

服务启动时会加载一套默认 GPT/SoVITS。若权重放在 `models/my-voice/` 下，独立二进制
可以用 `--voice my-voice` 选择音色绑定的权重。使用 Compose 时，在 `.env` 中将
`GPT_MODEL`、`SOVITS_MODEL` 分别设为 `/app/models/my-voice/gpt.safetensors`、
`/app/models/my-voice/sovits.safetensors`。
BERT、HuBERT 和 tokenizer 按模型文档放好。

### 添加音色

在服务的 `voices/` 下新建 `my-voice/`（首次体验用 `~/gpt-sovits-demo/voices/`）。
把有权使用的 3–10 秒录音存为 `my-voice/ref.wav`，逐字稿写入同目录的 `voice.json`。
通用 v2 可复用已有权重，新音色无需重训或重启。

想先试一次，可以用这个[公开中文样本](../examples/samples/local-assistant-zh.wav)作为 `ref.wav`，搭配下面的配置：

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "雨停了，河边的路灯亮了。我们慢慢走回家吧。",
  "language": "zh",
  "prompt_language": "zh"
}
```

`language` 是待朗读文字的语言，`prompt_language` 是参考录音的语言。

v2Pro 使用[与参考录音匹配的 SV](MODELS.md#v2pro-sv)；将 `ref_sv.safetensors`
放在音色目录，并在 JSON 中加入 `"sv_embedding": "ref_sv.safetensors"`。

服务已经在运行时，可以直接[试听这个音色](#验证服务)。首次部署选择下面一种启动方式；
CPU 和 CUDA Compose 共用容器名 `gpt-sovits-rs`。
已有服务按[升级与回滚](#升级与回滚)操作。

## CPU

CPU 镜像发布 `linux/amd64` 版本并启用静态链接的 MKL，适合没有 NVIDIA GPU 的
服务器或本机部署。

```bash
docker compose -f compose.cpu.yml pull
docker compose -f compose.cpu.yml up -d --wait --wait-timeout 900
docker compose -f compose.cpu.yml ps
```

CPU 固定版本镜像为 `ghcr.io/ricardomlee/gpt-sovits-rs:1.3.5`。

## CUDA

宿主机需有可用的 NVIDIA 驱动，且 Docker 能访问显卡（Linux 上使用 NVIDIA Container Toolkit）。
按[上表](#选择部署方式)设置 `.env` 的 `CUDA_IMAGE_TAG`，例如 `1.3.5-cuda-sm89`。
拉取后用同一镜像确认容器看得到目标 GPU，再启动：

```bash
docker compose -f compose.cuda.yml pull
docker compose -f compose.cuda.yml run --rm --pull never --no-deps --entrypoint nvidia-smi gpt-sovits -L
docker compose -f compose.cuda.yml up -d --wait --wait-timeout 900
```

自行构建时，把下面的 `80` 换成目标 CC 的数字（如 `120`），并在目标卡上验证合成：

```bash
docker build -f Dockerfile.cuda \
  --build-arg CUDA_COMPUTE_CAP=80 \
  -t ghcr.io/ricardomlee/gpt-sovits-rs:local-cuda-sm80 .
CUDA_IMAGE_TAG=local-cuda-sm80 docker compose -f compose.cuda.yml \
  up -d --pull never --wait --wait-timeout 900
```

这里的 `local-cuda-sm80` 是本地标签。长期使用时将它写入 `.env` 的 `CUDA_IMAGE_TAG`，
启动保留 `--pull never`，跳过远端拉取。

<a name="binary"></a>

## 二进制包

[Release](https://github.com/ricardomlee/gpt-sovits-rs/releases/latest) 中的 Linux x86_64 包携带 `libsoxr` 和 `libmp3lame`。
干净运行环境的验收使用 Ubuntu 24.04，安装系统运行库 `libstdc++6` 和 `libgomp1`，无需 Python 或 Rust 工具链。
Opus 编码器编入程序。
Docker 镜像支持相同的编码格式。第三方库见[音频编解码许可说明](../THIRD_PARTY.md)。

| 程序 | 用途 |
| --- | --- |
| `gpt-sovits` | 本机推理或常驻 HTTP 服务 |
| `gpt-sovits-convert` | 模型和 SV 文件转换 |
| `gpt-sovits-client` | 可选的 HTTP 命令行客户端，适用于脚本和通过外部命令接入的工具 |

下载[对应平台](#选择部署方式)的压缩包和 `SHA256SUMS`。尚未转换权重时，先用包内的
`gpt-sovits-convert` [准备官方权重](MODELS.md#准备官方-v2-模型)或[转换自己的权重](MODELS.md#自训练音色模型)。
以下为 Linux 示例，用的是官方权重的默认文件名：

```bash
sha256sum --check SHA256SUMS --ignore-missing
tar -xzf gpt-sovits-1.3.5-linux-x86_64.tar.gz
cd gpt-sovits-1.3.5-linux-x86_64
./gpt-sovits-convert --version
./gpt-sovits --models-dir /path/to/models \
  --voices-dir /path/to/voices \
  --http --port 9880
```

macOS 用 `shasum -a 256 压缩包文件名` 与 `SHA256SUMS` 中的同名条目核对。解压后保留程序旁的 `.dylib` 文件，在该目录运行同一条 `./gpt-sovits` 启动命令。1.3.2 的包如果提示找不到
`libmpg123.0.dylib`，运行 `brew install mpg123`。首次运行若被 Gatekeeper 阻止，在系统设置中允许该程序。

已发布的 Linux/macOS 二进制是 CPU 构建；macOS 构建未启用 Metal，`--device mps`
不会让这些发行物自动获得 GPU 加速。NVIDIA 推理使用 CUDA 镜像，或带 `cuda` feature 的源码构建。

## 验证服务

下面沿用当前终端的 `TTS_URL`，未设置时使用正式部署的 `9880`。在新终端连接首次运行的示例，先设 `TTS_URL=http://127.0.0.1:9881`；远程服务或自定义端口请换成实际地址。

```bash
TTS_URL=${TTS_URL:-http://127.0.0.1:9880}
curl -fsS --max-time 5 "$TTS_URL/voices"
```

列表中出现 `my-voice` 后，用它读一句中文：

```bash
curl --fail-with-body --max-time 300 "$TTS_URL/tts" \
  -H 'Content-Type: application/json' \
  -d '{"voice":"my-voice","text":"你好，今天有什么安排？"}' --output my-voice.wav
```

打开当前目录的 `my-voice.wav`，确认读出了整句。要让助手使用这个音色，见 [Agent 接入](AGENT_INTEGRATION.md)。

## 监听地址与版本兼容

独立二进制默认监听 `127.0.0.1`。局域网访问时用 `--host 0.0.0.0` 或
`GPT_SOVITS_HOST=0.0.0.0` 指定监听地址；命令行优先于环境变量。IPv6 使用
`--host ::1` 或 `--host ::`，不带方括号和端口，端口单独通过 `--port` 设置。

容器内部监听 `0.0.0.0`；CPU 和 CUDA Compose 默认将端口发布到宿主机的 `127.0.0.1:9880`。
要从局域网访问，在 `.env` 中填写宿主机的局域网地址，例如 `GPT_SOVITS_PORT=192.168.1.10:9880`。
已有配置 `GPT_SOVITS_PORT=9880` 仍会发布到所有网卡。服务无内置认证，跨机器使用时请用防火墙或带认证的反向代理控制访问。

## 升级与回滚

升级时先在 `.env` 中设置确切的镜像标签，再拉取并等待服务就绪：

```bash
docker compose -f compose.cuda.yml pull
docker compose -f compose.cuda.yml up -d --wait --wait-timeout 180
curl --fail http://localhost:9880/status
```

CPU 部署改用 `compose.cpu.yml`。回滚时把 `.env` 的镜像标签改回之前的固定版本，
再运行相同命令。模型、音色和输出均为宿主机目录挂载，重建容器不会覆盖它们。

服务收到 SIGTERM 或 Ctrl+C 后停止接收新任务，排队请求返回 503，在途请求最多有
30 秒收尾时间。超过期限时服务以非零状态退出，未完成的响应中断。Compose 的
`stop_grace_period` 为 45 秒，为服务收尾预留余量。停止时使用 `docker compose stop`
或 `docker compose down`；`/status` 在收尾期间显示 `stopping`。

## 音色目录

每个音色放在 `voices/<name>/voice.json`。参考音频和 SV embedding 的相对路径按该 voice
目录解析。

```text
voices/
  demo/
    voice.json
    ref.wav
```

以下 v2Pro 音色在 `voice.json` 中绑定自己的 GPT/SoVITS 和 SV；v2 音色省略 `sv_embedding`。
模型字段的相对路径按
`MODELS_DIR`（容器内的 `/app/models`）解析。服务在第一次请求该音色时加载并缓存对应的推理实例：

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频对应的文字",
  "sv_embedding": "ref_sv.safetensors",
  "gpt_model": "demo/gpt.safetensors",
  "sovits_model": "demo/sovits.safetensors",
  "language": "zh",
  "split_sentences": true
}
```

这意味着一个容器可以同时服务多个微调音色；请求仍然只需要传 `voice` 和 `text`。
GPT/SoVITS 推理实例使用有上限的 LRU 缓存，BERT 和 HuBERT 在音色间共享，GPU 推理保持
串行。Compose 默认缓存两套模型（包括启动时的默认模型）；显存足够时可以调整 `.env`
里的 `MAX_CACHED_PIPELINES`，显存较小时设为 `1`。

HTTP 请求直接传入的参考音频和 SV embedding 默认必须位于 `VOICES_DIR` 内。日常接入应
只传 `voice + text`；只有明确需要读取其他挂载目录进行调试时，才给启动命令增加
`--allow-external-reference-paths`。

## 排障

容器通过 `/health` 检查服务状态。启动后先看状态和日志：

```bash
docker compose -f compose.cpu.yml ps
docker compose -f compose.cpu.yml logs --tail=100 gpt-sovits
```

如果希望提前承担模型加载和参考特征计算，可以在启动后预热常用音色：

```bash
curl --fail-with-body http://127.0.0.1:9880/warmup \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo"}'
```

常用音色也可以在 `.env` 里自动预热：

```env
PRELOAD_VOICES=demo
```

多个音色用逗号分隔。预热列表最好不要超过 `MAX_CACHED_PIPELINES`，否则前面的音色会在服务开放前就被 LRU
淘汰。任一预热音色配置错误时，服务启动失败，容器也不会通过健康检查。

HTTP 默认限制每条合成文本最多 10,000 个 Unicode 字符、每个 batch 最多 64 条。可以在
`.env` 中通过 `MAX_TEXT_CHARS` 和 `MAX_BATCH_ITEMS` 调整；长期运行的个人服务建议保留
上限，避免误发的大请求长时间占用 GPU 队列。

请求等待串行推理槽位的默认超时是 120 秒，可通过 `QUEUE_TIMEOUT_SECS` 调整。超时返回
HTTP 503，只终止还未开始的排队请求，不会中断已经开始的模型加载或推理。

常见问题：

- `models/... not found`：模型目录没有挂载到容器，或 `.env` 里的模型路径和实际文件名不一致。
- `voices` 为空：`VOICES_DIR` 没有挂载，或缺少 `voices/<name>/voice.json`。
- 请求返回 `reference audio does not exist`：音色配置中的 `reference_audio` 相对路径按该音色目录解析。
- CPU 看起来很慢：先用短句验证；长文本建议走 `/tts/stream` 或启用分句配置。
