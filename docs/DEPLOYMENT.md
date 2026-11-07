# 部署

第一次试用且没有现成模型，可先走[Docker 首次运行](FIRST_RUN.zh-CN.md)，
无需安装宿主机转换器。下面的流程用于已有模型和自己的音色配置。

下面的 Compose 命令在仓库根目录执行。按顺序完成目录和音色配置，再选择 CPU、CUDA
或二进制启动，最后[验证服务](#验证服务)。

## 准备目录和音色

按[模型文档](MODELS.md)取得转换器并准备权重。部署使用以下目录：

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
GPT_SOVITS_IMAGE_TAG=1.2.2
```

路径按配置位置解析：

| 配置 | 路径所在位置 | 示例 |
| --- | --- | --- |
| `MODELS_DIR`、`VOICES_DIR`、`OUTPUTS_DIR` | 宿主机目录，挂载进容器 | `MODELS_DIR=/home/me/tts/models` |
| `GPT_MODEL`、`SOVITS_MODEL`、`BERT_MODEL`、`HUBERT_MODEL` | 容器内模型文件 | `GPT_MODEL=/app/models/gpt-model.safetensors` |
| `voice.json` 的 `reference_audio`、`sv_embedding` | 相对于该音色目录 | `ref.wav`、`ref_sv.safetensors` |
| `voice.json` 的 `gpt_model`、`sovits_model` | 相对于模型根目录 | `demo/gpt.safetensors` |

服务启动时会加载一套默认 GPT/SoVITS。若只有放在 `models/demo/` 下的微调模型，
将 `.env` 的 `GPT_MODEL`、`SOVITS_MODEL` 分别设为 `/app/models/demo/gpt.safetensors`、
`/app/models/demo/sovits.safetensors`。BERT、HuBERT 和 tokenizer 按模型文档放好。

创建一个本地 voice profile：

```bash
mkdir -p voices/demo
cp /path/to/your-3-to-10-second-reference.wav voices/demo/ref.wav
cat > voices/demo/voice.json <<'JSON'
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频里逐字对应的文字",
  "language": "zh",
  "mode": "auto",
  "split_sentences": true
}
JSON
```

v2Pro 还需[转换匹配的 SV](MODELS.md#v2pro-sv)，放在 `voices/demo/ref_sv.safetensors`，
并在 JSON 中加入 `"sv_embedding": "ref_sv.safetensors"`。Compose 挂载整个音色目录。

CPU 和 CUDA Compose 使用同一个容器名 `gpt-sovits-rs`，选择其中一种启动。
已有服务按[升级与回滚](#升级与回滚)操作。

## CPU

CPU 镜像发布 `linux/amd64` 版本并启用静态链接的 MKL，适合没有 NVIDIA GPU 的
服务器或本机部署。

```bash
docker compose -f compose.cpu.yml pull
docker compose -f compose.cpu.yml up -d --wait --wait-timeout 900
docker compose -f compose.cpu.yml ps
```

CPU 固定版本镜像为 `ghcr.io/ricardomlee/gpt-sovits-rs:1.2.2`。

## CUDA

宿主机需要 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。按 GPU 架构在 `.env`
里设置 `CUDA_IMAGE_TAG`：

```text
RTX 20: 1.2.2-cuda-sm75
RTX 30: 1.2.2-cuda-sm86
RTX 40: 1.2.2-cuda-sm89
H100:   1.2.2-cuda-sm90
```

```bash
docker compose -f compose.cuda.yml pull
docker compose -f compose.cuda.yml up -d --wait --wait-timeout 900
docker compose -f compose.cuda.yml ps
```

其他 compute capability 可以本地构建：

```bash
docker build -f Dockerfile.cuda \
  --build-arg CUDA_COMPUTE_CAP=80 \
  -t ghcr.io/ricardomlee/gpt-sovits-rs:local-cuda-sm80 .
CUDA_IMAGE_TAG=local-cuda-sm80 docker compose -f compose.cuda.yml \
  up -d --pull never --wait --wait-timeout 900
```

这里的 `local-cuda-sm80` 是本地标签。长期使用时将它写入 `.env` 的 `CUDA_IMAGE_TAG`，
启动保留 `--pull never`，跳过远端拉取。

## Binary

[Release](https://github.com/ricardomlee/gpt-sovits-rs/releases/latest) 中的 Linux x86_64 包携带 `libsoxr`，解压后可直接运行。
1.3.0 的发布包还携带 MP3 编码所需的 `libmp3lame`，Opus 编码器编入程序。
Docker 镜像包含相同的编码能力。第三方库说明见 [Audio Codec Notices](../THIRD_PARTY.md)。

| 程序 | 用途 |
| --- | --- |
| `gpt-sovits` | 本机推理或常驻 HTTP 服务 |
| `gpt-sovits-convert` | 模型和 SV 文件转换 |
| `gpt-sovits-client`（1.2.2 起） | 调用已有 HTTP 服务，适用于脚本和 Agent |

以下使用 1.2.2 包演示服务启动；客户端用法见 [HTTP 客户端](../client/README.md)。

```bash
tar -xzf gpt-sovits-1.2.2-linux-x86_64.tar.gz
cd gpt-sovits-1.2.2-linux-x86_64
./gpt-sovits-convert --version
./gpt-sovits --models-dir /path/to/models \
  --voices-dir /path/to/voices \
  --http --port 9880
```

macOS 包同时携带对应的 `libsoxr.0.dylib`；1.3.0 还包含 `libmp3lame.0.dylib`。首次运行若被 Gatekeeper 阻止，需要在
系统设置中确认允许该二进制。

已发布的 Linux/macOS 二进制是 CPU 构建；macOS 构建未启用 Metal，`--device mps`
不会让这些发行物自动获得 GPU 加速。NVIDIA 推理使用 CUDA 镜像，或带 `cuda` feature 的源码构建。

## 验证服务

在另一个终端执行。`/health` 应返回 `OK`，`/voices` 应包含刚创建的 `demo`：

```bash
curl --fail http://127.0.0.1:9880/health
curl --fail http://127.0.0.1:9880/voices
curl --fail-with-body http://127.0.0.1:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"你好，这是一次语音测试。","text_language":"zh"}' \
  --output output.wav
```

试听当前目录的 `output.wav`。HTTP 返回的音频由客户端保存；容器的 `/app/outputs` 挂载
用于容器内程序写文件。API 参数和其他输出方式见 [API](API.md#http-server)。

## 监听地址与版本兼容

从 `1.2.1` 开始，独立二进制默认监听 `127.0.0.1`。从 `1.2.0` 升级且需要局域网
访问时，必须显式配置监听地址，否则其他机器将无法连接。可以用 `--host 0.0.0.0` 或
`GPT_SOVITS_HOST=0.0.0.0` 指定监听地址；命令行优先于环境变量。IPv6 使用
`--host ::1` 或 `--host ::`，不带方括号和端口，端口单独通过 `--port` 设置。

Dockerfile 和 Compose 都设置 `GPT_SOVITS_HOST=0.0.0.0`，保证 Docker 的端口转发可用。
已发布的 `1.2.0` 镜像会忽略该变量，本身就监听所有接口，因此同一份 Compose 也能运行
固定版本 `1.2.0`。不要给 `1.2.0` 添加 `--host`，该版本尚不支持这个参数。

仓库 Compose 的默认端口映射是所有宿主机接口；上面的 `127.0.0.1:9880` 设置将其限制为本机。
跨机器访问时配置宿主机绑定地址，并通过防火墙或带认证的反向代理控制访问；服务本身无内置认证。

## 升级与回滚

升级时先在 `.env` 中设置确切的镜像标签，再拉取并等待服务就绪：

```bash
docker compose -f compose.cuda.yml pull
docker compose -f compose.cuda.yml up -d --wait --wait-timeout 180
curl --fail http://localhost:9880/status
```

CPU 部署改用 `compose.cpu.yml`。回滚时把 `.env` 的镜像标签改回之前的固定版本，
再运行相同命令。模型、音色和输出均为宿主机目录挂载，重建容器不会覆盖它们。

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
`MODELS_DIR` / 容器内 `/app/models` 解析；服务会在第一次请求该 voice 时加载并缓存对应
pipeline：

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
GPT/SoVITS pipeline 使用有上限的 LRU 缓存，BERT 和 HuBERT 在音色间共享，GPU 推理保持
串行。Compose 默认缓存两套模型（包括启动时的默认模型）；显存足够时可以调整 `.env`
里的 `MAX_CACHED_PIPELINES`，显存较小时设为 `1`。

HTTP 请求直接传入的参考音频和 SV embedding 默认必须位于 `VOICES_DIR` 内。日常接入应
只传 `voice + text`；只有明确需要读取其他挂载目录进行调试时，才给启动命令增加
`--allow-external-reference-paths`。

## 排障

容器带有 `/health` healthcheck。启动后先看状态：

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
淘汰。任一预热音色配置错误时服务会启动失败，Docker healthcheck 不会错误地显示 ready。

HTTP 默认限制每条合成文本最多 10,000 个 Unicode 字符、每个 batch 最多 64 条。可以在
`.env` 中通过 `MAX_TEXT_CHARS` 和 `MAX_BATCH_ITEMS` 调整；长期运行的个人服务建议保留
上限，避免误发的大请求长时间占用 GPU 队列。

请求等待串行 GPU 槽位的默认超时是 120 秒，可通过 `QUEUE_TIMEOUT_SECS` 调整。超时返回
HTTP 503，只终止还未开始的排队请求，不会中断已经开始的模型加载或推理。

常见问题：

- `models/... not found`：模型目录没有挂载到容器，或 `.env` 里的模型路径和实际文件名不一致。
- `voices` 为空：`VOICES_DIR` 没有挂载，或缺少 `voices/<name>/voice.json`。
- 请求返回 `reference audio does not exist`：voice profile 里的 `reference_audio` 相对路径按该 voice 目录解析。
- CPU 看起来很慢：先用短句验证；长文本建议走 `/tts/stream` 或启用分句配置。
