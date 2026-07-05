# 部署

这个服务按本地 HTTP TTS 来部署。建议挂载三个目录：

```text
models/   模型权重
voices/   音色配置和参考音频
outputs/  输出文件和测试报告，可选
```

先复制 `.env.example`：

```bash
cp .env.example .env
```

模型按 README 中的目录和文件名放好即可。文件名不同时，在 `.env` 里改对应路径。

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

v2Pro 音色还应把 SV embedding 放进同一个 voice 目录，并在 `voice.json` 里引用。compose
只挂载整个 `VOICES_DIR`，不需要给每个音色单独加环境变量：

```bash
gpt-sovits-convert sv \
  /path/to/logs/demo_v2pro/7-sv_cn/ref.wav.pt \
  voices/demo/ref_sv.safetensors
```

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频里逐字对应的文字",
  "sv_embedding": "ref_sv.safetensors",
  "language": "zh",
  "mode": "auto",
  "split_sentences": true
}
```

官方 v2 模型或自训练模型使用 Rust converter 准备。项目不分发模型权重，用户需要自行
下载官方模型、复用已有 GPT-SoVITS 安装目录，或使用自己训练出的 checkpoint：

```bash
mkdir -p models/bert models/hubert
gpt-sovits-convert gpt /path/to/s1bert25hz.ckpt models/gpt-model.safetensors
gpt-sovits-convert sovits /path/to/s2G2333k.pth models/sovits-model.safetensors
gpt-sovits-convert bert /path/to/chinese-roberta-wwm-ext-large/pytorch_model.bin models/bert/bert.safetensors
cp /path/to/chinese-roberta-wwm-ext-large/tokenizer.json models/bert/tokenizer.json
gpt-sovits-convert hubert /path/to/chinese-hubert-base/pytorch_model.bin models/hubert/hubert.safetensors
```

## CPU

CPU 镜像发布 `linux/amd64` 版本并启用静态链接的 MKL，适合没有 NVIDIA GPU 的
服务器或本机部署。

```bash
docker compose -f compose.cpu.yml pull
docker compose -f compose.cpu.yml up -d
docker compose -f compose.cpu.yml ps
curl http://localhost:9880/health
curl http://localhost:9880/voices
```

发布镜像：

```text
ghcr.io/ricardomlee/gpt-sovits-rs:latest
ghcr.io/ricardomlee/gpt-sovits-rs:1.0
ghcr.io/ricardomlee/gpt-sovits-rs:1.1.0
```

## CUDA

宿主机需要 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。按 GPU 架构在 `.env`
里设置 `CUDA_IMAGE_TAG`：

```text
RTX 20: latest-cuda-sm75
RTX 30: latest-cuda-sm86
RTX 40: latest-cuda-sm89
H100:   latest-cuda-sm90
```

```bash
docker compose -f compose.cuda.yml pull
docker compose -f compose.cuda.yml up -d
docker compose -f compose.cuda.yml ps
curl http://localhost:9880/health
curl http://localhost:9880/voices
```

版本固定标签使用 `1.1.0-cuda-sm89` 这种格式。其他 compute capability 可以本地构建：

```bash
docker build -f Dockerfile.cuda \
  --build-arg CUDA_COMPUTE_CAP=80 \
  -t gpt-sovits-rs:cuda-sm80 .
```

## Binary

Release 中的 Linux x86_64 包已经携带 `libsoxr`，并包含 `gpt-sovits` 与
`gpt-sovits-convert` 两个可执行文件，解压后可直接运行：

```bash
tar -xzf gpt-sovits-1.1.0-linux-x86_64.tar.gz
cd gpt-sovits-1.1.0-linux-x86_64
./gpt-sovits-convert --version
./gpt-sovits --models-dir /path/to/models \
  --http --port 9880
```

macOS 包同时携带对应的 `libsoxr.0.dylib`。首次运行若被 Gatekeeper 阻止，需要在
系统设置中确认允许该二进制。

## 音色目录

每个音色放在 `voices/<name>/voice.json`。`voice.json` 里的相对路径按该目录解析。

```text
voices/
  mao/
    voice.json
    ref.wav
```

助手调用示例：

```bash
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"你好，这是 GPT-SoVITS-RS 的本地语音服务。"}' \
  --output output.wav
```

也可以直接传参考音频和参考文本，主要用于调试：

```json
{
  "text": "你好世界",
  "refer_wav_path": "/app/voices/mao/ref.wav",
  "prompt_text": "参考音频对应的文字",
  "sv_embedding": "/app/voices/mao/ref_sv.safetensors",
  "text_language": "zh"
}
```

## 排障

容器带有 `/health` healthcheck。启动后先看状态：

```bash
docker compose -f compose.cpu.yml ps
docker compose -f compose.cpu.yml logs --tail=100 gpt-sovits
```

常见问题：

- `models/... not found`：模型目录没有挂载到容器，或 `.env` 里的模型路径和实际文件名不一致。
- `voices` 为空：`VOICES_DIR` 没有挂载，或缺少 `voices/<name>/voice.json`。
- 请求返回 `reference audio does not exist`：voice profile 里的 `reference_audio` 相对路径按该 voice 目录解析。
- CPU 看起来很慢：先用短句验证；长文本建议走 `/tts/stream` 或启用分句配置。
