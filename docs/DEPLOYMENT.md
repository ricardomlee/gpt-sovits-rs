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

模型按 README 中的目录和文件名放好即可。文件名不同时，在 `.env` 里改对应路径。想试 voice profile，可以把 `voices.example/mao` 复制到 `voices/mao`，再把对应参考音频放到 `voices/mao/ref.wav`。

官方 v2 模型可以自动准备：

```bash
python3 -m venv .venv-models
. .venv-models/bin/activate
pip install -r requirements-models.txt
python prepare_models.py
```

## CPU

CPU 镜像同时发布 `linux/amd64` 和 `linux/arm64`。x86_64 版本启用静态链接的 MKL，
ARM64 版本使用通用 Candle CPU 后端。

```bash
docker compose -f compose.cpu.yml pull
docker compose -f compose.cpu.yml up -d
curl http://localhost:9880/health
curl http://localhost:9880/voices
```

发布镜像：

```text
ghcr.io/ricardomlee/gpt-sovits-rs:latest
ghcr.io/ricardomlee/gpt-sovits-rs:1.0
ghcr.io/ricardomlee/gpt-sovits-rs:1.0.0
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
curl http://localhost:9880/health
curl http://localhost:9880/voices
```

版本固定标签使用 `1.0.0-cuda-sm89` 这种格式。其他 compute capability 可以本地构建：

```bash
docker build -f Dockerfile.cuda \
  --build-arg CUDA_COMPUTE_CAP=80 \
  -t gpt-sovits-rs:cuda-sm80 .
```

## Binary

Release 中的 Linux x86_64 包已经携带 `libsoxr`，解压后可直接运行：

```bash
tar -xzf gpt-sovits-1.0.0-linux-x86_64.tar.gz
cd gpt-sovits-1.0.0-linux-x86_64
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
  -d '{"voice":"mao","text":"人民，只有人民，才是创造世界历史的动力。"}' \
  --output output.wav
```

也可以直接传参考音频和参考文本，主要用于调试：

```json
{
  "text": "你好世界",
  "refer_wav_path": "/app/voices/mao/ref.wav",
  "prompt_text": "参考音频对应的文字",
  "text_language": "zh"
}
```
