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

如果模型路径或文件名不一样，改 `.env`。想试 voice profile，可以把 `voices.example/mao` 复制到 `voices/mao`，再把对应参考音频放到 `voices/mao/ref.wav`。

## CPU

```bash
docker compose -f compose.cpu.yml up -d --build
curl http://localhost:9880/health
curl http://localhost:9880/voices
```

## CUDA

先在 `.env` 里设置 `CUDA_COMPUTE_CAP`，再启动：

```bash
docker compose -f compose.cuda.yml up -d --build
curl http://localhost:9880/health
curl http://localhost:9880/voices
```

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
