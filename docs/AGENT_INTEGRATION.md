# Agent 接入策略

这个项目不应该先发明一套要求别人兼容的新 TTS 协议。

更稳妥的路线是：

1. 保留自己的原生接口：`/tts`、`/tts/stream`、`/tts/batch`。
2. 为常见生态提供 adapter：先做 OpenAI-compatible `/v1/audio/speech`。
3. 对具体 agent 写配置示例，而不是要求它们理解 GPT-SoVITS 内部参数。

## 为什么不先定义新协议

agent 框架通常已经有自己的 provider 层、persona、voice id、输出格式和 fallback 逻辑。让它们为了一个 TTS 服务新增协议，集成成本太高。

我们的原生协议可以继续存在，用来表达 GPT-SoVITS 特有能力；但对外集成时，优先贴近已有 provider 形状。

## OpenClaw

调研到的 OpenClaw TTS 形态：

- TTS 配置在 `messages.tts` 下。
- 它有多 provider 和 fallback 机制。
- Talk mode 最终会走 `talk.speak`。
- 文本回复可以带 voice/model/speed 等语音 directive。
- 它支持 OpenAI TTS provider，也支持 Local CLI provider。
- 它按不同渠道偏好不同输出格式：普通附件常用 MP3，语音消息常用 Opus，Talk/telephony 可走 PCM。

因此当前推荐两条接入路径：

### 路径 A：OpenAI-Compatible Adapter

把 OpenClaw 的 OpenAI TTS provider base URL 指到：

```text
http://localhost:9880/v1
```

请求会落到：

```text
POST /v1/audio/speech
```

本服务支持：

- `input`：要合成的文本。
- `voice`：映射到 `voices/<name>/voice.json`。
- `response_format`：当前支持 `wav` 和 `pcm`。
- `speed`：映射到推理选项。
- `languageCode` / `lang` / `language`：可选语言覆盖。

如果 OpenClaw 默认请求 `mp3` 或 `opus`，需要在配置里覆盖成 `wav` 或 `pcm`。本服务暂时不伪装支持 MP3/Opus，避免返回格式和声明不一致。

### 路径 B：Local CLI Provider

如果 OpenAI provider 不方便覆盖输出格式，可以先用 OpenClaw 的 Local CLI provider 调本项目 CLI，把输出格式设为 `wav`。

这条路径兼容性强，但每次调用都走进程启动和模型加载，不适合低延迟长期使用。后续如果需要，可以补一个专门的轻量 CLI client，只负责请求本地 HTTP 服务并写出 WAV。

## Hermes

目前还没有确认 Hermes Agent 的官方 TTS provider 接口细节。先不要为它写定制代码。

下一步应该确认：

- 是否支持 OpenAI-compatible speech endpoint。
- 是否支持 Local CLI TTS provider。
- 是否有 MCP/tool 方式返回音频文件。
- 默认请求的输出格式是 MP3、WAV、PCM 还是 Opus。

在确认前，只保留通用 adapter，不写 Hermes 专用协议。

## 我们自己的协议

原生协议继续以本项目需求为中心：

- `/voices`：列出本地音色。
- `/tts`：单句或短文本合成。
- `/tts/stream`：逐句流式输出。
- `/tts/batch`：批量合成和测试。

它们可以比 OpenAI-compatible adapter 暴露更多 GPT-SoVITS 能力，但不要求外部 agent 优先兼容这套协议。
