# Agent 接入策略

先不要发明一套新 TTS 协议让别人来兼容。

当前做法：

1. 原生接口继续保留：`/tts`、`/tts/stream`、`/tts/batch`。
2. 常见接口做 adapter：先提供 OpenAI-compatible `/v1/audio/speech`。
3. 对具体 agent 写配置示例，不让它们理解 GPT-SoVITS 的内部参数。

## 推荐的实时接入方式

agent 不应该把一整段很长的回复一次性塞给 TTS。更稳的方式是：

```text
LLM token stream
  -> 客户端句子边界检测
  -> 朗读 chunk 队列
  -> 逐句调用 POST /tts
  -> 按顺序播放返回的 WAV
```

这个模式比一次性合成长段更适合 GPT-SoVITS：

- 每个 TTS 请求只处理一句或一个短语，降低提前 EOS、吞字和哼唱的概率。
- 首句能更快播出来，不用等 agent 完整回复结束。
- 某一句失败时，只需要重试这一句，不影响已经播放或后续排队的句子。
- 单个本地服务实例通常只有一个 GPU 推理管线，多个请求会排队；盲目并行不会让单实例更快。

建议 chunk 尺寸：

- 日常中文：每句 10-25 个汉字左右。
- 古文、公告、长解释：优先按 `。！？；` 切；逗号后如果已经超过 15-20 个汉字，也可以切。
- 不要把多个完整句子合成一个请求。
- 不要把只有一两个字的碎片立即送 TTS，除非是很明确的语气词。

`/tts/stream` 的定位是：客户端已经有完整文本时，让服务端按句子切分并流式返回一个 WAV 响应。它不是 LLM token streaming 输入接口。agent 正在生成回复时，推荐在 agent/client 侧切句，然后逐句调用 `/tts`。

一个简单的 agent 输出协议可以是 JSON Lines：

```json
{"type":"speech_chunk","index":1,"text":"先帝创业未半而中道崩殂。"}
{"type":"speech_chunk","index":2,"text":"今天下三分，益州疲弊。"}
{"type":"speech_chunk","index":3,"text":"此诚危急存亡之秋也。"}
```

播放端必须按 `index` 顺序播放。单实例下不要为了长文本并行请求同一个服务；如果以后有多个 TTS worker 或多张 GPU，再由调度层按 chunk 分发，并保持播放顺序。

### Agent system prompt 建议

可以给 agent 加一段约束，让它的回复更适合实时朗读：

```text
When producing speech for TTS, emit short speech chunks instead of one long paragraph.
Each Chinese chunk should usually contain 10-25 Chinese characters.
Prefer splitting at 。！？； and split at ， when the clause is already long.
For Chinese polyphonic characters that may be misread, annotate only the ambiguous character as 字[pinyin+tone], for example 好[hao4]学 or 行[hang2]长.
Do not annotate every character.
```

如果 agent 框架不支持结构化 speech chunk，也可以在客户端读取普通文本 token 流，遇到句子边界后自行切分。

## 为什么不先定义新协议

agent 框架通常已经有 provider、persona、voice id、输出格式和 fallback。让它们为了一个 TTS 服务新增协议，成本太高。

原生协议用来暴露 GPT-SoVITS 特有能力。对外接入时，优先贴近别人已经支持的接口。

## OpenClaw

OpenClaw 的 TTS 大致是这样：

- TTS 配置在 `messages.tts` 下。
- 它有多 provider 和 fallback 机制。
- Talk mode 最终会走 `talk.speak`。
- 文本回复可以带 voice/model/speed 等语音指令。
- 它支持 OpenAI TTS provider，也支持 Local CLI provider。
- 它按不同渠道偏好不同输出格式：普通附件常用 MP3，语音消息常用 Opus，Talk/telephony 可走 PCM。

所以先支持两种接法。

### 路径 A：OpenAI-Compatible Adapter

把 OpenClaw 的 OpenAI TTS provider base URL 指到：

```text
http://localhost:9880/v1
```

请求会落到：

```text
POST /v1/audio/speech
```

本服务接收：

- `input`：要合成的文本。
- `voice`：映射到 `voices/<name>/voice.json`。
- `response_format`：当前支持 `wav` 和 `pcm`。
- `speed`：映射到推理选项。
- `languageCode` / `lang` / `language`：可选语言覆盖。

如果 OpenClaw 默认请求 `mp3` 或 `opus`，需要在配置里改成 `wav` 或 `pcm`。本服务暂时不假装支持 MP3/Opus，避免返回内容和声明不一致。

### 文本读音标注

agent 可以在回复文本里给中文多音字加拼音标注，服务会在文本前端解析并执行：

```text
这个人很好[hao4]学，银行的行[hang2]长正在重[zhong4]新安排会议。
```

规则：

- 格式是 `字[pinyin+声调数字]`，例如 `好[hao4]`、`行[hang2]`、`的[de5]`。
- 标注只影响前一个中文字符，最终送进 BERT/G2P 的可见文本会移除方括号内容。
- 声调必须是 `1` 到 `5`，`5` 表示轻声。
- 未标注的字继续走默认分词、拼音和变调逻辑。

推荐在 agent system prompt 里只要求它标注容易误读的多音字，不要给每个字都标。

### 长文本问题排查

如果用户听到吞句或某个 chunk 变成哼唱，先不要自动改采样参数。打开 chunk 诊断日志定位是哪一句出问题：

```bash
GPT_SOVITS_LOG_CHUNK_TEXT=1 RUST_LOG=info ./gpt-sovits --http --port 9880
```

日志会把每个 `tts_chunk` 的 index、字数、生成 semantic token 数、音频时长和 chars/sec 串起来。通常有问题的 chunk 会表现为 token 数或音频时长相对文本明显偏短。定位后优先调整切句，而不是全局调 `temperature`、`top_p`。

### 路径 B：Local CLI Provider

如果 OpenAI provider 不方便改输出格式，可以先用 OpenClaw 的 Local CLI provider 调本项目 CLI，把输出格式设为 `wav`。

这条路兼容性强，但每次调用都要启动进程和加载模型，不适合低延迟长期使用。后续如果需要，可以补一个轻量 CLI client，只负责请求本地 HTTP 服务并写出 WAV。

## Hermes

目前还没有确认 Hermes Agent 的官方 TTS provider 接口。先不要为它写定制代码。

下一步应该确认：

- 是否支持 OpenAI-compatible speech endpoint。
- 是否支持 Local CLI TTS provider。
- 是否有 MCP/tool 方式返回音频文件。
- 默认请求的输出格式是 MP3、WAV、PCM 还是 Opus。

确认前，只保留通用 adapter，不写 Hermes 专用协议。

## 我们自己的协议

原生协议按本项目需求来：

- `/voices`：列出本地音色。
- `/tts`：单句或短文本合成。
- `/tts/stream`：逐句流式输出。
- `/tts/batch`：批量合成和测试。

这些接口可以暴露更多 GPT-SoVITS 能力，但不要求外部 agent 优先兼容它们。
