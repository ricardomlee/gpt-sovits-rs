# Agent 接入

助手通过 HTTP 发送文本和音色名；需要逐句播放时，由助手端组织句子、排队和播放。
接口字段见 [API](API.md)，版本差异见 [文档入口](README.md)。
能发送 HTTP 请求的助手可直接调用接口；通过外部命令生成音频的助手可使用
[`gpt-sovits-client`](../client/README.md)。[Hermes 配置](#hermes)见下文。

## 逐句生成和播放

推荐在客户端按句子组织 LLM 输出：

```text
LLM token stream -> 句子边界检测 -> 有界队列
  -> 依次 POST /tts -> 按顺序播放 WAV
```

中文通常每段 10-25 字，优先在句末或较长从句的标点处分开，避免把一两个字拆成孤立请求。
根据试听结果调整片段长度，留意漏读和句间衔接。
不要切开数字、小数、百分比、版本号或 `字[pinyin+tone]` 标注。

单个服务使用串行推理槽，多个请求会排队。
队列应有上限；用户打断时丢弃尚未播放的旧回复。客户端断开不保证已经开始的 GPU
合成立刻取消。收到 503 可按客户端策略有限重试，不能无限重发过期内容。

`/tts/stream` 接收完整文本，服务端分段后在一个 WAV 响应中逐段输出 PCM。
它不接收 LLM token 流。响应头的 200 和首字节不代表已经拿到可播放音频；
流中途失败时也无法再换成 JSON 错误。播放端需支持未知长度 WAV，不能把每个网络块
当成独立 WAV。简单客户端逐句调用普通 `/tts` 即可。

需要结构化输出时，可约定客户端自己的 JSON Lines（不是服务端输入协议）：

```jsonl
{"type":"speech_chunk","index":1,"text":"先帝创业未半而中道崩殂。"}
{"type":"speech_chunk","index":2,"text":"今天下三分，益州疲弊。"}
```

## HTTP 契约

音色目录绑定模型、参考文字/音频、SV 和默认参数。Agent 只需稳定传 `voice` 和文字，
不必暴露本机文件路径。可用 `GET /voices` 查询，`POST /warmup` 预加载。
下面使用已配置的 `demo` 音色和默认端口 `9880`；首次体验服务改用 `9881`。

```bash
curl --fail-with-body http://127.0.0.1:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"demo","text":"你好，今天有什么安排？","text_language":"auto"}' \
  --output /tmp/reply.wav
```

支持 OpenAI-compatible speech 接口的客户端可把 base URL 指向
`http://127.0.0.1:9880/v1`，调用 `POST /v1/audio/speech`：

```json
{"model":"tts-1","voice":"demo","input":"你好，今天有什么安排？","text_language":"auto","response_format":"wav"}
```

1.3.0 支持 `wav`、`pcm`、`mp3` 和 `opus`，省略格式时仍返回 WAV。
Opus 使用 Ogg 容器，输出 48 kHz 单声道；当前 WAV/PCM 为 32 kHz、单声道、16-bit little-endian。
`model` 和 `instructions` 字段当前忽略；模型随音色配置选择。
兼容字段和错误响应见 [speech API](API.md#post-v1audiospeech)。
`/tts/batch` 的 HTTP 200 也可能包含逐项失败，应检查每一行的 `error` 和 `wav_base64`；
批量结果没有 `success` 字段。

长期使用常驻 HTTP 服务。每句重新启动推理 CLI 都会重新加载模型。
服务无内置认证。独立二进制默认监听 loopback，Compose 的访问范围由宿主机端口映射决定，
配置见[部署](DEPLOYMENT.md#监听地址与版本兼容)。

## Hermes

1.3.0 服务可直接使用 Hermes 内置的 OpenAI TTS provider：

```text
Hermes text_to_speech -> POST /v1/audio/speech -> 常驻推理服务
                        文本、音色、格式          返回 MP3 或 Ogg/Opus
```

### 验证服务

按[部署文档](DEPLOYMENT.md)启动服务、准备音色。以下命令在 Hermes 所在机器执行，
将 `demo` 换成 `/voices` 返回的音色名，大小写保持一致：

```bash
curl --fail-with-body http://127.0.0.1:9880/health
curl --fail-with-body http://127.0.0.1:9880/voices
curl --fail-with-body http://127.0.0.1:9880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"demo","input":"你好，今天有什么安排？","response_format":"mp3"}' \
  --output /tmp/reply.mp3
```

Hermes 与服务分开部署时，地址应填写 Hermes 能访问的服务地址。
模型、参考音频和 SV 留在服务端，客户端只传音色名和文本。
首次加载某个音色会比后续调用慢，可用 `POST /warmup` 预热。

### 配置 provider

在 Hermes 所在环境执行，将 `demo` 换成实际音色名：

```bash
hermes config set tts.openai.base_url http://127.0.0.1:9880/v1
hermes config set tts.openai.api_key local
hermes config set tts.openai.model tts-1
hermes config set tts.openai.voice demo
hermes config set tts.use_gateway false
hermes config set tts.provider openai
```

`api_key` 的 `local` 用于满足 SDK 的必填字段；服务本身没有认证功能。
`use_gateway false` 让语音请求使用这里配置的本地地址和凭据。
模型、参考音频和 SV 均留在服务端。用户端只需 Hermes 及其 OpenAI provider 的依赖。

然后让 Hermes 调用 `text_to_speech`，传入一小段 `text`，检查工具结果和生成的音频。
TTS provider 配置在每次工具调用时读取，修改这些配置后可直接重试。

需要收到语音后自动语音回复时，设置：

```bash
hermes config set voice.auto_tts true
```

此项属于 `voice` 配置。已运行的网关还受会话语音模式影响；可在会话中用 `/voice on`
开启，或重启网关加载新的全局默认值。

### 输出格式和音色设置

Hermes 默认获取 MP3；在 Telegram 等语音平台中会请求 Opus，服务直接返回 Ogg/Opus。
这条路径无需安装 `gpt-sovits-client` 或调用 FFmpeg 转码。
`voice.auto_tts` 控制何时自动回复。

分句、语速和采样参数统一由服务端音色配置管理。需要调整长句表现时，将以下字段
合并进该音色现有的 `voice.json`，保留原来的模型和参考音频设置：

```json
{"split_sentences":true,"split_method":"sentence","min_sentence_chars":12,"sentence_gap_ms":120}
```

原 command provider 的请求级分句或采样覆盖需要移到音色配置。
不要设置 `tts.openai.instructions` 来控制情绪，该字段当前被服务忽略。

### 命令行接入（可选）

使用 1.2.1/1.2.2 服务，或已有 command provider 脚本时，可继续使用
[`gpt-sovits-client`](../client/README.md)：

```bash
hermes config set tts.providers.sovits.type command
hermes config set tts.providers.sovits.command \
  "\"$(command -v gpt-sovits-client)\" --url http://127.0.0.1:9880 --voice demo --timeout 120 -o {output_path} < {input_path}"
hermes config set tts.providers.sovits.output_format wav
hermes config set tts.providers.sovits.timeout 150
hermes config set tts.provider sovits
```

先确认 `command -v gpt-sovits-client` 能找到程序。Hermes 将文本写入临时文件，
客户端从 stdin 读取并保存 WAV。长文本按实际耗时同时调整 provider 和客户端超时。
此路径需要语音气泡时，在 Hermes 环境安装 FFmpeg，再设置
`hermes config set tts.providers.sovits.voice_compatible true`。

本节按 Hermes [`0a62610` 的 TTS 实现](https://github.com/NousResearch/hermes-agent/blob/0a62610f10cc34d696b2239b2c69fa1ba0f1ca63/tools/tts_tool.py)
核对。若工具被权限策略拒绝，通过 Hermes 的工具授权或管理员配置处理。

## 读音和数字

只在确实歧义的字后加注音：

```text
这个人很好[hao4]学，银行的行[hang2]长正在重[chong2]新安排会议。
```

格式为 `字[pinyin+声调数字]`，1-5 表示声调，5 为轻声。只影响前一个中文字符，
方括号标注不会送给 BERT 朗读。其他文字仍按正常分词、G2P 和变调处理。

1.2.2 支持中文百分比、小数和全角数字展开，修复了数字附近的注音位置，并在切句时保护数字。
例如 `50%` 读“百分之五十”，`3.140` 读“三点一四零”，
`-3.5` 读“负三点五”。`50%分[fen4]成` 的标注仍只作用于原文的“分”。

数字规则用于中文。纯数字的 auto 默认中文；混合语言需要确定读法时显式指定
`text_language`。日期、时间、金额、电话、版本号和英文数字仍没有完整的上下文读法。
Agent 可先输出希望读出的文字，例如“二零二六年十月二十五日”“下午三点半”。

## 排查吞句

先定位具体 chunk，不要直接全局调整 temperature/top_p：

```bash
GPT_SOVITS_LOG_CHUNK_TEXT=1 ./gpt-sovits --http --port 9880 --voice demo
```

日志包含 chunk 序号、字数、semantic token 数、时长和 chars/sec。
明显短的音频只是线索，不是漏读的自动判定。保留请求、音色配置、模型哈希和失败样本，
用 [语音回归](RELEASE_ACCEPTANCE.md#speech-regression) 做 ASR 对照并试听。
文本日志可能包含私人对话，默认不要开启或对外上传。
