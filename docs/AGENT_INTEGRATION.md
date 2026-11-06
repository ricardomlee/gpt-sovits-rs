# Agent 接入

本页描述服务端契约，不依赖某个 agent 框架的配置格式。接口字段见 [API](API.md)，
版本差异见 [文档入口](README.md)。框架的 provider、播放与格式设置应以其对应版本为准。

## 逐句生成和播放

推荐在客户端按句子组织 LLM 输出：

```text
LLM token stream -> 句子边界检测 -> 有界队列
  -> 依次 POST /tts -> 按顺序播放 WAV
```

中文通常每段 10-25 字，优先在句末或较长从句的标点处分开，避免把一两个字拆成孤立请求。
这是接入建议，不是模型保证；分句仍可能漏读或产生不自然的衔接。
不要切开数字、小数、百分比、版本号或 `字[pinyin+tone]` 标注。

单个服务使用串行推理槽，多个请求会排队，并行提交不等于更快。
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

```bash
curl --fail-with-body http://127.0.0.1:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice":"sun","text":"你好，今天有什么安排？","text_language":"auto"}' \
  --output /tmp/reply.wav
```

支持 OpenAI-compatible speech 接口的客户端可把 base URL 指向
`http://127.0.0.1:9880/v1`，调用 `POST /v1/audio/speech`：

```json
{"model":"tts-1","voice":"sun","input":"你好，今天有什么安排？","response_format":"wav"}
```

这里只兼容 speech 请求，不提供完整 OpenAI API。支持 `wav` 和 `pcm`，不提供 MP3/Opus；
当前 PCM 为 32 kHz、单声道、16-bit little-endian，播放端必须按该格式处理。
不应给需要 MP3 的客户端返回标着 MP3 的 WAV。
`/tts/batch` 的 HTTP 200 也可能包含逐项失败，应检查每一行的 `error` 和 `wav_base64`；
批量结果没有 `success` 字段。

长期使用常驻 HTTP 服务。每句重新启动推理 CLI 都会重新加载模型。
服务无内置认证；默认限制在 loopback，跨机器访问需自己提供访问控制。

## 读音和数字

只在确实歧义的字后加注音：

```text
这个人很好[hao4]学，银行的行[hang2]长正在重[chong2]新安排会议。
```

格式为 `字[pinyin+声调数字]`，1-5 表示声调，5 为轻声。只影响前一个中文字符，
方括号标注不会送给 BERT 朗读。其他文字仍按正常分词、G2P 和变调处理。

**主线已合入、v1.2.1 尚不包含：** 中文百分比/小数/全角数字展开、数字附近注音位置修复，
以及切句时对数字的保护。例如 `50%` 读“百分之五十”，`3.140` 读“三点一四零”，
`-3.5` 读“负三点五”。`50%分[fen4]成` 的标注仍只作用于原文的“分”。

数字规则用于中文。纯数字的 auto 默认中文；混合语言需要确定读法时显式指定
`text_language`。日期、时间、金额、电话、版本号和英文数字仍没有完整的上下文读法。
Agent 可先输出希望读出的文字，例如“二零二六年十月二十五日”“下午三点半”。

## 排查吞句

先定位具体 chunk，不要直接全局调整 temperature/top_p：

```bash
GPT_SOVITS_LOG_CHUNK_TEXT=1 ./gpt-sovits --http --port 9880 --voice sun
```

日志包含 chunk 序号、字数、semantic token 数、时长和 chars/sec。
明显短的音频只是线索，不是漏读的自动判定。保留请求、音色配置、模型哈希和失败样本，
用 [语音回归](RELEASE_ACCEPTANCE.md#speech-regression) 做 ASR 对照并试听。
文本日志可能包含私人对话，默认不要开启或对外上传。
