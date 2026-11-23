# 下载和转换 GPT-SoVITS v2、v2Pro 模型

| 你手头有什么？ | 需要准备什么 | 转换与空间 |
| --- | --- | --- |
| 没有模型，想先听一句 | [Docker 首次运行](FIRST_RUN.zh-CN.md)会下载官方 v2、BERT、HuBERT 和公开英文参考录音 | 自动校验约 1.1 GB 的下载并转换；预留 6 GiB 磁盘、4 GiB 内存，另需下载镜像。 |
| 官方或微调的 v2 | [准备四个组件及 tokenizer](#准备官方-v2-模型)；微调音色只[替换 GPT 和 SoVITS](#自训练音色模型)，可复用 BERT、HuBERT | 原始权重需转换，已经转换好的 safetensors 可直接用。官方源文件加转换结果约 3 GiB；转换建议预留 6 GiB 磁盘、4 GiB 内存。 |
| v2Pro 微调权重 | [接入自训练音色](#自训练音色模型)：匹配的 GPT/SoVITS、BERT、HuBERT，以及参考录音对应的说话人向量（SV） | 与 v2 使用相同的转换器，另准备各音色的 SV 文件。 |

模型可以从官方渠道下载、从已有 GPT-SoVITS 安装目录复制，或使用自己训练的权重。
它们单独准备，转换后放在本机 `models/` 目录。默认推理使用四部分：

| 组件 | 默认来源 | 转换后路径 |
|---|---|---|
| GPT v2 | `lj1995/GPT-SoVITS` | `models/gpt-model.safetensors` |
| SoVITS v2 | `lj1995/GPT-SoVITS` | `models/sovits-model.safetensors` |
| Chinese RoBERTa large | GPT-SoVITS 模型仓库内副本 | `models/bert/bert.safetensors` |
| Chinese HuBERT base | GPT-SoVITS 模型仓库内副本 | `models/hubert/hubert.safetensors` |

## 获取转换器

[Release 压缩包](https://github.com/ricardomlee/gpt-sovits-rs/releases/latest)同时包含
`gpt-sovits` 和 `gpt-sovits-convert`，也可[从源码构建](DEVELOPMENT.md#build-from-source)。
下面的命令假设这两个程序已加入 `PATH`；在解压目录直接运行时，命令前加 `./`。
仅使用 Docker 的用户看[镜像内转换](#docker-转换)。

## 准备官方 v2 模型

先从 [GPT-SoVITS 官方模型仓库](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)
或 Hugging Face 缓存中取得这些源文件：

```text
gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
gsv-v2final-pretrained/s2G2333k.pth
chinese-roberta-wwm-ext-large/pytorch_model.bin
chinese-roberta-wwm-ext-large/tokenizer.json
chinese-hubert-base/pytorch_model.bin
```

然后用 Rust converter 生成运行时文件：

```bash
mkdir -p models/bert models/hubert

gpt-sovits-convert gpt \
  /path/to/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt \
  models/gpt-model.safetensors

gpt-sovits-convert sovits \
  /path/to/gsv-v2final-pretrained/s2G2333k.pth \
  models/sovits-model.safetensors

gpt-sovits-convert bert \
  /path/to/chinese-roberta-wwm-ext-large/pytorch_model.bin \
  models/bert/bert.safetensors

cp /path/to/chinese-roberta-wwm-ext-large/tokenizer.json models/bert/tokenizer.json

gpt-sovits-convert hubert \
  /path/to/chinese-hubert-base/pytorch_model.bin \
  models/hubert/hubert.safetensors
```

`tokenizer.json` 与转换后的 BERT 权重放在同一目录，保留原文件即可。

## Docker 转换

镜像中的转换器使用相同的子命令。把源目录只读挂载到 `/source`，输出目录挂载到 `/models`：

```bash
mkdir -p models
docker run --rm \
  -v "$PWD/models:/models" \
  -v "/path/to/source-models:/source:ro" \
  --entrypoint gpt-sovits-convert \
  ghcr.io/ricardomlee/gpt-sovits-rs:1.3.4 \
  sovits /source/gsv-v2final-pretrained/s2G2333k.pth /models/sovits-model.safetensors
```

## 自训练音色模型

以下用一个名为 `my-voice` 的音色，从训练权重跑到 HTTP 出声。准备同一套 v2 或 v2Pro
的 GPT `.ckpt`、SoVITS `.pth`，以及一段 3 到 10 秒的 WAV 和它的逐字稿。用训练
`.list` 文件时，取同一行的音频路径（第一列）和逐字稿（最后一列）。

把 `DATA_DIR` 设为 `models/` 和 `voices/` 的上级目录。按上表将已有的 BERT、HuBERT
和 tokenizer 放到 `DATA_DIR/models/`。需要转换时，在 `DATA_DIR` 运行上面的
[BERT、HuBERT 转换命令](#准备官方-v2-模型)，无需下载另一套 GPT 或 SoVITS。

### v2Pro SV

在训练的 `logs/<训练任务>/7-sv_cn/` 下按参考录音的原文件名找 SV：
`line.wav` 对应 `line.wav.pt`。纯 v2 不需要这个文件。没有 SV 时，v2Pro 会用零向量，
音色可能与训练结果不同。

给已有音色补 SV 时，可单独运行 `gpt-sovits-convert sv line.wav.pt ref_sv.safetensors`。

### 生成音色

从仓库目录运行[准备脚本](../examples/trained-voice/prepare.sh)。需要 Bash、`jq`，以及
已加入 `PATH` 的 `gpt-sovits-convert`。音色名以英文字母或数字开头，可含连字符或下划线。
仅使用 Docker 时，可沿用
[镜像内转换](#docker-转换)和[音色配置示例](DEPLOYMENT.md#音色目录)。

```bash
DATA_DIR="$HOME/tts"
bash examples/trained-voice/prepare.sh "$DATA_DIR" my-voice \
  /path/to/my-voice.ckpt /path/to/my-voice.pth /path/to/line.wav \
  zh '参考录音的逐字稿' /path/to/logs/my-voice_v2pro/7-sv_cn/line.wav.pt
```

v2 省略最后一个 SV 参数。脚本将权重写到 `models/my-voice/`，并在
`voices/my-voice/` 保存录音、`voice.json` 和可选的 SV。命令成功后会打印
`voice.json` 的路径；已有同名目录时会退出，不覆盖。

### 启动并接入 Hermes

检查和启动都传 `--voice my-voice`，这样服务就会用该音色绑定的权重启动，不需要再准备
`models/gpt-model.safetensors` 和 `models/sovits-model.safetensors`。两个命令都从
刚生成音色的 `DATA_DIR` 运行：

```bash
cd "$DATA_DIR"
gpt-sovits --device cpu --models-dir models --voices-dir voices --doctor --voice my-voice
gpt-sovits --device cpu --models-dir models --voices-dir voices --http --port 9880 --voice my-voice
```

在另一个终端[用 `my-voice` 试听](DEPLOYMENT.md#验证服务)，然后按 [Hermes 接入步骤](AGENT_INTEGRATION.md#hermes)
将 `tts.openai.voice` 设置为 `my-voice`。

已有服务时，把模型及音色目录加到已有的挂载目录，直接按新的音色名请求。多音色缓存设置见
[部署](DEPLOYMENT.md#音色目录)，用 Compose 启动时的默认权重路径见[目录配置](DEPLOYMENT.md#准备目录和音色)。

GPT、SoVITS 须来自兼容的 v2 或 v2Pro 架构。v3、v4 不在这条推理路径中，设置 `bigvgan_model`
也不能启用它们。

## 验证

以下使用官方 v2 的默认文件名和[添加音色](DEPLOYMENT.md#添加音色)一节的 `my-voice`：

```bash
gpt-sovits --inspect models/gpt-model.safetensors
gpt-sovits --inspect models/sovits-model.safetensors
gpt-sovits --doctor --models-dir models --voices-dir voices --voice my-voice
gpt-sovits --models-dir models --voices-dir voices --voice my-voice \
  --text '你好，这是一次语音测试。' --output test.wav
```

检查 doctor 输出中的模型、参考音频和 SV 路径，并试听 `test.wav`。
首次加载会校验 Flow 层数与 WaveNet 层组的完整性。缺层或索引不连续时，错误会包含
对应权重前缀；请核对源 checkpoint 和转换结果，重新取得或转换完整权重。
通过后可[启动 HTTP 服务](DEPLOYMENT.md)。

## 许可证

项目源码使用 MIT License。模型权重来自独立发布者，不因转换为 safetensors 而改变
原有许可证、授权范围或使用限制。发布产品前应分别核对 GPT-SoVITS 和所用自训练模型的
许可条件。
