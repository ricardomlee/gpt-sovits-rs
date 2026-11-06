# 模型下载与转换

只想先试一次？[首次运行](FIRST_RUN.zh-CN.md)提供固定版本的 Docker-only 路径，
直接从上游下载并校验全部所需文件，不需要先安装宿主机转换器。
已有模型的用户继续阅读下面的转换说明。

模型可以从官方渠道下载、从已有 GPT-SoVITS 安装目录复制，或使用自己训练的权重。
它们单独准备，转换后放在本机 `models/` 目录。默认推理使用四部分：

| 组件 | 默认来源 | 转换后路径 |
|---|---|---|
| GPT v2 | `lj1995/GPT-SoVITS` | `models/gpt-model.safetensors` |
| SoVITS v2 | `lj1995/GPT-SoVITS` | `models/sovits-model.safetensors` |
| Chinese RoBERTa large | GPT-SoVITS 模型仓库内副本 | `models/bert/bert.safetensors` |
| Chinese HuBERT base | GPT-SoVITS 模型仓库内副本 | `models/hubert/hubert.safetensors` |

模型源文件和转换结果合计需要约 3 GiB。转换时建议至少保留 6 GiB 可用磁盘空间和
4 GiB 可用内存。

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
  ghcr.io/ricardomlee/gpt-sovits-rs:1.2.1 \
  sovits /source/gsv-v2final-pretrained/s2G2333k.pth /models/sovits-model.safetensors
```

## 自训练音色模型

自训练或微调的 GPT、SoVITS 模型只需要替换前两项：

```bash
gpt-sovits-convert gpt \
  /path/to/custom-gpt.ckpt \
  models/gpt-model.safetensors

gpt-sovits-convert sovits \
  /path/to/custom-sovits.pth \
  models/sovits-model.safetensors
```

一个服务可以使用多个微调音色。把权重放在 `models/` 下，并在对应的
`voices/<name>/voice.json` 里绑定模型。例如 `voices/demo/voice.json`：

```text
models/
  demo/
    gpt.safetensors
    sovits.safetensors
  second/
    gpt.safetensors
    sovits.safetensors
```

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频对应的文字",
  "gpt_model": "demo/gpt.safetensors",
  "sovits_model": "demo/sovits.safetensors",
  "language": "zh"
}
```

`gpt_model`、`sovits_model` 的相对路径按 `--models-dir` 解析。没有配置
这些字段的 voice 会使用服务启动时加载的默认模型。CLI 的 `--voice` 使用绑定模型，
显式的 `--gpt-model` / `--sovits-model` 可覆盖该次 CLI 选择。HTTP 则按请求中的 voice
选择绑定模型；启动参数配置默认模型对。服务启动时仍需一套可用的默认模型，
路径配置见[部署](DEPLOYMENT.md#准备目录和音色)。

`bigvgan_model` 字段虽可解析，但 BigVGAN 尚未接入主 SoVITS 合成路径，不能用它启用 v3/v4。

HTTP 服务共享 BERT 和 HuBERT，并默认用容量为 2 的 LRU 缓存 GPT/SoVITS pipeline。
超过容量的音色仍然可用，但下次切回时会重新加载；用 `--max-cached-pipelines` 调整容量。

GPT 与 SoVITS 必须来自兼容的 GPT-SoVITS v2 或 v2Pro 架构。v3、v4 和改过网络结构的
checkpoint 不能仅靠改扩展名使用。

## v2Pro SV

v2Pro 的 SoVITS `.pth` 使用官方 `05`/`06` 版本头，`gpt-sovits-convert sovits` 会自动处理。
如果训练预处理目录里有 `logs/<voice>_v2pro/7-sv_cn/<ref>.wav.pt`，可以把它转换成 Rust
runtime 可读的 speaker-verification embedding：

```bash
gpt-sovits-convert sv \
  logs/demo_v2pro/7-sv_cn/ref.wav.pt \
  voices/demo/ref_sv.safetensors
```

然后在 `voices/demo/voice.json` 中配置：

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频对应的文字",
  "sv_embedding": "ref_sv.safetensors",
  "language": "zh"
}
```

不提供 `sv_embedding` 时，v2Pro 仍可运行，但会使用零 SV embedding，音色相似度通常不如官方完整路径。

转换器读取已有训练 SV 文件，不会从新参考音频提取 SV。更换参考音频时应使用与之匹配的
embedding；1.2.2 需要用户提供匹配的 embedding 文件。

自训练音色使用 3 到 10 秒的清晰参考音频及准确转写。可先用 ASR 转写，再人工核对。

## 验证

转换后检查文件，再创建[音色配置](DEPLOYMENT.md#准备目录和音色)。以下使用默认文件名，
自定义模型将 `--inspect` 后的路径改为实际文件：

```bash
gpt-sovits --inspect models/gpt-model.safetensors
gpt-sovits --inspect models/sovits-model.safetensors
gpt-sovits --doctor --models-dir models --voices-dir voices --voice demo
gpt-sovits --models-dir models --voices-dir voices --voice demo \
  --text '你好，这是一次语音测试。' --output test.wav
```

检查 doctor 输出中的模型、参考音频和 SV 路径，并试听 `test.wav`。
通过后可[启动 HTTP 服务](DEPLOYMENT.md)。

## 许可证

项目源码使用 MIT License。模型权重来自独立发布者，不因转换为 safetensors 而改变
原有许可证、授权范围或使用限制。发布产品前应分别核对 GPT-SoVITS 和所用自训练模型的
许可条件。
