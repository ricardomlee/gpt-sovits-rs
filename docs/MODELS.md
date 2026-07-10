# 模型下载与转换

`gpt-sovits-rs` 不在源码仓库、release binary 或 Docker 镜像中分发模型权重。用户应自行
从官方渠道下载、从已有 GPT-SoVITS 安装目录复制，或使用自己训练好的模型。默认推理需要四部分：

| 组件 | 默认来源 | 转换后路径 |
|---|---|---|
| GPT v2 | `lj1995/GPT-SoVITS` | `models/gpt-model.safetensors` |
| SoVITS v2 | `lj1995/GPT-SoVITS` | `models/sovits-model.safetensors` |
| Chinese RoBERTa large | GPT-SoVITS 模型仓库内副本 | `models/bert/bert.safetensors` |
| Chinese HuBERT base | GPT-SoVITS 模型仓库内副本 | `models/hubert/hubert.safetensors` |

模型源文件和转换结果合计需要约 3 GiB。转换时建议至少保留 6 GiB 可用磁盘空间和
4 GiB 可用内存。

## 准备官方 v2 模型

先从 GPT-SoVITS 官方模型仓库或 Hugging Face 缓存中取得这些源文件：

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

当前仓库不内置下载器，也不镜像第三方模型。下载可以通过浏览器、`huggingface-cli`、
`git lfs`、系统包管理器或你已有的 GPT-SoVITS 安装目录完成。转换本身不需要 Python。

如果只想使用 Docker 镜像里的转换器，可以把源模型目录只读挂载进去，把输出写入本机
`models/`：

```bash
docker run --rm \
  -v "$PWD/models:/models" \
  -v "/path/to/source-models:/source:ro" \
  --entrypoint gpt-sovits-convert \
  ghcr.io/ricardomlee/gpt-sovits-rs:latest \
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

如果一个服务里要放多个微调音色，不需要为每个音色起一个容器。把权重放在 `models/`
下，并在对应的 `voices/<name>/voice.json` 里绑定模型：

```text
models/
  carol/
    gpt.safetensors
    sovits.safetensors
  sun/
    gpt.safetensors
    sovits.safetensors
```

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频对应的文字",
  "sv_embedding": "ref_sv.safetensors",
  "gpt_model": "carol/gpt.safetensors",
  "sovits_model": "carol/sovits.safetensors",
  "language": "zh"
}
```

`gpt_model`、`sovits_model` 和 `bigvgan_model` 的相对路径按 `--models-dir` 解析。没有配置
这些字段的 voice 会继续使用服务启动时加载的默认模型。CLI 的 `--voice` 与 HTTP API
都会使用绑定模型，显式的 `--gpt-model` / `--sovits-model` 参数优先。

HTTP 服务共享 BERT 和 HuBERT，并默认用容量为 2 的 LRU 缓存 GPT/SoVITS pipeline。
超过容量的音色仍然可用，但下次切回时会重新加载；用 `--max-cached-pipelines` 调整容量。

GPT 与 SoVITS 必须来自兼容的 GPT-SoVITS v2 或 v2Pro 架构。v3、v4 和改过网络结构的
checkpoint 不能仅靠改扩展名使用。

v2Pro 的 SoVITS `.pth` 使用官方 `05`/`06` 版本头，`gpt-sovits-convert sovits` 会自动处理。
如果训练预处理目录里有 `logs/<voice>_v2pro/7-sv_cn/<ref>.wav.pt`，可以把它转换成 Rust
runtime 可读的 speaker-verification embedding：

```bash
gpt-sovits-convert sv \
  logs/diana_v2pro/7-sv_cn/ref.wav.pt \
  voices/diana/ref_sv.safetensors
```

然后在 `voices/diana/voice.json` 中配置：

```json
{
  "reference_audio": "ref.wav",
  "reference_text": "参考音频对应的文字",
  "sv_embedding": "ref_sv.safetensors",
  "language": "zh"
}
```

不提供 `sv_embedding` 时，v2Pro 仍可运行，但会使用零 SV embedding，音色相似度通常不如官方完整路径。

自训练音色仍然需要一段与文本严格对应的参考音频。可以用 ASR 先转写，再人工修正成 3 到
10 秒的短参考文本。

## 验证

转换后检查文件：

```bash
gpt-sovits --inspect models/gpt-model.safetensors
gpt-sovits --inspect models/sovits-model.safetensors
```

再用一段 3 到 10 秒的清晰参考音频做实际推理。参考文本必须与音频内容一致，否则音色
和韵律会明显下降。

## 许可证

项目源码使用 MIT License。模型权重来自独立发布者，不因转换为 safetensors 而改变
原有许可证、授权范围或使用限制。发布产品前应分别核对 GPT-SoVITS 和所用自训练模型的
许可条件。
