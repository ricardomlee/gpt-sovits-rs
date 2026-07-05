# 模型下载与转换

`gpt-sovits-rs` 不在 binary 或 Docker 镜像中分发模型权重。默认推理需要四部分：

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

当前仓库不内置下载器；下载可以通过浏览器、`huggingface-cli`、`git lfs`、系统包管理器或
你已有的 GPT-SoVITS 安装目录完成。转换本身不需要 Python。

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
