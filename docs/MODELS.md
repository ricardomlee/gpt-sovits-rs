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

## 自动下载

创建独立 Python 环境：

```bash
python3 -m venv .venv-models
. .venv-models/bin/activate
pip install -r requirements-models.txt
python prepare_models.py
```

脚本下载这些官方 v2 文件：

```text
gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
gsv-v2final-pretrained/s2G2333k.pth
chinese-roberta-wwm-ext-large/pytorch_model.bin
chinese-roberta-wwm-ext-large/tokenizer.json
chinese-hubert-base/pytorch_model.bin
```

已存在的输出默认不会覆盖。重新生成时使用：

```bash
python prepare_models.py --force
```

Hugging Face 下载缓存遵循 `HF_HOME`。例如把缓存放到大容量磁盘：

```bash
HF_HOME=/data/huggingface python prepare_models.py
```

## 使用本地源模型

如果已经安装原版 GPT-SoVITS：

```bash
python prepare_models.py \
  --source-dir /path/to/GPT-SoVITS/GPT_SoVITS/pretrained_models
```

脚本优先读取本地文件，缺失项才会下载。

也可以逐项覆盖：

```bash
python prepare_models.py \
  --gpt-checkpoint /path/to/model.ckpt \
  --sovits-checkpoint /path/to/model.pth \
  --bert-checkpoint /path/to/bert/pytorch_model.bin \
  --bert-tokenizer /path/to/bert/tokenizer.json \
  --hubert-checkpoint /path/to/hubert/pytorch_model.bin \
  --force
```

## 自训练音色模型

自训练或微调的 GPT、SoVITS 模型只需要替换前两项：

```bash
python convert_gpt_weights.py \
  /path/to/custom-gpt.ckpt \
  models/gpt-model.safetensors

python convert_sovits_weights.py \
  /path/to/custom-sovits.pth \
  models/sovits-model.safetensors
```

GPT 与 SoVITS 必须来自兼容的 GPT-SoVITS v2 架构。v3、v4、v2Pro 和改过网络结构的
checkpoint 不能仅靠改扩展名使用。

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
