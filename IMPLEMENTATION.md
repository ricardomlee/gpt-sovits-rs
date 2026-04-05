# GPT-SoVITS Rust 实现总结

## 项目位置
`/home/ric/gpt-sovits-rs/`

## 已完成的工作

### 1. 项目结构搭建 ✅

```
gpt-sovits-rs/
├── Cargo.toml                 # 项目配置和依赖
├── README.md                  # 项目文档
├── .gitignore
├── src/
│   ├── lib.rs                 # 库入口，导出公共 API
│   ├── main.rs                # CLI 入口
│   ├── config/mod.rs          # 配置管理
│   ├── text_frontend/         # 文本前端模块
│   │   ├── mod.rs
│   │   ├── g2p.rs            # Grapheme-to-Phoneme 转换
│   │   ├── lang_detect.rs    # 语言检测
│   │   ├── normalizer.rs     # 文本规范化
│   │   └── symbols.rs        # 音素符号表
│   ├── models/                # 神经网络模型
│   │   ├── mod.rs
│   │   ├── bert.rs           # BERT 特征提取
│   │   ├── bigvgan.rs        # BigVGAN 声码器
│   │   ├── gpt.rs            # GPT 语义模型
│   │   ├── hubert.rs         # Hubert 特征提取
│   │   └── sovits.rs         # SoVITS 音频合成
│   ├── inference/             # 推理流程
│   │   └── mod.rs            # Pipeline 实现
│   └── utils/                 # 工具函数
│       ├── mod.rs
│       └── audio.rs          # 音频 I/O
├── examples/
│   └── cli_inference.rs      # CLI 推理示例
└── scripts/
    ├── download_and_convert.py  # 模型下载转换脚本
    └── export_onnx.py           # ONNX 导出脚本
```

### 2. 依赖配置 ✅

**核心依赖**:
- `candle-core` - HuggingFace 的轻量级 ML 框架
- `candle-nn` - 神经网络模块
- `candle-transformers` - Transformer 模型支持
- `half` - FP16 半精度支持
- `hound` - WAV 音频 I/O
- `symphonia` - 音频解码
- `serde/serde_json` - 序列化
- `safetensors` - 模型权重格式
- `pinyin/jieba-rs` - 中文文本处理
- `clap` - CLI 参数解析
- `tracing` - 日志记录
- `tokio/axum` (可选) - HTTP API 支持

**已禁用的依赖** (需要系统安装 protoc):
- `candle-onnx` - 需要 protobuf 编译器
- `ort` - ONNX Runtime (可选，用于 BERT/Hubert)

### 3. 核心功能实现 ✅

#### Config 模块
- `Config` 结构体：设备选择、精度设置、模型版本
- `ConfigBuilder`：流式构建器 API
- 支持 CUDA/CPU/MPS 设备

#### Text Frontend 模块
- `TextFrontend`：文本处理主入口
- `TextNormalizer`：文本规范化（ whitespace、标点、数字）
- `LanguageDetector`：基于字符范围的语言检测
- `G2PConverter`：多语言 G2P 转换框架
- `SymbolTable`：音素符号映射表

#### Models 模块
- `Model` trait：所有模型的通用接口
- `GPTModel`：语义 token 预测（placeholder）
- `SoVITSModel`：Mel 频谱合成（placeholder）
- `BertModel`：BERT 特征提取（placeholder）
- `HubertModel`：音频特征提取（placeholder）
- `BigVGAN`：神经声码器（placeholder）

#### Inference 模块
- `Pipeline`：主推理流程
- `InferenceOptions`：推理参数配置
- `InferenceOptionsBuilder`：流式构建器
- 支持模型加载、文本处理、特征提取、音频生成

#### Utils 模块
- `AudioBuffer`：音频数据容器
- 方法：`load()`, `save()`, `normalize()`, `resample()`, `fade_in/out()`
- 单元测试覆盖

### 4. 工具脚本 ✅

#### download_and_convert.py
- 从 HuggingFace 下载 pretrained models
- 转换 `.ckpt`/`.pth` 到 `.safetensors` 格式
- 支持 GPT、SoVITS、BigVGAN 模型

#### export_onnx.py
- 导出 BERT 模型到 ONNX 格式
- 导出 Hubert 模型到 ONNX 格式
- 验证导出模型的 correctness

### 5. 编译验证 ✅

```bash
$ cargo check
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.46s
```

**警告**: 15 个未使用变量警告（placeholder 代码预期内）

## 待完成的工作

### 短期 (1-2 周)
1. **安装 protoc** - 启用 candle-onnx 和 ONNX Runtime
2. **实现真实的模型加载** - 从 safetensors 文件加载权重
3. **实现 GPT 推理** - Transformer 前向传播 + 采样
4. **实现 SoVITS 推理** - Flow decoder + duration predictor

### 中期 (2-4 周)
1. **文本前端完善** - 集成真实的 G2P 模型
2. **BigVGAN 实现** - AMP/ALiBi 激活函数
3. **性能优化** - KV Cache、批处理、量化
4. **测试覆盖** - 单元测试 + 端到端测试

### 长期 (1-2 月)
1. **HTTP API** - 完整版 axum 服务器
2. **CUDA 加速** - 自定义 kernel 优化
3. **模型量化** - INT8/FP16 推理
4. **流式推理** - 边生成边播放

## 使用示例

### CLI 使用
```bash
# 下载并转换模型
python scripts/download_and_convert.py --output-dir models

# 运行推理
cargo run --release -- \
    --gpt-model models/gpt-s1bert.safetensors \
    --sovits-model models/sovits-s2G.safetensors \
    --text "你好世界" \
    --reference-audio ref.wav \
    --reference-text "参考文本" \
    --output output.wav
```

### Rust API
```rust
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};

let config = Config::builder()
    .with_device("cuda")
    .with_half_precision(true)
    .build();

let mut pipeline = Pipeline::new(config)?;
pipeline.load_gpt("models/gpt-model.safetensors")?;
pipeline.load_sovits("models/sovits-model.safetensors")?;

let options = InferenceOptions::builder()
    .top_k(15)
    .top_p(0.95)
    .temperature(0.8)
    .language(Language::Chinese)
    .build();

let audio = pipeline.inference(
    "你好，这是测试文本",
    "ref.wav",
    "参考文本",
    &options
)?;

audio.save("output.wav")?;
```

## 技术栈对比

| 特性 | Python 原版 | Rust 实现 |
|------|-----------|----------|
| 推理速度 | RTF ~0.028 | 目标 RTF <0.02 |
| 内存占用 | ~2GB | 目标 ~500MB |
| 启动时间 | ~5s | 目标 <1s |
| 部署复杂度 | Python 环境 + PyTorch | 单一二进制 |
| 平台支持 | Windows/Linux/Mac | Windows/Linux/Mac |

## 下一步行动

1. **Protoc 安装**: `apt-get install protobuf-compiler`
2. **启用 ONNX 支持**: 取消注释 `candle-onnx` 和 `ort`
3. **模型权重加载**: 实现真实的 safetensors 加载
4. **GPT 推理**: 实现 Transformer decoder 前向传播

## 参考资源

- Candle 文档：https://docs.rs/candle-core
- HuggingFace GPT-SoVITS: https://huggingface.co/lj1995/GPT-SoVITS
- BigVGAN 论文：https://arxiv.org/abs/2306.00814
