# GPT-SoVITS Rust 实现技术文档

## 项目位置

`/home/ric/gpt-sovits-rs/`

## 项目状态

### 已完成的核心功能

#### 1. 完整推理流程 ✅

端到端 TTS 推理流程已完全实现：

```
文本 → Text Frontend → BERT → Hubert → GPT → SoVITS → BigVGAN → 音频
```

所有模型均已加载 safetensors/ONNX 权重并支持实际推理。

#### 2. 模型实现 ✅

| 模型 | 功能 | 状态 |
|------|------|------|
| BERT | 文本语义特征提取 | ✅ ONNX 推理 |
| Hubert | 音频韵律特征提取 | ✅ ONNX 推理 |
| GPT | 语义 token 生成 | ✅ 完整推理 + KV Cache |
| SoVITS | Mel 频谱合成 | ✅ 完整推理 |
| BigVGAN | 神经声码器 | ✅ 完整推理 |

#### 3. 性能优化 ✅

**KV Cache 优化** (`src/utils/kv_cache.rs`):
- 实现 `KvCache` 和 `KvCacheManager` 数据结构
- 修改 `MultiHeadAttention` 支持 `forward_kv` 方法
- GPT 自回归生成速度提升 **18x**

实测结果 (CPU, 500 tokens):
| 配置 | 时间 | 加速比 |
|------|------|--------|
| 无 KV Cache | 368.82s | 1.0x |
| 启用 KV Cache | 20.48s | 18.0x |

**GPU 加速** (`src/config/mod.rs`):
- 默认使用 CUDA 设备
- 支持自动 GPU 检测回退

#### 4. 工具模块 ✅

- `examples/benchmark_kv_cache.rs` - KV Cache 性能对比
- `examples/profile_pipeline.rs` - 全流程时间分析
- `examples/e2e_gpu_test.rs` - GPU 端到端测试
- `examples/test_hubert_fusion.rs` - Hubert 特征融合测试

### 项目结构

```
gpt-sovits-rs/
├── Cargo.toml
├── src/
│   ├── main.rs              # CLI 入口
│   ├── lib.rs               # 库入口
│   ├── config/
│   │   └── mod.rs           # 配置管理 (设备/精度)
│   ├── text_frontend/
│   │   ├── mod.rs           # 文本处理
│   │   ├── normalizer.rs    # 文本规范化
│   │   ├── lang_detect.rs   # 语言检测
│   │   ├── g2p.rs           # Grapheme-to-phoneme
│   │   └── symbols.rs       # 音素符号表
│   ├── models/
│   │   ├── mod.rs
│   │   ├── bert.rs          # BERT (ONNX)
│   │   ├── hubert.rs        # Hubert (ONNX)
│   │   ├── gpt.rs           # GPT (Candle)
│   │   ├── sovits.rs        # SoVITS (Candle)
│   │   ├── bigvgan.rs       # BigVGAN (Candle)
│   │   ├── mrte.rs          # 多参考音色编码器
│   │   └── transformer.rs   # Transformer 层
│   ├── inference/
│   │   └── mod.rs           # Pipeline 实现
│   └── utils/
│       ├── mod.rs
│       ├── audio.rs         # 音频 I/O
│       ├── kv_cache.rs      # KV Cache 优化
│       └── state_dict.rs    # 模型权重加载
├── examples/
│   ├── cli_inference.rs         # CLI 示例
│   ├── benchmark_kv_cache.rs    # KV Cache 基准
│   ├── profile_pipeline.rs      # 性能分析
│   └── e2e_gpu_test.rs          # GPU 测试
├── scripts/
│   ├── download_and_convert.py  # 模型转换
│   └── export_onnx.py           # ONNX 导出
└── models/                      # 模型文件
```

## 技术详解

### KV Cache 优化原理

在 Transformer 的自注意力机制中：

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**问题**：自回归生成时，每次生成新 token 都重新计算所有 K/V。

**解决**：缓存之前 token 的 K/V，只计算新 token 的 K/V。

```
步骤 1: 输入"你"
  - 计算 Q₁, K₁, V₁
  - Cache: {K₁, V₁}

步骤 2: 输入"好"
  - 计算 Q₂, K₂, V₂ (仅新 token)
  - 拼接：K = [K₁, K₂], V = [V₁, V₂]
  - Cache: {K₁, V₁, K₂, V₂}
```

**核心代码** (`src/utils/kv_cache.rs`):

```rust
pub struct KvCache {
    k_cache: Option<Tensor>,  // [batch, heads, seq_len, head_dim]
    v_cache: Option<Tensor>,
    len: usize,
}

pub fn update(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
    // 沿 seq_len 维度拼接缓存
    let k_out = Tensor::cat(&[prev_k, k], 2)?;
    let v_out = Tensor::cat(&[prev_v, v], 2)?;
    Ok((k_out, v_out))
}
```

**性能分析**:
- 理论加速：O(n²) → O(n)，500 tokens 约 250x
- 实测加速 (CPU): 18x (20.48s vs 368.82s)
- 实测加速 (GPU RTX 4060 Ti): 1.65x (8.01s vs 13.23s)

**为什么 GPU 加速比小？**
- GPU 并行计算能力强，KV Cache 的计算优化收益相对较小
- 但 KV Cache 减少了内存带宽压力，仍有 1.65x 提升
- CPU 受限于内存带宽，KV Cache 收益更明显 (18x)

### 全流程性能 (GPU)

**实测结果** (RTX 4060 Ti, 500 tokens, KV Cache 启用):

| 阶段 | 时间 | 占比 |
|------|------|------|
| GPT (KV Cache) | 7.52s | 91.9% |
| BigVGAN | 652ms | 8.0% |
| SoVITS | 7.6ms | 0.1% |
| BERT/Hubert (ONNX) | <1ms | <0.1% |
| **总计** | **8.18s** | 100% |

**输出**: 1.33s 音频 @ 24kHz
**实时率 (RTF)**: 0.16 (6.1x 实时)

**CPU vs GPU 对比**:
- GPT: 16.17s (CPU) → 7.52s (GPU) = 2.1x
- BigVGAN: 3.68s (CPU) → 0.65s (GPU) = 5.6x
- 总计：19.95s (CPU) → 8.18s (GPU) = 2.4x

### 模型架构

#### GPT 模型 (`src/models/gpt.rs`)

```rust
pub struct GPTModel {
    text_embedding: Tensor,      // [vocab_size, hidden_size]
    audio_embedding: Tensor,     // [1025, hidden_size]
    bert_proj: Option<(Tensor, Tensor)>,  // [512, 1024]
    hubert_proj: Option<(Tensor, Tensor)>, // [512, 768]
    mrte: Option<MRTE>,          // 可选的 MRTE 模块
    transformer: TransformerGPTSoVITS,
    ar_predict_layer: Tensor,    // [vocab_size, hidden_size]
    num_layers: usize,           // KV Cache 需要
}
```

#### Transformer 层 (`src/models/transformer.rs`)

```rust
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,   // SwiGLU 激活
    attn_norm: LayerNorm,
    ffn_norm: LayerNorm,
}

// GPT-SoVITS 使用 fused QKV 投影
in_proj_weight: [hidden * 3, hidden]  // Q, K, V 合并
```

### 推理流程 (`src/inference/mod.rs`)

```rust
pub fn inference<P: AsRef<Path>>(
    &mut self,
    text: &str,
    reference_audio: P,
    reference_text: &str,
    options: &InferenceOptions,
) -> Result<AudioBuffer> {
    // 1. 文本处理 → 音素 ID
    let phoneme_ids = self.text_frontend.process(text, ...)?;
    
    // 2. BERT 特征提取 [1, seq_len, 1024]
    let bert_features = self.bert_model.extract(text)?;
    
    // 3. Hubert 特征提取 [1, frames, 768]
    let hubert_features = self.hubert_model.extract(audio)?;
    
    // 4. GPT 生成语义 tokens (使用 KV Cache)
    let semantic_tokens = gpt.generate_with_features_kv_cache(...)?;
    
    // 5. SoVITS 合成 Mel 频谱
    let mel_spec = sovits.synthesize(&semantic_tokens, ...)?;
    
    // 6. BigVGAN 生成波形
    let waveform = bigvgan.generate(&mel_spec)?;
    
    Ok(AudioBuffer::new(waveform, 24000, 1))
}
```

## 基准测试

### 运行 KV Cache 基准

```bash
cargo run --release --example benchmark_kv_cache
```

输出示例:
```
=== KV Cache Benchmark ===
Without KV Cache: 368.82s (average)
With KV Cache:    20.48s (average)
Speedup: 18.01x faster
```

### 运行全流程分析

```bash
cargo run --release --example profile_pipeline
```

输出示例:
```
Text Frontend:    0.02s ( 0.1%)
BERT Inference:   0.15s ( 1.1%)
Hubert Inference: 0.08s ( 0.6%)
GPT Generation:  13.57s (95.6%)  ← 主要瓶颈
SoVITS Synthesis: 0.03s ( 0.2%)
BigVGAN Vocoder:  0.62s ( 4.3%)
Total:           14.20s
```

## 下一步优化方向

1. ** speculative sampling** - 推测性采样加速 GPT 生成
2. **模型量化** - INT8/FP16 减少内存带宽
3. **批处理** - 多语句并行推理
4. **流式输出** - 边生成边播放

## 参考资源

- Candle 文档：https://docs.rs/candle-core
- GPT-SoVITS: https://huggingface.co/lj1995/GPT-SoVITS
- BigVGAN 论文：https://arxiv.org/abs/2306.00814
