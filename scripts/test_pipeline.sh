#!/bin/bash
# GPT-SoVITS Rust Pipeline Test Script

echo "=== GPT-SoVITS Rust 测试脚本 ==="
echo ""

# 1. 检查编译
echo "1. 检查编译状态..."
cargo check --release 2>&1 | tail -3
echo ""

# 2. 运行测试
echo "2. 运行所有测试..."
cargo test --release 2>&1 | grep "test result"
echo ""

# 3. 显示 CLI 帮助
echo "3. CLI 工具帮助信息:"
cargo run --release -- --help | head -20
echo ""

# 4. 检查模型文件
echo "4. 检查模型目录:"
if [ -d "models" ]; then
    ls -la models/ 2>/dev/null || echo "   models/ 目录为空"
else
    echo "   models/ 目录不存在"
    echo "   需要下载模型文件到 models/ 目录"
fi
echo ""

# 5. 下载模型说明
echo "=== 模型下载指南 ==="
echo ""
echo "方法 1: 使用 Python 脚本下载 (推荐)"
echo "  python scripts/download_and_convert.py --output-dir models"
echo ""
echo "方法 2: 手动下载"
echo "  1. 访问 https://huggingface.co/lj1995/GPT-SoVITS"
echo "  2. 下载以下文件:"
echo "     - s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt (GPT)"
echo "     - s2G488k.pth (SoVITS)"
echo "     - BERT 和 Hubert ONNX 模型 (可选)"
echo "  3. 使用 scripts/convert_models.py 转换为 safetensors 格式"
echo ""
echo "方法 3: 使用 ONNX Runtime (需要 --features onnx)"
echo "  cargo build --release --features onnx"
echo "  然后放置 BERT/Hubert 的 .onnx 模型文件"
echo ""
