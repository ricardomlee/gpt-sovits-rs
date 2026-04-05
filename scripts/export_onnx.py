#!/usr/bin/env python3
"""
Export BERT and Hubert models to ONNX format.

These models will be used with ONNX Runtime in the Rust implementation.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

try:
    import onnx
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install onnx transformers torch")
    import onnx
    from transformers import AutoModel, AutoTokenizer


def export_bert_onnx(model_path: str, output_path: str):
    """Export BERT model to ONNX format."""
    print(f"Exporting BERT model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    # Create dummy input
    text = "测试文本"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Export to ONNX
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_path,
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
    )

    # Verify export
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"Exported BERT to: {output_path}")
    return True


def export_hubert_onnx(model_path: str, output_path: str):
    """Export Hubert model to ONNX format."""
    print(f"Exporting Hubert model from {model_path}...")

    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2Model.from_pretrained(model_path)
    model.eval()

    # Create dummy input (16kHz audio)
    import numpy as np
    dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")

    # Export to ONNX
    torch.onnx.export(
        model,
        (inputs["input_values"],),
        output_path,
        opset_version=14,
        input_names=["input_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "time"},
            "last_hidden_state": {0: "batch", 1: "time"},
        },
    )

    # Verify export
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"Exported Hubert to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX format")
    parser.add_argument("--model-type", type=str, choices=["bert", "hubert", "all"],
                        default="all", help="Type of model to export")
    parser.add_argument("--model-path", type=str,
                        help="Path to pretrained model")
    parser.add_argument("--output", type=str,
                        help="Output ONNX file path")

    args = parser.parse_args()

    output_dir = Path("models/onnx")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type in ["bert", "all"]:
        if args.model_path:
            bert_path = args.model_path
        else:
            bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

        if os.path.exists(bert_path):
            output = args.output or str(output_dir / "bert.onnx")
            export_bert_onnx(bert_path, output)
        else:
            print(f"BERT model not found at {bert_path}")

    if args.model_type in ["hubert", "all"]:
        if args.model_path:
            hubert_path = args.model_path
        else:
            hubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"

        if os.path.exists(hubert_path):
            output = args.output or str(output_dir / "hubert.onnx")
            export_hubert_onnx(hubert_path, output)
        else:
            print(f"Hubert model not found at {hubert_path}")

    print("\n=== Export Complete ===")
    print("\nONNX models can be used with ONNX Runtime in Rust:")
    print("  ort::Session::builder()?.commit_from_file(\"models/onnx/bert.onnx\")")


if __name__ == "__main__":
    main()
