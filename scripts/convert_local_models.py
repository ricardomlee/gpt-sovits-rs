#!/usr/bin/env python3
"""
Convert existing GPT-SoVITS models to safetensors format.
Usage: python scripts/convert_local_models.py --src /path/to/gpt-sovits/GPT_SoVITS/pretrained_models --output models
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install torch safetensors --break-system-packages -q")
    import torch
    from safetensors.torch import save_file


def convert_ckpt(ckpt_path: str, output_path: str):
    """Convert .ckpt file to safetensors."""
    print(f"Converting {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "weight" in checkpoint:
        state_dict = checkpoint["weight"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    save_file(state_dict, output_path)
    print(f"Saved to: {output_path}")
    return state_dict


def convert_pth(pth_path: str, output_path: str):
    """Convert .pth file to safetensors."""
    print(f"Converting {pth_path}...")
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)

    if "weight" in checkpoint:
        state_dict = checkpoint["weight"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    else:
        state_dict = checkpoint

    save_file(state_dict, output_path)
    print(f"Saved to: {output_path}")
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert local GPT-SoVITS models to safetensors")
    parser.add_argument("--src", type=str, required=True, help="Source directory containing .ckpt and .pth files")
    parser.add_argument("--output", type=str, default="models", help="Output directory for safetensors models")
    args = parser.parse_args()

    src = Path(args.src)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Source: {src}")
    print(f"Output: {output}")
    print()

    # Find and convert GPT model (.ckpt)
    gpt_ckpt = src / "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    if gpt_ckpt.exists():
        convert_ckpt(str(gpt_ckpt), str(output / "gpt-model.safetensors"))
    else:
        print(f"GPT .ckpt not found at {gpt_ckpt}")

    # Find and convert SoVITS model (.pth)
    sovits_pth = src / "s2G488k.pth"
    if sovits_pth.exists():
        convert_pth(str(sovits_pth), str(output / "sovits-model.safetensors"))
    else:
        print(f"SoVITS .pth not found at {sovits_pth}")

    # Convert BigVGAN if available
    bigvgan_dir = src / "models--nvidia--bigvgan_v2_24khz_100band_256x"
    if bigvgan_dir.exists():
        generator_path = bigvgan_dir / "bigvgan_generator.pt"
        if generator_path.exists():
            convert_pth(str(generator_path), str(output / "bigvgan.safetensors"))
        else:
            print(f"BigVGAN generator not found at {generator_path}")

    print()
    print("Conversion complete!")
    print(f"Models saved to: {output}")
    print()
    print("Usage:")
    print(f"  cargo run --release -- \\")
    print(f"    --gpt-model {output}/gpt-model.safetensors \\")
    print(f"    --sovits-model {output}/sovits-model.safetensors \\")
    print(f"    --text \"你好世界\" \\")
    print(f"    --reference-audio ref.wav \\")
    print(f"    --reference-text \"参考文本\" \\")
    print(f"    --output output.wav \\")
    print(f"    --device cpu")


if __name__ == "__main__":
    main()
