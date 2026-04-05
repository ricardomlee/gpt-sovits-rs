#!/usr/bin/env python3
"""
Model Download Script for GPT-SoVITS Rust

Downloads pretrained models from HuggingFace and converts them to safetensors format.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    import torch
    from safetensors.torch import save_file
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install huggingface_hub safetensors torch")
    from huggingface_hub import snapshot_download
    import torch
    from safetensors.torch import save_file


def download_models(output_dir: str, mirror: bool = True):
    """Download GPT-SoVITS pretrained models from HuggingFace."""

    if mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using HuggingFace mirror...")

    models_to_download = [
        {
            "repo_id": "lj1995/GPT-SoVITS",
            "allow_patterns": [
                "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                "s1v3.ckpt",
                "s2G488k.pth",
                "s2Gv3.pth",
                "gsv-v2final-pretrained/*.ckpt",
                "gsv-v2final-pretrained/*.pth",
                "gsv-v4-pretrained/*.pth",
                "v2Pro/*.pth",
                "models--nvidia--bigvgan_v2_24khz_100band_256x/**",
            ],
            "output_subdir": "GPT-SoVITS"
        },
        {
            "repo_id": "XXXXRT/GPT-SoVITS-Pretrained",
            "allow_patterns": [
                "chinese-hubert-base/**",
                "chinese-roberta-wwm-ext-large/**",
                "G2PWModel/**",
            ],
            "output_subdir": "additional"
        }
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_config in models_to_download:
        print(f"\nDownloading {model_config['repo_id']}...")
        try:
            download_path = snapshot_download(
                repo_id=model_config["repo_id"],
                allow_patterns=model_config["allow_patterns"],
                local_dir=output_path / model_config["output_subdir"],
                local_dir_use_symlinks=False,
            )
            print(f"Downloaded to: {download_path}")
        except Exception as e:
            print(f"Error downloading {model_config['repo_id']}: {e}")

    return output_path


def convert_ckpt_to_safetensors(ckpt_path: str, output_path: str):
    """Convert PyTorch .ckpt file to safetensors format."""
    print(f"Converting {ckpt_path}...")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract state dict based on format
    if "weight" in checkpoint:
        state_dict = checkpoint["weight"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Save as safetensors
    save_file(state_dict, output_path)
    print(f"Saved to: {output_path}")

    return state_dict


def convert_pth_to_safetensors(pth_path: str, output_path: str):
    """Convert PyTorch .pth file to safetensors format."""
    print(f"Converting {pth_path}...")

    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if "weight" in checkpoint:
        state_dict = checkpoint["weight"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    else:
        state_dict = checkpoint

    # Save as safetensors
    save_file(state_dict, output_path)
    print(f"Saved to: {output_path}")

    return state_dict


def convert_bigvgan_to_safetensors(model_dir: str, output_path: str):
    """Convert BigVGAN model to safetensors format."""
    print(f"Converting BigVGAN from {model_dir}...")

    # Load BigVGAN generator
    generator_path = Path(model_dir) / "bigvgan_generator.pt"
    if not generator_path.exists():
        print(f"Generator not found at {generator_path}")
        return None

    state_dict = torch.load(str(generator_path), map_location="cpu", weights_only=False)

    # Save as safetensors
    save_file(state_dict, output_path)
    print(f"Saved to: {output_path}")

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Download and convert GPT-SoVITS models")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Output directory for downloaded models")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Don't use HuggingFace mirror")
    parser.add_argument("--convert-only", type=str,
                        help="Convert existing model file to safetensors")

    args = parser.parse_args()

    # Convert single file mode
    if args.convert_only:
        path = Path(args.convert_only)
        output = path.with_suffix(".safetensors")

        if path.suffix == ".ckpt":
            convert_ckpt_to_safetensors(str(path), str(output))
        elif path.suffix == ".pth":
            if "bigvgan" in path.name.lower() or "vocoder" in path.name.lower():
                convert_bigvgan_to_safetensors(str(path.parent), str(output))
            else:
                convert_pth_to_safetensors(str(path), str(output))
        else:
            print(f"Unsupported file format: {path.suffix}")
        return

    # Download and convert all models
    print("=== GPT-SoVITS Model Downloader ===\n")

    download_dir = download_models(args.output_dir, mirror=not args.no_mirror)

    # Convert downloaded models
    print("\n=== Converting models to safetensors ===\n")

    conversions = [
        # GPT models
        ("s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", "gpt-s1bert.safetensors"),
        ("s1v3.ckpt", "gpt-s1v3.safetensors"),
        # SoVITS models
        ("s2G488k.pth", "sovits-s2G.safetensors"),
        ("s2Gv3.pth", "sovits-s2Gv3.safetensors"),
    ]

    for src_name, dst_name in conversions:
        src = download_dir / "GPT-SoVITS" / src_name
        if src.exists():
            convert_ckpt_to_safetensors(str(src), str(download_dir / dst_name))
        else:
            print(f"Skipping {src_name} (not found)")

    print("\n=== Download Complete ===")
    print(f"\nModels saved to: {download_dir}")
    print("\nTo use with gpt-sovits-rs:")
    print("  gpt-sovits --gpt-model models/gpt-s1bert.safetensors \\")
    print("             --sovits-model models/sovits-s2G.safetensors \\")
    print("             --text '你好世界' \\")
    print("             --reference-audio ref.wav \\")
    print("             --reference-text '参考文本' \\")
    print("             --output output.wav")


if __name__ == "__main__":
    main()
