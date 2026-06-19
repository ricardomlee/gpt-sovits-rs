#!/usr/bin/env python3
"""Convert GPT-SoVITS v2 checkpoint to safetensors format."""
import torch
from safetensors.torch import save_file
import sys

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ric/gpt-sovits/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
output_path = sys.argv[2] if len(sys.argv) > 2 else "/home/ric/gpt-sovits-rs/models/gpt-model-v2.safetensors"

print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

weights = {}
if "weight" in ckpt:
    raw = ckpt["weight"]
elif "state_dict" in ckpt:
    raw = ckpt["state_dict"]
else:
    raw = ckpt

print(f"Total keys: {len(raw)}")

for key, value in raw.items():
    # Keep key names as-is (Rust expects 'model.' prefix)
    clean_key = key

    if isinstance(value, torch.Tensor):
        # Convert to float32 for safetensors
        value = value.float().contiguous()
        weights[clean_key] = value
        if "text_embedding" in clean_key or "audio_embedding" in clean_key or "ar_predict" in clean_key:
            print(f"  {clean_key}: {value.shape}")

print(f"\nConverted {len(weights)} tensors")

# Verify key dims
if "ar_text_embedding.word_embeddings.weight" in weights:
    shape = weights["ar_text_embedding.word_embeddings.weight"].shape
    print(f"\nText embedding: {shape} (should be [732, 512] for v2)")

save_file(weights, output_path)
print(f"\nSaved to: {output_path}")
