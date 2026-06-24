#!/usr/bin/env python3
"""Convert GPT-SoVITS v2 SoVITS checkpoint to safetensors format."""
import torch
from safetensors.torch import save_file
import sys

pth_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ric/gpt-sovits/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
output_path = sys.argv[2] if len(sys.argv) > 2 else "/home/ric/gpt-sovits-rs/models/sovits-model-v2.safetensors"

print(f"Loading: {pth_path}")
ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)

raw = ckpt.get("weight", ckpt.get("state_dict", ckpt))
print(f"Total keys: {len(raw)}")

weights = {}
mismatches = []

for key, value in raw.items():
    if isinstance(value, torch.Tensor):
        value = value.float().contiguous()
        weights[key] = value
        # Print key layers
        if any(x in key for x in ['text_embedding', 'spectral.0', 'ssl_proj']):
            print(f"  {key}: {list(value.shape)}")

print(f"\nConverted {len(weights)} tensors")

# Verify critical dims
for name in ['enc_p.text_embedding.weight', 'ref_enc.spectral.0.fc.weight']:
    if name in weights:
        print(f"\n{name}: {list(weights[name].shape)}")

save_file(weights, output_path)
print(f"\nSaved to: {output_path}")
