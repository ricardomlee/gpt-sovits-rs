#!/usr/bin/env python3
"""Convert a GPT-SoVITS v2Pro SV embedding .pt file to safetensors."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embedding", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    tensor = torch.load(args.embedding, map_location="cpu", weights_only=True)
    if isinstance(tensor, dict):
        tensors = [value for value in tensor.values() if isinstance(value, torch.Tensor)]
        if len(tensors) != 1:
            raise ValueError(f"{args.embedding} must contain exactly one tensor")
        tensor = tensors[0]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{args.embedding} did not contain a tensor")
    if tuple(tensor.shape) == (20480,):
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != (1, 20480):
        raise ValueError(f"SV embedding must have shape [20480] or [1, 20480], got {tuple(tensor.shape)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    save_file(
        {"sv_embedding": tensor.detach().float().contiguous()},
        temporary,
        metadata={"model_type": "sv_embedding", "shape": "1,20480"},
    )
    os.replace(temporary, args.output)
    print(f"Saved SV embedding to {args.output}")


if __name__ == "__main__":
    main()
