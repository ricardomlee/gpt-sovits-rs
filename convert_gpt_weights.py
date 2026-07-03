#!/usr/bin/env python3
"""Convert one GPT-SoVITS GPT checkpoint to safetensors."""

import argparse
from pathlib import Path

from prepare_models import convert_gpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("models/gpt-model.safetensors"),
    )
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "Allow torch pickle loading for trusted legacy GPT-SoVITS "
            "checkpoints that cannot be read with weights_only=True."
        ),
    )
    args = parser.parse_args()
    convert_gpt(
        args.checkpoint,
        args.output,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
    )


if __name__ == "__main__":
    main()
