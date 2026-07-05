#!/usr/bin/env python3
"""Download and convert the GPT-SoVITS v2 models used by gpt-sovits-rs."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import types
from io import BytesIO
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import save_file

DEFAULT_REPO = "lj1995/GPT-SoVITS"
DEFAULT_FILES = {
    "gpt": "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "sovits": "gsv-v2final-pretrained/s2G2333k.pth",
    "bert": "chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "tokenizer": "chinese-roberta-wwm-ext-large/tokenizer.json",
    "hubert": "chinese-hubert-base/pytorch_model.bin",
}

SOVITS_HEADER_VERSION = {
    b"00": "v1",
    b"01": "v2",
    b"02": "v3",
    b"03": "v3",
    b"04": "v4",
    b"05": "v2Pro",
    b"06": "v2ProPlus",
}


def install_legacy_hparams_stub() -> None:
    """Provide utils.HParams for trusted legacy GPT-SoVITS pickle checkpoints."""
    module = sys.modules.get("utils")
    if module is None:
        module = types.ModuleType("utils")
        sys.modules["utils"] = module
    if hasattr(module, "HParams"):
        return

    class HParams(dict):
        pass

    HParams.__module__ = "utils"
    module.HParams = HParams


def load_state_dict(
    path: Path,
    *,
    allow_unsafe_pickle: bool = False,
) -> dict[str, torch.Tensor]:
    checkpoint = load_checkpoint(path, allow_unsafe_pickle=allow_unsafe_pickle)
    return checkpoint_tensors(path, checkpoint)


def load_checkpoint(
    path: Path,
    *,
    allow_unsafe_pickle: bool = False,
):
    data = path.read_bytes()
    header = data[:2]
    is_versioned_sovits = header in SOVITS_HEADER_VERSION and header != b"PK"
    source = BytesIO(b"PK" + data[2:]) if is_versioned_sovits else path

    if allow_unsafe_pickle or is_versioned_sovits:
        install_legacy_hparams_stub()
        return torch.load(source, map_location="cpu", weights_only=False)
    else:
        return torch.load(source, map_location="cpu", weights_only=True)


def checkpoint_tensors(path: Path, checkpoint) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} did not contain a state dictionary")

    for key in ("weight", "state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            checkpoint = nested
            break

    return {
        key: value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def sovits_version_from_checkpoint(source: Path, weights: dict[str, torch.Tensor]) -> str:
    header = source.read_bytes()[:2]
    if header in SOVITS_HEADER_VERSION:
        return SOVITS_HEADER_VERSION[header]
    if "ge_to512.weight" in weights and "sv_emb.weight" in weights:
        return "v2Pro"
    embedding = weights.get("enc_p.text_embedding.weight")
    if embedding is not None and embedding.shape[0] == 322:
        return "v1"
    return "v2"


def save_state_dict(
    weights: dict[str, torch.Tensor],
    output: Path,
    metadata: dict[str, str] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    converted = {
        key: value.detach().to(dtype=torch.float32, device="cpu").contiguous()
        for key, value in weights.items()
    }
    save_file(converted, temporary, metadata=metadata)
    os.replace(temporary, output)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Saved {len(converted)} tensors to {output} ({size_mb:.1f} MiB)")


def convert_gpt(
    source: Path,
    output: Path,
    *,
    allow_unsafe_pickle: bool = False,
) -> None:
    print(f"Converting GPT checkpoint: {source}")
    weights = load_state_dict(source, allow_unsafe_pickle=allow_unsafe_pickle)
    required = "model.ar_text_embedding.word_embeddings.weight"
    if required not in weights:
        raise KeyError(f"GPT checkpoint is missing {required}")
    save_state_dict(weights, output)


def convert_sovits(
    source: Path,
    output: Path,
    *,
    allow_unsafe_pickle: bool = False,
) -> None:
    print(f"Converting SoVITS checkpoint: {source}")
    checkpoint = load_checkpoint(source, allow_unsafe_pickle=allow_unsafe_pickle)
    weights = checkpoint_tensors(source, checkpoint)
    required = "enc_p.text_embedding.weight"
    if required not in weights:
        raise KeyError(f"SoVITS checkpoint is missing {required}")
    version = sovits_version_from_checkpoint(source, weights)
    save_state_dict(weights, output, metadata={"model_type": "sovits", "model_version": version})


def convert_bert(source: Path, output: Path) -> None:
    print(f"Converting Chinese RoBERTa checkpoint: {source}")
    raw = load_state_dict(source)
    weights: dict[str, torch.Tensor] = {}

    for key, value in raw.items():
        if key == "bert.embeddings.position_ids":
            continue
        if key.startswith("bert.embeddings."):
            weights[key.removeprefix("bert.")] = value
            continue
        if key.startswith("bert.encoder.layer."):
            layer = int(key.split(".")[3])
            if layer < 22:
                weights[key.removeprefix("bert.")] = value

    required = "encoder.layer.21.output.LayerNorm.bias"
    if required not in weights:
        raise KeyError(f"BERT checkpoint is missing {required}")
    save_state_dict(weights, output)


def convert_hubert(source: Path, output: Path) -> None:
    print(f"Converting Chinese HuBERT checkpoint: {source}")
    raw = load_state_dict(source)
    weights = {
        key: value
        for key, value in raw.items()
        if key != "masked_spec_embed"
        and not key.startswith("encoder.pos_conv_embed.conv.weight_")
    }

    weight_g = raw["encoder.pos_conv_embed.conv.weight_g"]
    weight_v = raw["encoder.pos_conv_embed.conv.weight_v"]
    norm = torch.linalg.vector_norm(weight_v.float(), dim=(0, 1), keepdim=True)
    weights["encoder.pos_conv_embed.conv.weight"] = (
        weight_g.float() * weight_v.float() / norm.clamp_min(1e-12)
    )

    required = "encoder.layers.11.final_layer_norm.bias"
    if required not in weights:
        raise KeyError(f"HuBERT checkpoint is missing {required}")
    save_state_dict(weights, output)


def resolve_source(
    explicit: Path | None,
    source_dir: Path | None,
    relative_path: str,
    repo_id: str,
) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit

    if source_dir is not None:
        local = source_dir / relative_path
        if local.is_file():
            return local

    from huggingface_hub import hf_hub_download

    print(f"Downloading {repo_id}/{relative_path}")
    return Path(hf_hub_download(repo_id=repo_id, filename=relative_path))


def convert_unless_present(
    source: Path,
    output: Path,
    force: bool,
    converter: Callable[[Path, Path], None],
) -> None:
    if output.is_file() and not force:
        print(f"Keeping existing {output}; pass --force to replace it")
        return
    converter(source, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and convert the GPT-SoVITS v2 models for gpt-sovits-rs."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Existing GPT_SoVITS/pretrained_models directory; missing files are downloaded.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--gpt-checkpoint", type=Path)
    parser.add_argument("--sovits-checkpoint", type=Path)
    parser.add_argument("--bert-checkpoint", type=Path)
    parser.add_argument("--bert-tokenizer", type=Path)
    parser.add_argument("--hubert-checkpoint", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "Allow torch pickle loading for trusted legacy GPT-SoVITS "
            "checkpoints that cannot be read with weights_only=True."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {
        "gpt": resolve_source(
            args.gpt_checkpoint,
            args.source_dir,
            DEFAULT_FILES["gpt"],
            args.repo_id,
        ),
        "sovits": resolve_source(
            args.sovits_checkpoint,
            args.source_dir,
            DEFAULT_FILES["sovits"],
            args.repo_id,
        ),
        "bert": resolve_source(
            args.bert_checkpoint,
            args.source_dir,
            DEFAULT_FILES["bert"],
            args.repo_id,
        ),
        "tokenizer": resolve_source(
            args.bert_tokenizer,
            args.source_dir,
            DEFAULT_FILES["tokenizer"],
            args.repo_id,
        ),
        "hubert": resolve_source(
            args.hubert_checkpoint,
            args.source_dir,
            DEFAULT_FILES["hubert"],
            args.repo_id,
        ),
    }

    output_dir = args.output_dir
    convert_unless_present(
        sources["gpt"],
        output_dir / "gpt-model.safetensors",
        args.force,
        lambda source, output: convert_gpt(
            source,
            output,
            allow_unsafe_pickle=args.allow_unsafe_pickle,
        ),
    )
    convert_unless_present(
        sources["sovits"],
        output_dir / "sovits-model.safetensors",
        args.force,
        lambda source, output: convert_sovits(
            source,
            output,
            allow_unsafe_pickle=args.allow_unsafe_pickle,
        ),
    )
    convert_unless_present(
        sources["bert"],
        output_dir / "bert" / "bert.safetensors",
        args.force,
        convert_bert,
    )
    tokenizer_output = output_dir / "bert" / "tokenizer.json"
    if args.force or not tokenizer_output.is_file():
        tokenizer_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sources["tokenizer"], tokenizer_output)
        print(f"Copied tokenizer to {tokenizer_output}")
    else:
        print(f"Keeping existing {tokenizer_output}; pass --force to replace it")
    convert_unless_present(
        sources["hubert"],
        output_dir / "hubert" / "hubert.safetensors",
        args.force,
        convert_hubert,
    )

    print("\nModels are ready:")
    for path in (
        output_dir / "gpt-model.safetensors",
        output_dir / "sovits-model.safetensors",
        output_dir / "bert" / "bert.safetensors",
        tokenizer_output,
        output_dir / "hubert" / "hubert.safetensors",
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
