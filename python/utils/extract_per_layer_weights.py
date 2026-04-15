"""Extract Gemma 4 per-layer weights from model.safetensors into standalone .npy files.

Usage:
    python extract_per_layer_weights.py \
        --input_path /path/to/gemma-4-E2B-it \
        --output_path /path/to/axmodel_dir

Produces:
    {output_path}/embed_tokens_per_layer.weight.npy   (~4.4 GB, bfloat16)
    {output_path}/per_layer_model_projection.weight.npy (~26 MB, float32)
    {output_path}/per_layer_projection_norm.weight.npy  (<1 KB, float32)
"""
import argparse
from pathlib import Path

import numpy as np
from safetensors import safe_open


EXTRACTED_KEYS = [
    "embed_tokens_per_layer.weight",
    "per_layer_model_projection.weight",
    "per_layer_projection_norm.weight",
]


def _infer_prefix(safe_file) -> str:
    key = next((k for k in safe_file.keys() if k.endswith("embed_tokens.weight")), None)
    if key is None:
        return "model.language_model."
    return key[: -len("embed_tokens.weight")]


def extract(input_path: str, output_path: str):
    model_safetensors = Path(input_path) / "model.safetensors"
    if not model_safetensors.exists():
        raise FileNotFoundError(f"model.safetensors not found in {input_path}")

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use PyTorch framework to handle bfloat16 tensors, then convert to numpy
    with safe_open(str(model_safetensors), framework="pt") as f:
        prefix = _infer_prefix(f)
        for short_key in EXTRACTED_KEYS:
            full_key = prefix + short_key
            out_file = out_dir / f"{short_key}.npy"
            tensor = f.get_tensor(full_key)
            # Large embedding: save as bfloat16 via ml_dtypes to keep file small
            # Projection/norm: save as float32 for direct use
            if "embed_tokens_per_layer" in short_key:
                data = tensor.float().numpy().astype(np.float16)
            else:
                data = tensor.float().numpy()
            np.save(str(out_file), data)
            size_mb = out_file.stat().st_size / 1024 / 1024
            print(f"  {full_key} -> {out_file.name} ({size_mb:.1f} MB, {data.dtype})")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Gemma 4 per-layer weights")
    parser.add_argument("--input_path", required=True, help="HF model directory with model.safetensors")
    parser.add_argument("--output_path", required=True, help="axmodel output directory")
    args = parser.parse_args()
    extract(args.input_path, args.output_path)
