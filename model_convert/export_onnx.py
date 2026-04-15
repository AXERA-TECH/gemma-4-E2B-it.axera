import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx.shape_inference import infer_shapes
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.gemma4_multimodal import DEFAULT_MAX_SOFT_TOKENS  # noqa: E402
from utils.gemma4_multimodal import make_image_position_ids  # noqa: E402
from utils.gemma4_multimodal import resolve_resize  # noqa: E402


def _onnx_simplify(onnx_path: Path):
    try:
        from onnxsim import simplify
    except Exception:
        return

    model = onnx.load(str(onnx_path))
    model = infer_shapes(model)
    model_simp, check = simplify(model)
    if not check:
        raise RuntimeError(f"Failed to simplify ONNX: {onnx_path}")
    onnx.save(model_simp, str(onnx_path))


def _onnx_slim(onnx_path: Path):
    if shutil.which("onnxslim") is None:
        return
    subprocess.run(
        ["onnxslim", str(onnx_path), str(onnx_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def _load_vision_modules(model_path: Path):
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    if hasattr(config.vision_config, "_attn_implementation"):
        config.vision_config._attn_implementation = "eager"

    vision_tower = Gemma4VisionModel(config.vision_config)
    projector = Gemma4MultimodalEmbedder(config.vision_config, config.text_config)

    vision_state = {}
    projector_state = {}
    with safe_open(str(model_path / "model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("model.vision_tower."):
                vision_state[key[len("model.vision_tower."):]] = handle.get_tensor(key)
            elif key.startswith("model.embed_vision."):
                projector_state[key[len("model.embed_vision."):]] = handle.get_tensor(key)

    missing, unexpected = vision_tower.load_state_dict(vision_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Failed loading Gemma 4 vision tower. missing={missing}, unexpected={unexpected}")
    missing, unexpected = projector.load_state_dict(projector_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Failed loading Gemma 4 projector. missing={missing}, unexpected={unexpected}")

    vision_tower.eval()
    projector.eval()
    return config, vision_tower.to(dtype=torch.float32), projector.to(dtype=torch.float32)


class ExportFriendlyRMSNorm(nn.Module):
    def __init__(self, reference_module: nn.Module):
        super().__init__()
        self.eps = reference_module.eps
        # Always provide a weight so the ONNX structure is identical to scaled RMSNorm.
        # pulsar2 can then fuse it into a single AxRMSNormalization op.
        # For with_scale=False, use all-ones (mathematically equivalent to no scale).
        if reference_module.with_scale:
            self.weight = nn.Parameter(reference_module.weight.detach().clone(), requires_grad=False)
        else:
            dim = reference_module._parameters.get("weight", None)
            if dim is not None:
                dim = dim.shape[0]
            else:
                # Infer dimension from the module's eps context; caller must ensure correct dim
                dim = None
            # Defer dimension detection to forward if unknown
            self._deferred_ones = dim is None
            if not self._deferred_ones:
                self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = hidden_states.float() * torch.rsqrt(hidden_states.float().pow(2).mean(-1, keepdim=True) + self.eps)
        if hasattr(self, "_deferred_ones") and self._deferred_ones:
            self.weight = nn.Parameter(torch.ones(hidden_states.shape[-1], dtype=torch.float32, device=hidden_states.device), requires_grad=False)
            self._deferred_ones = False
        normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


def _replace_rmsnorm_modules(module: nn.Module):
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "Gemma4RMSNorm":
            setattr(module, name, ExportFriendlyRMSNorm(child))
        else:
            _replace_rmsnorm_modules(child)


class VisionModelWrapper(nn.Module):
    def __init__(self, vision_tower: nn.Module, projector: nn.Module, image_position_ids: torch.LongTensor):
        super().__init__()
        self.rotary_emb = vision_tower.encoder.rotary_emb
        self.layers = vision_tower.encoder.layers
        self.pooler = vision_tower.pooler
        self.projector = projector
        self.register_buffer("image_position_ids", image_position_ids, persistent=False)
        patch_count = image_position_ids.shape[1]
        self.output_length = patch_count // 9
        self.register_buffer("padding_positions", torch.zeros((1, patch_count), dtype=torch.bool), persistent=False)
        self.register_buffer("attention_mask", torch.zeros((1, 1, patch_count, patch_count), dtype=torch.float32), persistent=False)
        self.register_buffer("input_proj_weight", vision_tower.patch_embedder.input_proj.weight.detach().to(torch.float32), persistent=False)
        fixed_position_embeddings = vision_tower.patch_embedder._position_embeddings(
            image_position_ids,
            self.padding_positions,
        ).detach().to(torch.float32)
        self.register_buffer("position_embeddings", fixed_position_embeddings, persistent=False)

    def forward(self, pixel_values: torch.Tensor):
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = F.linear(pixel_values[0].to(torch.float32), self.input_proj_weight)
        hidden_states = hidden_states.unsqueeze(0)
        hidden_states = hidden_states + self.position_embeddings
        position_embeddings = self.rotary_emb(hidden_states, self.image_position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=self.attention_mask,
                position_embeddings=position_embeddings,
                position_ids=self.image_position_ids,
            )
        hidden_states, pooler_mask = self.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=self.image_position_ids,
            padding_positions=self.padding_positions,
            output_length=self.output_length,
        )
        hidden_states = hidden_states[pooler_mask]
        return self.projector(hidden_states)


def _seq_len_from_output(output: np.ndarray):
    if output.ndim < 2:
        return None
    if output.ndim == 2:
        return int(output.shape[0])
    return int(output.shape[-2])


def verify_vit_onnx_output(onnx_path: Path, sample_pixel_values: np.ndarray, expected_tokens: int, hidden_size: int):
    providers = ["CPUExecutionProvider"]
    available = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(str(onnx_path), providers=providers)
    outputs = session.run(None, {"pixel_values": sample_pixel_values})
    output_shapes = [tuple(int(v) for v in out.shape) for out in outputs]
    has_expected = any(
        _seq_len_from_output(out) == int(expected_tokens) and out.shape[-1] == int(hidden_size)
        for out in outputs
        if out.ndim >= 2
    )
    if not has_expected:
        raise RuntimeError(
            "Exported ONNX does not contain projected token output. "
            f"expected_tokens={expected_tokens}, hidden_size={hidden_size}, output_shapes={output_shapes}"
        )
    print(f"ONNX output verification passed. output_shapes={output_shapes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Gemma 4 vision tower + projector to ONNX")
    parser.add_argument("-m", "--model", type=str, default="../python/gemma-4-E2B-it",
                        help="Path to the original Gemma 4 model directory")
    parser.add_argument("-o", "--onnx_save_dir", type=str, default="./vit-models",
                        help="Directory used for ONNX outputs")
    parser.add_argument("--max_soft_tokens", type=int, default=DEFAULT_MAX_SOFT_TOKENS,
                        help="Fixed number of projected image soft tokens")
    parser.add_argument("--resize_h", type=int, default=None,
                        help="Fixed input image height")
    parser.add_argument("--resize_w", type=int, default=None,
                        help="Fixed input image width")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--optimize", action="store_true",
                        help="Run onnxsim / onnxslim after export")
    args = parser.parse_args()

    model_path = Path(args.model)
    onnx_save_dir = Path(args.onnx_save_dir)
    onnx_save_dir.mkdir(parents=True, exist_ok=True)

    resize_h, resize_w, expected_tokens = resolve_resize(
        max_soft_tokens=args.max_soft_tokens,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
    )
    patch_size = 16
    patch_count = (resize_h // patch_size) * (resize_w // patch_size)
    pixel_dim = 3 * patch_size * patch_size
    onnx_stem = f"gemma4_vision_h{resize_h}_w{resize_w}_t{expected_tokens}"
    onnx_path = onnx_save_dir / f"{onnx_stem}.onnx"
    meta_path = onnx_save_dir / f"{onnx_stem}.json"

    config, vision_tower, projector = _load_vision_modules(model_path)
    _replace_rmsnorm_modules(vision_tower)
    _replace_rmsnorm_modules(projector)
    image_position_ids = make_image_position_ids(resize_h=resize_h, resize_w=resize_w, patch_size=patch_size)
    wrapper = VisionModelWrapper(vision_tower, projector, image_position_ids=image_position_ids).eval()

    sample_pixel_values = torch.randn(1, patch_count, pixel_dim, dtype=torch.float32)
    with torch.no_grad():
        reference = wrapper(sample_pixel_values).cpu().numpy()
    expected_hidden = config.text_config.hidden_size
    valid_reference_shapes = {
        (expected_tokens, expected_hidden),
        (1, expected_tokens, expected_hidden),
    }
    if tuple(reference.shape) not in valid_reference_shapes:
        raise RuntimeError(
            f"Unexpected PyTorch output shape: {reference.shape}, expected "
            f"one of {sorted(valid_reference_shapes)}"
        )

    torch.onnx.export(
        wrapper,
        sample_pixel_values,
        str(onnx_path),
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
        verbose=False,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
    )

    verify_vit_onnx_output(
        onnx_path,
        sample_pixel_values.cpu().numpy().astype(np.float32),
        expected_tokens=expected_tokens,
        hidden_size=config.text_config.hidden_size,
    )

    if args.optimize:
        _onnx_simplify(onnx_path)
        _onnx_slim(onnx_path)
        verify_vit_onnx_output(
            onnx_path,
            sample_pixel_values.cpu().numpy().astype(np.float32),
            expected_tokens=expected_tokens,
            hidden_size=config.text_config.hidden_size,
        )

    meta = {
        "model_type": "gemma4_vision_projector",
        "resize_h": resize_h,
        "resize_w": resize_w,
        "patch_size": patch_size,
        "patch_count": patch_count,
        "pixel_dim": pixel_dim,
        "expected_tokens": expected_tokens,
        "hidden_size": config.text_config.hidden_size,
        "onnx_path": str(onnx_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
