import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from onnx.shape_inference import infer_shapes
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel
from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder


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


def _load_audio_modules(model_path: Path):
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    if hasattr(config.audio_config, "_attn_implementation"):
        config.audio_config._attn_implementation = "eager"

    audio_tower = Gemma4AudioModel(config.audio_config)
    projector = Gemma4MultimodalEmbedder(config.audio_config, config.text_config)

    audio_state = {}
    projector_state = {}
    with safe_open(str(model_path / "model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("model.audio_tower."):
                audio_state[key[len("model.audio_tower."):]] = handle.get_tensor(key)
            elif key.startswith("model.embed_audio."):
                projector_state[key[len("model.embed_audio."):]] = handle.get_tensor(key)

    missing, unexpected = audio_tower.load_state_dict(audio_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Failed loading Gemma 4 audio tower. missing={missing}, unexpected={unexpected}")
    missing, unexpected = projector.load_state_dict(projector_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Failed loading Gemma 4 audio projector. missing={missing}, unexpected={unexpected}")

    audio_tower.eval()
    projector.eval()
    return config, audio_tower.to(dtype=torch.float32), projector.to(dtype=torch.float32)


class ExportFriendlyRMSNorm(nn.Module):
    def __init__(self, reference_module: nn.Module):
        super().__init__()
        self.eps = reference_module.eps
        if reference_module.with_scale:
            self.weight = nn.Parameter(reference_module.weight.detach().clone(), requires_grad=False)
        else:
            dim = reference_module._parameters.get("weight", None)
            dim = None if dim is None else dim.shape[0]
            self._deferred_ones = dim is None
            if not self._deferred_ones:
                self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = hidden_states.float() * torch.rsqrt(hidden_states.float().pow(2).mean(-1, keepdim=True) + self.eps)
        if hasattr(self, "_deferred_ones") and self._deferred_ones:
            self.weight = nn.Parameter(
                torch.ones(hidden_states.shape[-1], dtype=torch.float32, device=hidden_states.device),
                requires_grad=False,
            )
            self._deferred_ones = False
        normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


def _replace_rmsnorm_modules(module: nn.Module):
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "Gemma4RMSNorm":
            setattr(module, name, ExportFriendlyRMSNorm(child))
        else:
            _replace_rmsnorm_modules(child)


class ExportFriendlyAudioAttention(nn.Module):
    def __init__(self, reference_module: nn.Module, seq_length: int):
        super().__init__()
        self.config = reference_module.config
        self.layer_idx = reference_module.layer_idx
        self.attention_logits_soft_cap = reference_module.attention_logits_soft_cap
        self.head_dim = reference_module.head_dim
        self.num_heads = reference_module.num_heads

        self.q_scale = reference_module.q_scale
        self.k_scale = reference_module.k_scale

        self.chunk_size = reference_module.chunk_size
        self.max_past_horizon = reference_module.max_past_horizon
        self.max_future_horizon = reference_module.max_future_horizon
        self.context_size = reference_module.context_size

        self.q_proj = reference_module.q_proj
        self.k_proj = reference_module.k_proj
        self.v_proj = reference_module.v_proj
        self.post = reference_module.post
        self.relative_k_proj = reference_module.relative_k_proj
        self.per_dim_scale = reference_module.per_dim_scale
        self.register_buffer("softcap", reference_module.softcap.detach().clone(), persistent=False)

        num_blocks = (seq_length + self.chunk_size - 1) // self.chunk_size
        gather_indices = []
        for block_idx in range(num_blocks):
            block_start = block_idx * self.chunk_size
            gather_indices.extend(range(block_start, block_start + self.context_size))
        self.register_buffer(
            "context_gather_indices",
            torch.tensor(gather_indices, dtype=torch.long),
            persistent=False,
        )
        self.num_blocks = num_blocks

    def _convert_to_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, 0, 0, pad))
        return hidden_states.reshape(batch_size, num_blocks, self.chunk_size, num_heads, head_dim).contiguous()

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_heads, head_dim = hidden_states.shape
        hidden_states = torch.nn.functional.pad(
            hidden_states,
            (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1),
        )
        gathered = hidden_states.index_select(1, self.context_gather_indices.to(hidden_states.device))
        return gathered.reshape(batch_size, self.num_blocks, self.context_size, num_heads, head_dim).contiguous()

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, num_blocks, block_size, position_length = x.shape
        context_size = self.context_size
        x = torch.nn.functional.pad(x, (0, context_size + 1 - position_length))
        x = x.view(batch_size, num_heads, num_blocks, block_size * (context_size + 1))
        x = x[..., : block_size * context_size]
        return x.view(batch_size, num_heads, num_blocks, block_size, context_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).float().view(hidden_shape)
        key_states = self.k_proj(hidden_states).float().view(hidden_shape)
        value_states = self.v_proj(hidden_states).float().view(hidden_shape)

        query_states = query_states * self.q_scale * torch.nn.functional.softplus(self.per_dim_scale)
        key_states = key_states * self.k_scale

        query_states = self._convert_to_block(query_states)
        key_states = self._extract_block_context(key_states)
        value_states = self._extract_block_context(value_states)
        num_blocks = query_states.shape[1]

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(-1, self.num_heads, self.head_dim)
        relative_key_states = relative_key_states.to(dtype=query_states.dtype)

        queries = query_states.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)

        queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd
        attn_weights = attn_weights / self.softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.softcap

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.logical_not(), self.config.attention_invalid_logits_value
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, num_blocks * self.chunk_size, -1)
        attn_output = attn_output[:, :seq_length].contiguous()
        attn_output = self.post(attn_output.to(dtype=self.post.linear.weight.dtype))
        return attn_output, attn_weights


def _replace_audio_attention_modules(module: nn.Module, seq_length: int):
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "Gemma4AudioAttention":
            setattr(module, name, ExportFriendlyAudioAttention(child, seq_length))
        else:
            _replace_audio_attention_modules(child, seq_length)


def _num_mel_frames(audio_duration_sec: float, sampling_rate: int, frame_length: int, hop_length: int) -> int:
    num_samples = int(round(audio_duration_sec * sampling_rate))
    frame_size_for_unfold = frame_length + 1
    pad_left = frame_length // 2
    num_mel_frames = (num_samples + pad_left - frame_size_for_unfold) // hop_length + 1
    if num_mel_frames <= 0:
        raise ValueError(f"audio_duration_sec={audio_duration_sec} is too short for Gemma 4 audio framing.")
    return int(num_mel_frames)


def _num_audio_tokens(num_mel_frames: int) -> int:
    tokens = int(num_mel_frames)
    for _ in range(2):
        tokens = (tokens + 2 - 3) // 2 + 1
    return int(tokens)


class AudioModelWrapper(nn.Module):
    def __init__(self, audio_tower: nn.Module, projector: nn.Module, input_features_mask: torch.Tensor, seq_length: int):
        super().__init__()
        self.audio_tower = audio_tower
        self.projector = projector
        self.register_buffer("input_features_mask", input_features_mask.to(dtype=torch.bool), persistent=False)
        self.register_buffer(
            "blocked_attention_mask",
            self._build_blocked_attention_mask(
                seq_length=seq_length,
                chunk_size=audio_tower.config.attention_chunk_size,
                max_past_horizon=audio_tower.config.attention_context_left - 1,
                max_future_horizon=audio_tower.config.attention_context_right,
            ),
            persistent=False,
        )

    @staticmethod
    def _build_blocked_attention_mask(
        seq_length: int,
        chunk_size: int,
        max_past_horizon: int,
        max_future_horizon: int,
    ) -> torch.Tensor:
        context_size = chunk_size + max_past_horizon + max_future_horizon
        num_blocks = (seq_length + chunk_size - 1) // chunk_size
        blocked = torch.zeros((1, 1, num_blocks, chunk_size, context_size), dtype=torch.bool)

        for block_idx in range(num_blocks):
            block_start = block_idx * chunk_size
            context_start = block_start - max_past_horizon
            for query_offset in range(chunk_size):
                query_idx = block_start + query_offset
                if query_idx >= seq_length:
                    continue
                min_kv = query_idx - max_past_horizon
                max_kv = query_idx + max_future_horizon
                for ctx_offset in range(context_size):
                    kv_idx = context_start + ctx_offset
                    if 0 <= kv_idx < seq_length and min_kv <= kv_idx <= max_kv:
                        blocked[0, 0, block_idx, query_offset, ctx_offset] = True

        return blocked

    def forward(self, input_features: torch.Tensor):
        hidden_states, _ = self.audio_tower.subsample_conv_projection(input_features, self.input_features_mask)
        position_embeddings = self.audio_tower.rel_pos_enc(hidden_states)
        attention_mask = self.blocked_attention_mask.to(device=hidden_states.device)

        for encoder_layer in self.audio_tower.layers[: self.audio_tower.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.audio_tower.output_proj(hidden_states)
        return self.projector(hidden_states)


def verify_audio_onnx_output(
    onnx_path: Path,
    sample_input_features: np.ndarray,
    expected_tokens: int,
    hidden_size: int,
):
    providers = ["CPUExecutionProvider"]
    available = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(str(onnx_path), providers=providers)
    outputs = session.run(None, {"input_features": sample_input_features})
    output_shapes = [tuple(int(v) for v in out.shape) for out in outputs]
    has_expected = any(tuple(out.shape) == (1, expected_tokens, hidden_size) for out in outputs)
    if not has_expected:
        raise RuntimeError(
            "Exported ONNX does not contain projected audio token output. "
            f"expected_shape={(1, expected_tokens, hidden_size)}, output_shapes={output_shapes}"
        )
    print(f"ONNX output verification passed. output_shapes={output_shapes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Gemma 4 audio tower + projector to ONNX")
    parser.add_argument("-m", "--model", type=str, default="../gemma-4-hf-original/gemma-4-E2B-it",
                        help="Path to the original Gemma 4 model directory")
    parser.add_argument("-o", "--onnx_save_dir", type=str, default="./audio-models",
                        help="Directory used for ONNX outputs")
    parser.add_argument("--audio_duration_sec", type=float, default=30.0,
                        help="Fixed audio duration in seconds")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--optimize", action="store_true",
                        help="Run onnxsim / onnxslim after export")
    args = parser.parse_args()

    model_path = Path(args.model)
    onnx_save_dir = Path(args.onnx_save_dir)
    onnx_save_dir.mkdir(parents=True, exist_ok=True)

    config, audio_tower, projector = _load_audio_modules(model_path)
    _replace_rmsnorm_modules(audio_tower)
    _replace_rmsnorm_modules(projector)

    processor_config = json.loads((model_path / "processor_config.json").read_text(encoding="utf-8"))
    feature_cfg = processor_config["feature_extractor"]
    sampling_rate = int(feature_cfg["sampling_rate"])
    feature_size = int(feature_cfg["feature_size"])
    frame_length = int(feature_cfg["frame_length"])
    hop_length = int(feature_cfg["hop_length"])

    num_mel_frames = _num_mel_frames(args.audio_duration_sec, sampling_rate, frame_length, hop_length)
    expected_tokens = _num_audio_tokens(num_mel_frames)
    _replace_audio_attention_modules(audio_tower, expected_tokens)

    onnx_stem = f"gemma4_audio_{int(args.audio_duration_sec)}s"
    onnx_path = onnx_save_dir / f"{onnx_stem}.onnx"
    meta_path = onnx_save_dir / f"{onnx_stem}.json"

    input_features_mask = torch.ones((1, num_mel_frames), dtype=torch.bool)
    wrapper = AudioModelWrapper(
        audio_tower,
        projector,
        input_features_mask=input_features_mask,
        seq_length=expected_tokens,
    ).eval()

    sample_input_features = torch.randn(1, num_mel_frames, feature_size, dtype=torch.float32)
    with torch.no_grad():
        reference = wrapper(sample_input_features).cpu().numpy()
    expected_hidden = config.text_config.hidden_size
    expected_shape = (1, expected_tokens, expected_hidden)
    if tuple(reference.shape) != expected_shape:
        raise RuntimeError(f"Unexpected PyTorch output shape: {reference.shape}, expected {expected_shape}")

    torch.onnx.export(
        wrapper,
        sample_input_features,
        str(onnx_path),
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
        verbose=False,
        input_names=["input_features"],
        output_names=["audio_embeds"],
    )

    verify_audio_onnx_output(
        onnx_path,
        sample_input_features.cpu().numpy().astype(np.float32),
        expected_tokens=expected_tokens,
        hidden_size=expected_hidden,
    )

    if args.optimize:
        _onnx_simplify(onnx_path)
        _onnx_slim(onnx_path)
        verify_audio_onnx_output(
            onnx_path,
            sample_input_features.cpu().numpy().astype(np.float32),
            expected_tokens=expected_tokens,
            hidden_size=expected_hidden,
        )

    meta = {
        "model_type": "gemma4_audio_projector",
        "audio_duration_sec": args.audio_duration_sec,
        "sampling_rate": sampling_rate,
        "feature_size": feature_size,
        "frame_length": frame_length,
        "hop_length": hop_length,
        "num_mel_frames": num_mel_frames,
        "num_audio_tokens": expected_tokens,
        "hidden_size": expected_hidden,
        "onnx_path": onnx_path.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {onnx_path}")
    print(f"saved {meta_path}")
