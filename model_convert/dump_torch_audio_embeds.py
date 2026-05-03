"""Extract torch-reference audio_embeds (1, 750, 1536) for a wav.

Mirrors AudioModelWrapper.forward() from model_convert/export_audio_onnx.py,
but keeps the un-optimised torch path (eager attention, standard RMSNorm).
Result can be compared to ONNX / axmodel outputs and fed directly into the
on-board LLM axmodel (bypassing audio encoder) to isolate audio export error
vs LLM axmodel precision.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel
from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder


def load_audio_modules(model_path: Path):
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    if hasattr(config.audio_config, "_attn_implementation"):
        config.audio_config._attn_implementation = "eager"

    audio_tower = Gemma4AudioModel(config.audio_config)
    projector = Gemma4MultimodalEmbedder(config.audio_config, config.text_config)

    audio_state, projector_state = {}, {}
    with safe_open(str(model_path / "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("model.audio_tower."):
                audio_state[key[len("model.audio_tower."):]] = f.get_tensor(key)
            elif key.startswith("model.embed_audio."):
                projector_state[key[len("model.embed_audio."):]] = f.get_tensor(key)

    miss, unex = audio_tower.load_state_dict(audio_state, strict=False)
    if miss or unex:
        raise RuntimeError(f"audio_tower missing={miss} unexpected={unex}")
    miss, unex = projector.load_state_dict(projector_state, strict=False)
    if miss or unex:
        raise RuntimeError(f"projector missing={miss} unexpected={unex}")

    return config, audio_tower.to(dtype=torch.float32).eval(), projector.to(dtype=torch.float32).eval()


def load_audio_feature_extractor(model_path: Path):
    processor_config = json.loads((model_path / "processor_config.json").read_text(encoding="utf-8"))
    feature_cfg = processor_config["feature_extractor"].copy()
    feature_cfg.pop("feature_extractor_type", None)
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

    return Gemma4AudioFeatureExtractor(**feature_cfg)


def build_blocked_attention_mask(seq_length: int, chunk_size: int, max_past_horizon: int, max_future_horizon: int):
    context_size = chunk_size + max_past_horizon + max_future_horizon
    num_blocks = (seq_length + chunk_size - 1) // chunk_size
    blocked = torch.zeros((1, 1, num_blocks, chunk_size, context_size), dtype=torch.bool)
    for block_idx in range(num_blocks):
        block_start = block_idx * chunk_size
        context_start = block_start - max_past_horizon
        for q_off in range(chunk_size):
            q_idx = block_start + q_off
            if q_idx >= seq_length:
                continue
            min_kv, max_kv = q_idx - max_past_horizon, q_idx + max_future_horizon
            for ctx_off in range(context_size):
                kv_idx = context_start + ctx_off
                if 0 <= kv_idx < seq_length and min_kv <= kv_idx <= max_kv:
                    blocked[0, 0, block_idx, q_off, ctx_off] = True
    return blocked


@torch.no_grad()
def run_audio_pipeline(audio_tower, projector, input_features: torch.Tensor, input_features_mask: torch.Tensor):
    cfg = audio_tower.config
    hidden_states, _ = audio_tower.subsample_conv_projection(input_features, input_features_mask)
    position_embeddings = audio_tower.rel_pos_enc(hidden_states)
    seq_length = hidden_states.shape[1]
    attention_mask = build_blocked_attention_mask(
        seq_length=seq_length,
        chunk_size=cfg.attention_chunk_size,
        max_past_horizon=cfg.attention_context_left - 1,
        max_future_horizon=cfg.attention_context_right,
    ).to(hidden_states.device)
    for layer in audio_tower.layers[: cfg.num_hidden_layers]:
        hidden_states = layer(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)
    hidden_states = audio_tower.output_proj(hidden_states)
    return projector(hidden_states)


def load_waveform_16k_mono(wav_path: Path) -> np.ndarray:
    y, sr = sf.read(str(wav_path), dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"expected 16kHz wav, got {sr}")
    return y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF gemma-4 model dir")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--duration_sec", type=float, default=30.0)
    ap.add_argument("--out", required=True, help=".npy path for (1, 750, 1536) audio_embeds")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compare_onnx", default="", help="optional onnx path for sanity cosine check")
    args = ap.parse_args()

    target_samples = int(args.duration_sec * 16000)
    y = load_waveform_16k_mono(Path(args.wav))
    if len(y) >= target_samples:
        y = y[:target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)))

    extractor = load_audio_feature_extractor(Path(args.model))
    mel, _ = extractor._extract_spectrogram(y[None, :], np.ones(target_samples, dtype=np.int64))
    input_features = torch.from_numpy(mel.astype(np.float32)[None, ...]).to(args.device)
    num_mel_frames = input_features.shape[1]
    input_features_mask = torch.ones((1, num_mel_frames), dtype=torch.bool, device=args.device)
    print(f"input_features: {tuple(input_features.shape)}, range=[{input_features.min():.3f}, {input_features.max():.3f}]")

    config, audio_tower, projector = load_audio_modules(Path(args.model))
    audio_tower = audio_tower.to(args.device)
    projector = projector.to(args.device)

    audio_embeds = run_audio_pipeline(audio_tower, projector, input_features, input_features_mask)
    audio_embeds = audio_embeds.cpu().numpy().astype(np.float32)
    print(f"audio_embeds: shape={audio_embeds.shape}, dtype={audio_embeds.dtype}, range=[{audio_embeds.min():.3f}, {audio_embeds.max():.3f}]")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, audio_embeds)
    print(f"saved torch audio_embeds -> {args.out}")

    if args.compare_onnx:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(args.compare_onnx, providers=providers)
        onnx_out = sess.run(None, {"input_features": input_features.cpu().numpy()})[0]
        a = audio_embeds.reshape(-1)
        b = onnx_out.reshape(-1)
        cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        diff = np.abs(audio_embeds - onnx_out)
        print(f"torch vs ONNX  cos={cos:.6f}  mean_abs_diff={diff.mean():.4f}  max_abs_diff={diff.max():.4f}")


if __name__ == "__main__":
    main()
