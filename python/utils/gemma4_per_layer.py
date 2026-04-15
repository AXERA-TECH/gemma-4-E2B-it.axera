from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


class Gemma4PerLayerInputs:
    """Matches torch Gemma4TextModel.project_per_layer_inputs():

        per_layer_embed  = embed_tokens_per_layer(input_ids) * sqrt(per_layer_dim)
        per_layer_proj   = per_layer_model_projection(inputs_embeds) * projection_scale
        per_layer_proj   = per_layer_projection_norm(per_layer_proj)
        per_layer_inputs = (per_layer_proj + per_layer_embed) * merge_scale

    IMPORTANT: For multimodal, torch uses pad_embedding (not VIT features) at
    image positions when computing per_layer_proj.  Callers must replace image
    positions in input_embeds with scaled pad_embedding before calling compute().
    """

    def __init__(self, axmodel_dir: str, config):
        self.axmodel_dir = Path(axmodel_dir)

        self.num_hidden_layers = int(config.num_hidden_layers)
        self.hidden_size = int(config.hidden_size)
        self.hidden_size_per_layer_input = int(getattr(config, "hidden_size_per_layer_input", 0) or 0)
        if self.hidden_size_per_layer_input <= 0:
            raise RuntimeError("Current config does not enable Gemma 4 per-layer inputs.")

        self.embed_scale = float(self.hidden_size_per_layer_input**0.5)
        self.projection_scale = float(self.hidden_size**-0.5)
        self.merge_scale = float(2.0**-0.5)
        self.rms_norm_eps = float(config.rms_norm_eps)
        self._decode_cache = {}

        embed_path = self.axmodel_dir / "embed_tokens_per_layer.weight.npy"
        proj_path = self.axmodel_dir / "per_layer_model_projection.weight.npy"
        norm_path = self.axmodel_dir / "per_layer_projection_norm.weight.npy"
        if not embed_path.exists():
            raise FileNotFoundError(
                f"{embed_path} not found. Run utils/extract_per_layer_weights.py first."
            )

        self._embed_mmap = np.load(str(embed_path), mmap_mode="r")
        self.vocab_size_per_layer_input = int(self._embed_mmap.shape[0])

        self.per_layer_model_projection = np.load(str(proj_path)).astype(np.float32)
        self.per_layer_projection_norm = np.load(str(norm_path)).astype(np.float32)

    def _gather_embed_rows(self, token_ids):
        rows = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id < 0 or token_id >= self.vocab_size_per_layer_input:
                raise IndexError(
                    f"Token id {token_id} out of range for per-layer embeddings "
                    f"(vocab_size={self.vocab_size_per_layer_input})"
                )
            rows.append(self._embed_mmap[token_id : token_id + 1].astype(np.float32))
        return np.concatenate(rows, axis=0) * self.embed_scale

    def _rms_norm(self, values: np.ndarray):
        values = np.asarray(values, dtype=np.float32)
        mean_squared = np.mean(np.square(values), axis=-1, keepdims=True) + self.rms_norm_eps
        normed = values * np.power(mean_squared, -0.5)
        return normed * self.per_layer_projection_norm.reshape(1, 1, -1)

    def compute(self, token_ids, input_embeds):
        token_ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        input_embeds = np.asarray(input_embeds, dtype=np.float32)
        if input_embeds.ndim == 3:
            if input_embeds.shape[0] != 1:
                raise ValueError(f"Expected batch=1 inputs_embeds, got shape={input_embeds.shape}")
            input_embeds = input_embeds[0]
        if input_embeds.ndim != 2:
            raise ValueError(f"Expected 2D inputs_embeds, got shape={input_embeds.shape}")
        if input_embeds.shape[0] != token_ids.shape[0]:
            raise ValueError(
                f"Token ids and inputs_embeds length mismatch: tokens={token_ids.shape[0]}, embeds={input_embeds.shape[0]}"
            )

        per_layer_embed = self._gather_embed_rows(token_ids).reshape(
            token_ids.shape[0],
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = (input_embeds @ self.per_layer_model_projection.T) * self.projection_scale
        per_layer_projection = per_layer_projection.reshape(
            token_ids.shape[0],
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self._rms_norm(per_layer_projection)
        merged = (per_layer_projection + per_layer_embed) * self.merge_scale
        return merged.astype(bfloat16)

    def decode_input(self, token_id: int, input_embed):
        token_id = int(token_id)
        if token_id not in self._decode_cache:
            merged = self.compute([token_id], np.asarray(input_embed, dtype=np.float32).reshape(1, -1))
            self._decode_cache[token_id] = merged[0]
        return self._decode_cache[token_id]
