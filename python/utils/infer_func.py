import atexit
import os
import re

import numpy as np
from axengine import InferenceSession
from ml_dtypes import bfloat16
from tqdm import tqdm


def release_ax_inference_session(session):
    inner = getattr(session, "_sess", None)
    unload = getattr(inner, "_unload", None)
    if not callable(unload):
        return

    try:
        unload()
    except Exception as exc:
        print(f"[WARN] Failed to unload axengine session cleanly: {exc}")
    finally:
        try:
            inner._unload = lambda: None
        except Exception:
            pass


def _layer_head_dim(config, layer_idx: int) -> int:
    if getattr(config, "layer_types", None) and layer_idx < len(config.layer_types):
        if config.layer_types[layer_idx] == "full_attention":
            return int(getattr(config, "global_head_dim", config.head_dim or (config.hidden_size // config.num_attention_heads)))
    return int(config.head_dim or (config.hidden_size // config.num_attention_heads))


def _build_shared_kv_source_layers(config):
    num_layers = int(config.num_hidden_layers)
    source_layers = [None] * num_layers
    layer_types = getattr(config, "layer_types", None)
    num_shared_layers = int(getattr(config, "num_kv_shared_layers", 0) or 0)
    if not layer_types or num_shared_layers <= 0:
        return source_layers

    first_shared_layer = num_layers - num_shared_layers
    if first_shared_layer <= 0:
        return source_layers

    prev_layers = list(layer_types[:first_shared_layer])
    for layer_idx in range(first_shared_layer, min(num_layers, len(layer_types))):
        layer_type = layer_types[layer_idx]
        source_layers[layer_idx] = len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
    return source_layers


def detect_prefill_len(model_dir: str, default: int = 128) -> int:
    """Auto-detect prefill_len (aka slice_len) from axmodel filenames.

    Matches files named like ``<prefix>_p<N>_l<idx>_together.axmodel`` and returns
    ``N``. Falls back to ``default`` when no layer files match.
    """
    layer_pattern = re.compile(r"^.*_p(?P<prefill>\d+)_l\d+_together\.axmodel$")
    prefill_counts = {}
    try:
        for fname in os.listdir(model_dir):
            match = layer_pattern.match(fname)
            if match:
                prefill = int(match.group("prefill"))
                prefill_counts[prefill] = prefill_counts.get(prefill, 0) + 1
    except FileNotFoundError:
        return default

    if not prefill_counts:
        return default
    return max(prefill_counts.items(), key=lambda kv: kv[1])[0]


def _find_axmodel_files(base_dir: str, expected_layers: int = None, expected_prefill: int = 128):
    files = os.listdir(base_dir)
    layer_pattern = re.compile(r"^(?P<prefix>.*)_p(?P<prefill>\d+)_l(?P<idx>\d+)_together\.axmodel$")
    post_pattern = re.compile(r"^(?P<prefix>.*)_post\.axmodel$")

    prefix_map = {}
    for fname in files:
        match = layer_pattern.match(fname)
        if match:
            prefix = match.group("prefix")
            idx = int(match.group("idx"))
            prefix_map.setdefault(prefix, []).append((idx, fname))

    if not prefix_map:
        prefix = "gemma3_text"
        layer_files = [(i, f"{prefix}_p{expected_prefill}_l{i}_together.axmodel") for i in range(expected_layers or 0)]
    else:
        prefix = max(prefix_map.items(), key=lambda kv: len(kv[1]))[0]
        print(f"Detected prefixes: {list(prefix_map.keys())}, chosen: {prefix}, layers: {len(prefix_map[prefix])}")
        layer_files = sorted(prefix_map[prefix], key=lambda it: it[0])

    post_file = None
    for fname in files:
        match = post_pattern.match(fname)
        if match and match.group("prefix") == prefix:
            post_file = fname
            break
    if post_file is None:
        candidate = os.path.join(base_dir, f"{prefix}_post.axmodel")
        if os.path.exists(candidate):
            post_file = f"{prefix}_post.axmodel"
        else:
            for fname in files:
                if fname.endswith("_post.axmodel"):
                    post_file = fname
                    break

    return layer_files, post_file, prefix


class InferManager:
    def __init__(self, config, model_dir, max_seq_len=2047, per_layer_helper=None):
        self.config = config
        self.max_seq_len = int(max_seq_len)
        self.per_layer_helper = per_layer_helper
        self.hidden_size_per_layer_input = int(getattr(config, "hidden_size_per_layer_input", 0) or 0)
        self.hidden_size = int(config.hidden_size)
        self.text_embed_scale = float(self.hidden_size**0.5)
        self.external_input_scaling = "gemma4" in str(getattr(config, "model_type", ""))

        rope_scaling = getattr(config, "rope_scaling", None)
        self.use_mrope = rope_scaling is not None and isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling

        self.layer_head_dims = [_layer_head_dim(config, idx) for idx in range(config.num_hidden_layers)]
        self.layer_kv_dims = [head_dim * config.num_key_value_heads for head_dim in self.layer_head_dims]
        self.shared_kv_source_layers = _build_shared_kv_source_layers(config)

        layer_files, post_file, prefix = _find_axmodel_files(model_dir, config.num_hidden_layers)

        self.decoder_sessions = []
        for _, fname in tqdm(layer_files, desc="Init InferenceSession"):
            session = InferenceSession(os.path.join(model_dir, fname))
            self.decoder_sessions.append(session)

        self.decode_cache_lens = [self._decode_cache_len(session) for session in self.decoder_sessions]
        self.cache_len = max(self.decode_cache_lens, default=self.max_seq_len + 1)
        self.k_caches = [np.zeros((1, self.cache_len, kv_dim), dtype=bfloat16) for kv_dim in self.layer_kv_dims]
        self.v_caches = [np.zeros((1, self.cache_len, kv_dim), dtype=bfloat16) for kv_dim in self.layer_kv_dims]

        if post_file is None:
            raise FileNotFoundError("Cannot find post process .axmodel file in model_dir")
        self.post_process_session = InferenceSession(os.path.join(model_dir, post_file))
        self._closed = False
        atexit.register(self.close)
        print("Model loaded successfully!")

    def close(self):
        if self._closed:
            return

        sessions = list(getattr(self, "decoder_sessions", []))
        post_process_session = getattr(self, "post_process_session", None)
        if post_process_session is not None:
            sessions.append(post_process_session)

        for session in sessions:
            release_ax_inference_session(session)

        self.decoder_sessions = []
        self.post_process_session = None
        self._closed = True

    @staticmethod
    def _compute_mm_group_ids(mm_token_type_ids):
        if mm_token_type_ids is None:
            return None

        mm_token_type_ids = np.asarray(mm_token_type_ids, dtype=np.int32).reshape(-1)
        is_multimodal = np.isin(mm_token_type_ids, (1, 2, 3))
        prev_is_multimodal = np.roll(is_multimodal, 1)
        prev_is_multimodal[0] = False
        new_group_starts = is_multimodal & ~prev_is_multimodal
        group_ids = np.cumsum(new_group_starts.astype(np.int32)) - 1
        group_ids[~is_multimodal] = -1
        return group_ids

    @staticmethod
    def _validate_mm_groups_fit_single_slice(mm_group_ids, slice_len: int):
        if mm_group_ids is None:
            return

        valid_group_ids = np.unique(mm_group_ids[mm_group_ids >= 0])
        for group_id in valid_group_ids:
            positions = np.flatnonzero(mm_group_ids == group_id)
            if positions.size == 0:
                continue
            start = int(positions[0])
            end = int(positions[-1])
            if start // slice_len != end // slice_len:
                num_slices = end // slice_len - start // slice_len + 1
                print(
                    f"[WARN] Multimodal token block (group_id={group_id}, pos {start}-{end}) "
                    f"spans {num_slices} prefill slices. Bidirectional attention within "
                    f"earlier slices is partial (chunked prefill limitation)."
                )

    @staticmethod
    def _apply_mm_bidirectional_mask(mask, mm_group_ids, slice_idx: int, slice_len: int, seq_len: int):
        if mm_group_ids is None:
            return

        ctx_len = min(seq_len, slice_len * (slice_idx + 1))
        query_start = slice_idx * slice_len
        query_end = min(query_start + slice_len, seq_len)
        if query_start >= query_end:
            return

        visible_group_ids = mm_group_ids[:ctx_len]
        for local_row, query_pos in enumerate(range(query_start, query_end)):
            group_id = int(mm_group_ids[query_pos])
            if group_id < 0:
                continue
            kv_positions = np.flatnonzero(visible_group_ids == group_id)
            if kv_positions.size == 0:
                continue
            mask[0, local_row, kv_positions] = 0

    @staticmethod
    def _session_output_names(session):
        try:
            return tuple(output.name for output in session.get_outputs())
        except Exception:
            return ()

    @staticmethod
    def _session_input_names(session):
        try:
            return tuple(input_meta.name for input_meta in session.get_inputs())
        except Exception:
            return ()

    @staticmethod
    def _session_input_shapes(session):
        try:
            return {input_meta.name: tuple(input_meta.shape) for input_meta in session.get_inputs()}
        except Exception:
            return {}

    def _decode_cache_len(self, session):
        input_shapes = self._session_input_shapes(session)
        k_shape = input_shapes.get("K_cache")
        if k_shape is not None and len(k_shape) >= 2 and k_shape[1] is not None:
            return int(k_shape[1])
        return self.max_seq_len + 1

    @staticmethod
    def _match_cache_len(cache: np.ndarray, expected_len: int) -> np.ndarray:
        if cache.shape[1] == expected_len:
            return cache
        if cache.shape[1] > expected_len:
            return cache[:, :expected_len, :]

        padded = np.zeros((cache.shape[0], expected_len, cache.shape[2]), dtype=cache.dtype)
        padded[:, : cache.shape[1], :] = cache
        return padded

    def _decoder_output_names(self, session, shape_group: int):
        available_names = self._session_output_names(session)
        base_names = ("K_cache_out", "V_cache_out", "output")
        if shape_group == 0:
            return base_names

        grouped_names = (
            f"K_cache_out_{shape_group}",
            f"V_cache_out_{shape_group}",
            f"output_{shape_group}",
        )
        if all(name in available_names for name in grouped_names):
            return grouped_names
        return base_names

    def _decoder_input_name_map(self, session, shape_group: int):
        available_names = set(self._session_input_names(session))
        logical_names = ["K_cache", "V_cache", "indices", "input", "mask"]
        if self.hidden_size_per_layer_input:
            logical_names.append("per_layer_input")

        mapped_names = {}
        for logical_name in logical_names:
            grouped_name = f"{logical_name}_{shape_group}" if shape_group != 0 else logical_name
            if grouped_name in available_names:
                mapped_names[logical_name] = grouped_name
            elif logical_name in available_names:
                mapped_names[logical_name] = logical_name
            elif not available_names:
                mapped_names[logical_name] = grouped_name

        return mapped_names

    def _prepare_decoder_input(self, session, input_feed, shape_group: int):
        name_map = self._decoder_input_name_map(session, shape_group)
        if self.hidden_size_per_layer_input and "per_layer_input" not in name_map:
            raise RuntimeError("Decoder axmodel is missing `per_layer_input`; please rebuild Gemma 4 text axmodels.")
        return {name_map[key]: value for key, value in input_feed.items() if key in name_map}

    def _run_decoder(self, session, input_feed, shape_group: int):
        names = self._decoder_output_names(session, shape_group)
        outputs = None

        try:
            outputs = session.run(list(names), input_feed, shape_group=shape_group)
        except TypeError:
            try:
                outputs = session.run(list(names), input_feed, shape_group)
            except TypeError:
                outputs = session.run(None, input_feed, shape_group=shape_group)

        if isinstance(outputs, dict):
            return outputs[names[0]], outputs[names[1]], outputs[names[2]]

        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 3:
                return outputs[0], outputs[1], outputs[2]
            offset = shape_group * 3
            if len(outputs) >= offset + 3:
                return outputs[offset], outputs[offset + 1], outputs[offset + 2]
            return outputs[0], outputs[1], outputs[2]

        return outputs[0], outputs[1], outputs[2]

    @staticmethod
    def _top_p(probs: np.ndarray, p: float) -> np.ndarray:
        sorted_indices = np.argsort(probs)
        filtered = probs.copy()
        cumulative = 0
        for idx in sorted_indices[::-1]:
            if cumulative >= p:
                filtered[idx] = 0
            cumulative += filtered[idx]
        return filtered / cumulative

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return (exp_logits / np.sum(exp_logits)).astype(np.float64)

    def post_process(
        self,
        logits,
        top_k=1,
        top_p=0.9,
        temperature=0.6,
        repetition_penalty=1.0,
        token_ids=None,
    ):
        logits = logits.astype(np.float32).flatten()
        if repetition_penalty is not None and repetition_penalty != 1.0 and token_ids:
            for t in set(token_ids):
                if 0 <= t < logits.size:
                    if logits[t] < 0:
                        logits[t] *= repetition_penalty
                    else:
                        logits[t] /= repetition_penalty

        top_k = max(1, min(int(top_k), logits.size))
        temperature = max(float(temperature), 1e-6)
        top_p = min(max(float(top_p), 1e-6), 1.0)

        candidate_indices = np.argpartition(logits, -top_k)[-top_k:]
        candidate_logits = logits[candidate_indices] / temperature
        candidate_probs = self._softmax(candidate_logits)
        candidate_probs = self._top_p(candidate_probs, top_p)
        candidate_probs = candidate_probs.astype(np.float64) / candidate_probs.sum()
        chosen_idx = np.random.multinomial(1, candidate_probs).argmax()
        next_token = candidate_indices[chosen_idx]
        return next_token, candidate_indices, candidate_probs

    def gen_slice_indices(self, token_len, prefill=128, expand=128):
        remaining = max(0, token_len - prefill)
        extra_blocks = (remaining + expand - 1) // expand
        return list(range(extra_blocks + 1))

    def _get_prefill_per_layer_input(self, per_layer_inputs, slice_idx: int, slice_len: int, layer_idx: int, remain_len: int):
        if per_layer_inputs is None:
            return None

        layer_input = np.zeros((1, slice_len, self.hidden_size_per_layer_input), dtype=bfloat16)
        start = slice_idx * slice_len
        layer_slice = per_layer_inputs[start : start + remain_len, layer_idx, :]
        layer_input[:, :remain_len, :] = np.asarray(layer_slice, dtype=np.float32).reshape(1, remain_len, -1).astype(bfloat16)
        return layer_input

    def _get_decode_per_layer_input(self, token_id: int, embed_matrix, layer_idx: int):
        if self.per_layer_helper is None:
            return None
        scaled_embed = np.asarray(embed_matrix[int(token_id), :], dtype=np.float32) * self.text_embed_scale
        per_layer_input = self.per_layer_helper.decode_input(int(token_id), scaled_embed)
        return np.asarray(per_layer_input[layer_idx], dtype=np.float32).reshape(1, 1, -1).astype(bfloat16)

    @staticmethod
    def _is_text_token(mm_token_type_ids, token_pos: int) -> bool:
        if mm_token_type_ids is None:
            return True
        token_type = int(mm_token_type_ids[token_pos])
        return token_type not in (1, 2, 3)

    @staticmethod
    def _build_decode_mask(cache_len: int, visible_past_tokens: int):
        mask = np.full((1, 1, cache_len + 1), -65536, dtype=np.float32)
        if visible_past_tokens > 0:
            mask[:, :, :visible_past_tokens] = 0
        mask[:, :, cache_len] = 0
        return mask.astype(bfloat16)

    @staticmethod
    def _build_shared_decode_cache(cache: np.ndarray, expected_len: int, past_tokens: int, current_index: int):
        shared = np.zeros((cache.shape[0], expected_len, cache.shape[2]), dtype=cache.dtype)
        visible_past = min(max(past_tokens, 0), max(expected_len - 1, 0))
        if visible_past > 0:
            shared[:, :visible_past, :] = cache[:, :visible_past, :]
        if expected_len > 0 and current_index >= 0:
            shared[:, expected_len - 1 : expected_len, :] = cache[:, current_index : current_index + 1, :]
        return shared

    def prefill(
        self,
        tokenizer,
        token_ids,
        embed_data,
        mm_token_type_ids=None,
        slice_len=128,
        top_k=1,
        top_p=0.9,
        temperature=0.6,
        repetition_penalty=1.0,
        per_layer_inputs=None,
    ):
        seq_len = len(token_ids)
        if self.hidden_size_per_layer_input and per_layer_inputs is None:
            raise RuntimeError("Gemma 4 runtime requires `per_layer_inputs` during prefill.")

        mm_group_ids = self._compute_mm_group_ids(mm_token_type_ids)
        self._validate_mm_groups_fit_single_slice(mm_group_ids, slice_len)
        slice_indices = [i for i in range(seq_len // slice_len + 1)]
        print(f"slice_indices: {slice_indices}")
        total_prefill_len = slice_len * (slice_indices[-1] + 1)

        if total_prefill_len > 0:
            for slice_idx in slice_indices:
                base_indices = np.arange(slice_idx * slice_len, (slice_idx + 1) * slice_len, dtype=np.uint32)
                if self.use_mrope:
                    indices = np.tile(base_indices, (3, 1))
                else:
                    indices = base_indices.reshape(1, -1)

                mask = np.zeros((1, slice_len, slice_len * (slice_idx + 1)), dtype=np.float32) - 65536
                data = np.zeros((1, slice_len, self.config.hidden_size), dtype=bfloat16)
                for i, token_pos in enumerate(range(slice_idx * slice_len, (slice_idx + 1) * slice_len)):
                    if token_pos < seq_len:
                        mask[:, i, : slice_idx * slice_len + i + 1] = 0
                        token_embed = np.asarray(embed_data[token_pos], dtype=np.float32)
                        if self.external_input_scaling and self._is_text_token(mm_token_type_ids, token_pos):
                            token_embed = token_embed * self.text_embed_scale
                        data[:, i : i + 1, :] = token_embed.reshape((1, 1, self.config.hidden_size)).astype(bfloat16)

                self._apply_mm_bidirectional_mask(mask, mm_group_ids, slice_idx, slice_len, seq_len)
                remain_len = seq_len - slice_idx * slice_len if slice_idx == slice_indices[-1] else slice_len
                mask = mask.astype(bfloat16)
                latest_k_out = [None] * self.config.num_hidden_layers
                latest_v_out = [None] * self.config.num_hidden_layers

                for layer_idx in range(self.config.num_hidden_layers):
                    source_layer_idx = self.shared_kv_source_layers[layer_idx]
                    if source_layer_idx is None:
                        k_cache = (
                            self.k_caches[layer_idx][:, : slice_len * slice_idx, :]
                            if slice_idx
                            else np.zeros((1, 1, self.config.hidden_size), dtype=bfloat16)
                        )
                        v_cache = (
                            self.v_caches[layer_idx][:, : slice_len * slice_idx, :]
                            if slice_idx
                            else np.zeros((1, 1, self.config.hidden_size), dtype=bfloat16)
                        )
                    else:
                        if latest_k_out[source_layer_idx] is None or latest_v_out[source_layer_idx] is None:
                            raise RuntimeError(f"Shared-KV source layer {source_layer_idx} was not produced before layer {layer_idx}")
                        if slice_idx:
                            k_cache = np.concatenate(
                                [self.k_caches[source_layer_idx][:, : slice_len * slice_idx, :], latest_k_out[source_layer_idx]],
                                axis=1,
                            )
                            v_cache = np.concatenate(
                                [self.v_caches[source_layer_idx][:, : slice_len * slice_idx, :], latest_v_out[source_layer_idx]],
                                axis=1,
                            )
                        else:
                            k_cache = latest_k_out[source_layer_idx]
                            v_cache = latest_v_out[source_layer_idx]

                    input_feed = {
                        "K_cache": k_cache,
                        "V_cache": v_cache,
                        "indices": indices,
                        "input": data,
                        "mask": mask,
                    }
                    per_layer_input = self._get_prefill_per_layer_input(per_layer_inputs, slice_idx, slice_len, layer_idx, remain_len)
                    if per_layer_input is not None:
                        input_feed["per_layer_input"] = per_layer_input

                    input_feed = self._prepare_decoder_input(self.decoder_sessions[layer_idx], input_feed, shape_group=slice_idx + 1)
                    k_out, v_out, data = self._run_decoder(self.decoder_sessions[layer_idx], input_feed, shape_group=slice_idx + 1)
                    latest_k_out[layer_idx] = k_out
                    latest_v_out[layer_idx] = v_out
                    self.k_caches[layer_idx][:, slice_idx * slice_len : slice_idx * slice_len + remain_len, :] = k_out[:, :remain_len, :]
                    self.v_caches[layer_idx][:, slice_idx * slice_len : slice_idx * slice_len + remain_len, :] = v_out[:, :remain_len, :]

                print("Slice prefill done:", slice_idx)

            post_out = self.post_process_session.run(
                None,
                {"input": data[:, seq_len - (len(slice_indices) - 1) * slice_len - 1, None, :]},
            )[0]
            next_token, possible_tokens, possible_probs = self.post_process(
                post_out,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                token_ids=token_ids,
            )
            token_ids.append(next_token)
            return token_ids

    def decode(
        self,
        tokenizer,
        token_ids,
        embed_matrix,
        prefill_len=128,
        slice_len=128,
        eos_token_id=None,
        stream=True,
        top_k=1,
        top_p=0.9,
        temperature=0.6,
        repetition_penalty=1.0,
        max_new_tokens=None,
        stream_callback=None,
    ):
        if self.hidden_size_per_layer_input and self.per_layer_helper is None:
            raise RuntimeError("Gemma 4 runtime requires a per-layer helper during decode.")

        decoded_text = tokenizer.decode(token_ids[-1], skip_special_tokens=True)
        if stream:
            print("answer >>", decoded_text, end="", flush=True)
        if stream_callback is not None:
            stream_callback(decoded_text)

        seq_len = len(token_ids) - 1

        max_new_tokens = self.cache_len if max_new_tokens is None else int(max_new_tokens)
        generated = 0

        for step_idx in range(self.cache_len):
            if prefill_len > 0 and step_idx < seq_len:
                continue
            cur_token = token_ids[step_idx]
            indices = np.array([step_idx], np.uint32).reshape((1, 1))
            token_embed = np.asarray(embed_matrix[cur_token, :], dtype=np.float32)
            if self.external_input_scaling:
                token_embed = token_embed * self.text_embed_scale
            data = token_embed.reshape((1, 1, self.config.hidden_size)).astype(bfloat16)
            visible_past_tokens = step_idx
            mask = self._build_decode_mask(self.max_seq_len, visible_past_tokens)
            for layer_idx in range(self.config.num_hidden_layers):
                source_layer_idx = self.shared_kv_source_layers[layer_idx]
                decode_cache_len = self.decode_cache_lens[layer_idx]
                if source_layer_idx is None:
                    k_cache = self._match_cache_len(self.k_caches[layer_idx], decode_cache_len)
                    v_cache = self._match_cache_len(self.v_caches[layer_idx], decode_cache_len)
                else:
                    k_cache = self._build_shared_decode_cache(
                        self.k_caches[source_layer_idx],
                        decode_cache_len,
                        visible_past_tokens,
                        step_idx,
                    )
                    v_cache = self._build_shared_decode_cache(
                        self.v_caches[source_layer_idx],
                        decode_cache_len,
                        visible_past_tokens,
                        step_idx,
                    )
                input_feed = {
                    "K_cache": k_cache,
                    "V_cache": v_cache,
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                per_layer_input = self._get_decode_per_layer_input(cur_token, embed_matrix, layer_idx)
                if per_layer_input is not None:
                    input_feed["per_layer_input"] = per_layer_input

                input_feed = self._prepare_decoder_input(self.decoder_sessions[layer_idx], input_feed, shape_group=0)
                k_out, v_out, data = self._run_decoder(self.decoder_sessions[layer_idx], input_feed, shape_group=0)
                self.k_caches[layer_idx][:, step_idx : step_idx + 1, :] = k_out[:, :1, :]
                self.v_caches[layer_idx][:, step_idx : step_idx + 1, :] = v_out[:, :1, :]
            if step_idx < seq_len - 1:
                continue

            post_out = self.post_process_session.run(None, {"input": data})[0]
            next_token, possible_tokens, possible_probs = self.post_process(
                post_out,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                token_ids=token_ids,
            )
            if eos_token_id is not None and next_token in eos_token_id:
                break
            if next_token == tokenizer.eos_token_id:
                break
            token_ids.append(next_token)
            generated += 1
            if generated >= max_new_tokens:
                break

            decoded_piece = tokenizer.decode(next_token, skip_special_tokens=True)
            decoded_text += decoded_piece
            if stream:
                print(decoded_piece, end="", flush=True)
            if stream_callback is not None:
                stream_callback(decoded_text)

        return decoded_text

    def decode_stream(
        self,
        tokenizer,
        token_ids,
        embed_matrix,
        prefill_len=128,
        slice_len=128,
        eos_token_id=None,
        top_k=1,
        top_p=0.9,
        temperature=0.6,
        repetition_penalty=1.0,
        max_new_tokens=None,
    ):
        if self.hidden_size_per_layer_input and self.per_layer_helper is None:
            raise RuntimeError("Gemma 4 runtime requires a per-layer helper during decode.")

        decoded_text = tokenizer.decode(token_ids[-1], skip_special_tokens=True)
        yield decoded_text

        seq_len = len(token_ids) - 1

        max_new_tokens = self.cache_len if max_new_tokens is None else int(max_new_tokens)
        generated = 0

        for step_idx in range(self.cache_len):
            if prefill_len > 0 and step_idx < seq_len:
                continue
            cur_token = token_ids[step_idx]
            indices = np.array([step_idx], np.uint32).reshape((1, 1))
            token_embed = np.asarray(embed_matrix[cur_token, :], dtype=np.float32)
            if self.external_input_scaling:
                token_embed = token_embed * self.text_embed_scale
            data = token_embed.reshape((1, 1, self.config.hidden_size)).astype(bfloat16)
            visible_past_tokens = step_idx
            mask = self._build_decode_mask(self.max_seq_len, visible_past_tokens)
            for layer_idx in range(self.config.num_hidden_layers):
                source_layer_idx = self.shared_kv_source_layers[layer_idx]
                decode_cache_len = self.decode_cache_lens[layer_idx]
                if source_layer_idx is None:
                    k_cache = self._match_cache_len(self.k_caches[layer_idx], decode_cache_len)
                    v_cache = self._match_cache_len(self.v_caches[layer_idx], decode_cache_len)
                else:
                    k_cache = self._build_shared_decode_cache(
                        self.k_caches[source_layer_idx],
                        decode_cache_len,
                        visible_past_tokens,
                        step_idx,
                    )
                    v_cache = self._build_shared_decode_cache(
                        self.v_caches[source_layer_idx],
                        decode_cache_len,
                        visible_past_tokens,
                        step_idx,
                    )
                input_feed = {
                    "K_cache": k_cache,
                    "V_cache": v_cache,
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                per_layer_input = self._get_decode_per_layer_input(cur_token, embed_matrix, layer_idx)
                if per_layer_input is not None:
                    input_feed["per_layer_input"] = per_layer_input

                input_feed = self._prepare_decoder_input(self.decoder_sessions[layer_idx], input_feed, shape_group=0)
                k_out, v_out, data = self._run_decoder(self.decoder_sessions[layer_idx], input_feed, shape_group=0)
                self.k_caches[layer_idx][:, step_idx : step_idx + 1, :] = k_out[:, :1, :]
                self.v_caches[layer_idx][:, step_idx : step_idx + 1, :] = v_out[:, :1, :]
            if step_idx < seq_len - 1:
                continue

            post_out = self.post_process_session.run(None, {"input": data})[0]
            next_token, possible_tokens, possible_probs = self.post_process(
                post_out,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                token_ids=token_ids,
            )
            if eos_token_id is not None and next_token in eos_token_id:
                break
            if next_token == tokenizer.eos_token_id:
                break
            token_ids.append(next_token)
            generated += 1
            if generated >= max_new_tokens:
                break

            decoded_piece = tokenizer.decode(next_token, skip_special_tokens=True)
            decoded_text += decoded_piece
            yield decoded_text
