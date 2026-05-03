import argparse
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from utils.gemma4_compat import load_text_runtime_config
from utils.gemma4_compat import load_tokenizer
from utils.gemma4_multimodal import DEFAULT_MAX_SOFT_TOKENS
from utils.gemma4_multimodal import prepare_audio_inputs
from utils.gemma4_multimodal import build_messages
from utils.gemma4_multimodal import detect_soft_tokens_from_vit_path
from utils.gemma4_multimodal import load_image
from utils.gemma4_multimodal import load_processor
from utils.gemma4_multimodal import prepare_multimodal_inputs
from utils.gemma4_multimodal import replace_audio_tokens
from utils.gemma4_multimodal import replace_image_tokens
from utils.gemma4_multimodal import resolve_resize
from utils.gemma4_multimodal import to_numpy_fp32
from utils.gemma4_per_layer import Gemma4PerLayerInputs
from utils.infer_func import InferManager
from utils.infer_func import detect_prefill_len
from utils.infer_func import release_ax_inference_session
from utils.vision_output import describe_output_shapes
from utils.vision_output import select_vit_output


def _default_hf_model() -> str:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "gemma-4-E2B-it",
        script_dir / "gemma_4_e2b_it_tokenizer",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _default_axmodel_path() -> str:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "gemma-4-E2B-it_axmodel",
        script_dir / "gemma_4_e2b_it_ax650n_axmodel",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _default_vit_model_path() -> str:
    script_dir = Path(__file__).resolve().parent
    resize_h, resize_w, expected_tokens = resolve_resize(DEFAULT_MAX_SOFT_TOKENS)
    stem = f"gemma4_vision_h{resize_h}_w{resize_w}_t{expected_tokens}"
    candidates = [
        script_dir / "vit_models" / f"{stem}.axmodel",
        script_dir / "vit_models" / f"{stem}.onnx",
        script_dir.parent / "model_convert" / "compiled_output" / f"{stem}.axmodel",
        script_dir.parent / "model_convert" / "vit-models" / f"{stem}.onnx",
        script_dir.parent / "model_convert" / "compiled_output" / "compiled.axmodel",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _default_audio_profile() -> tuple[str, float, int]:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        (script_dir / "audio_models" / "gemma4_audio_30s.axmodel", 30.0, 750),
        (script_dir / "audio_models" / "gemma4_audio_5s.axmodel", 5.0, 125),
        (script_dir.parent / "model_convert" / "compiled_output_audio_30s" / "compiled.axmodel", 30.0, 750),
        (script_dir.parent / "model_convert" / "compiled_output_audio_5s" / "compiled.axmodel", 5.0, 125),
        (Path("/tmp/compiled_output_audio_30s/compiled.axmodel"), 30.0, 750),
        (Path("/tmp/compiled_output_audio_5s/compiled.axmodel"), 5.0, 125),
    ]
    for candidate, duration_sec, audio_tokens in candidates:
        if candidate.exists():
            return str(candidate), duration_sec, audio_tokens
    return str(candidates[0][0]), candidates[0][1], candidates[0][2]


def _infer_audio_profile_from_path(audio_model_path: str) -> tuple[float | None, int | None]:
    path_str = str(audio_model_path)
    if "audio_5s" in path_str or "compiled_output_audio_5s" in path_str:
        return 5.0, 125
    if "audio_30s" in path_str or "compiled_output_audio_30s" in path_str:
        return 30.0, 750
    return None, None



def _run_vit_axmodel(vit_model_path: str, pixel_values: np.ndarray, target_hidden_size: int, expected_tokens: int):
    from axengine import InferenceSession

    session = InferenceSession(vit_model_path)
    try:
        outputs = session.run(None, {"pixel_values": pixel_values})
        if isinstance(outputs, dict):
            outputs = [np.array(value, copy=True) for value in outputs.values()]
        else:
            outputs = [np.array(value, copy=True) for value in outputs]
    finally:
        release_ax_inference_session(session)
    return (
        select_vit_output(outputs, target_hidden_size, expected_tokens=expected_tokens),
        describe_output_shapes(outputs),
    )


def _run_vit_onnx(vit_model_path: str, pixel_values: np.ndarray, target_hidden_size: int, expected_tokens: int):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(vit_model_path, providers=providers)
    outputs = session.run(None, {"pixel_values": pixel_values})
    return (
        select_vit_output(outputs, target_hidden_size, expected_tokens=expected_tokens),
        describe_output_shapes(outputs),
    )


def _run_audio_axmodel(
    audio_model_path: str,
    input_features: np.ndarray,
    target_hidden_size: int,
    expected_tokens: int,
):
    from axengine import InferenceSession

    session = InferenceSession(audio_model_path)
    try:
        outputs = session.run(None, {"input_features": input_features})
        if isinstance(outputs, dict):
            outputs = [np.array(value, copy=True) for value in outputs.values()]
        else:
            outputs = [np.array(value, copy=True) for value in outputs]
    finally:
        release_ax_inference_session(session)
    return (
        select_vit_output(outputs, target_hidden_size, expected_tokens=expected_tokens),
        describe_output_shapes(outputs),
    )


def _run_audio_onnx(
    audio_model_path: str,
    input_features: np.ndarray,
    target_hidden_size: int,
    expected_tokens: int,
):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(audio_model_path, providers=providers)
    outputs = session.run(None, {"input_features": input_features})
    return (
        select_vit_output(outputs, target_hidden_size, expected_tokens=expected_tokens),
        describe_output_shapes(outputs),
    )


def _text_prefill_inputs(tokenizer, embeds: np.ndarray, prompt: str, system_prompt: str = "", enable_thinking: bool = False):
    messages = build_messages(prompt=prompt, system_prompt=system_prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="np")
    token_ids = inputs["input_ids"][0].tolist()
    prefill_data = np.take(embeds, token_ids, axis=0).astype(bfloat16)
    return token_ids, prefill_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 4 E2B AX model inference")
    parser.add_argument("--hf_model", type=str, default=_default_hf_model(),
                        help="Path to Gemma 4 tokenizer/config directory")
    parser.add_argument("--axmodel_path", type=str, default=_default_axmodel_path(),
                        help="Path to compiled LLM axmodel folder")
    parser.add_argument("--vit_model_path", type=str, default=_default_vit_model_path(),
                        help="Path to Gemma 4 vision ONNX model or .axmodel")
    default_audio_model_path, default_audio_duration_sec, default_audio_tokens = _default_audio_profile()
    parser.add_argument("--audio_model_path", type=str, default=default_audio_model_path,
                        help="Path to Gemma 4 audio ONNX model or .axmodel")
    parser.add_argument("--image_path", type=str, default="",
                        help="Optional input image path. If omitted, runs text-only generation.")
    parser.add_argument("--audio_path", type=str, default="",
                        help="Optional input audio path. Audio-only inference uses the fixed-duration audio encoder path.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.",
                        help="Optional system prompt")
    parser.add_argument("--prompt", type=str, default="Describe the image in detail.",
                        help="Input prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Render the thinking-aware chat template")
    parser.add_argument("--max_soft_tokens", type=int, default=DEFAULT_MAX_SOFT_TOKENS,
                        help="Fixed number of image soft tokens. Default 70 keeps the image block within one 128-token prefill slice.")
    parser.add_argument("--resize_h", type=int, default=None,
                        help="Optional fixed image height for the vision encoder")
    parser.add_argument("--resize_w", type=int, default=None,
                        help="Optional fixed image width for the vision encoder")
    parser.add_argument("--slice_len", type=int, default=None,
                        help="Prefill slice length. Must match the LLM `--prefill_len` used at build time. "
                        "When omitted, auto-detected from axmodel filenames under --axmodel_path.")
    parser.add_argument("--audio_duration_sec", type=float, default=None,
                        help="Fixed audio duration expected by the audio encoder model. "
                        "When omitted, infer it from --audio_model_path.")
    parser.add_argument("--audio_tokens", type=int, default=None,
                        help="Fixed number of audio soft tokens produced by the audio encoder model. "
                        "When omitted, infer it from --audio_model_path.")
    parser.add_argument("--audio_embeds_npy", type=str, default="",
                        help="Optional path to a pre-computed (1, audio_tokens, hidden_size) or "
                        "(audio_tokens, hidden_size) float32 npy. When set, the audio encoder step "
                        "(axmodel / onnx) is bypassed and these embeds are used directly for prefill. "
                        "Useful for A/B testing torch-reference audio embeds against the on-device audio axmodel.")
    args = parser.parse_args()

    if args.slice_len is None:
        args.slice_len = detect_prefill_len(args.axmodel_path, default=128)
        print(f"[INFO] Auto-detected slice_len={args.slice_len} from {args.axmodel_path}")

    if args.audio_duration_sec is None or args.audio_tokens is None:
        inferred_duration_sec, inferred_audio_tokens = _infer_audio_profile_from_path(args.audio_model_path)
        if args.audio_duration_sec is None:
            args.audio_duration_sec = inferred_duration_sec or default_audio_duration_sec
        if args.audio_tokens is None:
            args.audio_tokens = inferred_audio_tokens or default_audio_tokens

    if args.image_path and args.audio_path:
        # Gemma4 natively supports image+audio in the same prompt; this script's prefill
        # path currently only wires up a single modality's prepare_* helper. Lifting this
        # requires extending prepare_* to produce a combined inputs dict with both
        # pixel_values and input_features, and running both encoders before replacing
        # tokens. Left as future work.
        raise ValueError("Simultaneous image+audio inputs are not supported by this demo script yet.")

    config = load_text_runtime_config(args.hf_model)
    embeds = np.load(os.path.join(args.axmodel_path, "model.embed_tokens.weight.npy"))
    per_layer_helper = None
    if int(getattr(config, "hidden_size_per_layer_input", 0) or 0) > 0:
        per_layer_helper = Gemma4PerLayerInputs(args.axmodel_path, config)

    # Auto-detect max_soft_tokens from VIT model filename if not explicitly set
    if args.image_path and args.max_soft_tokens == DEFAULT_MAX_SOFT_TOKENS:
        detected = detect_soft_tokens_from_vit_path(args.vit_model_path)
        if detected is not None and detected != args.max_soft_tokens:
            print(f"[INFO] Auto-detected max_soft_tokens={detected} from VIT model: {args.vit_model_path}")
            args.max_soft_tokens = detected

    mm_token_type_ids = None
    prefill_per_layer_inputs = None
    if args.image_path:
        processor = load_processor(args.hf_model)
        tokenizer = processor.tokenizer
        image = load_image(args.image_path)
        mm = prepare_multimodal_inputs(
            processor,
            image=image,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            enable_thinking=args.enable_thinking,
            max_soft_tokens=args.max_soft_tokens,
            resize_h=args.resize_h,
            resize_w=args.resize_w,
        )
        inputs = mm["inputs"]
        token_ids = inputs["input_ids"][0].cpu().numpy().tolist()
        mm_token_type_ids = inputs["mm_token_type_ids"][0].cpu().numpy().tolist()
        pixel_values = to_numpy_fp32(inputs["pixel_values"])

        if not os.path.exists(args.vit_model_path):
            raise FileNotFoundError(
                f"Vision model not found: {args.vit_model_path}. "
                "Please export the Gemma 4 vision ONNX and compile it first."
            )

        if args.vit_model_path.endswith(".axmodel"):
            image_embeds, vit_output_shapes = _run_vit_axmodel(
                args.vit_model_path,
                pixel_values,
                target_hidden_size=config.hidden_size,
                expected_tokens=mm["expected_tokens"],
            )
        else:
            image_embeds, vit_output_shapes = _run_vit_onnx(
                args.vit_model_path,
                pixel_values,
                target_hidden_size=config.hidden_size,
                expected_tokens=mm["expected_tokens"],
            )

        if image_embeds.ndim == 3:
            image_embeds = image_embeds[0]
        if image_embeds.shape[0] != mm["expected_tokens"]:
            raise ValueError(
                "Unexpected vision output token count. "
                f"got={image_embeds.shape[0]}, expected={mm['expected_tokens']}, "
                f"vit_output_shapes={vit_output_shapes}"
            )

        prefill_data = np.take(embeds, token_ids, axis=0)
        prefill_data = replace_image_tokens(
            token_ids,
            prefill_data,
            image_embeds,
            image_token_id=config.image_token_id,
        ).astype(bfloat16)
    elif args.audio_path:
        processor = load_processor(args.hf_model)
        tokenizer = processor.tokenizer
        mm = prepare_audio_inputs(
            processor,
            audio_path=args.audio_path,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            enable_thinking=args.enable_thinking,
            audio_duration_sec=args.audio_duration_sec,
            fixed_audio_tokens=args.audio_tokens,
        )
        inputs = mm["inputs"]
        token_ids = inputs["input_ids"][0].cpu().numpy().tolist()
        mm_token_type_ids = inputs["mm_token_type_ids"][0].cpu().numpy().tolist()
        input_features = to_numpy_fp32(inputs["input_features"])

        if args.audio_tokens > args.slice_len:
            print(
                f"[WARN] audio_tokens={args.audio_tokens} exceeds slice_len={args.slice_len}. "
                "Cross-slice audio blocks are supported, but the current chunked prefill path does not provide "
                "full future-token visibility to earlier slices within the same multimodal block."
            )

        if args.audio_embeds_npy:
            audio_embeds = np.load(args.audio_embeds_npy).astype(np.float32)
            audio_output_shapes = [tuple(int(v) for v in audio_embeds.shape)]
            print(f"[INFO] loaded precomputed audio_embeds from {args.audio_embeds_npy}, shape={audio_embeds.shape}")
        elif not os.path.exists(args.audio_model_path):
            raise FileNotFoundError(
                f"Audio model not found: {args.audio_model_path}. "
                "Please export and compile the Gemma 4 audio encoder first, "
                "or pass --audio_embeds_npy to bypass the audio encoder."
            )
        elif args.audio_model_path.endswith(".axmodel"):
            audio_embeds, audio_output_shapes = _run_audio_axmodel(
                args.audio_model_path,
                input_features,
                target_hidden_size=config.hidden_size,
                expected_tokens=args.audio_tokens,
            )
        else:
            audio_embeds, audio_output_shapes = _run_audio_onnx(
                args.audio_model_path,
                input_features,
                target_hidden_size=config.hidden_size,
                expected_tokens=args.audio_tokens,
            )

        if audio_embeds.ndim == 3:
            audio_embeds = audio_embeds[0]
        if audio_embeds.shape[0] != args.audio_tokens:
            raise ValueError(
                "Unexpected audio output token count. "
                f"got={audio_embeds.shape[0]}, expected={args.audio_tokens}, "
                f"audio_output_shapes={audio_output_shapes}"
            )

        prefill_data = np.take(embeds, token_ids, axis=0)
        prefill_data = replace_audio_tokens(
            token_ids,
            prefill_data,
            audio_embeds,
            audio_token_id=config.audio_token_id,
        ).astype(bfloat16)
    else:
        tokenizer = load_tokenizer(args.hf_model)
        token_ids, prefill_data = _text_prefill_inputs(
            tokenizer,
            embeds,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            enable_thinking=args.enable_thinking,
        )

    if per_layer_helper is not None:
        # Matches torch Gemma4TextModel.forward():
        #   inputs_embeds *= sqrt(hidden_size)  (scale ALL including image features)
        #   per_layer_inputs = project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        # Token IDs: pad_token_id at image positions (for embed lookup)
        # Embeddings: ALL scaled by sqrt(hidden_size) (text + image features)
        per_layer_source_embeds = np.asarray(prefill_data, dtype=np.float32)
        per_layer_token_ids = list(token_ids)
        scale = float(config.hidden_size**0.5)
        pad_token_id = int(getattr(config, "pad_token_id", 0) or 0)
        for i in range(len(token_ids)):
            per_layer_source_embeds[i] *= scale
            if mm_token_type_ids is not None and int(mm_token_type_ids[i]) != 0:
                per_layer_token_ids[i] = pad_token_id
        prefill_per_layer_inputs = per_layer_helper.compute(per_layer_token_ids, per_layer_source_embeds)

    eos_token_id = config.eos_token_id if isinstance(config.eos_token_id, list) else None

    kv_cache_len = int(getattr(config, "kv_cache_len", 2047) or 2047)
    imer = InferManager(config, args.axmodel_path, max_seq_len=kv_cache_len, per_layer_helper=per_layer_helper)
    try:
        token_ids = imer.prefill(
            tokenizer,
            token_ids,
            prefill_data,
            mm_token_type_ids=mm_token_type_ids,
            slice_len=args.slice_len,
            per_layer_inputs=prefill_per_layer_inputs,
        )
        imer.decode(
            tokenizer,
            token_ids,
            embeds,
            slice_len=args.slice_len,
            eos_token_id=eos_token_id,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        imer.close()
    print("\n")
