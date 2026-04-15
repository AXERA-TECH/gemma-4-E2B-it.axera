import argparse
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from utils.gemma4_compat import load_text_runtime_config
from utils.gemma4_compat import load_tokenizer
from utils.gemma4_multimodal import DEFAULT_MAX_SOFT_TOKENS
from utils.gemma4_multimodal import build_messages
from utils.gemma4_multimodal import detect_soft_tokens_from_vit_path
from utils.gemma4_multimodal import load_image
from utils.gemma4_multimodal import load_processor
from utils.gemma4_multimodal import prepare_multimodal_inputs
from utils.gemma4_multimodal import replace_image_tokens
from utils.gemma4_multimodal import resolve_resize
from utils.gemma4_multimodal import to_numpy_fp32
from utils.gemma4_per_layer import Gemma4PerLayerInputs
from utils.infer_func import InferManager
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



def _run_vit_axmodel(vit_model_path: str, pixel_values: np.ndarray, target_hidden_size: int, expected_tokens: int):
    from axengine import InferenceSession

    session = InferenceSession(vit_model_path)
    outputs = session.run(None, {"pixel_values": pixel_values})
    if isinstance(outputs, dict):
        outputs = list(outputs.values())
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
    parser.add_argument("--image_path", type=str, default="",
                        help="Optional input image path. If omitted, runs text-only generation.")
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
    parser.add_argument("--slice_len", type=int, default=128,
                        help="Prefill slice length. Must match the LLM `--prefill_len` used at build time.")
    args = parser.parse_args()

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
    print("\n")
