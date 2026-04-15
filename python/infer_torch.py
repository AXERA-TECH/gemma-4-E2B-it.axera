import argparse
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM

from utils.gemma4_multimodal import DEFAULT_MAX_SOFT_TOKENS
from utils.gemma4_multimodal import build_messages
from utils.gemma4_multimodal import load_image
from utils.gemma4_multimodal import load_processor
from utils.gemma4_multimodal import prepare_multimodal_inputs


def _default_model_path() -> str:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "gemma-4-E2B-it",
        script_dir.parent.parent / "gemma-4-hf-original" / "gemma-4-E2B-it",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 4 E2B PyTorch inference")
    parser.add_argument("--model_path", type=str, default=_default_model_path(),
                        help="Path to the original Gemma 4 model directory with weights")
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
                        help="Fixed number of image soft tokens used for image prompts")
    parser.add_argument("--resize_h", type=int, default=None,
                        help="Optional fixed image height for the vision encoder")
    parser.add_argument("--resize_w", type=int, default=None,
                        help="Optional fixed image width for the vision encoder")
    args = parser.parse_args()

    processor = load_processor(args.model_path)
    if torch.cuda.is_available():
        model = AutoModelForMultimodalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        input_device = next(model.parameters()).device
    else:
        input_device = torch.device("cpu")
        model = AutoModelForMultimodalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
        ).to(input_device).eval()

    if args.image_path:
        image = load_image(args.image_path)
        prepared = prepare_multimodal_inputs(
            processor,
            image=image,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            enable_thinking=args.enable_thinking,
            max_soft_tokens=args.max_soft_tokens,
            resize_h=args.resize_h,
            resize_w=args.resize_w,
        )
        inputs = prepared["inputs"].to(input_device)
    else:
        messages = build_messages(args.prompt, system_prompt=args.system_prompt)
        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        inputs = processor(text=text, return_tensors="pt").to(input_device)

    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    try:
        print(processor.parse_response(response))
    except Exception:
        print(response)
