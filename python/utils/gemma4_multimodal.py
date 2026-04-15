from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor


DEFAULT_MAX_SOFT_TOKENS = 70
DEFAULT_RESIZE_BY_SOFT_TOKENS = {
    70: (336, 480),
    140: (480, 672),
    280: (672, 960),
}


def load_processor(model_dir: str):
    return AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)


def resolve_resize(
    max_soft_tokens: int = DEFAULT_MAX_SOFT_TOKENS,
    resize_h: int | None = None,
    resize_w: int | None = None,
    patch_size: int = 16,
    pooling_kernel_size: int = 3,
) -> tuple[int, int, int]:
    if (resize_h is None) != (resize_w is None):
        raise ValueError("`resize_h` and `resize_w` must be provided together.")

    if resize_h is None and resize_w is None:
        if max_soft_tokens not in DEFAULT_RESIZE_BY_SOFT_TOKENS:
            raise ValueError(
                "No default fixed resolution for this `max_soft_tokens`. "
                "Please pass both `resize_h` and `resize_w` explicitly."
            )
        resize_h, resize_w = DEFAULT_RESIZE_BY_SOFT_TOKENS[max_soft_tokens]

    if resize_h % patch_size != 0 or resize_w % patch_size != 0:
        raise ValueError(
            f"Fixed resize must be divisible by patch_size={patch_size}, got resize_h={resize_h}, resize_w={resize_w}."
        )

    pooling_stride = patch_size * pooling_kernel_size
    if resize_h % pooling_stride != 0 or resize_w % pooling_stride != 0:
        raise ValueError(
            f"Fixed resize must be divisible by patch_size * pooling_kernel_size = {pooling_stride}, "
            f"got resize_h={resize_h}, resize_w={resize_w}."
        )

    expected_tokens = expected_tokens_from_resize(
        resize_h,
        resize_w,
        patch_size=patch_size,
        pooling_kernel_size=pooling_kernel_size,
    )
    if expected_tokens != int(max_soft_tokens):
        raise ValueError(
            f"Fixed resize implies {expected_tokens} projected tokens, but max_soft_tokens={max_soft_tokens}."
        )
    return resize_h, resize_w, expected_tokens


def detect_soft_tokens_from_vit_path(vit_model_path: str) -> int | None:
    """Extract soft token count from VIT model filename like gemma4_vision_h336_w480_t70.axmodel."""
    import re
    stem = Path(vit_model_path).stem
    m = re.search(r"_t(\d+)$", stem)
    if m:
        return int(m.group(1))
    return None


def expected_tokens_from_resize(
    resize_h: int,
    resize_w: int,
    patch_size: int = 16,
    pooling_kernel_size: int = 3,
) -> int:
    patch_count = (resize_h // patch_size) * (resize_w // patch_size)
    return int(patch_count // (pooling_kernel_size * pooling_kernel_size))


def make_image_position_ids(
    resize_h: int,
    resize_w: int,
    patch_size: int = 16,
) -> torch.LongTensor:
    patch_h = resize_h // patch_size
    patch_w = resize_w // patch_size
    grid_x, grid_y = torch.meshgrid(
        torch.arange(patch_w, dtype=torch.long),
        torch.arange(patch_h, dtype=torch.long),
        indexing="xy",
    )
    position_ids = torch.stack([grid_x, grid_y], dim=-1).reshape(1, patch_h * patch_w, 2)
    return position_ids


def load_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def resize_image(image: Image.Image, resize_h: int, resize_w: int) -> Image.Image:
    return image.convert("RGB").resize((resize_w, resize_h), resample=Image.BICUBIC)


def build_messages(prompt: str, image: Image.Image | None = None, system_prompt: str = "") -> list[dict]:
    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    if image is None:
        messages.append({"role": "user", "content": prompt})
    else:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        )
    return messages


def build_processor_messages(
    prompt: str,
    image: Image.Image | None = None,
    system_prompt: str = "",
    history=None,
) -> list[dict]:
    """Build messages for processor.apply_chat_template.

    Gemma4's chat template renders the first system message content with a raw
    ``| trim`` filter (line 166 of the jinja template), which stringifies a
    Python list.  Meanwhile the processor's visual-extraction loop (line 144 of
    processing_utils.py) crashes on plain-string content.  To sidestep both
    issues we emit the system prompt as a plain-string system message and guard
    the processor call with ``tokenize=False`` first, then tokenize separately.
    """
    messages: list[dict] = []
    # System prompt must be a plain string — the jinja template expects this
    # for messages[0]. The processor visual-scan loop only crashes when
    # tokenize=True, so callers that need tokenization should pass
    # tokenize=False first or use _safe_processor_apply_chat_template.
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    history = history or []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})

    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def _safe_apply_chat_template(processor, messages, **kwargs):
    """Call processor.apply_chat_template while working around two bugs:

    1. transformers processing_utils.py visual-extraction loop (line 144)
       crashes on messages whose ``content`` is a plain string.
    2. Gemma4's jinja chat template (line 166) renders the first system
       message content with ``| trim``, which stringifies a Python list.

    Fix: temporarily wrap string content as list for the visual scan, and
    patch the chat template to correctly extract text from list content in
    the system message block.
    """
    # Fix the jinja template to handle list content for system messages
    orig_template = processor.chat_template
    if "messages[0]['content'] | trim" in orig_template:
        fixed_template = orig_template.replace(
            "{{- messages[0]['content'] | trim -}}",
            "{% if messages[0]['content'] is string %}{{- messages[0]['content'] | trim -}}"
            "{% elif messages[0]['content'] is sequence %}"
            "{% for _sys_item in messages[0]['content'] %}"
            "{% if _sys_item['type'] == 'text' %}{{- _sys_item['text'] | trim -}}{% endif %}"
            "{% endfor %}{% endif %}",
        )
        processor.chat_template = fixed_template

    # Wrap string content to list for the visual scan
    patches = []
    for msg in messages:
        c = msg.get("content")
        if isinstance(c, str):
            patches.append((msg, c))
            msg["content"] = [{"type": "text", "text": c}]

    try:
        return processor.apply_chat_template(messages, **kwargs)
    finally:
        for msg, orig in patches:
            msg["content"] = orig
        processor.chat_template = orig_template


def prepare_multimodal_inputs(
    processor,
    image: Image.Image,
    prompt: str,
    system_prompt: str = "",
    enable_thinking: bool = False,
    max_soft_tokens: int = DEFAULT_MAX_SOFT_TOKENS,
    resize_h: int | None = None,
    resize_w: int | None = None,
):
    patch_size = getattr(processor.image_processor, "patch_size", 16)
    pooling_kernel_size = getattr(processor.image_processor, "pooling_kernel_size", 3)
    resize_h, resize_w, expected_tokens = resolve_resize(
        max_soft_tokens=max_soft_tokens,
        resize_h=resize_h,
        resize_w=resize_w,
        patch_size=patch_size,
        pooling_kernel_size=pooling_kernel_size,
    )

    fixed_image = resize_image(image, resize_h=resize_h, resize_w=resize_w)
    messages = build_processor_messages(prompt=prompt, image=fixed_image, system_prompt=system_prompt)
    inputs = _safe_apply_chat_template(
        processor,
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        processor_kwargs={
            "images_kwargs": {
                "do_resize": False,
                "max_soft_tokens": max_soft_tokens,
            }
        },
    )

    image_token_id = processor.image_token_id
    image_token_count = int((inputs["input_ids"] == image_token_id).sum().item())
    if image_token_count != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} image soft tokens from fixed resize, got {image_token_count}. "
            "Please verify the fixed resolution and max_soft_tokens pair."
        )

    return {
        "messages": messages,
        "fixed_image": fixed_image,
        "inputs": inputs,
        "resize_h": resize_h,
        "resize_w": resize_w,
        "expected_tokens": expected_tokens,
    }


def replace_image_tokens(token_ids, token_embeds, image_embeds, image_token_id: int):
    image_positions = [idx for idx, token_id in enumerate(token_ids) if int(token_id) == int(image_token_id)]
    if not image_positions:
        return token_embeds

    flat_image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
    if len(image_positions) != flat_image_embeds.shape[0]:
        raise ValueError(
            f"Image tokens and image features do not match: tokens={len(image_positions)}, "
            f"features={flat_image_embeds.shape[0]}"
        )
    if token_embeds.shape[-1] != flat_image_embeds.shape[-1]:
        raise ValueError(
            f"Embedding dim mismatch: token_dim={token_embeds.shape[-1]}, image_dim={flat_image_embeds.shape[-1]}"
        )
    token_embeds[image_positions, :] = flat_image_embeds
    return token_embeds


def to_numpy_fp32(tensor_like) -> np.ndarray:
    if isinstance(tensor_like, np.ndarray):
        return tensor_like.astype(np.float32)
    return tensor_like.detach().cpu().numpy().astype(np.float32)
