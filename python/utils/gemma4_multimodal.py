from __future__ import annotations

import os
from pathlib import Path
import wave

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


def _resample_waveform(waveform: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    # Linear interpolation; introduces aliasing when downsampling (e.g. 44.1kHz -> 16kHz).
    # For best quality pass mono 16kHz WAV and skip this path; otherwise librosa.load
    # with its polyphase resampler is preferred.
    if src_rate == dst_rate or waveform.size == 0:
        return waveform.astype(np.float32)

    src_positions = np.arange(waveform.shape[0], dtype=np.float32) / float(src_rate)
    dst_length = max(1, int(round(waveform.shape[0] * float(dst_rate) / float(src_rate))))
    dst_positions = np.arange(dst_length, dtype=np.float32) / float(dst_rate)
    return np.interp(dst_positions, src_positions, waveform).astype(np.float32)


def load_audio_waveform(audio_path: str | Path, sampling_rate: int = 16000) -> np.ndarray:
    audio_path = Path(audio_path)
    if audio_path.suffix.lower() == ".wav":
        with wave.open(str(audio_path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError(f"Unsupported WAV compression type: {wav_file.getcomptype()}")

            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            src_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width == 1:
            waveform = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            waveform = (waveform - 128.0) / 128.0
        elif sample_width == 2:
            waveform = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            waveform = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        if channels > 1:
            waveform = waveform.reshape(-1, channels).mean(axis=1)
        waveform = _resample_waveform(waveform, src_rate=src_rate, dst_rate=sampling_rate)
        return np.asarray(np.clip(waveform, -1.0, 1.0), dtype=np.float32)

    try:
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
        import librosa
    except Exception as exc:
        raise RuntimeError(
            "Non-WAV audio loading requires librosa. Please convert the input to a mono 16kHz WAV file."
        ) from exc

    waveform, _ = librosa.load(str(audio_path), sr=sampling_rate, mono=True)
    return np.asarray(waveform, dtype=np.float32)


def resize_image(image: Image.Image, resize_h: int, resize_w: int) -> Image.Image:
    return image.convert("RGB").resize((resize_w, resize_h), resample=Image.BICUBIC)


def build_messages(
    prompt: str,
    image: Image.Image | None = None,
    audio=None,
    system_prompt: str = "",
) -> list[dict]:
    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    if image is None and audio is None:
        messages.append({"role": "user", "content": prompt})
    else:
        user_content = []
        if audio is not None:
            user_content.append({"type": "audio", "audio": audio})
        if image is not None:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": prompt})
        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )
    return messages


def build_processor_messages(
    prompt: str,
    image: Image.Image | None = None,
    audio=None,
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
    if audio is not None:
        user_content.append({"type": "audio", "audio": audio})
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


def prepare_audio_inputs(
    processor,
    audio_path: str | Path,
    prompt: str,
    system_prompt: str = "",
    enable_thinking: bool = False,
    audio_duration_sec: float = 30.0,
    fixed_audio_tokens: int | None = None,
):
    sampling_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    max_length = int(round(audio_duration_sec * sampling_rate))
    waveform = load_audio_waveform(audio_path, sampling_rate=sampling_rate)

    if waveform.shape[0] < max_length:
        padded_waveform = np.pad(waveform, (0, max_length - waveform.shape[0]), mode="constant")
    else:
        padded_waveform = waveform[:max_length]

    messages = build_processor_messages(prompt=prompt, audio=padded_waveform, system_prompt=system_prompt)
    inputs = _safe_apply_chat_template(
        processor,
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        processor_kwargs={
            "audio_kwargs": {
                "padding": "max_length",
                "max_length": max_length,
                "truncation": True,
                "pad_to_multiple_of": None,
            }
        },
    )

    audio_token_id = processor.audio_token_id
    audio_token_count = int((inputs["input_ids"] == audio_token_id).sum().item())
    expected_tokens = int(fixed_audio_tokens or processor.audio_seq_length)
    if audio_token_count != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} audio soft tokens from fixed audio preprocessing, got {audio_token_count}."
        )

    return {
        "messages": messages,
        "waveform": waveform,
        "padded_waveform": padded_waveform,
        "inputs": inputs,
        "audio_duration_sec": audio_duration_sec,
        "expected_tokens": expected_tokens,
    }


def replace_special_tokens(
    token_ids,
    token_embeds,
    modality_embeds,
    special_token_id: int,
    modality_name: str,
):
    positions = [idx for idx, token_id in enumerate(token_ids) if int(token_id) == int(special_token_id)]
    if not positions:
        return token_embeds

    flat_embeds = modality_embeds.reshape(-1, modality_embeds.shape[-1])
    if len(positions) != flat_embeds.shape[0]:
        raise ValueError(
            f"{modality_name.capitalize()} tokens and features do not match: "
            f"tokens={len(positions)}, features={flat_embeds.shape[0]}"
        )
    if token_embeds.shape[-1] != flat_embeds.shape[-1]:
        raise ValueError(
            f"Embedding dim mismatch: token_dim={token_embeds.shape[-1]}, "
            f"{modality_name}_dim={flat_embeds.shape[-1]}"
        )
    token_embeds[positions, :] = flat_embeds
    return token_embeds


def replace_image_tokens(token_ids, token_embeds, image_embeds, image_token_id: int):
    return replace_special_tokens(
        token_ids,
        token_embeds,
        image_embeds,
        special_token_id=image_token_id,
        modality_name="image",
    )


def replace_audio_tokens(token_ids, token_embeds, audio_embeds, audio_token_id: int):
    return replace_special_tokens(
        token_ids,
        token_embeds,
        audio_embeds,
        special_token_id=audio_token_id,
        modality_name="audio",
    )


def to_numpy_fp32(tensor_like) -> np.ndarray:
    if isinstance(tensor_like, np.ndarray):
        return tensor_like.astype(np.float32)
    return tensor_like.detach().cpu().numpy().astype(np.float32)
