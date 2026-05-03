import json
from pathlib import Path
from types import SimpleNamespace

from transformers import PreTrainedTokenizerFast


def load_text_runtime_config(model_dir: str):
    model_path = Path(model_dir)
    with open(model_path / "config.json", encoding="utf-8") as f:
        raw_config = json.load(f)

    text_config = dict(raw_config.get("text_config") or raw_config)
    text_config["model_type"] = text_config.get("model_type", "gemma4_text")
    text_config["eos_token_id"] = raw_config.get("eos_token_id", text_config.get("eos_token_id"))
    text_config["audio_token_id"] = raw_config.get("audio_token_id")
    text_config["image_token_id"] = raw_config.get("image_token_id")
    text_config["vision_config"] = raw_config.get("vision_config")
    text_config["vision_soft_tokens_per_image"] = raw_config.get("vision_soft_tokens_per_image")
    return SimpleNamespace(**text_config)


def load_tokenizer(model_dir: str):
    model_path = Path(model_dir)
    with open(model_path / "tokenizer_config.json", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    init_kwargs = {
        "tokenizer_file": str(model_path / "tokenizer.json"),
        "bos_token": tokenizer_config.get("bos_token"),
        "eos_token": tokenizer_config.get("eos_token"),
        "pad_token": tokenizer_config.get("pad_token"),
        "unk_token": tokenizer_config.get("unk_token"),
        "mask_token": tokenizer_config.get("mask_token"),
        "padding_side": tokenizer_config.get("padding_side", "left"),
    }
    tokenizer = PreTrainedTokenizerFast(**{k: v for k, v in init_kwargs.items() if v is not None})

    chat_template_path = model_path / "chat_template.jinja"
    if chat_template_path.exists():
        tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")

    return tokenizer
