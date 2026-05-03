"""x86 torch reference end-to-end: waveform -> audio_embeds -> LLM decode.

Uses the official AutoModelForMultimodalLM + processor.apply_chat_template
pipeline (transformers 5.5.0). Bypasses the broken default audio_kwargs by
passing pad_to_multiple_of=None. Goal: produce the model's own unquantized
transcription so we can tell whether on-device "short" outputs are
LLM axmodel quantization artefacts or prompt/greedy-decoding behavior.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


def load_waveform_mono(wav_path: Path, expected_sr: int) -> np.ndarray:
    y, sr = sf.read(str(wav_path), dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != expected_sr:
        raise SystemExit(f"expected {expected_sr} Hz wav, got {sr}")
    return y.astype(np.float32)


def load_audio_feature_extractor(model_path: Path):
    processor_config = json.loads((model_path / "processor_config.json").read_text(encoding="utf-8"))
    feature_cfg = processor_config["feature_extractor"].copy()
    feature_cfg.pop("feature_extractor_type", None)
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

    return Gemma4AudioFeatureExtractor(**feature_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--duration_sec", type=float, default=30.0)
    ap.add_argument("--prompt", default="Transcribe the speech in its original language. Output only the transcription.")
    ap.add_argument("--system_prompt", default="")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    feature_extractor = load_audio_feature_extractor(Path(args.model))
    sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))

    target_samples = int(args.duration_sec * sampling_rate)
    y = load_waveform_mono(Path(args.wav), sampling_rate)
    if len(y) >= target_samples:
        y = y[:target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)))

    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype="auto", device_map=args.device)

    user_content = [{"type": "audio", "audio": y}, {"type": "text", "text": args.prompt}]
    messages = []
    if args.system_prompt.strip():
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": user_content})

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        audio_kwargs={
            "padding": "max_length",
            "max_length": target_samples,
            "truncation": True,
            "pad_to_multiple_of": None,
        },
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    print(f"prompt tokens: {input_len}  input_features: {tuple(inputs['input_features'].shape)}")

    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    print("answer >>", text)


if __name__ == "__main__":
    main()
