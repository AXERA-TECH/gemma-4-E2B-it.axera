import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor


def _collect_audio_paths(root: Path):
    suffixes = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)


def _num_mel_frames(num_samples: int, frame_length: int, hop_length: int) -> int:
    frame_size_for_unfold = frame_length + 1
    pad_left = frame_length // 2
    return (num_samples + pad_left - frame_size_for_unfold) // hop_length + 1


def _build_synthetic_input_features(index: int, num_mel_frames: int, feature_size: int) -> np.ndarray:
    rng = np.random.default_rng(index)
    time_axis = np.linspace(0.0, 1.0, num_mel_frames, endpoint=False, dtype=np.float32)
    freq_axis = np.linspace(0.0, 1.0, feature_size, endpoint=False, dtype=np.float32)

    temporal = 0.8 * np.sin(2.0 * np.pi * (1.2 + 0.15 * index) * time_axis)[:, None]
    spectral = 0.6 * np.cos(2.0 * np.pi * (2.4 + 0.1 * index) * freq_axis)[None, :]
    bands = 0.35 * np.sin(2.0 * np.pi * (time_axis[:, None] * (3.0 + 2.0 * freq_axis[None, :])))
    noise = 0.12 * rng.standard_normal((num_mel_frames, feature_size), dtype=np.float32)

    features = -4.2 + temporal + spectral + bands + noise
    features = np.clip(features, -12.0, 6.0).astype(np.float32)
    return features[None, ...]


def _load_audio_with_librosa(audio_path: Path, sampling_rate: int) -> np.ndarray:
    import librosa

    waveform, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
    return waveform.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Gemma 4 audio calibration inputs")
    parser.add_argument("--model_path", type=str, default="../gemma-4-hf-original/gemma-4-E2B-it",
                        help="Path to the original Gemma 4 model directory")
    parser.add_argument("--audio_dir", type=str, default="../assets/gemma4_audio_test.mp3",
                        help="Path to an audio file, a directory of audio files, or a tar archive")
    parser.add_argument("--output_dir", type=str, default="./datasets/gemma4_audio_calibration",
                        help="Directory used for intermediate .npy files")
    parser.add_argument("--audio_duration_sec", type=float, default=30.0,
                        help="Fixed audio duration in seconds")
    parser.add_argument("--repeat", type=int, default=4,
                        help="Repeat each source audio this many times when building the calibration set")
    parser.add_argument("--synthetic_count", type=int, default=0,
                        help="Generate this many synthetic waveforms instead of loading real audio when > 0")
    parser.add_argument("--tar_name", type=str, default="",
                        help="Optional output tar file name")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    processor_config = json.loads((model_path / "processor_config.json").read_text(encoding="utf-8"))
    feature_cfg = processor_config["feature_extractor"].copy()
    feature_cfg.pop("feature_extractor_type", None)
    feature_extractor = Gemma4AudioFeatureExtractor(**feature_cfg)

    sampling_rate = int(feature_cfg["sampling_rate"])
    max_length = int(round(args.audio_duration_sec * sampling_rate))
    num_mel_frames = _num_mel_frames(
        num_samples=max_length,
        frame_length=int(feature_cfg["frame_length"]),
        hop_length=int(feature_cfg["hop_length"]),
    )
    feature_size = int(feature_cfg["feature_size"])

    input_path = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _save_features(input_features: np.ndarray, stem: str, rep: int, index: int):
        output_path = output_dir / f"{stem}_{rep:02d}_{index:04d}.input_features.npy"
        np.save(output_path, np.asarray(input_features, dtype=np.float32))
        print(f"saved {output_path}")
        return output_path

    def _extract_features(waveform: np.ndarray):
        features = feature_extractor(
            waveform.astype(np.float32),
            padding="max_length",
            max_length=max_length,
            truncation=True,
            pad_to_multiple_of=None,
            return_tensors="np",
        )
        return np.asarray(features["input_features"], dtype=np.float32)

    def _prepare_from_paths(audio_paths):
        saved_files = []
        index = 0
        for audio_path in audio_paths:
            waveform = _load_audio_with_librosa(audio_path, sampling_rate)
            for rep in range(max(1, args.repeat)):
                input_features = _extract_features(waveform)
                saved_files.append(_save_features(input_features, audio_path.stem, rep, index))
                index += 1
        return saved_files

    def _prepare_synthetic():
        saved_files = []
        for index in range(max(0, args.synthetic_count)):
            input_features = _build_synthetic_input_features(index, num_mel_frames=num_mel_frames, feature_size=feature_size)
            saved_files.append(_save_features(input_features, "synthetic_mel", 0, index))
        return saved_files

    if args.synthetic_count > 0:
        saved_files = _prepare_synthetic()
    elif input_path.is_file() and tarfile.is_tarfile(input_path):
        with tempfile.TemporaryDirectory(prefix="gemma4_audio_calib_") as tmp_dir:
            extract_dir = Path(tmp_dir)
            with tarfile.open(input_path, "r") as tar:
                tar.extractall(extract_dir)
            audio_paths = _collect_audio_paths(extract_dir)
            if not audio_paths:
                raise FileNotFoundError(f"No audio files found inside tar file {input_path}")
            saved_files = _prepare_from_paths(audio_paths)
    elif input_path.is_dir():
        audio_paths = _collect_audio_paths(input_path)
        if not audio_paths:
            raise FileNotFoundError(f"No audio files found under {input_path}")
        saved_files = _prepare_from_paths(audio_paths)
    elif input_path.is_file():
        saved_files = _prepare_from_paths([input_path])
    else:
        raise FileNotFoundError(f"`audio_dir` must be an audio file, audio directory, or tar archive, got: {input_path}")

    tar_name = args.tar_name or f"gemma4_audio_{int(args.audio_duration_sec)}s_calibration.tar"
    tar_path = output_dir.parent / tar_name
    with tarfile.open(tar_path, "w") as tar:
        for npy_path in saved_files:
            tar.add(npy_path, arcname=npy_path.name)

    print(f"packed {len(saved_files)} files -> {tar_path}")
