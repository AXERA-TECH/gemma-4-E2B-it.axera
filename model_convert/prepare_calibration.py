import argparse
import tarfile
from pathlib import Path
import sys
import tempfile

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils.gemma4_multimodal import DEFAULT_MAX_SOFT_TOKENS  # noqa: E402
from utils.gemma4_multimodal import load_image  # noqa: E402
from utils.gemma4_multimodal import load_processor  # noqa: E402
from utils.gemma4_multimodal import prepare_multimodal_inputs  # noqa: E402
from utils.gemma4_multimodal import resolve_resize  # noqa: E402
from utils.gemma4_multimodal import to_numpy_fp32  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Gemma 4 vision calibration inputs")
    parser.add_argument("--model_path", type=str, default="../python/gemma-4-E2B-it",
                        help="Path to the original Gemma 4 model directory")
    parser.add_argument("--dataset_dir", type=str, default="./datasets/imagenet-calib.tar",
                        help="Path to a calibration image directory or a .tar image archive")
    parser.add_argument("--output_dir", type=str, default="./dataset/gemma4_vision_calibration_imagenet",
                        help="Directory used for intermediate .npy files")
    parser.add_argument("--prompt", type=str, default="Describe the image.",
                        help="Text prompt paired with each calibration image")
    parser.add_argument("--max_soft_tokens", type=int, default=DEFAULT_MAX_SOFT_TOKENS,
                        help="Fixed number of projected image soft tokens")
    parser.add_argument("--resize_h", type=int, default=None,
                        help="Optional fixed input image height")
    parser.add_argument("--resize_w", type=int, default=None,
                        help="Optional fixed input image width")
    parser.add_argument("--tar_name", type=str, default="",
                        help="Optional output tar file name")
    args = parser.parse_args()

    processor = load_processor(args.model_path)
    resize_h, resize_w, expected_tokens = resolve_resize(
        max_soft_tokens=args.max_soft_tokens,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
    )

    dataset_path = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _collect_image_paths(root: Path):
        return sorted(
            path for path in root.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

    def _prepare_from_image_paths(image_paths):
        saved_files = []
        for image_path in image_paths:
            image = load_image(image_path)
            prepared = prepare_multimodal_inputs(
                processor,
                image=image,
                prompt=args.prompt,
                max_soft_tokens=args.max_soft_tokens,
                resize_h=resize_h,
                resize_w=resize_w,
            )
            pixel_values = to_numpy_fp32(prepared["inputs"]["pixel_values"])
            output_path = output_dir / f"{image_path.stem}.pixel_values.npy"
            np.save(output_path, pixel_values)
            saved_files.append(output_path)
            print(f"saved {output_path}")
        return saved_files

    if dataset_path.is_dir():
        image_paths = _collect_image_paths(dataset_path)
        if not image_paths:
            raise FileNotFoundError(f"No images found under {dataset_path}")
        saved_files = _prepare_from_image_paths(image_paths)
    elif dataset_path.is_file() and tarfile.is_tarfile(dataset_path):
        with tempfile.TemporaryDirectory(prefix="gemma4_calib_") as tmp_dir:
            extract_dir = Path(tmp_dir)
            with tarfile.open(dataset_path, "r") as tar:
                tar.extractall(extract_dir)
            image_paths = _collect_image_paths(extract_dir)
            if not image_paths:
                raise FileNotFoundError(f"No images found inside tar file {dataset_path}")
            saved_files = _prepare_from_image_paths(image_paths)
    else:
        raise FileNotFoundError(
            f"`dataset_dir` must be an image directory or a tar archive, got: {dataset_path}"
        )

    tar_name = args.tar_name or f"gemma4_vision_h{resize_h}_w{resize_w}_t{expected_tokens}_calibration.tar"
    tar_path = output_dir.parent / tar_name
    with tarfile.open(tar_path, "w") as tar:
        for npy_path in saved_files:
            tar.add(npy_path, arcname=npy_path.name)

    print(f"packed {len(saved_files)} files -> {tar_path}")
