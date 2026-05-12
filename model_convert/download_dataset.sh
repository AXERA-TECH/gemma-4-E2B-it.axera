#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO="${DATASET_REPO:-AXERA-TECH/gemma-4-E2B-it.axera}"
TAG="${DATASET_RELEASE_TAG:-calibration}"
ASSET_NAME="${DATASET_ASSET_NAME:-gemma4-calibration-datasets.tar}"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET_NAME}"

REQUIRED_FILES=(
  "datasets/imagenet-calib.tar"
  "datasets/gemma4_vision_h336_w480_t70_calibration.tar"
  "datasets/gemma4_vision_h480_w672_t140_calibration.tar"
  "datasets/gemma4_vision_h672_w960_t280_calibration.tar"
  "datasets/gemma4_audio_5s_calibration.tar"
  "datasets/gemma4_audio_30s_calibration.tar"
)

all_present=1
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    all_present=0
    break
  fi
done

if [[ "$all_present" -eq 1 ]]; then
  echo "datasets 已完整存在，跳过下载。"
  exit 0
fi

mkdir -p datasets
archive_path="datasets/${ASSET_NAME}"

echo "将从当前仓库 GitHub Release 下载 calibration 数据集包："
echo "  ${URL}"

download_ok=0
if command -v wget >/dev/null 2>&1; then
  if wget -O "$archive_path" "$URL"; then
    download_ok=1
  fi
fi

if [[ "$download_ok" -ne 1 ]] && command -v curl >/dev/null 2>&1; then
  if curl -L "$URL" -o "$archive_path"; then
    download_ok=1
  fi
fi

if [[ "$download_ok" -ne 1 ]]; then
  echo "错误：下载失败，请手动从以下地址下载："
  echo "  ${URL}"
  echo "下载后在 model_convert/ 目录执行："
  echo "  tar -xf /path/to/${ASSET_NAME}"
  exit 1
fi

tar -xf "$archive_path" -C "$SCRIPT_DIR"
rm -f "$archive_path"

for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "错误：解压完成后缺少文件 $f"
    exit 1
  fi
done

echo "下载并解压成功，datasets 目录已就绪。"
