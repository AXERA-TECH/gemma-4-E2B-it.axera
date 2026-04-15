# gemma-4-E2B-it Vision 模型转换

本文档描述 `google/gemma-4-E2B-it` 视觉分支在 AXERA 平台上的导出、校准与编译流程.

## 目录说明

```bash
model_convert/
├── README.md
├── requirements.txt
├── download_dataset.sh
├── export_onnx.py
├── prepare_calibration.py
├── datasets/               # calibration 原图 + 生成的校准包
│   ├── imagenet-calib.tar
│   ├── gemma4_vision_h336_w480_t70_calibration.tar
│   ├── gemma4_vision_h480_w672_t140_calibration.tar
│   └── gemma4_vision_h672_w960_t280_calibration.tar
├── pulsar2_configs/
│   ├── config.json          # 70 tokens
│   ├── config_140.json      # 140 tokens
│   └── config_280.json      # 280 tokens
├── vit-models/              # 导出的 ONNX 模型
├── compiled_output/         # 70 tokens 编译产物
├── compiled_output_140/     # 140 tokens 编译产物
└── compiled_output_280/     # 280 tokens 编译产物
```

## 环境准备

当前已在以下环境中验证 `export_onnx.py` 与 `prepare_calibration.py`:

- Python `3.12`
- `transformers==5.5.0`
- `torch==2.11.0`

在 `model_convert/` 目录执行:

```bash
python -m pip install -r requirements.txt
```

如果希望在导出后追加 `--optimize`, 还需要额外安装:

```bash
python -m pip install onnxsim onnxslim
```

## 当前已验证配置

三种 Vision profile 均已完成导出、量化、编译与板端验证:

| `max_soft_tokens` | 固定分辨率 (`H x W`) | ONNX |
|---|---|---|
| `70` | `336 x 480` | `gemma4_vision_h336_w480_t70.onnx` |
| `140` | `480 x 672` | `gemma4_vision_h480_w672_t140.onnx` |
| `280` | `672 x 960` | `gemma4_vision_h672_w960_t280.onnx` |

- 量化方式:DEFAULT U16
- calibration 来源:`datasets/imagenet-calib.tar`（128 张图, 实际使用 64 张）

## soft token 配置说明

默认映射定义在 `python/utils/gemma4_multimodal.py` 中. 当前 Gemma 4 Vision 路径使用 `patch_size=16`、`pooling_kernel_size=3`, 因此最终 soft token 数满足:

```text
soft_tokens = (resize_h / 16) * (resize_w / 16) / 9
```

以默认配置为例:`336/16=21`, `480/16=30`, 因此 `21 x 30 / 9 = 70`.

如果想使用其他 `max_soft_tokens`, 需要同时显式指定匹配的 `--resize_h` 和 `--resize_w`, 并保证计算得到的 token 数与 `max_soft_tokens` 完全一致.

## 导出 Vision ONNX

> 注意从 HF 下载原始模型: git clone https://huggingface.co/google/gemma-4-E2B-it

在 `model_convert/` 目录执行:

```bash
# 70 tokens（默认）
python export_onnx.py \
  --model ./gemma-4-E2B-it \
  --onnx_save_dir ./vit-models \
  --max_soft_tokens 70

# 140 tokens
python export_onnx.py \
  --model ./gemma-4-E2B-it \
  --onnx_save_dir ./vit-models \
  --max_soft_tokens 140

# 280 tokens
python export_onnx.py \
  --model ./gemma-4-E2B-it \
  --onnx_save_dir ./vit-models \
  --max_soft_tokens 280
```

导出完成后会生成 `.onnx` 和 `.json` 文件.

## 下载 calibration 图像

请在 `model_convert/` 目录执行:

```bash
bash download_dataset.sh
```

该脚本会下载 `datasets/imagenet-calib.tar`（原始 calibration 图像压缩包）.

## 生成校准数据

在 `model_convert/` 目录执行:

```bash
# 70 tokens
python prepare_calibration.py \
  --model_path ./gemma-4-E2B-it \
  --dataset_dir ./datasets/imagenet-calib.tar \
  --output_dir ./datasets/gemma4_vision_calibration_imagenet \
  --max_soft_tokens 70

# 140 tokens
python prepare_calibration.py \
  --model_path ./gemma-4-E2B-it \
  --dataset_dir ./datasets/imagenet-calib.tar \
  --output_dir ./datasets/gemma4_vision_calibration_imagenet_140 \
  --max_soft_tokens 140

# 280 tokens
python prepare_calibration.py \
  --model_path ./gemma-4-E2B-it \
  --dataset_dir ./datasets/imagenet-calib.tar \
  --output_dir ./datasets/gemma4_vision_calibration_imagenet_280 \
  --max_soft_tokens 280
```

说明:

- `prepare_calibration.py` 支持图片目录和 `.tar` 压缩包两种输入.
- 脚本会先生成中间 `*.pixel_values.npy` 文件, 再自动打包成 `.tar` 校准文件.
- 当前 `datasets/imagenet-calib.tar` 中包含 128 张 calibration 图像；`pulsar2_configs/` 中配置 `calibration_size=64`, 编译时实际使用 64 个样本.

## 使用 pulsar2 编译 Vision axmodel

在 `model_convert/` 目录执行:

```bash
# 70 tokens
pulsar2 build \
  --output_dir ./compiled_output \
  --config pulsar2_configs/config.json \
  --npu_mode NPU3 \
  --input vit-models/gemma4_vision_h336_w480_t70.onnx \
  --compiler.check 0 \
  --target_hardware AX650

# 140 tokens
pulsar2 build \
  --output_dir ./compiled_output_140 \
  --config pulsar2_configs/config_140.json \
  --npu_mode NPU3 \
  --input vit-models/gemma4_vision_h480_w672_t140.onnx \
  --compiler.check 0 \
  --target_hardware AX650

# 280 tokens
pulsar2 build \
  --output_dir ./compiled_output_280 \
  --config pulsar2_configs/config_280.json \
  --npu_mode NPU3 \
  --input vit-models/gemma4_vision_h672_w960_t280.onnx \
  --compiler.check 0 \
  --target_hardware AX650
```

编译输出:`compiled_output*/compiled.axmodel`

## 拷贝到推理目录

Vision 编译完成后, 将产物拷贝到推理目录并重命名:

```bash
cp compiled_output/compiled.axmodel ../python/vit_models/gemma4_vision_h336_w480_t70.axmodel
cp compiled_output_140/compiled.axmodel ../python/vit_models/gemma4_vision_h480_w672_t140.axmodel
cp compiled_output_280/compiled.axmodel ../python/vit_models/gemma4_vision_h672_w960_t280.axmodel

# 同步 JSON 配置
cp vit-models/*.json ../python/vit_models/
```

## LLM 编译 (1k prefill + 1k decoding)

> 注意在性能不足的机器上将 --parallel 设置为较小值, 如果性能足够, 可以直接设置为 35, 大幅度提升编译速度.

```sh
FLOAT_MATMUL_USE_CONV_EU=1 pulsar2 llm_build --input_path gemma-4-E2B-it --output_path gemma-4-E2B-it_axmodel/ --hidden_state_type bf16 --prefill_len 128 --kv_cache_len 2047 --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 --last_kv_cache_len 512 --last_kv_cache_len 640 --last_kv_cache_len 768 --last_kv_cache_len 896 --last_kv_cache_len 1024 --chip AX650 -c 0 --parallel 1
```

在 AX650 系列芯片上使用 `FLOAT_MATMUL_USE_CONV_EU=1` 可以大幅度提升 TTFT 性能, 建议在编译 LLM 时开启该环境变量.
