# gemma-4-E2B-it Vision / Audio 模型转换

本文档描述 `google/gemma-4-E2B-it` 的 Vision / Audio 分支在 AXERA 平台上的导出、校准与编译流程.

## 目录说明

```text
model_convert/
├── README.md
├── requirements.txt
├── download_dataset.sh
├── export_onnx.py                   # Vision ONNX 导出
├── export_audio_onnx.py             # Audio ONNX 导出
├── prepare_calibration.py           # Vision 校准数据生成
├── prepare_audio_calibration.py     # Audio 校准数据生成
├── dump_torch_audio_embeds.py       # 调试工具: 用 torch HF 模型 dump 参考 audio_embeds, 与 ONNX 做 cosine sanity check
├── infer_torch_audio_reference.py   # 调试工具: 用 torch HF 模型做完整 audio + LLM 转写, 作为板端 axmodel 的参考
├── audio-models/            # 导出的 Audio ONNX
│   ├── gemma4_audio_5s.onnx
│   ├── gemma4_audio_5s.json
│   ├── gemma4_audio_30s.onnx
│   └── gemma4_audio_30s.json
├── vit-models/              # 导出的 Vision ONNX
├── datasets/                # calibration 原始数据 + 生成的校准包
│   ├── imagenet-calib.tar
│   ├── gemma4_vision_h336_w480_t70_calibration.tar
│   ├── gemma4_vision_h480_w672_t140_calibration.tar
│   ├── gemma4_vision_h672_w960_t280_calibration.tar
│   ├── gemma4_audio_5s_calibration.tar
│   └── gemma4_audio_30s_calibration.tar
├── pulsar2_configs/
│   ├── config.json          # Vision 70  tokens
│   ├── config_140.json      # Vision 140 tokens
│   ├── config_280.json      # Vision 280 tokens
│   ├── config_audio_5s.json
│   └── config_audio_30s.json
├── compiled_output/         # Vision 70  tokens 编译产物
├── compiled_output_140/     # Vision 140 tokens 编译产物
├── compiled_output_280/     # Vision 280 tokens 编译产物
├── compiled_output_audio_5s/
└── compiled_output_audio_30s/
```

## 环境准备

本文档中的命令均在以下环境下验证:

- Python `3.12`
- `transformers==5.5.0`
- `torch==2.11.0`

在 `model_convert/` 目录执行:

```bash
python -m pip install -r requirements.txt
```

如果希望在 ONNX 导出后追加 `--optimize`, 还需要额外安装:

```bash
python -m pip install onnxsim onnxslim
```

下文所有脚本默认以 `model_convert/` 为当前工作目录. 原始 Gemma 4 HF 权重目录由用户自行准备, 用 `$HF_MODEL` 引用; 下载方式见 [导出 Vision ONNX](#导出-vision-onnx).

## 已验证精度

三种 Vision profile 均已完成导出、量化、编译与板端验证:

| `max_soft_tokens` | 固定分辨率 (`H x W`) | ONNX |
|---|---|---|
| `70`  | `336 x 480` | `vit-models/gemma4_vision_h336_w480_t70.onnx` |
| `140` | `480 x 672` | `vit-models/gemma4_vision_h480_w672_t140.onnx` |
| `280` | `672 x 960` | `vit-models/gemma4_vision_h672_w960_t280.onnx` |

- 量化方式: DEFAULT U16
- calibration 来源: `datasets/imagenet-calib.tar` (128 张图, 实际使用 64 张)

两种固定时长 Audio profile 均已完成导出、量化、编译与板端真实音频 A/B 验证:

| Audio duration | Mel frames | Audio tokens | ONNX | Encoder cosine vs ONNX |
|---|---|---|---|---|
| `5s`  | `499`  | `125` | `audio-models/gemma4_audio_5s.onnx`  | `0.996173` |
| `30s` | `2999` | `750` | `audio-models/gemma4_audio_30s.onnx` | `0.998999` / `0.999799` (chunk0 / chunk1) |

- 量化方式: DEFAULT U16
- calibration 来源: `datasets/gemma4_audio_{5s,30s}_calibration.tar`, 各含 `64` 条由 `LibriSpeech dev-clean` 生成的真实语音 log-mel 样本
- 板端转写输出与 `audio ONNX` 参考逐字符一致

## soft token 配置说明

默认映射定义在 `python/utils/gemma4_multimodal.py` 中. 当前 Gemma 4 Vision 路径使用 `patch_size=16`、`pooling_kernel_size=3`, 因此最终 soft token 数满足:

```text
soft_tokens = (resize_h / 16) * (resize_w / 16) / 9
```

以默认配置为例:`336/16=21`, `480/16=30`, 因此 `21 x 30 / 9 = 70`.

如果想使用其他 `max_soft_tokens`, 需要同时显式指定匹配的 `--resize_h` 和 `--resize_w`, 并保证计算得到的 token 数与 `max_soft_tokens` 完全一致.

## 导出 Vision ONNX

从 Hugging Face 下载原始模型权重, 以下示例假设克隆到仓库同级目录 `gemma-4-hf-original/gemma-4-E2B-it`, 并用 `$HF_MODEL` 引用:

```bash
git clone https://huggingface.co/google/gemma-4-E2B-it ../../gemma-4-hf-original/gemma-4-E2B-it
export HF_MODEL=../../gemma-4-hf-original/gemma-4-E2B-it
```

在 `model_convert/` 目录执行:

```bash
# 70 tokens (默认)
python export_onnx.py \
  --model "$HF_MODEL" \
  --onnx_save_dir ./vit-models \
  --max_soft_tokens 70

# 140 tokens
python export_onnx.py \
  --model "$HF_MODEL" \
  --onnx_save_dir ./vit-models \
  --max_soft_tokens 140

# 280 tokens
python export_onnx.py \
  --model "$HF_MODEL" \
  --onnx_save_dir ./vit-models \
  --max_soft_tokens 280
```

导出完成后会生成 `.onnx` 和对应 `.json` 配置文件.

## 导出 Audio ONNX

Audio ONNX 导出需要原始 Hugging Face 模型权重目录, 而不是仅包含 tokenizer 的运行时目录.

在 `model_convert/` 目录执行:

```bash
# 5 seconds / 125 audio tokens
python export_audio_onnx.py \
  --model "$HF_MODEL" \
  --onnx_save_dir ./audio-models \
  --audio_duration_sec 5

# 30 seconds / 750 audio tokens
python export_audio_onnx.py \
  --model "$HF_MODEL" \
  --onnx_save_dir ./audio-models \
  --audio_duration_sec 30
```

导出完成后会生成:

- `audio-models/gemma4_audio_5s.onnx` 与 `audio-models/gemma4_audio_5s.json`
- `audio-models/gemma4_audio_30s.onnx` 与 `audio-models/gemma4_audio_30s.json`

## 下载 Vision calibration 图像

在 `model_convert/` 目录执行:

```bash
bash download_dataset.sh
```

该脚本会下载 `datasets/imagenet-calib.tar` (原始 calibration 图像压缩包).

## 生成 Vision 校准数据

在 `model_convert/` 目录执行:

```bash
# 70 tokens
python prepare_calibration.py \
  --model_path "$HF_MODEL" \
  --dataset_dir ./datasets/imagenet-calib.tar \
  --output_dir ./datasets/gemma4_vision_calibration_imagenet \
  --max_soft_tokens 70

# 140 tokens
python prepare_calibration.py \
  --model_path "$HF_MODEL" \
  --dataset_dir ./datasets/imagenet-calib.tar \
  --output_dir ./datasets/gemma4_vision_calibration_imagenet_140 \
  --max_soft_tokens 140

# 280 tokens
python prepare_calibration.py \
  --model_path "$HF_MODEL" \
  --dataset_dir ./datasets/imagenet-calib.tar \
  --output_dir ./datasets/gemma4_vision_calibration_imagenet_280 \
  --max_soft_tokens 280
```

说明:

- `prepare_calibration.py` 支持图片目录和 `.tar` 压缩包两种输入.
- 脚本会先生成中间 `*.pixel_values.npy` 文件, 再自动打包成 `.tar` 校准文件.
- `datasets/imagenet-calib.tar` 中包含 128 张 calibration 图像; `pulsar2_configs/` 中配置 `calibration_size=64`, 编译时实际使用 64 个样本.

## 生成 Audio 校准数据

本仓库已经预打好 Audio 校准包 (`datasets/gemma4_audio_{5s,30s}_calibration.tar`), 每个各 `64` 条真实语音样本. 如果只是要复现编译过程, 可以直接跳到 [编译 Audio axmodel](#编译-audio-axmodel) 一节.

如果需要重新生成, 任意一组 `≥ 32` 条的真实语音源都可以使用. 以 [LibriSpeech dev-clean](https://www.openslr.org/resources/12/dev-clean.tar.gz) 为例 (在 `model_convert/` 目录执行):

```bash
# 1. 下载并解压
mkdir -p datasets/librispeech
wget -P datasets/librispeech https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf datasets/librispeech/dev-clean.tar.gz -C datasets/librispeech

# 2. 随机抽取 64 条 flac 到一个子目录 (shuf 需要 coreutils, Linux 默认自带)
mkdir -p datasets/librispeech_subset
find datasets/librispeech/LibriSpeech/dev-clean -name '*.flac' \
  | shuf -n 64 \
  | xargs -I{} cp {} datasets/librispeech_subset/

# 3. 为 5s 和 30s 各生成一份 .tar
python prepare_audio_calibration.py \
  --model_path "$HF_MODEL" \
  --audio_dir ./datasets/librispeech_subset \
  --output_dir ./datasets/gemma4_audio_5s_calibration \
  --audio_duration_sec 5 \
  --repeat 1

python prepare_audio_calibration.py \
  --model_path "$HF_MODEL" \
  --audio_dir ./datasets/librispeech_subset \
  --output_dir ./datasets/gemma4_audio_30s_calibration \
  --audio_duration_sec 30 \
  --repeat 1
```

说明:

- `prepare_audio_calibration.py` 支持单个音频文件、音频目录和 `.tar` 压缩包三种输入; 目录输入时脚本会递归收集 `wav / mp3 / flac / m4a / ogg`.
- 脚本会先生成中间 `*.input_features.npy`, 再自动打包成 `.tar` 校准文件.
- `pulsar2_configs/config_audio_{5s,30s}.json` 配置 `calibration_size=64`, 编译时实际使用 64 个样本; 生成得到的 `.tar` 内样本数需 `≥ 64`.
- 非 `WAV` 格式依赖 `librosa` 进行解码与重采样. 首次 `import librosa` 可能触发 `numba` JIT, 脚本内部已设置 `NUMBA_DISABLE_JIT=1` 规避.
- 单条 LibriSpeech clip 通常短于 `30s`, `prepare_audio_calibration.py` 会把不足时长的波形零填充到 `30s`. 这在当前验证中仍然给到 `≥ 0.998` 的 encoder cosine; 如果目标场景的语音分布与读书体差异较大, 建议使用对应领域的真实音频替换.
- **不要用合成 mel 或单条音频重复 4 次作为校准**: U16 量化的 scale 会在真实语音动态范围 (`log-mel ∈ [-6.91, +3.5]`) 上被严重低估, 导致 encoder cosine 退化到 `0.86 ~ 0.89` 级别.

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

## 使用 pulsar2 编译 Audio axmodel

在 `model_convert/` 目录执行:

```bash
# 5 seconds
pulsar2 build \
  --output_dir ./compiled_output_audio_5s \
  --config pulsar2_configs/config_audio_5s.json \
  --npu_mode NPU3 \
  --input audio-models/gemma4_audio_5s.onnx \
  --compiler.check 0 \
  --target_hardware AX650

# 30 seconds
pulsar2 build \
  --output_dir ./compiled_output_audio_30s \
  --config pulsar2_configs/config_audio_30s.json \
  --npu_mode NPU3 \
  --input audio-models/gemma4_audio_30s.onnx \
  --compiler.check 0 \
  --target_hardware AX650
```

编译输出:

- `compiled_output_audio_5s/compiled.axmodel`
- `compiled_output_audio_30s/compiled.axmodel`

## 拷贝到推理目录

Vision 编译完成后, 将产物拷贝到推理目录并重命名:

```bash
cp compiled_output/compiled.axmodel ../python/vit_models/gemma4_vision_h336_w480_t70.axmodel
cp compiled_output_140/compiled.axmodel ../python/vit_models/gemma4_vision_h480_w672_t140.axmodel
cp compiled_output_280/compiled.axmodel ../python/vit_models/gemma4_vision_h672_w960_t280.axmodel

# 同步 JSON 配置
cp vit-models/*.json ../python/vit_models/
```

Audio 编译完成后, 将产物拷贝到推理目录并重命名:

```bash
cp compiled_output_audio_5s/compiled.axmodel ../python/audio_models/gemma4_audio_5s.axmodel
cp compiled_output_audio_30s/compiled.axmodel ../python/audio_models/gemma4_audio_30s.axmodel
```

## LLM 编译 (1k prefill + 1k decoding)

LLM 编译依赖 `pulsar2 llm_build` 工具. 下面的命令假设 `$HF_MODEL` 指向原始 HF 模型目录 (参见 [导出 Vision ONNX](#导出-vision-onnx)):

```bash
FLOAT_MATMUL_USE_CONV_EU=1 pulsar2 llm_build \
  --input_path "$HF_MODEL" \
  --output_path gemma-4-E2B-it_axmodel/ \
  --hidden_state_type bf16 \
  --prefill_len 128 \
  --kv_cache_len 2047 \
  --last_kv_cache_len 128 \
  --last_kv_cache_len 256 \
  --last_kv_cache_len 384 \
  --last_kv_cache_len 512 \
  --last_kv_cache_len 640 \
  --last_kv_cache_len 768 \
  --last_kv_cache_len 896 \
  --last_kv_cache_len 1024 \
  --chip AX650 \
  -c 0 \
  --parallel 1
```

说明:

- `FLOAT_MATMUL_USE_CONV_EU=1` 在 AX650 系列上可显著提升 TTFT, 建议保留.
- `--parallel` 控制编译并行度. 如果机器性能充足, 可以设置为 `35` (对应 35 个 decoder layer), 可大幅提升编译速度; 性能不足时保留 `1`.
