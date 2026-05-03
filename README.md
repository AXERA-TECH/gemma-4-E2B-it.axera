# gemma-4-E2B-it.axera

> `google/gemma-4-E2B-it` 在 Axera NPU 上的推理与模型转换示例工程.

- 当前提供 `Python` 推理脚本与模型转换流程.
- 支持文本对话、图文多模态推理(单图)以及固定时长音频推理.
- 如需重新导出 ONNX 或重新编译 axmodel, 请参考 [模型转换](./model_convert/README.md).
- 音频模块已编译并在板端验证 `5s` 与 `30s` 两种固定时长版本; 真实音频上均与 `ONNX` 参考逐字符一致.

## 支持平台

- [x] AX650 / NPU3
- [ ] AX620E

## LLM 推理耗时统计

以下数字在 `AX650 DEMO Board` 上实测得到。

- `w8a16` 模型: TTFT 约 `1664 ms`, 解码速度约 `10.44 tokens/s`;
- `w4a16` 模型: TTFT 约 `1233.7 ms`, 解码速度约 `15.22 tokens/s`.

### 8 bit

该模型 prefill 阶段存在 9 个可用子图, 共 35 层 Decode Layer, 每个子图耗时如下:

```sh
g1: 4.234 ms
g2: 4.730 ms
g3: 4.755 ms
g4: 5.018 ms
g5: 5.297 ms
g6: 5.445 ms
g7: 5.472 ms
g8: 5.860 ms
g9: 6.089 ms
```

decode 阶段只有一个子图, 耗时如下:

```sh
g0: 2.165 ms
后处理耗时: 19.987 ms.
```

模型最大 TTFT 为: 46.97 * 35 + 19.987 约为 1664 ms.

模型解码速度为: 1000 / (2.165 * 35 + 19.987) = 10.44 tokens/s.

### 4 bit

该模型 prefill 阶段存在 9 个可用子图, 共 35 层 Decode Layer, 每个子图耗时如下:

```sh
g1: 3.035 ms
g2: 3.261 ms
g3: 3.404 ms
g4: 3.700 ms
g5: 3.860 ms
g6: 4.016 ms
g7: 4.252 ms
g8: 4.444 ms
g9: 4.701 ms
```

decode 阶段只有一个子图, 耗时如下:

```sh
g0: 1.302 ms
后处理耗时: 20.154 ms
```

模型最大 TTFT 为: 34.673 * 35 + 20.154 ms 约为 1233.7 ms.

模型解码速度为: 1000 / (1.302 * 35 + 20.154 ms) = 15.22 tokens/s.

## ViT 耗时统计

| Model | Resolution | Soft Tokens | Time (ms) |
|---|---|---|---|
| `gemma4_vision_h336_w480_t70.axmodel` | 336×480 | 70 | 87.966 ms |
| `gemma4_vision_h480_w672_t140.axmodel` | 480×672 | 140 | 258.329 ms |
| `gemma4_vision_h672_w960_t280.axmodel` | 672×960 | 280 | 750.429 ms |

## Audio Encoder 耗时与精度

两种固定时长 audio encoder 均已完成板端实测. 精度与 `audio ONNX` 参考对比:

| Model | Audio duration | Audio tokens | Encoder cosine vs ONNX | 板端转写 vs ONNX |
|---|---|---|---|---|
| `gemma4_audio_5s.axmodel`  | 5s  | 125 | `0.996173` | 逐字符一致 |
| `gemma4_audio_30s.axmodel` | 30s | 750 | `0.998999` / `0.999799` (chunk0 / chunk1) | 逐字符一致 |

单次推理耗时 (`ax_run_model --warmup 1 --repeat 5`, NPU3 单核亲和):

| Model | cmm size | min | max | avg |
|---|---|---|---|---|
| `gemma4_audio_5s.axmodel`  | `~310 MiB` | `28.905 ms`  | `28.955 ms`  | `28.930 ms` |
| `gemma4_audio_30s.axmodel` | `~359 MiB` | `170.848 ms` | `171.166 ms` | `170.978 ms` |

复现命令 (在板端执行):

```bash
cd python/audio_models
/opt/bin/ax_run_model -m gemma4_audio_5s.axmodel  -w 1 -r 5
/opt/bin/ax_run_model -m gemma4_audio_30s.axmodel -w 1 -r 5
```

## 仓库结构

```text
.
├── assets/                 # Demo 图像与音频
├── model_convert/
│   ├── README.md
│   ├── requirements.txt
│   ├── download_dataset.sh
│   ├── export_onnx.py
│   ├── export_audio_onnx.py
│   ├── prepare_calibration.py
│   ├── prepare_audio_calibration.py
│   ├── datasets/           # calibration 原始数据 + 生成的校准包
│   ├── pulsar2_configs/    # Vision 与 Audio 的编译配置
│   ├── vit-models/         # 导出的 Vision ONNX
│   ├── audio-models/       # 导出的 Audio ONNX
│   └── compiled_output*/   # pulsar2 编译产物
├── python/
│   ├── infer_axmodel.py    # CLI 推理脚本 (文本 / 图文 / 音频)
│   ├── infer_torch.py      # x86 torch 文本参考推理
│   ├── infer_torch_img.py  # x86 torch 图文参考推理
│   ├── infer_torch_audio.py # x86 torch 音频参考推理
│   ├── gradio_demo.py      # Web UI 交互体验
│   ├── gemma-4-E2B-it/     # tokenizer / config (不含模型权重)
│   ├── gemma-4-E2B-it_axmodel/  # LLM axmodel + 辅助权重
│   ├── audio_models/       # 部署用 audio encoder axmodel
│   ├── vit_models/         # 部署用 VIT axmodel
│   └── utils/              # 推理工具函数
└── README.md
```

## 环境准备

板端需要安装以下 Python 依赖:

- `pyaxengine`
- `transformers >= 5.5.0`
- `numpy`
- `ml_dtypes`
- `pillow`
- `gradio` (仅在运行 Web Demo 时需要)

板端默认 `transformers` 通常低于 `5.5.0`, 且 Gemma 4 的图像 / 音频 processor 需要 `transformers 5.5.0` 及以上. 如果板端无法直接 `pip install`, 请在有网络的主机上预先下载 wheel 并打包到某个目录, 再通过 `PYTHONPATH` 在板端注入. 假设离线依赖目录为 `$OFFLINE_PYDEPS`, 运行前执行:

```bash
export PYTHONPATH="$OFFLINE_PYDEPS:$PYTHONPATH"
```

`$OFFLINE_PYDEPS` 的具体位置由用户自行决定, 本仓库不做假设.

仓库 `assets/` 下预备了 `16 kHz` 单声道 `WAV` 测试片段, 推理脚本 (`infer_axmodel.py`) 用 Python stdlib `wave` 直接解析, 不依赖 `librosa` 或 `ffmpeg`. 如果想传入 `mp3 / flac / m4a / ogg` 等格式, 则额外需要 `librosa`.

## 快速运行

### 文本推理

```bash
cd python

python3 infer_axmodel.py \
  --prompt "美国首都是哪里" \
  --max_new_tokens 1024
```

输出:

```text
...
Init InferenceSession: 100%|██████████████████████████████████████████████████████████| 35/35 [00:09<00:00,  3.79it/s]
Model loaded successfully!
slice_indices: [0]
Slice prefill done: 0
answer >> 美国的首都**不是一个单一的城市**，而是**联邦政府的中心**，主要由**华盛顿特区 (Washington, D.C.)** 组成.

**华盛顿特区 (Washington, D.C.)** 是美国的**联邦首都**，因为那里有：

* **国会大厦 (The Capitol Building)**：美国国会所在地.
* **白宫 (The White House)**：美国总统的官邸.
* **联邦政府机构**：包括国务院、国防部等众多重要的联邦政府部门.

**总结来说：**

* **联邦首都：** 华盛顿特区 (Washington, D.C.)
* **人口最多的城市/经济中心：** 纽约市 (New York City) 或洛杉矶 (Los Angeles) 等，但它们不是首都.

所以，如果你问"美国的首都"，最准确的答案是**华盛顿特区**.
```

### 音频推理

提供两种固定时长 audio encoder, 运行前请确保以下文件已经就位:

- `python/audio_models/gemma4_audio_5s.axmodel`  (5 秒 / 125 audio tokens)
- `python/audio_models/gemma4_audio_30s.axmodel` (30 秒 / 750 audio tokens)
- `assets/gemma4_audio_test_5s.wav`              (5 秒测试片段)
- `assets/gemma4_audio_test_chunk0_30s.wav`      (30 秒测试片段, 由 `gemma4_audio_test.mp3` 前 30 秒切出)
- `assets/gemma4_audio_test_chunk1_30s.wav`      (30 秒测试片段, 由 `gemma4_audio_test.mp3` 30-60 秒切出)

推理脚本默认吃 `16 kHz` 单声道 `WAV` (使用 Python stdlib `wave` 直接解析, 无额外依赖). 非 `WAV` 格式 (`mp3 / flac / m4a / ogg`) 也可以, 但板端需要安装 `librosa` 才能解码与重采样; 不装时会抛 `Non-WAV audio loading requires librosa ...` 错误. 为了让 demo 无额外依赖, 本仓库已预先切好 `30s` 测试片段的 `WAV`.

`5s` 版本 (低延迟):

```bash
cd python

python3 infer_axmodel.py \
  --audio_path ../assets/gemma4_audio_test_5s.wav \
  --audio_model_path audio_models/gemma4_audio_5s.axmodel \
  --audio_duration_sec 5 \
  --audio_tokens 125 \
  --system_prompt "" \
  --prompt "Transcribe the speech in its original language. Output only the transcription." \
  --max_new_tokens 128
```

示例输出:

```text
answer >> Lesson 22, Knowledge and Progress.
```

`30s` 版本 (更长上下文):

```bash
cd python

python3 infer_axmodel.py \
  --audio_path ../assets/gemma4_audio_test_chunk0_30s.wav \
  --audio_model_path audio_models/gemma4_audio_30s.axmodel \
  --audio_duration_sec 30 \
  --audio_tokens 750 \
  --system_prompt "" \
  --prompt "Transcribe the speech in its original language. Output only the transcription." \
  --max_new_tokens 256
```

示例输出:

```text
answer >> In what two areas have people made no progress at all? Why does the idea of progress loom so large in the modern world? Surely because progress of a particular kind is actually taking place around us and is becoming more and more manifest
```

说明:

- 两种 profile 在板端真实音频上的转写与 `audio ONNX` 参考逐字符一致, encoder 级别 cosine `≥ 0.996`.
- `750` 个 audio tokens 会跨越 `7` 个 `128-token` prefill slices, 日志中会出现 `[WARN] Multimodal token block ... spans 6 prefill slices.` 这是 `chunked prefill` 对同一 multimodal block 内 bidirectional attention 的限制; 在当前测试集上 (LibriSpeech 读书体) 不影响转写质量.
- 如需验证模型编译结果, 见 [模型转换 — 已验证精度](./model_convert/README.md#已验证精度).

### 图文多模态推理

支持三种 VIT 分辨率, 通过 `--vit_model_path` 指定:

```bash
cd python

# 70 soft tokens (336×480), 速度最快, 推荐作为默认
python3 infer_axmodel.py \
  --image_path ../assets/sample_1.png \
  --prompt "请描述一下这幅图" \
  --system_prompt "" \
  --vit_model_path vit_models/gemma4_vision_h336_w480_t70.axmodel

# 140 soft tokens (480×672)
python3 infer_axmodel.py \
  --image_path ../assets/sample_1.png \
  --prompt "请描述一下这幅图" \
  --system_prompt "" \
  --vit_model_path vit_models/gemma4_vision_h480_w672_t140.axmodel

# 280 soft tokens (672×960), 分辨率最高
python3 infer_axmodel.py \
  --image_path ../assets/sample_1.png \
  --prompt "请描述一下这幅图" \
  --system_prompt "" \
  --vit_model_path vit_models/gemma4_vision_h672_w960_t280.axmodel
```

示例输出 (70 tokens):

```text
answer >> 这是一张女性的肖像照片，她有着非常柔和、甜美的外表.

**人物特征：**
* **面部：** 她的五官精致，眼睛大而明亮，表情看起来比较平静或略带微笑.
* **发型：** 她留着一头浅灰色或银灰色的长发，发丝柔顺，披散在肩上.
  她的发型上装饰着一些花朵或发饰，增添了一丝清新和自然的感觉.
* **服装：** 她穿着一套浅灰色的、低胸的连体衣或比基尼式的上衣和下装.

**场景与氛围：**
* **背景：** 背景显示这是一个海滩场景.可以看到沙滩、海浪和远处的海洋.
* **光线：** 光线充足，柔和地打在人物身上，突出了她的皮肤和服装的质感.
```

### Gradio 交互体验

```bash
cd python

python3 gradio_demo.py \
  --hf_model gemma-4-E2B-it \
  --axmodel_path gemma-4-E2B-it_axmodel/ \
  --vit_model_path vit_models/gemma4_vision_h672_w960_t280.axmodel
```

默认端口 `7860`, 启动后可通过 `http://<board-ip>:7860` 访问. 支持上传图片进行图文对话.

示例结果:

![demo](assets/gradio_img2text.png)

## 技术讨论

- GitHub Issues
- QQ 群: `139953715`
