# gemma-4-E2B-it.axera

> `google/gemma-4-E2B-it` 在 Axera NPU 上的推理与模型转换示例工程.

- 目前支持 `Python` 推理与模型转换.
- 支持文本对话和图文多模态推理（单图）.
- 如需重新导出 Vision ONNX 或重新编译 VIT，请参考 [模型转换](./model_convert/README.md).

## 支持平台

- [x] AX650 / NPU3
- [ ] AX620E

## LLM 推理耗时统计

**TL;DR:**

- w8a16 模型 TTFT 约为 1664 ms, 解码速度约为 10.44 tokens/s;
- w4a16 模型 TTFT 约为 1233.7 ms, 解码速度约为 15.22 tokens/s.

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


## 仓库结构

```bash
.
├── assets/                 # Demo 图像，仅用于展示与推理示例
├── model_convert/
│   ├── README.md
│   ├── requirements.txt
│   ├── download_dataset.sh
│   ├── export_onnx.py
│   ├── prepare_calibration.py
│   ├── datasets/           # calibration 原图 + 生成的校准包
│   ├── pulsar2_configs/    # 70/140/280 三种分辨率的编译配置
│   ├── vit-models/         # 导出的 ONNX 模型
│   └── compiled_output*/   # pulsar2 编译产物
├── python/
│   ├── infer_axmodel.py    # CLI 推理脚本（文本 + 图文多模态）
│   ├── infer_torch.py      # x86 torch 参考推理
│   ├── infer_torch_img.py  # x86 torch 图文参考推理
│   ├── gradio_demo.py      # Web UI 交互体验
│   ├── gemma-4-E2B-it/     # tokenizer / config（不含模型权重）
│   ├── gemma-4-E2B-it_axmodel/  # LLM axmodel + 辅助权重
│   ├── vit_models/         # 部署用 VIT axmodel
│   └── utils/              # 推理工具函数
└── README.md
```

## 环境准备

在 AX 开发板上需要准备：

- `pyaxengine`
- `transformers>=5.5.0`
- `numpy`
- `ml_dtypes`
- `pillow`
- `gradio`（仅在运行 Demo 时需要）

Gemma 4 依赖较新的 `transformers`.如果板端默认环境版本偏旧，可以通过覆盖目录方式运行：

```bash
export PYTHONPATH=/path/to/your/gemma4_pydeps:$PYTHONPATH
```

## 快速运行

### 文本推理

```bash
cd gemma-4-E2B-it.axera/python

python3 infer_axmodel.py \
  --prompt "美国首都是哪里" \
  --max_new_tokens 1024
```

输出:

```sh
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

### 图文多模态推理

支持三种 VIT 分辨率，通过 `--vit_model_path` 指定：

```bash
# 70 soft tokens (336×480) — 推荐，速度最快
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

# 280 soft tokens (672×960) — 分辨率最高
python3 infer_axmodel.py \
  --image_path ../assets/sample_1.png \
  --prompt "请描述一下这幅图" \
  --system_prompt "" \
  --vit_model_path vit_models/gemma4_vision_h672_w960_t280.axmodel
```

输出示例（70 tokens）：

```
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

### 3. Gradio 交互体验

```bash
python3 gradio_demo.py \
  --hf_model gemma-4-E2B-it \
  --axmodel_path gemma-4-E2B-it_axmodel/ \
  --vit_model_path vit_models/gemma4_vision_h672_w960_t280.axmodel
```

默认端口为 `7860`，启动后可通过 `http://<board-ip>:7860` 访问. 支持上传图片进行图文对话.

示例结果如下:

![demo](assets/gradio_img2text.png)

## 技术讨论

- GitHub Issues
- QQ 群: `139953715`
