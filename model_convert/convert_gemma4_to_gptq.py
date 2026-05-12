#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_gemma4_to_gptq.py

说明:
  - 使用 GPTQModel 将 gemma-4-E2B-it 量化为 GPTQ Int4
  - 支持验证量化前后 embedding 相似度
  - 建议单卡运行 (多卡可能中途报错)
"""

import os
import sys
import time
import json
import glob
import tarfile
import hashlib
import shutil
import argparse
import inspect
from typing import List

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["DISABLE_TF32"] = "1"

# -------------------------
# imports
# -------------------------
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from datasets import load_dataset
    from gptqmodel import GPTQModel, QuantizeConfig, get_best_device
    from gptqmodel.utils.model import get_module_by_name_prefix
except Exception as e:
    print("缺少依赖或导入失败, 请先安装必要包 (gptqmodel, transformers, datasets, torch, numpy). ")
    print("示例安装命令:")
    print("  pip install gptqmodel transformers datasets torch numpy")
    raise

if hasattr(torch, "compile"):
    torch.compile = lambda *args, **kwargs: args[0]

# -------------------------
# Monkey-patch: Gemma4TextDecoderLayer.forward
# 量化时 module_looper 有两个问题:
# 1. 只捕获 args[0] (hidden_states)，per_layer_input (args[1]) 丢失 → None * Tensor 报错
# 2. position_embeddings 从第一层缓存，不同 attention 类型 (sliding vs full) 维度不同
#    sliding: head_dim=256, full: global_head_dim=512 → 维度不匹配
# 修补方案:
#   - per_layer_input=None 时安全跳过 (非量化目标权重)
#   - position_embeddings 维度不匹配时，用挂载的 rotary_emb 重新计算
# -------------------------
def _compute_rope_embeddings(position_ids, head_dim, rope_theta, device, dtype):
    """从头计算 rotary position embeddings, 不依赖任何模型状态"""
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim))
    inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos, sin


try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

    def _patched_gemma4_decoder_forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        shared_kv_states=None,
        position_embeddings: torch.Tensor = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ) -> torch.Tensor:
        # 修复 position_embeddings 维度不匹配问题
        if position_embeddings is not None:
            cos, sin = position_embeddings
            expected_dim = self.self_attn.head_dim
            if cos.shape[-1] != expected_dim and position_ids is not None:
                # 根据层类型确定 rope_theta
                layer_type = getattr(self.self_attn, 'layer_type', None)
                rope_params = getattr(self.config, 'rope_parameters', {})
                rope_theta = 10000.0
                if layer_type and layer_type in rope_params:
                    rope_theta = rope_params[layer_type].get('rope_theta', rope_theta)
                position_embeddings = _compute_rope_embeddings(
                    position_ids, expected_dim, rope_theta,
                    device=hidden_states.device, dtype=hidden_states.dtype,
                )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states if shared_kv_states is not None else {},
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # per_layer_input block: 仅当 per_layer_input 实际可用时执行
        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states

    if os.environ.get("GPTQMODEL_DISABLE_GEMMA4_FORWARD_PATCH") == "1":
        print("[INFO] 跳过 Gemma4TextDecoderLayer.forward monkey-patch (GPTQMODEL_DISABLE_GEMMA4_FORWARD_PATCH=1)")
    else:
        Gemma4TextDecoderLayer.forward = _patched_gemma4_decoder_forward
        print("[INFO] 已 monkey-patch Gemma4TextDecoderLayer.forward (兼容量化 forward replay)")
except ImportError:
    print("[WARN] 无法 monkey-patch Gemma4TextDecoderLayer, 跳过")

if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
    try:
        torch._inductor.config.triton = False
    except Exception:
        pass


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask[:, -1].sum().item() == attention_mask.shape[0]:
        return last_hidden_states[:, -1]
    else:
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
        return last_hidden_states[batch_idx, seq_lens]


def compute_embeddings_with_model(model, tokenizer, texts: List[str], device, max_length=1024, batch_size=8):
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        embs = model.encode(texts)
        return np.array(embs)

    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            batch = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            if hasattr(out, "last_hidden_state"):
                last_hidden = out.last_hidden_state
            elif isinstance(out, (tuple, list)) and len(out) > 0:
                last_hidden = out[0]
            else:
                raise RuntimeError("无法从模型输出中获取 last_hidden_state")
            pooled = last_token_pool(last_hidden, batch["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1).cpu().numpy()
            all_embs.append(pooled)
    return np.concatenate(all_embs, axis=0)


def cosine_similarities(a: np.ndarray, b: np.ndarray):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sims = (a_n * b_n).sum(axis=1)
    return sims


def safe_cuda_empty_cache():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def materialize_meta_parameters(q_model, context=""):
    if not hasattr(q_model, "model"):
        return
    meta_params = []
    for name, param in q_model.model.named_parameters():
        if getattr(param, "is_meta", False):
            meta_params.append(name)
    if not meta_params:
        return
    if context:
        print(f"[WARN] {context}: 检测到 {len(meta_params)} 个 meta 参数, 正在 materialize...")
    else:
        print(f"[WARN] 检测到 {len(meta_params)} 个 meta 参数, 正在 materialize...")
    seen = set()
    for param_name in meta_params:
        module_name = param_name.rsplit('.', 1)[0] if '.' in param_name else param_name
        if module_name in seen:
            continue
        seen.add(module_name)
        module, _ = get_module_by_name_prefix(q_model.model, module_name)
        if module is None:
            continue
        try:
            q_model.shell_module_materialize(module, q_model.quantize_config.device)
        except Exception as exc:
            print(f"[WARN] materialize 模块 {module_name} 失败: {exc}")


IMAGE_CALIB_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
IMAGE_CALIB_TAR_EXTS = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def is_image_path(path: str) -> bool:
    return path.lower().endswith(IMAGE_CALIB_EXTS)


def is_tar_path(path: str) -> bool:
    return path.lower().endswith(IMAGE_CALIB_TAR_EXTS)


def extract_image_calib_tar(tar_path: str) -> List[str]:
    """Extract image members from a tar archive into a stable /tmp cache."""
    tar_path = os.path.abspath(tar_path)
    cache_key = hashlib.sha256(f"{tar_path}:{os.path.getmtime(tar_path)}".encode("utf-8")).hexdigest()[:16]
    extract_root = os.path.join("/tmp", f"gemma4_image_calib_{cache_key}")
    done_flag = os.path.join(extract_root, ".extract_done")

    if not os.path.exists(done_flag):
        os.makedirs(extract_root, exist_ok=True)
        with tarfile.open(tar_path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile() or not is_image_path(member.name):
                    continue
                member_name = member.name.lstrip("/").replace("\\", "/")
                parts = [p for p in member_name.split("/") if p not in ("", ".", "..")]
                if not parts:
                    continue
                target_path = os.path.abspath(os.path.join(extract_root, *parts))
                if not target_path.startswith(os.path.abspath(extract_root) + os.sep):
                    raise ValueError(f"Unsafe tar member path: {member.name}")
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                source = tf.extractfile(member)
                if source is None:
                    continue
                with source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
        with open(done_flag, "w", encoding="utf-8") as f:
            f.write(f"{tar_path}\n")

    image_paths = []
    for root, _, files in os.walk(extract_root):
        for name in files:
            path = os.path.join(root, name)
            if is_image_path(path):
                image_paths.append(path)
    return sorted(image_paths)


def resolve_image_calib_paths(image_calib_path: str) -> List[str]:
    """Resolve image files from files, directories, globs, tar archives, or comma-separated lists."""
    paths = []
    for item in image_calib_path.split(","):
        item = item.strip()
        if not item:
            continue
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                for name in files:
                    path = os.path.join(root, name)
                    if is_image_path(path):
                        paths.append(path)
            continue
        matched = glob.glob(item)
        if matched:
            for path in matched:
                if os.path.isdir(path):
                    paths.extend(resolve_image_calib_paths(path))
                elif is_tar_path(path):
                    paths.extend(extract_image_calib_tar(path))
                elif is_image_path(path):
                    paths.append(path)
        elif os.path.isfile(item):
            if is_tar_path(item):
                paths.extend(extract_image_calib_tar(item))
            elif is_image_path(item):
                paths.append(item)

    deduped = []
    seen = set()
    for path in paths:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return sorted(deduped)


def build_multimodal_calibration_samples(
    texts: List[str],
    image_paths: List[str],
    size: int,
    image_height: int,
    image_width: int,
) -> List[dict]:
    """Build Gemma4 image calibration samples that mimic image-after-text-history."""
    if not image_paths:
        raise ValueError("image_paths must not be empty for multimodal calibration.")

    image_cache = {}

    def load_resized_image(path: str) -> Image.Image:
        if path not in image_cache:
            image_cache[path] = Image.open(path).convert("RGB").resize(
                (image_width, image_height),
                resample=Image.BICUBIC,
            )
        return image_cache[path].copy()

    images_kwargs = {
        "do_convert_rgb": True,
        "size": {"height": image_height, "width": image_width},
        "max_soft_tokens": 70,
    }
    samples = []
    for i, text in enumerate(texts[:size]):
        history_text = " ".join(text.split()[:120]) or "Please summarize your core capabilities."
        image = load_resized_image(image_paths[i % len(image_paths)])
        samples.append({
            "messages": [
                {"role": "user", "content": history_text},
                {"role": "assistant", "content": "I understand."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image.copy()},
                        {"type": "text", "text": "Describe the visual elements of this image."},
                    ],
                },
            ],
            "images_kwargs": images_kwargs,
        })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Gemma-4-E2B-it GPTQ INT4 量化脚本")
    parser.add_argument("--model_id", type=str, default="gemma-4-E2B-it",
                        help="原始模型路径或 HuggingFace ID")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="量化模型输出目录 (默认: <model_id>-gptq-int4)")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8],
                        help="量化位宽 (默认: 4)")
    parser.add_argument("--group_size", type=int, default=128,
                        help="量化分组大小 (默认: 128)")
    parser.add_argument("--calib_size", type=int, default=512,
                        help="校准数据集大小 (默认: 512)")
    parser.add_argument("--eval_size", type=int, default=128,
                        help="验证集大小 (默认: 128)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="量化时的 batch size (默认: 2, 显存不足可设为 1)")
    parser.add_argument("--device", type=str, default=None,
                        help="指定设备 (默认: 自动选择)")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="校准文本最大长度 (默认: 1024)")
    parser.add_argument("--calibration_concat_size", type=int, default=0,
                        help="Concatenate text-only calibration rows into fixed token blocks. 0 disables concatenation.")
    parser.add_argument("--calibration_concat_separator", type=str, default="\n\n",
                        help="Separator inserted when concatenating text-only calibration rows.")
    parser.add_argument("--skip_eval", action="store_true",
                        help="跳过验证阶段 (节省时间和显存)")
    parser.add_argument("--image_calib_path", type=str, default="",
                        help="Optional image file, directory, glob, tar archive, or comma-separated list for Gemma4 multimodal calibration.")
    parser.add_argument("--image_calib_size", type=int, default=128,
                        help="Number of image-after-text-history calibration samples to prepend when --image_calib_path is set.")
    parser.add_argument("--image_calib_height", type=int, default=336,
                        help="Fixed image calibration height. Default matches the AX 70-token vision profile.")
    parser.add_argument("--image_calib_width", type=int, default=480,
                        help="Fixed image calibration width. Default matches the AX 70-token vision profile.")
    parser.add_argument("--skip_per_layer_adapter_quant", action="store_true",
                        help=(
                            "Skip GPTQ quantization for Gemma4 per_layer_input_gate and "
                            "per_layer_projection modules. This matches the current AXERA "
                            "Gemma4 export/runtime contract, where per-layer adapter inputs "
                            "are computed outside the decoder axmodels."
                        ))
    args = parser.parse_args()

    model_id = args.model_id
    out_dir = args.out_dir or f"{model_id.rstrip('/').replace('/', '_')}-gptq-int{args.bits}"
    os.makedirs(out_dir, exist_ok=True)
    done_flag = os.path.join(out_dir, "quantize_done.txt")

    # select device
    if args.device:
        device = torch.device(args.device)
    else:
        try:
            device = torch.device(get_best_device())
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] target device = {device}")

    # load tokenizer
    print("[STEP] 加载 tokenizer ...")
    # Workaround: gemma-4 tokenizer_config.json 中 extra_special_tokens 为 list,
    # 但 transformers<4.57 要求 dict。在加载前临时修正。
    _tok_cfg_path = os.path.join(model_id, "tokenizer_config.json")
    _tok_cfg_patched = False
    _tok_cfg_backup = None
    if os.path.isfile(_tok_cfg_path):
        with open(_tok_cfg_path, "r", encoding="utf-8") as _f:
            _tok_cfg = json.load(_f)
        est = _tok_cfg.get("extra_special_tokens")
        if isinstance(est, list):
            print("[INFO] 修正 extra_special_tokens: list -> dict (transformers 兼容)")
            _tok_cfg_backup = json.dumps(_tok_cfg, ensure_ascii=False)
            _tok_cfg["extra_special_tokens"] = {t: t for t in est}
            with open(_tok_cfg_path, "w", encoding="utf-8") as _f:
                json.dump(_tok_cfg, _f, indent=2, ensure_ascii=False)
            _tok_cfg_patched = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", use_fast=True, trust_remote_code=True)
    finally:
        # 恢复原始 tokenizer_config.json
        if _tok_cfg_patched and _tok_cfg_backup is not None:
            with open(_tok_cfg_path, "w", encoding="utf-8") as _f:
                _f.write(_tok_cfg_backup)

    # calibration data
    print(f"[STEP] 准备 calibration 数据 (size={args.calib_size}) ...")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if isinstance(t, str) and len(t.strip()) > 10]
        calib_texts = texts[:args.calib_size]
    except Exception:
        print("[WARN] 无法加载 wikitext 数据集, 使用备用文本")
        calib_texts = [
            "Deep learning is transforming AI.",
            "Large language models have billions of parameters.",
        ] * args.calib_size
        calib_texts = calib_texts[:args.calib_size]
    # GPTQModel 5.7.0 支持直接传入原始文本字符串，无需预先 tokenize。
    # For Gemma4 VLM deployment, prepend image-after-text-history samples so
    # GPTQ Hessian statistics cover image soft-token activations at non-zero
    # absolute positions, which is the axllm failure mode being debugged.
    calib_data = calib_texts
    if args.image_calib_path:
        image_paths = resolve_image_calib_paths(args.image_calib_path)
        if not image_paths:
            raise FileNotFoundError(f"--image_calib_path did not resolve any images: {args.image_calib_path}")
        mm_calib = build_multimodal_calibration_samples(
            calib_texts,
            image_paths=image_paths,
            size=max(0, args.image_calib_size),
            image_height=args.image_calib_height,
            image_width=args.image_calib_width,
        )
        calib_data = mm_calib + calib_texts
        print(f"[INFO] 已添加 Gemma4 多模态校准样本: {len(mm_calib)} (images={len(image_paths)})")

    # quantization
    if not os.path.exists(done_flag):
        print("[STEP] 构建 QuantizeConfig ...")
        qc_sig = inspect.signature(QuantizeConfig)
        qc_kwargs = {}
        if 'bits' in qc_sig.parameters:
            qc_kwargs['bits'] = args.bits
        if 'group_size' in qc_sig.parameters:
            qc_kwargs['group_size'] = args.group_size
        if 'calibration_enable_gpu_cache' in qc_sig.parameters:
            qc_kwargs['calibration_enable_gpu_cache'] = False
        if 'desc_act' in qc_sig.parameters:
            qc_kwargs['desc_act'] = False
        if 'use_accelerate' in qc_sig.parameters:
            qc_kwargs['use_accelerate'] = False
        if 'static_groups' in qc_sig.parameters:
            qc_kwargs['static_groups'] = False
        if 'sym' in qc_sig.parameters:
            qc_kwargs['sym'] = True
        if 'true_sequential' in qc_sig.parameters:
            qc_kwargs['true_sequential'] = True
        if 'damp_percent' in qc_sig.parameters:
            qc_kwargs['damp_percent'] = 0.01
        if args.skip_per_layer_adapter_quant:
            qc_kwargs['dynamic'] = {
                r"-:.*\.per_layer_input_gate$": {},
                r"-:.*\.per_layer_projection$": {},
            }
            print("[INFO] 将跳过 per_layer_input_gate/per_layer_projection 的 GPTQ 量化")

        try:
            quant_config = QuantizeConfig(**qc_kwargs)
        except TypeError:
            # Older GPTQModel builds may not expose dynamic quantization control.
            # Keep the script usable there, but fail loudly for the AXERA-specific
            # mode that depends on skipping Gemma4 per-layer adapter modules.
            if args.skip_per_layer_adapter_quant and 'dynamic' in qc_kwargs:
                raise RuntimeError(
                    "当前 GPTQModel 不支持 QuantizeConfig.dynamic，无法安全跳过 "
                    "Gemma4 per_layer adapter 量化"
                )
            qc_kwargs.pop('dynamic', None)
            quant_config = QuantizeConfig(**qc_kwargs)

        print("[STEP] 调用 GPTQModel.from_pretrained ...")
        # Workaround: Gemma4Config 没有 bos_token_id/eos_token_id 顶层属性,
        # tokenicer 库会因此报错。预先给 config 打补丁。
        _cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if not hasattr(_cfg, "bos_token_id"):
            text_cfg = getattr(_cfg, "text_config", None)
            _cfg.bos_token_id = getattr(text_cfg, "bos_token_id", 2) if text_cfg else 2
        if not hasattr(_cfg, "eos_token_id"):
            text_cfg = getattr(_cfg, "text_config", None)
            _cfg.eos_token_id = getattr(text_cfg, "eos_token_id", 1) if text_cfg else 1
        if not hasattr(_cfg, "pad_token_id"):
            text_cfg = getattr(_cfg, "text_config", None)
            _cfg.pad_token_id = getattr(text_cfg, "pad_token_id", 0) if text_cfg else 0
        # 写回 config.json 供 GPTQModel 读取
        _cfg_path = os.path.join(model_id, "config.json")
        _cfg_backup = None
        if os.path.isfile(_cfg_path):
            with open(_cfg_path, "r", encoding="utf-8") as _f:
                _cfg_raw = json.load(_f)
            _cfg_backup = json.dumps(_cfg_raw, ensure_ascii=False)
            _patched = False
            for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
                if key not in _cfg_raw:
                    _cfg_raw[key] = getattr(_cfg, key)
                    _patched = True
            if _patched:
                print("[INFO] 向 config.json 注入缺失的 bos/eos/pad_token_id")
                with open(_cfg_path, "w", encoding="utf-8") as _f:
                    json.dump(_cfg_raw, _f, indent=2, ensure_ascii=False)
        model = GPTQModel.from_pretrained(model_id, quant_config, trust_remote_code=True, device_map="auto")

        safe_cuda_empty_cache()
        print("[STEP] 开始量化 (这可能需要较长时间) ...")
        t0 = time.time()
        try:
            quantize_kwargs = {}
            if args.calibration_concat_size > 0:
                quantize_kwargs["calibration_concat_size"] = args.calibration_concat_size
                quantize_kwargs["calibration_concat_separator"] = args.calibration_concat_separator
            model.quantize(calib_data, batch_size=args.batch_size, **quantize_kwargs)
            materialize_meta_parameters(model, "GPU save")
            model.save(out_dir)
            elapsed = time.time() - t0
            with open(done_flag, "w") as f:
                f.write(f"done (GPU, {elapsed:.1f}s)\n")
            print(f"[INFO] GPU 量化完成, 耗时 {elapsed:.1f}s")
        except Exception as e_gpu:
            print("[WARN] GPU 量化失败, 尝试 CPU:", repr(e_gpu))
            safe_cuda_empty_cache()
            if 'device' in qc_sig.parameters:
                qc_kwargs['device'] = 'cpu'
            quant_config_cpu = QuantizeConfig(**qc_kwargs)
            model_cpu = GPTQModel.load(model_id, quant_config_cpu, trust_remote_code=True)
            quantize_kwargs = {}
            if args.calibration_concat_size > 0:
                quantize_kwargs["calibration_concat_size"] = args.calibration_concat_size
                quantize_kwargs["calibration_concat_separator"] = args.calibration_concat_separator
            model_cpu.quantize(calib_data, batch_size=1, **quantize_kwargs)
            materialize_meta_parameters(model_cpu, "CPU save")
            model_cpu.save(out_dir)
            elapsed = time.time() - t0
            with open(done_flag, "w") as f:
                f.write(f"done (CPU, {elapsed:.1f}s)\n")
            print(f"[INFO] CPU 量化完成, 耗时 {elapsed:.1f}s")
    else:
        print(f"[INFO] 已检测到量化完成标记 ({done_flag}), 跳过量化步骤")

    print(f"[INFO] 量化模型已保存到: {out_dir}")

    # 恢复原始 config.json (如果被补丁修改过)
    if _cfg_backup is not None:
        _cfg_path = os.path.join(model_id, "config.json")
        with open(_cfg_path, "w", encoding="utf-8") as _f:
            _f.write(_cfg_backup)
        print("[INFO] 已恢复原始 config.json")

    # validation
    if args.skip_eval:
        print("[INFO] 已跳过验证阶段 (--skip_eval)")
        return

    print("[STEP] 加载参考模型 ...")
    dtype = torch.float16 if "cuda" in str(device) else None
    ref_model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    ref_model.to(device)

    print("[STEP] 加载量化模型 ...")
    try:
        quant_model = AutoModel.from_pretrained(out_dir, trust_remote_code=True)
        quant_model.to(device)
    except Exception as e_auto:
        print("[WARN] AutoModel 无法加载量化模型, 尝试 GPTQModel.load:", repr(e_auto))
        quant_model = GPTQModel.load(out_dir)

    eval_texts = calib_texts[:args.eval_size]

    print("[STEP] 计算参考模型 embeddings ...")
    ref_embs = compute_embeddings_with_model(ref_model, tokenizer, eval_texts, device, args.max_length, batch_size=8)

    print("[STEP] 计算量化模型 embeddings ...")
    q_embs = compute_embeddings_with_model(quant_model, tokenizer, eval_texts, device, args.max_length, batch_size=8)

    sims = cosine_similarities(ref_embs, q_embs)
    summary = {
        "cosine_mean": float(np.mean(sims)),
        "cosine_std": float(np.std(sims)),
        "cosine_min": float(np.min(sims)),
        "cosine_max": float(np.max(sims)),
    }
    print("=== 验证结果 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    results_path = os.path.join(out_dir, "quant_eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[INFO] 验证结果保存到:", results_path)


if __name__ == "__main__":
    main()
