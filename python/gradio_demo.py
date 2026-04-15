import argparse
import os
import socket
import time
from pathlib import Path
from typing import Generator

import gradio as gr
import numpy as np
from ml_dtypes import bfloat16

from utils.gemma4_compat import load_text_runtime_config
from utils.gemma4_per_layer import Gemma4PerLayerInputs
from utils.gemma4_multimodal import DEFAULT_MAX_SOFT_TOKENS
from utils.gemma4_multimodal import DEFAULT_RESIZE_BY_SOFT_TOKENS
from utils.gemma4_multimodal import _safe_apply_chat_template
from utils.gemma4_multimodal import build_messages
from utils.gemma4_multimodal import build_processor_messages
from utils.gemma4_multimodal import detect_soft_tokens_from_vit_path
from utils.gemma4_multimodal import load_processor
from utils.gemma4_multimodal import replace_image_tokens
from utils.gemma4_multimodal import resolve_resize
from utils.gemma4_multimodal import resize_image
from utils.gemma4_multimodal import to_numpy_fp32
from utils.infer_func import InferManager
from utils.vision_output import describe_output_shapes
from utils.vision_output import select_vit_output


def _list_host_ips():
    ips = set()
    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET)
        for info in infos:
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    if not ips:
        ips.add("127.0.0.1")
    return sorted(ips)


def _default_hf_model() -> str:
    script_dir = Path(__file__).resolve().parent
    for candidate in [script_dir / "gemma-4-E2B-it", script_dir / "gemma_4_e2b_it_tokenizer"]:
        if candidate.exists():
            return str(candidate)
    return str(script_dir / "gemma-4-E2B-it")


def _default_axmodel_path() -> str:
    script_dir = Path(__file__).resolve().parent
    for candidate in [script_dir / "gemma-4-E2B-it_axmodel", script_dir / "gemma_4_e2b_it_ax650n_axmodel"]:
        if candidate.exists():
            return str(candidate)
    return str(script_dir / "gemma-4-E2B-it_axmodel")


def _default_vit_model_path() -> str:
    script_dir = Path(__file__).resolve().parent
    resize_h, resize_w, expected_tokens = resolve_resize(DEFAULT_MAX_SOFT_TOKENS)
    stem = f"gemma4_vision_h{resize_h}_w{resize_w}_t{expected_tokens}"
    candidates = [
        script_dir / "vit_models" / f"{stem}.axmodel",
        script_dir / "vit_models" / f"{stem}.onnx",
        script_dir.parent / "model_convert" / "compiled_output" / f"{stem}.axmodel",
        script_dir.parent / "model_convert" / "vit-models" / f"{stem}.onnx",
        script_dir.parent / "model_convert" / "compiled_output" / "compiled.axmodel",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _run_vit_model(vit_model_path: str, pixel_values: np.ndarray, target_hidden_size: int, expected_tokens: int):
    if vit_model_path.endswith(".axmodel"):
        from axengine import InferenceSession

        session = InferenceSession(vit_model_path)
        outputs = session.run(None, {"pixel_values": pixel_values})
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
    else:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(vit_model_path, providers=providers)
        outputs = session.run(None, {"pixel_values": pixel_values})

    return (
        select_vit_output(outputs, target_hidden_size, expected_tokens=expected_tokens),
        describe_output_shapes(outputs),
    )


class Gemma4GradioDemo:
    def __init__(self, hf_model: str, axmodel_path: str, vit_model_path: str, max_seq_len: int = 2047):
        self.processor = load_processor(hf_model)
        self.tokenizer = self.processor.tokenizer
        self.config = load_text_runtime_config(hf_model)
        self.embeds = np.load(os.path.join(axmodel_path, "model.embed_tokens.weight.npy"))
        self.axmodel_path = axmodel_path
        self.vit_model_path = vit_model_path
        self.max_seq_len = max_seq_len
        self.slice_len = 128

        self.per_layer_helper = None
        if int(getattr(self.config, "hidden_size_per_layer_input", 0) or 0) > 0:
            self.per_layer_helper = Gemma4PerLayerInputs(axmodel_path, self.config)

        # Auto-detect soft token count from VIT model filename
        self.vit_soft_tokens = detect_soft_tokens_from_vit_path(vit_model_path) or DEFAULT_MAX_SOFT_TOKENS
        print(f"[INFO] VIT soft tokens: {self.vit_soft_tokens} (from {os.path.basename(vit_model_path)})")

        # Pre-load LLM InferManager FIRST (must grab NPU before onnxruntime import)
        self.infer_manager = InferManager(
            self.config, self.axmodel_path,
            max_seq_len=self.max_seq_len,
            per_layer_helper=self.per_layer_helper,
        )
        print(f"[INFO] LLM InferManager loaded: {axmodel_path}")

        # Pre-load VIT model AFTER LLM (onnxruntime import may interfere with NPU)
        self.vit_session = None
        self._vit_backend = None
        if vit_model_path and os.path.exists(vit_model_path):
            if vit_model_path.endswith(".axmodel"):
                from axengine import InferenceSession as AxInferenceSession
                self.vit_session = AxInferenceSession(vit_model_path)
                self._vit_backend = "axmodel"
            elif vit_model_path.endswith(".onnx"):
                import onnxruntime as ort
                self.vit_session = ort.InferenceSession(vit_model_path, providers=["CPUExecutionProvider"])
                self._vit_backend = "onnx"
                for inp in self.vit_session.get_inputs():
                    print(f"[INFO] VIT ONNX input: name={inp.name}, shape={inp.shape}, dtype={inp.type}")
            print(f"[INFO] VIT model loaded: {vit_model_path} ({self._vit_backend})")

    def _reset_kv_cache(self):
        for k_cache in self.infer_manager.k_caches:
            k_cache.fill(0)
        for v_cache in self.infer_manager.v_caches:
            v_cache.fill(0)

    def _run_vit(self, pixel_values: np.ndarray, expected_tokens: int):
        if self.vit_session is None:
            raise RuntimeError(f"VIT model not loaded: {self.vit_model_path}")
        if self._vit_backend == "onnx":
            input_name = self.vit_session.get_inputs()[0].name
            outputs = self.vit_session.run(None, {input_name: pixel_values})
        else:
            outputs = self.vit_session.run(None, {"pixel_values": pixel_values})
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
        return select_vit_output(outputs, self.config.hidden_size, expected_tokens=expected_tokens)

    def generate_stream(
        self,
        prompt: str,
        history,
        system_prompt: str,
        image,
        enable_thinking: bool = False,
        max_new_tokens: int = 1024,
        max_soft_tokens: int = None,
    ) -> Generator[str, None, None]:
        # Always use the VIT model's actual soft token count
        max_soft_tokens = self.vit_soft_tokens

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

        mm_token_type_ids = None
        if image is not None:
            resize_h, resize_w, expected_tokens = resolve_resize(max_soft_tokens)
            fixed_image = resize_image(image, resize_h=resize_h, resize_w=resize_w)
            mm_messages = build_processor_messages(
                prompt=prompt,
                image=fixed_image,
                system_prompt=system_prompt,
                history=history,
            )
            inputs = _safe_apply_chat_template(
                self.processor,
                mm_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                processor_kwargs={
                    "images_kwargs": {
                        "do_resize": False,
                        "max_soft_tokens": max_soft_tokens,
                    }
                },
            )
            image_token_count = int((inputs["input_ids"] == self.processor.image_token_id).sum().item())
            if image_token_count != expected_tokens:
                raise ValueError(
                    f"Expected {expected_tokens} image soft tokens from fixed resize, got {image_token_count}."
                )
            token_ids = inputs["input_ids"][0].cpu().numpy().tolist()
            mm_token_type_ids = inputs["mm_token_type_ids"][0].cpu().numpy().tolist()
            pixel_values = to_numpy_fp32(inputs["pixel_values"])

            image_embeds = self._run_vit(pixel_values, expected_tokens=expected_tokens)
            if image_embeds.ndim == 3:
                image_embeds = image_embeds[0]
            if image_embeds.shape[0] != expected_tokens:
                raise ValueError(
                    "Unexpected vision output token count. "
                    f"got={image_embeds.shape[0]}, expected={expected_tokens}"
                )

            prefill_data = np.take(self.embeds, token_ids, axis=0)
            prefill_data = replace_image_tokens(
                token_ids,
                prefill_data,
                image_embeds,
                image_token_id=self.config.image_token_id,
            ).astype(bfloat16)
        else:
            text_messages = build_messages(prompt=prompt, system_prompt=system_prompt)
            text = self.tokenizer.apply_chat_template(
                text_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = self.tokenizer(text, return_tensors="np")
            token_ids = inputs["input_ids"][0].tolist()
            prefill_data = np.take(self.embeds, token_ids, axis=0).astype(bfloat16)

        prefill_per_layer_inputs = None
        if self.per_layer_helper is not None:
            per_layer_source_embeds = np.asarray(prefill_data, dtype=np.float32)
            per_layer_token_ids = list(token_ids)
            scale = float(self.config.hidden_size ** 0.5)
            pad_token_id = int(getattr(self.config, "pad_token_id", 0) or 0)
            for i in range(len(token_ids)):
                per_layer_source_embeds[i] *= scale
                if mm_token_type_ids is not None and int(mm_token_type_ids[i]) != 0:
                    per_layer_token_ids[i] = pad_token_id
            prefill_per_layer_inputs = self.per_layer_helper.compute(per_layer_token_ids, per_layer_source_embeds)

        self._reset_kv_cache()
        t0 = time.perf_counter()
        token_ids = self.infer_manager.prefill(
            self.tokenizer,
            token_ids,
            prefill_data,
            mm_token_type_ids=mm_token_type_ids,
            slice_len=self.slice_len,
            per_layer_inputs=prefill_per_layer_inputs,
        )
        ttft = time.perf_counter() - t0

        yield f"[TTFT: {ttft * 1000:.1f} ms] "

        eos_token_id = self.config.eos_token_id if isinstance(self.config.eos_token_id, list) else None
        t_decode_start = time.perf_counter()
        decoded_count = 0

        for text_so_far in self.infer_manager.decode_stream(
            self.tokenizer,
            token_ids,
            self.embeds,
            slice_len=self.slice_len,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        ):
            decoded_count += 1
            elapsed = time.perf_counter() - t_decode_start
            speed = decoded_count / elapsed if elapsed > 0 else 0.0
            yield f"[TTFT: {ttft * 1000:.1f} ms | {speed:.1f} tok/s] {text_so_far}"

    def chat(self, user_input, image, system_prompt, enable_thinking, max_new_tokens):
        user_text = (user_input or "").strip()
        if not user_text and image is None:
            yield [], gr.update(), gr.update(), gr.update(), gr.update()
            return

        yield (
            [(user_text, "...")],
            gr.update(value=""),
            gr.update(),
            gr.update(value="<div style='text-align: right; font-size: 13px; color: #6b7280; font-family: monospace;'>TTFT -- ms&nbsp;&nbsp;|&nbsp;&nbsp;Speed -- tok/s&nbsp;&nbsp;|&nbsp;&nbsp;Tokens --</div>"),
            gr.update(interactive=False),
        )

        chatbot_history = [(user_text, "")]
        for chunk in self.generate_stream(
            prompt=user_text,
            history=[],
            system_prompt=system_prompt,
            image=image,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
        ):
            # chunk format: "[TTFT: xx ms | yy tok/s] actual_text"
            # parse metrics from chunk prefix
            display_text = chunk
            ttft_disp, speed_disp, tok_disp = "--", "--", "--"
            if chunk.startswith("["):
                bracket_end = chunk.find("]")
                if bracket_end > 0:
                    metrics_str = chunk[1:bracket_end]
                    display_text = chunk[bracket_end + 1:].lstrip()
                    parts = metrics_str.split("|")
                    for p in parts:
                        p = p.strip()
                        if "TTFT" in p:
                            ttft_disp = p.replace("TTFT:", "").replace("ms", "").strip()
                        elif "tok/s" in p:
                            speed_disp = p.replace("tok/s", "").strip()

            chatbot_history[-1] = (user_text, display_text)
            metrics_html = (
                f"<div style='text-align: right; font-size: 13px; color: #6b7280; font-family: monospace;'>"
                f"TTFT {ttft_disp} ms&nbsp;&nbsp;|&nbsp;&nbsp;Speed {speed_disp} tok/s</div>"
            )
            yield chatbot_history, gr.update(value=""), gr.update(), gr.update(value=metrics_html), gr.update(interactive=False)

        yield chatbot_history, gr.update(value=""), gr.update(), gr.update(value=metrics_html), gr.update(interactive=True)


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 E2B Gradio Demo")
    parser.add_argument("--hf_model", type=str, default=_default_hf_model(),
                        help="Path to Gemma 4 tokenizer/config directory")
    parser.add_argument("--axmodel_path", type=str, default=_default_axmodel_path(),
                        help="Path to compiled LLM axmodel folder")
    parser.add_argument("--vit_model_path", type=str, default=_default_vit_model_path(),
                        help="Path to Gemma 4 vision ONNX model or .axmodel")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio listen address")
    parser.add_argument("--share", action="store_true", help="Enable gradio share")
    args = parser.parse_args()

    demo_engine = Gemma4GradioDemo(args.hf_model, args.axmodel_path, args.vit_model_path)

    custom_js = """
    function() {
        setTimeout(() => {
            const textareas = document.querySelectorAll('#user-input textarea');
            textareas.forEach(textarea => {
                textarea.removeEventListener('keydown', textarea._customKeyHandler);
                textarea._customKeyHandler = function(e) {
                    if (e.key === 'Enter') {
                        if (e.shiftKey) {
                            e.preventDefault();
                            const start = this.selectionStart;
                            const end = this.selectionEnd;
                            const value = this.value;
                            this.value = value.substring(0, start) + '\\n' + value.substring(end);
                            this.selectionStart = this.selectionEnd = start + 1;
                            this.dispatchEvent(new Event('input', { bubbles: true }));
                        } else {
                            e.preventDefault();
                            const sendBtn = document.querySelector('#send-btn');
                            if (sendBtn) { sendBtn.click(); }
                        }
                    }
                };
                textarea.addEventListener('keydown', textarea._customKeyHandler);
            });
        }, 500);
    }
    """

    with gr.Blocks(title="Gemma-4-E2B-it AX NPU Demo", theme=gr.themes.Soft(), js=custom_js) as iface:
        gr.HTML("""<style>
        #image-pane img {object-fit: contain; max-height: 380px;}
        #chat-wrap {position: relative;}
        #metrics-display {position: absolute; right: 12px; bottom: 12px; z-index: 5; pointer-events: none; text-align: right;}
        #metrics-display > div {display: inline-block;}
        </style>""")
        gr.Markdown("### Gemma-4-E2B-it on AX NPU\n Upload an image (optional), type your question, and get a response.")

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Group(elem_id="chat-wrap"):
                    chatbot = gr.Chatbot(height=500, label="Chat", type="tuples")
                    metrics_md = gr.Markdown(
                        "<div style='text-align: right; font-size: 13px; color: #6b7280; font-family: monospace;'>"
                        "TTFT -- ms&nbsp;&nbsp;|&nbsp;&nbsp;Speed -- tok/s</div>",
                        elem_id="metrics-display",
                    )

                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Press Enter to send, Shift+Enter for newline",
                        lines=2,
                        scale=7,
                        max_lines=5,
                        show_label=False,
                        elem_id="user-input",
                    )
                    with gr.Column(scale=1, min_width=100):
                        send_btn = gr.Button("Send", variant="primary", size="sm", elem_id="send-btn")
                        clear_btn = gr.Button("Clear", variant="secondary", size="sm")

            with gr.Column(scale=3):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image (optional)",
                    height=380,
                    image_mode="RGB",
                    show_download_button=False,
                    elem_id="image-pane",
                )
                with gr.Accordion("Advanced Settings", open=False):
                    system_prompt = gr.Textbox(label="System Prompt", value="You are a helpful assistant.", lines=2)
                    enable_thinking = gr.Checkbox(label="Enable Thinking", value=False)
                    max_new_tokens = gr.Slider(label="Max New Tokens", minimum=32, maximum=2048, step=32, value=1024)
                vit_h, vit_w = DEFAULT_RESIZE_BY_SOFT_TOKENS.get(demo_engine.vit_soft_tokens, (0, 0))
                gr.Markdown(
                    f"- VIT: **{demo_engine.vit_soft_tokens}** soft tokens ({vit_h}x{vit_w})\n"
                    f"- Supports single-image understanding\n"
                    f"- Single-turn only, no history"
                )

        def _clear():
            return (
                [],
                gr.update(value=""),
                gr.update(),
                gr.update(value="<div style='text-align: right; font-size: 13px; color: #6b7280; font-family: monospace;'>TTFT -- ms&nbsp;&nbsp;|&nbsp;&nbsp;Speed -- tok/s</div>"),
                gr.update(interactive=True),
            )

        send_btn.click(
            fn=demo_engine.chat,
            inputs=[user_input, image_input, system_prompt, enable_thinking, max_new_tokens],
            outputs=[chatbot, user_input, image_input, metrics_md, send_btn],
            show_progress=False,
            queue=True,
        )
        clear_btn.click(fn=_clear, inputs=None, outputs=[chatbot, user_input, image_input, metrics_md, send_btn])

    ips = _list_host_ips()
    print(f"Starting Gradio server on port {args.port}")
    for ip in ips:
        print(f"  http://{ip}:{args.port}")
    iface.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
