#!/usr/bin/env python3
"""Memory-staged Z-Image-Turbo inference for Jetson Orin NX 16GB."""

import argparse
import gc
import inspect
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from diffusers import ZImagePipeline
from transformers import AutoTokenizer, T5EncoderModel


def fmt_bytes(value):
    return f"{value / (1024 ** 3):.2f} GiB"


def read_meminfo():
    values = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0]) * 1024
    except OSError:
        pass
    return values


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup(label=None):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if label:
        snapshot(label)


def snapshot(label):
    cuda_sync()
    mem = read_meminfo()
    parts = [f"[memory] {label}"]
    if torch.cuda.is_available():
        parts.append(f"cuda_alloc={fmt_bytes(torch.cuda.memory_allocated())}")
        parts.append(f"cuda_reserved={fmt_bytes(torch.cuda.memory_reserved())}")
        parts.append(f"cuda_peak={fmt_bytes(torch.cuda.max_memory_allocated())}")
    if mem:
        parts.append(f"ram_available={fmt_bytes(mem.get('MemAvailable', 0))}")
        parts.append(f"ram_total={fmt_bytes(mem.get('MemTotal', 0))}")
    print(" | ".join(parts), flush=True)


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def is_oom(error):
    text = str(error).lower()
    return "out of memory" in text or "cuda error: out of memory" in text or "cublas_status_alloc_failed" in text


def pick_text_encoder_dir(model, base_model):
    for root in [Path(base_model) if base_model else None, Path(model)]:
        if root and (root / "text_encoder").exists():
            return root / "text_encoder"
    raise FileNotFoundError("Could not find text_encoder in --base_model or --model")


def pick_tokenizer_dir(model, base_model):
    for root in [Path(base_model) if base_model else None, Path(model)]:
        for name in ["tokenizer", "tokenizer_2"]:
            if root and (root / name).exists():
                return root / name
    raise FileNotFoundError("Could not find tokenizer/tokenizer_2 in --base_model or --model")


def chat_format_prompts(tokenizer, texts):
    formatted = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        if hasattr(tokenizer, "apply_chat_template"):
            formatted.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )
        else:
            formatted.append(text)
    return formatted


@torch.inference_mode()
def encode_prompts(args):
    tokenizer_dir = pick_tokenizer_dir(args.model, args.base_model)
    text_encoder_dir = pick_text_encoder_dir(args.model, args.base_model)
    print(f"Loading tokenizer: {tokenizer_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)

    max_length = args.max_sequence_length
    if max_length is None:
        max_length = min(getattr(tokenizer, "model_max_length", 512), 512)
        if max_length > 10000:
            max_length = 512

    texts = [args.prompt]
    need_negative = args.guidance_scale > 0
    negative_texts = [args.negative_prompt or ""] if need_negative else []

    def encode_text_batch(encoder, device, batch_texts):
        formatted = chat_format_prompts(tokenizer, batch_texts)
        tokens = tokenizer(
            formatted,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device).bool()
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-2]
        embeds = [hidden[i][attention_mask[i]].detach().to("cpu", dtype=torch.float16) for i in range(hidden.shape[0])]
        del outputs, hidden, input_ids, attention_mask, tokens
        return embeds

    def run_on(device):
        print(f"Loading T5 text encoder on {device}: {text_encoder_dir}", flush=True)
        encoder = T5EncoderModel.from_pretrained(
            text_encoder_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        encoder.eval()
        encoder.to(device)
        snapshot(f"text encoder loaded on {device}")

        prompt_embeds = encode_text_batch(encoder, device, texts)
        negative_prompt_embeds = encode_text_batch(encoder, device, negative_texts) if need_negative else None

        del encoder
        cleanup(f"text encoder released from {device}")
        return prompt_embeds, negative_prompt_embeds

    reset_peak()
    snapshot("before text encoder load")
    if torch.cuda.is_available() and not args.force_text_encoder_cpu:
        try:
            return run_on("cuda")
        except RuntimeError as error:
            if not is_oom(error):
                raise
            print(f"CUDA OOM while encoding prompt, retrying T5 on CPU: {error}", flush=True)
            cleanup("after failed CUDA text encoding")

    return run_on("cpu")


def configure_pipeline(pipe):
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")
    if hasattr(pipe, "vae") and pipe.vae is not None:
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

    transformer = getattr(pipe, "transformer", None)
    if transformer is not None and hasattr(transformer, "set_attention_backend"):
        for backend in ["flash", "_flash_3", "sdpa"]:
            try:
                transformer.set_attention_backend(backend)
                print(f"attention backend: {backend}", flush=True)
                break
            except Exception:
                pass


def instantiate_pipeline(args, torch_dtype):
    print(f"Loading ZImagePipeline without T5/tokenizer: {args.model}", flush=True)
    kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "text_encoder": None,
        "tokenizer": None,
    }
    try:
        return ZImagePipeline.from_pretrained(args.model, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("torch_dtype")
        return ZImagePipeline.from_pretrained(args.model, torch_dtype="auto", **kwargs)


def load_pipeline(args):
    dtype_map = {"transformer": torch.float8_e4m3fn, "vae": torch.float16, "default": torch.float16}
    pipe = instantiate_pipeline(args, dtype_map)
    configure_pipeline(pipe)

    try:
        pipe.to("cuda")
        return pipe
    except RuntimeError as error:
        if not is_oom(error):
            raise
        print(f"CUDA OOM while moving pipeline to GPU, retrying with model CPU offload: {error}", flush=True)
        del pipe
        cleanup("after failed pipe.to cuda")

    pipe = instantiate_pipeline(args, "auto")
    configure_pipeline(pipe)
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload(gpu_id=0)
        print("enabled model CPU offload fallback", flush=True)
        return pipe

    pipe.to("cuda")
    return pipe


def filtered_call(pipe, kwargs):
    signature = inspect.signature(pipe.__call__)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return pipe(**kwargs)
    return pipe(**{k: v for k, v in kwargs.items() if k in signature.parameters})


@torch.inference_mode()
def move_prompt_list(prompt_embeds, device):
    if prompt_embeds is None:
        return None
    return [item.to(device, non_blocking=True) for item in prompt_embeds]


def generate_with_retries(pipe, prompt_embeds, negative_prompt_embeds, args):
    sizes = [(args.height, args.width)]
    if args.height > 768 or args.width > 768:
        sizes.append((768, 768))
    if args.height > 512 or args.width > 512:
        sizes.append((512, 512))

    last_error = None
    for height, width in sizes:
        reset_peak()
        cleanup(f"before generate {height}x{width}")
        try:
            generator = torch.Generator("cuda").manual_seed(args.seed)
            call_kwargs = {
                "prompt": None,
                "prompt_embeds": move_prompt_list(prompt_embeds, "cuda"),
                "negative_prompt_embeds": move_prompt_list(negative_prompt_embeds, "cuda"),
                "height": height,
                "width": width,
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "generator": generator,
            }

            started = time.time()
            result = filtered_call(pipe, call_kwargs)
            cuda_sync()
            print(f"generation_time_sec={time.time() - started:.1f}", flush=True)
            snapshot("after generate")
            return result.images[0], (height, width)
        except RuntimeError as error:
            last_error = error
            if "call_kwargs" in locals():
                del call_kwargs
            if not is_oom(error):
                raise
            print(f"CUDA OOM at {height}x{width}, retrying smaller if available: {error}", flush=True)
            cleanup(f"after OOM {height}x{width}")

    raise RuntimeError(f"All generation attempts failed; last error: {last_error}")


def main():
    parser = argparse.ArgumentParser(description="Run Z-Image-Turbo on a 16 GB Jetson Orin NX")
    parser.add_argument("--model", default="/models/z-image-turbo-fp8-diffusers")
    parser.add_argument("--base_model", default="/models/z-image-turbo", help="Base Diffusers model containing tokenizer/text_encoder")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--output", default="/output/zimage.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--force_text_encoder_cpu", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this script.", file=sys.stderr)
        sys.exit(2)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}", flush=True)
    snapshot("startup")

    prompt_embeds, negative_prompt_embeds = encode_prompts(args)
    snapshot("prompt embeddings ready on CPU")

    reset_peak()
    snapshot("before loading pipe")
    pipe = load_pipeline(args)
    snapshot("after loading pipe")

    image, final_size = generate_with_retries(pipe, prompt_embeds, negative_prompt_embeds, args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"saved={output} size={final_size[0]}x{final_size[1]}", flush=True)

    del image, pipe, prompt_embeds, negative_prompt_embeds
    cleanup("shutdown")


if __name__ == "__main__":
    main()
