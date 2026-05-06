#!/usr/bin/env python3
"""Text-to-image TensorRT runner that avoids importing PyTorch.

This is an experimental runtime-only path for 384/512 TRT engines. It uses
TensorRT Python bindings, CUDA Runtime via ctypes, NumPy, tokenizers, and PIL.
"""
import ctypes
import gc
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import tensorrt as trt
from PIL import Image
from tokenizers import Tokenizer


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
RUNTIME = trt.Runtime(TRT_LOGGER)


class Cuda:
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2
    DEVICE_TO_DEVICE = 3

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        stream = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        self.stream = stream

    def _check(self, code, name):
        if code != 0:
            raise RuntimeError(f"{name} failed with cuda error {code}")

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr

    def free(self, ptr):
        if ptr:
            self._check(self.lib.cudaFree(ptr), "cudaFree")

    def memcpy_h2d(self, dst, src):
        arr = np.ascontiguousarray(src)
        self._check(
            self.lib.cudaMemcpy(dst, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, self.HOST_TO_DEVICE),
            "cudaMemcpyH2D",
        )

    def memcpy_d2h(self, dst, src):
        arr = np.ascontiguousarray(dst)
        self._check(
            self.lib.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), src, arr.nbytes, self.DEVICE_TO_HOST),
            "cudaMemcpyD2H",
        )
        return arr

    def memcpy_d2d(self, dst, src, nbytes):
        self._check(self.lib.cudaMemcpy(dst, src, nbytes, self.DEVICE_TO_DEVICE), "cudaMemcpyD2D")

    def sync(self):
        self._check(self.lib.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")


CUDA = Cuda()


def trt_dtype_info(dtype):
    if dtype == trt.DataType.HALF:
        return np.float16, 2
    if hasattr(trt.DataType, "BF16") and dtype == trt.DataType.BF16:
        return np.uint16, 2
    if dtype == trt.DataType.FLOAT:
        return np.float32, 4
    if dtype == trt.DataType.INT64:
        return np.int64, 8
    if dtype == trt.DataType.INT32:
        return np.int32, 4
    if dtype == trt.DataType.BOOL:
        return np.bool_, 1
    raise TypeError(f"Unsupported TRT dtype: {dtype}")


def nbytes_for(shape, dtype):
    _, itemsize = trt_dtype_info(dtype)
    return int(np.prod(shape)) * itemsize


class DeviceBuffer:
    def __init__(self, shape, dtype, ptr=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.nbytes = nbytes_for(self.shape, dtype)
        self.ptr = ptr or CUDA.malloc(self.nbytes)
        self.owned = ptr is None

    def free(self):
        if self.owned and self.ptr:
            CUDA.free(self.ptr)
            self.ptr = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class TRTEngine:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.engine = RUNTIME.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = {}
        self.dtypes = {}
        self.shapes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            self.dtypes[name] = dtype
            self.shapes[name] = shape
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs[name] = shape

    def close(self):
        self.context = None
        self.engine = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def run(self, outputs=None, **inputs):
        outputs = outputs or {}
        temp_inputs = []
        result = {}
        for name in self.inputs:
            value = inputs[name]
            if isinstance(value, DeviceBuffer):
                buf = value
            else:
                np_dtype, _ = trt_dtype_info(self.dtypes[name])
                arr = np.ascontiguousarray(value, dtype=np_dtype)
                buf = DeviceBuffer(arr.shape, self.dtypes[name])
                CUDA.memcpy_h2d(buf.ptr, arr)
                temp_inputs.append(buf)
            self.context.set_input_shape(name, tuple(buf.shape))
            self.context.set_tensor_address(name, int(buf.ptr.value))
        for name, shape in self.outputs.items():
            buf = outputs.get(name)
            if buf is None:
                buf = DeviceBuffer(shape, self.dtypes[name])
            self.context.set_tensor_address(name, int(buf.ptr.value))
            result[name] = buf
        ok = self.context.execute_async_v3(int(CUDA.stream.value))
        if not ok:
            raise RuntimeError(f"TensorRT execute failed: {self.path}")
        CUDA.sync()
        for buf in temp_inputs:
            buf.free()
        return result


class LightweightQwenTokenizer:
    def __init__(self, model_dir):
        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        self.tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.pad_token_id = self.tokenizer.token_to_id(cfg.get("pad_token", "<|endoftext|>"))

    def apply_chat_template(self, prompt):
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, prompt, max_length=128):
        ids = self.tokenizer.encode(self.apply_chat_template(prompt)).ids[:max_length]
        attention = [1] * len(ids)
        pad = max_length - len(ids)
        ids += [self.pad_token_id] * pad
        attention += [0] * pad
        return SimpleNamespace(
            input_ids=np.asarray([ids], dtype=np.int64),
            attention_mask=np.asarray([attention], dtype=np.int64),
        )


class MinimalFlowMatchEulerScheduler:
    order = 1

    def __init__(self, model_dir):
        path = os.path.join(model_dir, "scheduler", "scheduler_config.json")
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.num_train_timesteps = int(cfg.get("num_train_timesteps", 1000))
        self.shift = float(cfg.get("shift", 1.0))
        self.base_shift = float(cfg.get("base_shift", 0.5))
        self.max_shift = float(cfg.get("max_shift", 1.15))
        self.base_image_seq_len = int(cfg.get("base_image_seq_len", 256))
        self.max_image_seq_len = int(cfg.get("max_image_seq_len", 4096))
        sigmas = np.linspace(1, self.num_train_timesteps, self.num_train_timesteps, dtype=np.float32)[::-1]
        sigmas = sigmas / self.num_train_timesteps
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.sigma_min = float(sigmas[-1])
        self.sigma_max = float(sigmas[0])
        self.set_timesteps(4)

    def set_timesteps(self, steps, mu=None):
        timesteps = np.linspace(self.sigma_max * self.num_train_timesteps, 0.0, steps, dtype=np.float32)
        sigmas = timesteps / self.num_train_timesteps
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = sigmas * self.num_train_timesteps
        self.sigmas = np.concatenate([sigmas.astype(np.float32), np.zeros(1, dtype=np.float32)])
        self._step_index = 0
        self._begin_index = 0

    def set_begin_index(self, begin_index):
        self._begin_index = int(begin_index)
        self._step_index = int(begin_index)

    def scale_noise(self, sample, timestep, noise):
        del timestep  # The caller has already selected the scheduler begin index.
        sigma = np.float32(self.sigmas[self._begin_index])
        return sigma * noise + (np.float32(1.0) - sigma) * sample

    def step(self, model_output, sample):
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        self._step_index += 1
        return sample + (sigma_next - sigma) * model_output


def precompute_freqs_cis_3d(img_h, img_w, text_len=128, theta=256.0):
    axes_dims = [32, 48, 48]
    axes_lens = [1536, 512, 512]
    total_len = img_h * img_w + text_len
    pose_ids = np.zeros((total_len, 3), dtype=np.int64)
    for i in range(img_h * img_w):
        pose_ids[i, 0] = text_len + 1
        pose_ids[i, 1] = i // img_w
        pose_ids[i, 2] = i % img_w
    for i in range(text_len):
        pose_ids[img_h * img_w + i, 0] = i + 1
    all_freqs = []
    for dim, length in zip(axes_dims, axes_lens):
        freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        t = np.arange(length, dtype=np.float32)
        angles = np.outer(t, freqs)
        cos_sin = np.empty((length, dim), dtype=np.float32)
        cos_sin[:, 0::2] = np.cos(angles)
        cos_sin[:, 1::2] = np.sin(angles)
        all_freqs.append(cos_sin)
    parts = [all_freqs[i][pose_ids[:, i]] for i in range(3)]
    return np.concatenate(parts, axis=-1)[None].astype(np.float16)


def bf16_device_to_fp16_device_cpu(buf):
    host = np.empty(buf.shape, dtype=np.uint16)
    CUDA.memcpy_d2h(host, buf.ptr)
    fp32 = (host.astype(np.uint32) << 16).view(np.float32)
    fp16 = fp32.astype(np.float16)
    out = DeviceBuffer(fp16.shape, trt.DataType.HALF)
    CUDA.memcpy_h2d(out.ptr, fp16)
    return out


def bf16_device_to_fp16_device(buf, engine_dir=None):
    if engine_dir:
        try:
            cast_engine = TRTEngine(engine_path(engine_dir, "bf16_to_fp16_1x128x2560"))
            out = cast_engine.run(input=buf)["output"]
            cast_engine.close()
            return out
        except FileNotFoundError:
            pass
    return bf16_device_to_fp16_device_cpu(buf)


def device_to_host(buf):
    dtype, _ = trt_dtype_info(buf.dtype)
    arr = np.empty(buf.shape, dtype=dtype)
    return CUDA.memcpy_d2h(arr, buf.ptr)


def load_init_image(image_path, resolution):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None]
    return (arr * 2.0 - 1.0).astype(np.float16)


def engine_path(engine_dir, base):
    for suffix in ("_bf16.engine", "_fp16.engine", ".engine"):
        path = os.path.join(engine_dir, base + suffix)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(base)


class NoTorchPipeline:
    def __init__(self):
        self.model_dir = os.environ.get("MODEL_DIR", "/models/z-image-turbo-fp8-diffusers")
        self.engine_dir = os.environ.get("ENGINE_DIR", "/engines-384-bf16")
        self.text_dir = os.environ.get("TEXT_ENCODER_ENGINE_DIR", "/models/axera-onnx/trt-text-encoder-split-g4")
        self.vae_dir = os.environ.get("VAE_ENGINE_DIR", self.engine_dir)
        self.resolution = int(os.environ.get("RESOLUTION", "384"))
        self.text_tokens = 128
        self.latent_h = self.resolution // 8
        self.latent_w = self.resolution // 8
        self.image_tokens = (self.latent_h // 2) * (self.latent_w // 2)
        self.seq_len = self.image_tokens + self.text_tokens
        self.tokenizer = LightweightQwenTokenizer(self.model_dir)
        self.scheduler = MinimalFlowMatchEulerScheduler(self.model_dir)
        self.freqs = precompute_freqs_cis_3d(self.latent_h // 2, self.latent_w // 2)
        self.freqs_full = DeviceBuffer(self.freqs.shape, trt.DataType.HALF)
        CUDA.memcpy_h2d(self.freqs_full.ptr, self.freqs)
        self.freqs_image = DeviceBuffer((1, self.image_tokens, 128), trt.DataType.HALF)
        self.freqs_prompt = DeviceBuffer((1, self.text_tokens, 128), trt.DataType.HALF)
        CUDA.memcpy_h2d(self.freqs_image.ptr, self.freqs[:, : self.image_tokens, :])
        CUDA.memcpy_h2d(self.freqs_prompt.ptr, self.freqs[:, self.image_tokens :, :])
        self.mask_full = np.ones((1, self.seq_len), dtype=np.bool_)
        self.mask_image = np.ones((1, self.image_tokens), dtype=np.bool_)
        self.mask_prompt = np.ones((1, self.text_tokens), dtype=np.bool_)
        default_cache = 30 if self.resolution == 384 else 18
        cache_env = os.environ.get("MAX_CACHED_LAYERS")
        self.max_cached_layers = int(cache_env) if cache_env else default_cache
        self.loaded_layers = {}
        self.layer_output_buffers = [
            DeviceBuffer((1, self.seq_len, 3840), trt.DataType.HALF),
            DeviceBuffer((1, self.seq_len, 3840), trt.DataType.HALF),
        ]

    def get_layer(self, idx):
        if idx in self.loaded_layers:
            return self.loaded_layers[idx]
        while len(self.loaded_layers) >= self.max_cached_layers:
            old_idx = min(self.loaded_layers.keys())
            self.loaded_layers.pop(old_idx).close()
            gc.collect()
        self.loaded_layers[idx] = TRTEngine(engine_path(self.engine_dir, f"layer_{idx:02d}"))
        return self.loaded_layers[idx]

    def encode_prompt(self, prompt):
        text = self.tokenizer(prompt)
        hidden = None
        ping = DeviceBuffer((1, self.text_tokens, 2560), trt.DataType.BF16)
        pong = DeviceBuffer((1, self.text_tokens, 2560), trt.DataType.BF16)
        groups = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19), (20, 23), (24, 27), (28, 31), (32, 35)]
        for idx, (start, end) in enumerate(groups):
            eng = TRTEngine(os.path.join(self.text_dir, f"text_encoder_group_{start:02d}_{end:02d}_bf16.engine"))
            out_buf = ping if idx % 2 == 0 else pong
            kwargs = {"input_ids": text.input_ids, "attention_mask": text.attention_mask}
            for name in eng.inputs:
                if name.startswith("hidden_states"):
                    kwargs[name] = hidden
            hidden = eng.run(outputs={"hidden_states": out_buf}, **kwargs)["hidden_states"]
            eng.close()
            del eng
            gc.collect()
        prompt_fp16 = bf16_device_to_fp16_device(hidden, self.text_dir)
        ping.free()
        pong.free()
        return prompt_fp16

    def prepare_img2img_latent(self, image_path, strength, rng):
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"STRENGTH must be in [0, 1], got {strength}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Init image not found: {image_path}")
        print(f"Encoding init image: {image_path} (strength={strength})", flush=True)
        init_image = load_init_image(image_path, self.resolution)
        vae_encoder = TRTEngine(engine_path(self.vae_dir, "vae_encoder"))
        encoded = vae_encoder.run(image=init_image)
        mean = device_to_host(encoded["latent_mean"]).astype(np.float32)
        std = device_to_host(encoded["latent_std"]).astype(np.float32)
        for buf in encoded.values():
            buf.free()
        vae_encoder.close()
        del vae_encoder
        gc.collect()
        eps = rng.standard_normal(mean.shape, dtype=np.float32)
        image_latent = mean + std * eps
        with open(os.path.join(self.model_dir, "vae", "config.json"), "r", encoding="utf-8") as f:
            vae_cfg = json.load(f)
        image_latent = (
            (image_latent - float(vae_cfg.get("shift_factor", 0.0)))
            * float(vae_cfg.get("scaling_factor", 1.0))
        ).astype(np.float32)

        total_steps = len(self.scheduler.timesteps)
        init_timestep = min(total_steps, int(total_steps * strength))
        t_start = max(total_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]
        if len(timesteps) < 1:
            raise ValueError(f"STRENGTH={strength} leaves no denoising steps")
        self.scheduler.set_begin_index(t_start)
        noise = rng.standard_normal(image_latent.shape, dtype=np.float32)
        latent = self.scheduler.scale_noise(image_latent, timesteps[:1], noise).astype(np.float32)
        return latent, timesteps, t_start

    def save_img2img_latent(self, image_path, strength, steps, seed, output_path):
        self.scheduler.set_timesteps(steps)
        rng = np.random.default_rng(seed)
        latent, timesteps, t_start = self.prepare_img2img_latent(image_path, strength, rng)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(
            output_path,
            latent=latent,
            t_start=np.asarray(t_start, dtype=np.int32),
            steps=np.asarray(steps, dtype=np.int32),
            timesteps=np.asarray(timesteps, dtype=np.float32),
        )
        print(f"INIT LATENT SAVED: {output_path}", flush=True)

    def load_img2img_latent(self, path, steps):
        data = np.load(path)
        latent = np.asarray(data["latent"], dtype=np.float32)
        saved_steps = int(np.asarray(data["steps"]).item())
        if saved_steps != steps:
            raise ValueError(f"INIT_LATENT_PATH was prepared with NUM_STEPS={saved_steps}, got {steps}")
        t_start = int(np.asarray(data["t_start"]).item())
        self.scheduler.set_begin_index(t_start)
        return latent, self.scheduler.timesteps[t_start:]

    def run(self, prompt, steps=4, seed=42, init_image=None, strength=0.6):
        t0 = time.time()
        print(f"Encoding: '{prompt[:60]}...'", flush=True)
        prompt_embeds = self.encode_prompt(prompt)
        print(f"  Encoded in {time.time() - t0:.1f}s", flush=True)

        prompt_pre = TRTEngine(engine_path(self.engine_dir, "prompt_preprocessor"))
        processed = prompt_pre.run(prompt_embeds=prompt_embeds)["processed_prompt"]
        prompt_embeds.free()
        prompt_pre.close()
        del prompt_pre
        for name in ("context_refiner_00", "context_refiner_01"):
            path = os.path.join(self.engine_dir, f"{name}.engine")
            if os.path.exists(path):
                out = DeviceBuffer((1, self.text_tokens, 3840), trt.DataType.HALF)
                refiner = TRTEngine(path)
                processed = refiner.run(
                    outputs={"output": out},
                    x=processed,
                    attn_mask=self.mask_prompt,
                    freqs_cis=self.freqs_prompt,
                )["output"]
                refiner.close()
                del refiner

        self.scheduler.set_timesteps(steps)
        rng = np.random.default_rng(seed)
        init_latent_path = os.environ.get("INIT_LATENT_PATH") or None
        if init_latent_path:
            print(f"Loading init latent: {init_latent_path}", flush=True)
            latent, timesteps = self.load_img2img_latent(init_latent_path, steps)
        elif init_image:
            latent, timesteps, _ = self.prepare_img2img_latent(init_image, strength, rng)
        else:
            latent = rng.standard_normal((1, 16, self.latent_h, self.latent_w), dtype=np.float32)
            timesteps = self.scheduler.timesteps
            self.scheduler.set_begin_index(0)
        active_steps = len(timesteps)

        latent_pre = TRTEngine(engine_path(self.engine_dir, "latent_preprocessor"))
        t_embed = TRTEngine(engine_path(self.engine_dir, "t_embedder"))
        final_proj = TRTEngine(engine_path(self.engine_dir, "final_projection"))
        noise_refiners = [TRTEngine(os.path.join(self.engine_dir, f"noise_refiner_{i:02d}.engine")) for i in range(2)]
        gc.collect()

        total_trt = 0.0
        for step_idx, timestep in enumerate(timesteps):
            st = time.time()
            print(f"  Step {step_idx + 1}/{active_steps} (t={timestep:.0f})...", end=" ", flush=True)
            adaln = t_embed.run(timestep=np.asarray([1000.0 - float(timestep)], dtype=np.float32))["adaln_input"]
            image_tokens = latent_pre.run(latent=latent.astype(np.float32))["image_tokens"]
            for refiner in noise_refiners:
                out = DeviceBuffer((1, self.image_tokens, 3840), trt.DataType.HALF)
                image_tokens = refiner.run(
                    outputs={"output": out},
                    x=image_tokens,
                    attn_mask=self.mask_image,
                    freqs_cis=self.freqs_image,
                    adaln_input=adaln,
                )["output"]

            x = DeviceBuffer((1, self.seq_len, 3840), trt.DataType.HALF)
            image_nbytes = nbytes_for((1, self.image_tokens, 3840), trt.DataType.HALF)
            prompt_nbytes = nbytes_for((1, self.text_tokens, 3840), trt.DataType.HALF)
            CUDA.memcpy_d2d(x.ptr, image_tokens.ptr, image_nbytes)
            dst_prompt = ctypes.c_void_p(x.ptr.value + image_nbytes)
            CUDA.memcpy_d2d(dst_prompt, processed.ptr, prompt_nbytes)

            for layer_idx in range(30):
                out = self.layer_output_buffers[layer_idx % 2]
                x = self.get_layer(layer_idx).run(
                    outputs={"output": out},
                    x=x,
                    attn_mask=self.mask_full,
                    freqs_cis=self.freqs_full,
                    adaln_input=adaln,
                )["output"]
            noise_dev = final_proj.run(hidden=x, adaln_input=adaln)["noise_pred"]
            noise = -device_to_host(noise_dev).astype(np.float32)
            latent = self.scheduler.step(noise, latent).astype(np.float32)
            dt = time.time() - st
            total_trt += dt
            print(f"{dt:.1f}s", flush=True)

        print(f"Total TRT: {total_trt:.1f}s", flush=True)
        with open(os.path.join(self.model_dir, "vae", "config.json"), "r", encoding="utf-8") as f:
            vae_cfg = json.load(f)
        latent_decode = ((latent / float(vae_cfg.get("scaling_factor", 1.0))) + float(vae_cfg.get("shift_factor", 0.0))).astype(np.float16)
        image_dev = TRTEngine(engine_path(self.vae_dir, "vae_decoder")).run(latent=latent_decode)["image"]
        image = device_to_host(image_dev).astype(np.float32)
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image[0], (1, 2, 0))
        return image


def main():
    prompt = os.environ.get(
        "PROMPT",
        "A cute orange tabby cat sitting on a sunny windowsill, soft natural lighting, photorealistic, high detail",
    )
    steps = int(os.environ.get("NUM_STEPS", "4"))
    init_image = os.environ.get("INIT_IMAGE") or None
    strength = float(os.environ.get("STRENGTH", "0.6"))
    seed = int(os.environ.get("SEED", "42"))
    pipe = NoTorchPipeline()
    if os.environ.get("IMG2IMG_ENCODE_ONLY", "0") == "1":
        if not init_image:
            raise ValueError("IMG2IMG_ENCODE_ONLY=1 requires INIT_IMAGE")
        init_latent_path = os.environ.get("INIT_LATENT_PATH", "/output/init_latent_no_torch.npz")
        pipe.save_img2img_latent(init_image, strength, steps, seed=seed, output_path=init_latent_path)
        return
    t0 = time.time()
    image = pipe.run(prompt, steps=steps, seed=seed, init_image=init_image, strength=strength)
    output_path = os.environ.get("OUTPUT_PATH", "/output/output_no_torch.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray((image * 255).astype(np.uint8)).save(output_path)
    print(f"IMAGE SAVED ({time.time() - t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
