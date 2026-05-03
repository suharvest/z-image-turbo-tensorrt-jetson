"""TRT-accelerated Z-Image-Turbo using BF16 TensorRT engines.

Pipeline:
  1. Text Encoder (PyTorch) → prompt_embeds [1,128,2560]
  2. TRT prompt_preprocessor → processed_prompt [1,128,3840]
  3. Random latent [1,16,resolution/8,resolution/8]
  4. TRT latent_preprocessor → image_tokens [1,(resolution/16)^2,3840]
  5. Context/noise refiners → concat image+text tokens
  6. Pre-compute 3D RoPE for the active resolution
  7. For each denoising step:
     a. TRT t_embedder: timestep → adaln_input [1,256]
     b. TRT layer_0..29: x → x (30 layers)
     c. TRT final_projection: x → noise_pred [1,16,resolution/8,resolution/8]
     d. Scheduler step
  8. VAE Decoder (PyTorch or TensorRT) → image

512 and 384 modes use separate static-shape engines; select with RESOLUTION.
"""
import json, os, time, math, numpy as np
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
RUNTIME = trt.Runtime(TRT_LOGGER)

class TRTEngine:
    """Wrapper for a TRT engine with named I/O."""
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            self.engine = RUNTIME.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.inputs = []
        self.outputs = {}
        self.output_dtypes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs[name] = shape
                self.output_dtypes[name] = self._torch_dtype(self.engine.get_tensor_dtype(name))

    @staticmethod
    def _torch_dtype(dtype):
        if dtype == trt.DataType.HALF:
            return torch.float16
        if hasattr(trt.DataType, "BF16") and dtype == trt.DataType.BF16:
            return torch.bfloat16
        if dtype == trt.DataType.FLOAT:
            return torch.float32
        if dtype == trt.DataType.INT32:
            return torch.int32
        if dtype == trt.DataType.INT64:
            return torch.int64
        if dtype == trt.DataType.BOOL:
            return torch.bool
        raise TypeError(f"Unsupported TRT output dtype for torch allocation: {dtype}")

    def __call__(self, **tensors):
        """Run inference. Keys must match engine input names."""
        output_tensors = tensors.pop("_outputs", None)
        for name in self.inputs:
            t = tensors[name].contiguous().cuda()
            self.context.set_tensor_address(name, t.data_ptr())
            self.context.set_input_shape(name, tuple(t.shape))
        result = {}
        for name, shape in self.outputs.items():
            if output_tensors is not None and name in output_tensors:
                t = output_tensors[name]
            else:
                s = [1 if v == -1 else v for v in list(shape)]
                t = torch.empty(tuple(s), dtype=self.output_dtypes[name], device="cuda")
            self.context.set_tensor_address(name, t.data_ptr())
            result[name] = t
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return result


def precompute_freqs_cis_3d(img_h=32, img_w=32, text_len=128, theta=256.0):
    """3D RoPE matching diffusers Z-Image basic mode.

    Official token order is [image, caption], but position IDs are built before
    concatenation:
      caption: [1..text_len, 0, 0]
      image:   [text_len + 1, h, w]

    Returns freqs_cis [1, total_len, 128] FP16 in interleaved format.
    """
    axes_dims = [32, 48, 48]
    axes_lens = [1536, 512, 512]
    total_len = img_h * img_w + text_len

    # Build pose_ids for unified [image, caption] sequence.
    pose_ids = torch.zeros(total_len, 3, dtype=torch.long)
    for i in range(img_h * img_w):
        pose_ids[i, 0] = text_len + 1  # frame/sequence axis, after caption
        pose_ids[i, 1] = i // img_w    # height
        pose_ids[i, 2] = i % img_w     # width
    for i in range(text_len):
        pose_ids[img_h * img_w + i, 0] = i + 1

    # Generate per-axis freqs_cis
    all_freqs = []
    for dim, length in zip(axes_dims, axes_lens):
        with torch.no_grad():
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(length).float()
            angles = torch.outer(t, freqs)  # [length, dim/2]
            cos_sin = torch.empty(length, dim)
            cos_sin[:, 0::2] = torch.cos(angles)
            cos_sin[:, 1::2] = torch.sin(angles)
            all_freqs.append(cos_sin)

    # Index by pose_ids and concatenate
    parts = []
    for axis_idx in range(3):
        idx = pose_ids[:, axis_idx]  # [total_len]
        parts.append(all_freqs[axis_idx][idx])  # [total_len, dim]

    freqs_cis = torch.cat(parts, dim=-1).unsqueeze(0)  # [1, total_len, 128]
    return freqs_cis.to(torch.float16)


class TRTZImagePipelineV2:
    """TRT-accelerated Z-Image-Turbo using BF16 engines."""

    def __init__(self, engine_dir="/engines-v2", model_dir="/models/z-image-turbo-fp8-diffusers"):
        self.engine_dir = engine_dir
        self.vae_engine_dir = os.environ.get("VAE_ENGINE_DIR", engine_dir)
        self.model_dir = model_dir
        self.device = "cuda"
        self.use_trt_vae = os.environ.get("USE_TRT_VAE", "0") == "1"
        self.vae_scale = None
        self.vae_shift = None
        self.resolution = int(os.environ.get("RESOLUTION", "512"))
        self.latent_h = self.resolution // 8
        self.latent_w = self.resolution // 8
        self.image_h_tokens = self.latent_h // 2
        self.image_w_tokens = self.latent_w // 2
        self.image_tokens = self.image_h_tokens * self.image_w_tokens
        self.text_tokens = int(os.environ.get("TEXT_TOKENS", "128"))
        self.seq_len = self.image_tokens + self.text_tokens

        # Monkey-patches for diffusers on Jetson
        import torch.distributed as dist
        if not hasattr(dist, "device_mesh"):
            dist.device_mesh = type("device_mesh", (), {"DeviceMesh": type("FakeDM", (), {})})
        import torch._dynamo.utils as du
        if not hasattr(du, "NP_SUPPORTED_MODULES"):
            du.NP_SUPPORTED_MODULES = {}
        import torch.nn.functional as F
        _orig_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = lambda *a, **kw: _orig_sdpa(*a, **{k:v for k,v in kw.items() if k!='enable_gqa'})
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        # Load PyTorch components first
        print("Loading PyTorch components...", flush=True)
        self._load_pytorch()

        self.engines = {}
        self.group_engines = {}
        self.loaded_layers = {}
        self.layer_output_buffers = None
        self.layer_paths = [f"{engine_dir}/layer_{i:02d}_fp16.engine" for i in range(30)]  # BF16
        self.group_layer_paths = {
            0: os.environ.get(
                "GROUP_00_04_ENGINE",
                "/models/axera-onnx/group-layers-00-04/layers_00_04_fp16.engine",
            )
        }
        default_cache = 23 if self.resolution == 384 else self.MAX_CACHED
        self.max_cached = int(os.environ.get("MAX_CACHED_LAYERS", str(default_cache)))
        self.use_group_layers = os.environ.get("USE_GROUP_LAYERS", "0") == "1"

        # Pre-compute freqs_cis (constant for all steps)
        self.freqs_cis = precompute_freqs_cis_3d(
            img_h=self.image_h_tokens, img_w=self.image_w_tokens, text_len=self.text_tokens
        ).to(self.device)
        self.image_freqs_cis = self.freqs_cis[:, :self.image_tokens, :]
        self.prompt_freqs_cis = self.freqs_cis[:, self.image_tokens:, :]
        self.attn_mask = torch.ones(1, self.seq_len, dtype=torch.bool, device=self.device)
        self.image_mask = torch.ones(1, self.image_tokens, dtype=torch.bool, device=self.device)
        self.prompt_mask = torch.ones(1, self.text_tokens, dtype=torch.bool, device=self.device)

    def _load_pytorch(self):
        if os.environ.get("MINIMAL_PYTORCH_LOAD", "1") == "1":
            from diffusers import FlowMatchEulerDiscreteScheduler
            from transformers import Qwen3Model, Qwen2Tokenizer

            self.text_encoder = Qwen3Model.from_pretrained(
                self.model_dir,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
            ).to(self.device).eval()
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                self.model_dir,
                subfolder="tokenizer",
            )
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.model_dir,
                subfolder="scheduler",
            )
            self.vae = None
        else:
            from diffusers import ZImagePipeline
            pipe = ZImagePipeline.from_pretrained(
                self.model_dir, torch_dtype=torch.bfloat16,
                device_map="balanced", max_memory={0: "8GB"},
            )
            self.text_encoder = pipe.text_encoder
            self.vae = pipe.vae
            self.scheduler = pipe.scheduler
            self.tokenizer = pipe.tokenizer
            del pipe.transformer
        import gc; gc.collect()
        torch.cuda.empty_cache()

    def _engine_path(self, base_name, engine_dir=None):
        engine_dir = engine_dir or self.engine_dir
        candidates = [
            f"{engine_dir}/{base_name}_fp16.engine",
            f"{engine_dir}/{base_name}.engine",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _load_engines(self):
        """Load all TRT engines (call AFTER freeing text_encoder)."""
        pre = [
            "latent_preprocessor", "t_embedder", "final_projection",
            "noise_refiner_00", "noise_refiner_01",
        ]
        for name in pre:
            path = self._engine_path(name)
            if os.path.exists(path):
                if name in self.engines:
                    continue
                t0 = time.time()
                self.engines[name] = TRTEngine(path)
                print(f"  {name}: {os.path.getsize(path)/1e6:.0f}MB ({time.time()-t0:.1f}s)", flush=True)
        # Load all layer engines
        for i in range(30):
            self._get_layer(i)
        print(f"  Layers: {len(self.loaded_layers)}/30 loaded", flush=True)

    MAX_CACHED = 18  # Validated on orin-nx 16GB with prompt/context release and layer output buffer reuse.

    def _get_layer(self, i):
        if i in self.loaded_layers:
            return self.loaded_layers[i]
        while len(self.loaded_layers) >= self.max_cached:
            del self.loaded_layers[min(self.loaded_layers.keys())]
        self.loaded_layers[i] = TRTEngine(self.layer_paths[i])
        return self.loaded_layers[i]

    def _get_group_layer(self, start):
        if start in self.group_engines:
            return self.group_engines[start]
        path = self.group_layer_paths[start]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing group layer engine: {path}")
        self.group_engines[start] = TRTEngine(path)
        return self.group_engines[start]

    def _release_engines(self, names):
        for name in names:
            self.engines.pop(name, None)

    def _release_text_encoder(self):
        if hasattr(self, "text_encoder"):
            del self.text_encoder

    def _calculate_shift(self):
        image_seq_len = self.image_tokens
        cfg = self.scheduler.config
        base_seq_len = cfg.get("base_image_seq_len", 256)
        max_seq_len = cfg.get("max_image_seq_len", 4096)
        base_shift = cfg.get("base_shift", 0.5)
        max_shift = cfg.get("max_shift", 1.15)
        mu = (max_shift - base_shift) / (max_seq_len - base_seq_len) * image_seq_len
        mu += base_shift - (max_shift - base_shift) / (max_seq_len - base_seq_len) * base_seq_len
        return mu

    def _set_timesteps(self, num_inference_steps):
        self.scheduler.sigma_min = 0.0
        try:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, mu=self._calculate_shift())
        except TypeError:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)

    def _load_vae(self):
        if self.vae is None:
            from diffusers import AutoencoderKL
            t0 = time.time()
            self.vae = AutoencoderKL.from_pretrained(
                self.model_dir,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
            ).to(self.device).eval()
            print(f"  VAE loaded in {time.time()-t0:.1f}s", flush=True)

    def _load_vae_config(self):
        if self.vae_scale is not None and self.vae_shift is not None:
            return
        config_path = os.path.join(self.model_dir, "vae", "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.vae_scale = float(cfg.get("scaling_factor", 1.0))
            self.vae_shift = float(cfg.get("shift_factor", 0.0))
            return
        self._load_vae()
        self.vae_scale = float(getattr(self.vae.config, "scaling_factor", 1.0))
        self.vae_shift = float(getattr(self.vae.config, "shift_factor", 0.0))

    def _load_init_image(self, image_path):
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        arr = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.mul(2.0).sub(1.0)

    def _prepare_img2img_latent(self, image_path, strength, generator):
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"STRENGTH must be in [0, 1], got {strength}")
        print(f"Encoding init image: {image_path} (strength={strength})", flush=True)
        self._load_vae_config()
        if self.use_trt_vae:
            path = self._engine_path("vae_encoder", self.vae_engine_dir)
            if not os.path.exists(path):
                raise FileNotFoundError(f"USE_TRT_VAE=1 but VAE encoder engine is missing: {path}")
            init_image = self._load_init_image(image_path).to(self.device, dtype=torch.float16)
            vae_encoder = TRTEngine(path)
            encoded = vae_encoder(image=init_image)
            mean = encoded["latent_mean"].to(torch.float32)
            std = encoded["latent_std"].to(torch.float32)
            eps = torch.randn(mean.shape, generator=generator, device=self.device, dtype=mean.dtype)
            image_latent = mean + std * eps
            image_latent = (image_latent - self.vae_shift) * self.vae_scale
            del vae_encoder
        else:
            self._load_vae()
            init_image = self._load_init_image(image_path).to(self.device, dtype=torch.bfloat16)
            with torch.no_grad():
                encoded = self.vae.encode(init_image)
                if hasattr(encoded, "latent_dist"):
                    image_latent = encoded.latent_dist.sample(generator=generator)
                elif hasattr(encoded, "latents"):
                    image_latent = encoded.latents
                else:
                    image_latent = encoded[0]
                image_latent = (image_latent - self.vae_shift) * self.vae_scale
                image_latent = image_latent.to(torch.float32)

        total_steps = len(self.scheduler.timesteps)
        init_timestep = min(total_steps, int(total_steps * strength))
        t_start = max(total_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]
        if len(timesteps) < 1:
            raise ValueError(f"STRENGTH={strength} leaves no denoising steps")
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
        timestep = timesteps[:1]
        noise = torch.randn(image_latent.shape, generator=generator, device=self.device, dtype=image_latent.dtype)
        latent = self.scheduler.scale_noise(image_latent, timestep, noise)
        return latent, timesteps

    def _decode_latent(self, latent):
        self._load_vae_config()
        latent = (latent / self.vae_scale) + self.vae_shift
        if self.use_trt_vae:
            path = self._engine_path("vae_decoder", self.vae_engine_dir)
            if not os.path.exists(path):
                raise FileNotFoundError(f"USE_TRT_VAE=1 but VAE decoder engine is missing: {path}")
            t0 = time.time()
            vae_decoder = TRTEngine(path)
            image = vae_decoder(latent=latent.to(torch.float16))["image"]
            print(f"  TRT VAE decoded in {time.time()-t0:.1f}s", flush=True)
            del vae_decoder
            return image

        if self.vae is None:
            self._load_vae()
        with torch.no_grad():
            image = self.vae.decode(latent.to(torch.bfloat16)).sample
        return image

    def encode_prompt(self, prompt):
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=128,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            output = self.text_encoder(
                input_ids=text_inputs.input_ids.to(self.device),
                attention_mask=text_inputs.attention_mask.to(self.device),
                output_hidden_states=True,
            )
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            return output.hidden_states[-2].to(torch.float16)
        return output[0].to(torch.float16)

    def run_step(self, x, adaln_input):
        """Run 30 TRT layers + final projection."""
        N_LAYERS = 30  # 30 layers work without NaN
        layer_buffers = None
        if os.environ.get("REUSE_LAYER_OUTPUT_BUFFERS", "1") == "1":
            if self.layer_output_buffers is None:
                self.layer_output_buffers = [
                    torch.empty((1, self.seq_len, 3840), dtype=torch.float16, device=self.device),
                    torch.empty((1, self.seq_len, 3840), dtype=torch.float16, device=self.device),
                ]
            layer_buffers = self.layer_output_buffers
        i = 0
        while i < N_LAYERS:
            if self.use_group_layers and i == 0:
                eng = self._get_group_layer(0)
                outputs = None
                if layer_buffers is not None:
                    outputs = {"output": layer_buffers[0]}
                out = eng(x=x, attn_mask=self.attn_mask, freqs_cis=self.freqs_cis,
                          adaln_input=adaln_input, _outputs=outputs)
                x = out["output"]
                i += 5
                continue
            eng = self._get_layer(i)
            outputs = None
            if layer_buffers is not None:
                outputs = {"output": layer_buffers[i % 2]}
            out = eng(x=x, attn_mask=self.attn_mask, freqs_cis=self.freqs_cis,
                      adaln_input=adaln_input, _outputs=outputs)
            x = out["output"]
            i += 1

        f_out = self.engines["final_projection"](
            hidden=x, adaln_input=adaln_input)
        return f_out["noise_pred"]  # [1, 16, 64, 64] FP16

    def generate(self, prompt, num_inference_steps=8, seed=42, init_image=None, strength=0.6):
        import gc

        # Phase 1: Encode prompt
        print(f"Encoding: '{prompt[:60]}...'", flush=True)
        t0 = time.time()
        prompt_embeds = self.encode_prompt(prompt)
        print(f"  Encoded in {time.time()-t0:.1f}s", flush=True)

        # Phase 2: Preprocess prompt and latent
        self.engines["prompt_preprocessor"] = TRTEngine(
            f"{self.engine_dir}/prompt_preprocessor_fp16.engine")
        self.engines["latent_preprocessor"] = TRTEngine(
            f"{self.engine_dir}/latent_preprocessor_fp16.engine")

        p_out = self.engines["prompt_preprocessor"](prompt_embeds=prompt_embeds)
        processed_prompt = p_out["processed_prompt"]  # [1, 128, 3840]
        for name in ("context_refiner_00", "context_refiner_01"):
            path = f"{self.engine_dir}/{name}.engine"
            if os.path.exists(path) and name not in self.engines:
                self.engines[name] = TRTEngine(path)
            if name in self.engines:
                processed_prompt = self.engines[name](
                    x=processed_prompt,
                    attn_mask=self.prompt_mask,
                    freqs_cis=self.prompt_freqs_cis,
                )["output"]
        self._release_engines(["prompt_preprocessor", "context_refiner_00", "context_refiner_01"])
        gc.collect(); torch.cuda.empty_cache()

        self._release_text_encoder()
        gc.collect(); torch.cuda.empty_cache()

        generator = torch.Generator(self.device).manual_seed(seed)
        self._set_timesteps(num_inference_steps)
        if init_image:
            latent, timesteps = self._prepare_img2img_latent(init_image, strength, generator)
            active_steps = len(timesteps)
        else:
            latent = torch.randn(1, 16, self.latent_h, self.latent_w, generator=generator, device=self.device)
            timesteps = self.scheduler.timesteps
            active_steps = num_inference_steps
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(0)

        # Phase 3: Free text encoder, load TRT engines
        print("Loading TRT engines...", flush=True)
        self._release_text_encoder()
        if self.vae is not None and os.environ.get("DELAY_LOAD_VAE", "1") == "1":
            del self.vae
            self.vae = None
        gc.collect(); torch.cuda.empty_cache()
        self._load_engines()

        # Phase 4: Denoising loop
        total_trt = 0
        for step_idx, t in enumerate(timesteps):
            print(f"  Step {step_idx+1}/{active_steps} (t={t:.0f})...", end=" ", flush=True)
            st = time.time()

            # diffusers passes normalized t=(1000-t)/1000 to the transformer,
            # and the transformer multiplies by t_scale=1000 before t_embedder.
            t_out = self.engines["t_embedder"](
                timestep=torch.tensor([1000.0 - float(t)], dtype=torch.float32)
            )
            adaln = t_out["adaln_input"]

            # Reprocess latent: current noisy latent → image tokens
            l_out = self.engines["latent_preprocessor"](latent=latent.to(torch.float32))
            image_tokens = l_out["image_tokens"]
            for name in ("noise_refiner_00", "noise_refiner_01"):
                image_tokens = self.engines[name](
                    x=image_tokens,
                    attn_mask=self.image_mask,
                    freqs_cis=self.image_freqs_cis,
                    adaln_input=adaln,
                )["output"]

            # Concat: image FIRST, then prompt (final_projection extracts image token span).
            x = torch.cat([image_tokens, processed_prompt], dim=1)

            noise_pred = -self.run_step(x, adaln).to(torch.float32)
            step_time = time.time() - st
            total_trt += step_time

            latent = self.scheduler.step(
                noise_pred, t, latent
            ).prev_sample
            print(f"{step_time:.1f}s", flush=True)

        print(f"Total TRT: {total_trt:.1f}s", flush=True)

        # Phase 5: Decode
        print("Decoding...", flush=True)
        if os.environ.get("FREE_TRT_BEFORE_VAE", "1") == "1":
            del self.engines
            del self.group_engines
            del self.loaded_layers
            self.layer_output_buffers = None
            gc.collect()
            torch.cuda.empty_cache()
        image = self._decode_latent(latent)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image[0]


if __name__ == "__main__":
    import numpy as np
    ENGINE_DIR = os.environ.get("ENGINE_DIR", "/engines-v2")
    MODEL_DIR = os.environ.get("MODEL_DIR", "/models/z-image-turbo-fp8-diffusers")

    pipeline = TRTZImagePipelineV2(ENGINE_DIR, MODEL_DIR)

    print(f"\nGenerating {os.environ.get('RESOLUTION', '512')}x{os.environ.get('RESOLUTION', '512')}...", flush=True)
    t0 = time.time()
    image = pipeline.generate(
        os.environ.get(
            "PROMPT",
            "A cute orange tabby cat sitting on a sunny windowsill, soft natural lighting, photorealistic, high detail",
        ),
        num_inference_steps=int(os.environ.get("NUM_STEPS", "8")), seed=42,
        init_image=os.environ.get("INIT_IMAGE") or None,
        strength=float(os.environ.get("STRENGTH", "0.6")),
    )
    total = time.time() - t0
    image = np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
    from PIL import Image as PILImage
    os.makedirs("/output", exist_ok=True)
    output_path = os.environ.get("OUTPUT_PATH", "/output/output_trt_v2.png")
    PILImage.fromarray((image * 255).astype(np.uint8)).save(output_path)
    print(f"IMAGE SAVED ({total:.1f}s total)", flush=True)
