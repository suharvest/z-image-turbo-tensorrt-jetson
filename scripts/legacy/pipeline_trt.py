"""TRT-accelerated Z-Image-Turbo inference using AXERA split ONNX engines.

Confirmed I/O (512x512 resolution):
  cfg_00(timestep [1]) → t_emb [1, 256]
  cfg_01(prompt_embeds [1, 128, 2560]) → prompt_out [128, 3840]
  cfg_02(latent [1, 16, 1, 64, 64], t_emb [1, 256]) → image_tokens [1024, 3840]
  cfg_03(prompt_out, image_tokens, t_emb) → layer_0 [1, 1152, 3840]
  cfg_04..cfg_32(layer_out, t_emb) → layer_1..layer_29 [1, 1152, 3840]
  auto_00(layer_29, t_emb) → noise_pred [1, 16, 1, 64, 64]

Each step: ~30 layers × 37.6ms = ~1.13s GPU time
8 steps: ~9s transformer + ~5s text encoder + VAE ≈ ~15s total
"""
import os, sys, time, numpy as np
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
RUNTIME = trt.Runtime(TRT_LOGGER)

class TRTEngine:
    """Wrapper for a single TRT engine with named I/O."""
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            self.engine = RUNTIME.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        self.input_names = []
        self.output_names = []
        self.output_shapes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = shape

    def __call__(self, **named_inputs):
        # Set inputs
        for name in self.input_names:
            tensor = named_inputs[name]
            tensor = tensor.contiguous().cuda()
            self.context.set_tensor_address(name, tensor.data_ptr())
            # Set input shape (handle dynamic dims)
            shape = tuple(tensor.shape)
            self.context.set_input_shape(name, shape)

        # Allocate and bind outputs
        outputs = {}
        for name in self.output_names:
            shape = list(self.output_shapes[name])
            # Replace -1 with actual batch from first input
            for j, s in enumerate(shape):
                if s == -1:
                    shape[j] = 1
            t = torch.empty(tuple(shape), dtype=torch.float32, device="cuda")
            self.context.set_tensor_address(name, t.data_ptr())
            outputs[name] = t

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs

class TRTZImagePipeline:
    """TRT-accelerated Z-Image-Turbo pipeline."""
    def __init__(self, engine_dir, model_dir):
        self.engine_dir = engine_dir
        self.model_dir = model_dir
        self.device = "cuda"

        # Monkey patches (needed for diffusers on Jetson)
        import torch.distributed as dist
        if not hasattr(dist, "device_mesh"):
            dist.device_mesh = type("device_mesh", (), {"DeviceMesh": type("FakeDM", (), {})})
        import torch._dynamo.utils as du
        if not hasattr(du, "NP_SUPPORTED_MODULES"):
            du.NP_SUPPORTED_MODULES = {}
        import torch.nn.functional as F
        _orig_sdpa = F.scaled_dot_product_attention
        def _patched_sdpa(*args, **kwargs):
            kwargs.pop("enable_gqa", False)
            return _orig_sdpa(*args, **kwargs)
        F.scaled_dot_product_attention = _patched_sdpa
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        # Load PyTorch components first (text encoder + scheduler, not VAE yet)
        print("Loading PyTorch components...", flush=True)
        self._load_pytorch()

        # Encode prompt now, while text encoder is in memory
        self._cached_prompt_embeds = None

        # Prepare TRT engine paths (not loaded yet)
        self.engine_dir = engine_dir
        self.engines = {}
        self.layer_paths = [self._engine_path(f"layer_{i}") for i in range(30)]
        self.loaded_layers = {}

    def _engine_path(self, name):
        """Return engine file path for a given logical name. Layer engines not preloaded."""
        b = self.engine_dir
        if name == "t_embedder":
            return f"{b}/cfg_00_timestep_to_model_t_embedder_mlp_mlp_2_Gemm_output_0_config.onnx.engine"
        if name == "prompt_prep":
            return f"{b}/cfg_01_prompt_embeds_to_model_Slice_1_output_0_config.onnx.engine"
        if name == "latent_prep":
            return f"{b}/cfg_02_latent_model_input_to_model_Slice_output_0_config.onnx.engine"
        if name == "final":
            return f"{b}/auto_00_model_layers_29_Add_4_output_0_to_sample_auto.onnx.engine"
        if name.startswith("layer_"):
            i = int(name.split("_")[1])
            if i == 0:
                return f"{b}/cfg_03_model_Slice_1_output_0_to_model_layers_0_Add_4_output_0_config.onnx.engine"
            return f"{b}/cfg_{i+3:02d}_model_layers_{i-1}_Add_4_output_0_to_model_layers_{i}_Add_4_output_0_config.onnx.engine"
        return None

    def _load_engines(self):
        """Load all TRT engines (call AFTER freeing PyTorch text encoder)."""
        import gc
        # Load non-layer engines
        for name in ["t_embedder", "prompt_prep", "latent_prep", "final"]:
            path = self._engine_path(name)
            if os.path.exists(path):
                t0 = time.time()
                self.engines[name] = TRTEngine(path)
                print(f"  {name}: {os.path.getsize(path)/1e6:.0f}MB ({time.time()-t0:.1f}s)", flush=True)
        # Load ALL layer engines now (PyTorch text_encoder is freed)
        for i in range(30):
            self._get_layer_engine(i)
        print(f"  Layers: {len(self.loaded_layers)}/30 loaded ({sum(1 for p in self.layer_paths if os.path.exists(p))} paths)", flush=True)

    def _load_pytorch(self):
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(
            self.model_dir,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            max_memory={0: "8GB"},
        )
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler
        self.tokenizer = pipe.tokenizer
        # Free transformer memory
        del pipe.transformer
        import gc; gc.collect()
        torch.cuda.empty_cache()

    def encode_prompt(self, prompt):
        """Qwen3 text encoder → prompt_embeds [1, 128, 2560]."""
        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=128,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            output = self.text_encoder(
                input_ids=text_inputs.input_ids.to(self.device),
                attention_mask=text_inputs.attention_mask.to(self.device),
            )
        # Return the raw hidden states [1, 128, 2560]
        return output[0].to(torch.float32)

    MAX_CACHED_LAYERS = 20

    def _get_layer_engine(self, i):
        """Load a specific layer engine on demand, evict oldest if cache full."""
        if i in self.loaded_layers:
            return self.loaded_layers[i]
        path = self.layer_paths[i]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing engine: {path}")
        # Evict oldest entries if cache full
        while len(self.loaded_layers) >= self.MAX_CACHED_LAYERS:
            oldest = min(self.loaded_layers.keys())
            del self.loaded_layers[oldest]
        engine = TRTEngine(path)
        self.loaded_layers[i] = engine
        return engine

    def _unload_layers(self):
        """Free all cached layer engines."""
        self.loaded_layers.clear()
        import gc; gc.collect()
        torch.cuda.empty_cache()

    def run_step(self, latent_4d, prompt_embeds, timestep):
        """Run one denoising step through TRT transformer (all engines preloaded)."""
        E = self.engines
        t_float = torch.tensor([float(timestep)], dtype=torch.float32)

        t_out = E["t_embedder"](timestep=t_float)
        t_emb = t_out["/model/t_embedder/mlp/mlp.2/Gemm_output_0"]

        p_out = E["prompt_prep"](prompt_embeds=prompt_embeds)
        prompt_tokens = p_out["/model/Slice_1_output_0"]

        latent_5d = latent_4d.unsqueeze(2).to(torch.float32)
        l_out = E["latent_prep"](
            **{"/model/t_embedder/mlp/mlp.2/Gemm_output_0": t_emb,
               "latent_model_input": latent_5d}
        )
        image_tokens = l_out["/model/Slice_output_0"]

        # Run all 30 transformer layers (all engines preloaded)
        layer_out = None
        for i in range(30):
            eng = self._get_layer_engine(i)
            if i == 0:
                out = eng(
                    **{"/model/Slice_1_output_0": prompt_tokens,
                       "/model/Slice_output_0": image_tokens,
                       "/model/t_embedder/mlp/mlp.2/Gemm_output_0": t_emb}
                )
            else:
                out = eng(
                    **{f"/model/layers.{i-1}/Add_4_output_0": layer_out,
                       "/model/t_embedder/mlp/mlp.2/Gemm_output_0": t_emb}
                )
            layer_out = out[f"/model/layers.{i}/Add_4_output_0"]

        f_out = E["final"](
            **{"/model/layers.29/Add_4_output_0": layer_out,
               "/model/t_embedder/mlp/mlp.2/Gemm_output_0": t_emb}
        )
        noise_pred = f_out["sample"].squeeze(2)
        return noise_pred

    def generate(self, prompt, height=512, width=512, num_inference_steps=8, seed=42):
        """Generate image from text prompt. Memory flow: encode→free→TRT→decode."""
        import gc

        # Phase 1: Encode prompt (PyTorch text encoder)
        print(f"Encoding prompt: '{prompt[:60]}...'", flush=True)
        t0 = time.time()
        prompt_embeds = self.encode_prompt(prompt)
        print(f"  Prompt encoded in {time.time()-t0:.1f}s", flush=True)

        # Free text encoder to make room for TRT engines
        print("Freeing text encoder...", flush=True)
        del self.text_encoder
        gc.collect()
        torch.cuda.empty_cache()

        # Phase 2: Load all TRT engines (now enough memory without text encoder)
        print("Loading TRT engines...", flush=True)
        t0 = time.time()
        self._load_engines()
        print(f"  All engines loaded in {time.time()-t0:.1f}s", flush=True)

        # Initial latent noise
        generator = torch.Generator(self.device).manual_seed(seed)
        latent = torch.randn(1, 16, 64, 64, generator=generator, device=self.device)
        latent = latent.to(torch.bfloat16)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Phase 3: Denoising loop (TRT only, no PyTorch transformer)
        total_trt_time = 0
        for step_idx, t in enumerate(timesteps):
            print(f"  Step {step_idx+1}/{len(timesteps)} (t={t:.0f})...", end=" ", flush=True)
            st = time.time()
            noise_pred = self.run_step(latent, prompt_embeds, t)
            step_time = time.time() - st
            total_trt_time += step_time
            latent = self.scheduler.step(
                noise_pred.to(torch.bfloat16), t, latent
            ).prev_sample
            print(f"{step_time:.1f}s", flush=True)

        print(f"Total TRT time: {total_trt_time:.1f}s", flush=True)

        # Phase 4: Decode latent with VAE (reload if needed)
        print("Decoding...", flush=True)
        with torch.no_grad():
            image = self.vae.decode(latent.to(torch.bfloat16)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image[0]


if __name__ == "__main__":
    ENGINE_DIR = "/engines"
    MODEL_DIR = "/models/z-image-turbo-fp8-diffusers"

    pipeline = TRTZImagePipeline(ENGINE_DIR, MODEL_DIR)

    print("\nGenerating 512x512 image...", flush=True)
    t0 = time.time()
    image = pipeline.generate(
        "A cute orange tabby cat sitting on a sunny windowsill, soft natural lighting, photorealistic, high detail",
        num_inference_steps=8, seed=42,
    )
    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s", flush=True)

    # Save
    import numpy as np
    image = np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
    from PIL import Image as PILImage
    os.makedirs("/output", exist_ok=True)
    PILImage.fromarray((image * 255).astype(np.uint8)).save("/output/output_trt.png")
    print(f"IMAGE SAVED to /output/output_trt.png ({total:.1f}s)", flush=True)
