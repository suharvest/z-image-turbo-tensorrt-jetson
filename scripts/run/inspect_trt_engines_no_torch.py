#!/usr/bin/env python3
"""Inspect TensorRT engine I/O without importing torch."""
import os

import tensorrt as trt


LOGGER = trt.Logger(trt.Logger.WARNING)
RUNTIME = trt.Runtime(LOGGER)


def inspect_engine(path):
    with open(path, "rb") as f:
        engine = RUNTIME.deserialize_cuda_engine(f.read())
    print(os.path.basename(path))
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        print(f"  {mode.name:6s} {name:24s} {str(dtype):12s} {tuple(shape)}")


def main():
    engine_dir = os.environ.get("ENGINE_DIR", "/engines-384-bf16")
    text_dir = os.environ.get("TEXT_ENCODER_ENGINE_DIR", "/models/axera-onnx/trt-text-encoder-split-g4")
    vae_dir = os.environ.get("VAE_ENGINE_DIR", engine_dir)
    names = [
        os.path.join(text_dir, "text_encoder_group_00_03_bf16.engine"),
        os.path.join(text_dir, "text_encoder_group_04_07_bf16.engine"),
        os.path.join(engine_dir, "prompt_preprocessor_fp16.engine"),
        os.path.join(engine_dir, "latent_preprocessor_fp16.engine"),
        os.path.join(engine_dir, "noise_refiner_00_fp16.engine"),
        os.path.join(engine_dir, "layer_00_fp16.engine"),
        os.path.join(engine_dir, "t_embedder_fp16.engine"),
        os.path.join(engine_dir, "final_projection_fp16.engine"),
        os.path.join(vae_dir, "vae_decoder_fp16.engine"),
    ]
    for path in names:
        if os.path.exists(path):
            inspect_engine(path)
        else:
            print(f"missing {path}")


if __name__ == "__main__":
    main()
