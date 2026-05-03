#!/usr/bin/env python3
"""Verify TRT engines produce clean (non-NaN) output for Z-Image-Turbo layers.

Usage: python3 verify_trt_engine.py /path/to/engine
"""

import sys, os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())

def verify_engine(engine_path, test_name):
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"Engine: {engine_path}")

    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    # Get binding info
    num_io = engine.num_io_tensors
    print(f"  IO tensors: {num_io}")

    io_info = {}
    for i in range(num_io):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        io_mode = engine.get_tensor_mode(name)
        io_info[name] = {"idx": i, "shape": shape, "dtype": dtype, "mode": io_mode}
        print(f"  {'IN' if io_mode == trt.TensorIOMode.INPUT else 'OUT'} [{i}] {name}: shape={shape}, dtype={dtype}")

    # Allocate buffers and create random FP16 inputs
    inputs = {}
    outputs = {}
    bindings = {}

    for name, info in io_info.items():
        shape = tuple(info["shape"])
        size = int(np.prod(shape))

        if info["mode"] == trt.TensorIOMode.INPUT:
            if str(info["dtype"]) == "DataType.BOOL":
                data = np.ones(shape, dtype=np.bool_)
            elif str(info["dtype"]) == "DataType.FLOAT":
                data = np.random.randn(*shape).astype(np.float32) * 0.1
            elif str(info["dtype"]) == "DataType.HALF":
                data = np.random.randn(*shape).astype(np.float16) * 0.1
            else:
                print(f"  WARNING: unknown input dtype {info['dtype']}, skipping")
                return False

            gpu_mem = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(gpu_mem, data)
            inputs[name] = data
            bindings[name] = gpu_mem
        else:
            gpu_mem = cuda.mem_alloc(size * 2)  # FP16 = 2 bytes
            outputs[name] = np.empty(shape, dtype=np.float16)
            bindings[name] = gpu_mem

    # Set tensor addresses
    for name, gpu_mem in bindings.items():
        context.set_tensor_address(name, int(gpu_mem))

    # Run inference
    stream = cuda.Stream()
    context.execute_async_v3(stream.handle)
    stream.synchronize()

    # Check outputs
    all_ok = True
    for name, info in io_info.items():
        if info["mode"] == trt.TensorIOMode.OUTPUT:
            cuda.memcpy_dtoh(outputs[name], bindings[name])
            out = outputs[name]
            nan_count = np.isnan(out).sum()
            inf_count = np.isinf(out).sum()
            total = out.size
            status = "OK" if (nan_count == 0 and inf_count == 0) else "FAIL"
            if status == "FAIL":
                all_ok = False
            print(f"  {name}: shape={out.shape} mean={out.astype(np.float32).mean():.6f} "
                  f"std={out.astype(np.float32).std():.6f} "
                  f"NaN={nan_count} Inf={inf_count} [{status}]")

    # Cleanup
    for gpu_mem in bindings.values():
        gpu_mem.free()

    if all_ok:
        print(f"  RESULT: PASS")
    else:
        print(f"  RESULT: FAIL (NaN/Inf detected)")
    return all_ok


if __name__ == "__main__":
    import random
    engines = sys.argv[1:] if len(sys.argv) > 1 else []

    if not engines:
        # Default: test 3 random layers
        import glob
        all_engines = sorted(glob.glob("/tmp/trt-engines/layer_*_fp16.engine"))
        # Pick: first, middle, last
        engines = [
            all_engines[0],           # layer_00
            all_engines[14],          # layer_14
            all_engines[-1],          # layer_29
        ]

    results = {}
    for engine_path in engines:
        name = os.path.basename(engine_path).replace(".engine", "")
        results[name] = verify_engine(engine_path, name)

    print(f"\n{'='*60}")
    print("SUMMARY")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    all_pass = all(results.values())
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
