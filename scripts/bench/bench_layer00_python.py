#!/usr/bin/env python3
import os
import time

import tensorrt as trt
import torch


ENGINE = os.environ.get(
    "ENGINE",
    "/home/harvest/models/axera-onnx/trt-engines-bf16/layer_00_fp16.engine",
)
WARMUP = int(os.environ.get("WARMUP", "10"))
ITERS = int(os.environ.get("ITERS", "100"))


class Engine:
    def __init__(self, path):
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.inputs = []
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append((name, shape))
            else:
                self.outputs[name] = shape

    def bind(self, tensors):
        for name, _ in self.inputs:
            t = tensors[name].contiguous()
            self.context.set_tensor_address(name, t.data_ptr())
            self.context.set_input_shape(name, tuple(t.shape))
        for name in self.outputs:
            t = tensors[name].contiguous()
            self.context.set_tensor_address(name, t.data_ptr())

    def run(self):
        self.context.execute_async_v3(self.stream.cuda_stream)


def main():
    torch.cuda.init()
    tensors = {
        "x": torch.randn((1, 1152, 3840), dtype=torch.float16, device="cuda"),
        "attn_mask": torch.ones((1, 1152), dtype=torch.bool, device="cuda"),
        "freqs_cis": torch.randn((1, 1152, 128), dtype=torch.float16, device="cuda"),
        "adaln_input": torch.randn((1, 256), dtype=torch.float16, device="cuda"),
        "output": torch.empty((1, 1152, 3840), dtype=torch.float16, device="cuda"),
    }
    engine = Engine(ENGINE)
    engine.bind(tensors)

    for _ in range(WARMUP):
        engine.run()
    engine.stream.synchronize()

    start = time.perf_counter()
    for _ in range(ITERS):
        engine.run()
    engine.stream.synchronize()
    elapsed = time.perf_counter() - start
    print(f"python engine={ENGINE}")
    print(f"iters={ITERS} total_ms={elapsed*1000:.3f} avg_ms={elapsed*1000/ITERS:.3f}")


if __name__ == "__main__":
    main()
