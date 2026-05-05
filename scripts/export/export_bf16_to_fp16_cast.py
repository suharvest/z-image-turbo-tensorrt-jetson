#!/usr/bin/env python3
"""Export a tiny BF16 -> FP16 Cast ONNX for no-torch runtime engine boundaries."""

import os

import onnx
from onnx import TensorProto, helper


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "onnx-cast")
SHAPE = [1, 128, 2560]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Cast",
                inputs=["input"],
                outputs=["output"],
                to=TensorProto.FLOAT16,
            )
        ],
        name="bf16_to_fp16_1x128x2560",
        inputs=[helper.make_tensor_value_info("input", TensorProto.BFLOAT16, SHAPE)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT16, SHAPE)],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 17)],
        producer_name="z-image-turbo-jetson-trt",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = os.path.join(OUTPUT_DIR, "bf16_to_fp16_1x128x2560.onnx")
    onnx.save(model, path)
    print(path)


if __name__ == "__main__":
    main()
