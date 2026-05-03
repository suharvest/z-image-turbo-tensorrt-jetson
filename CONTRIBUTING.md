# Contributing

Contributions are welcome, especially:

- Jetson Orin Nano / 8GB validation
- Additional resolution profiles
- Engine build automation
- Memory footprint reductions
- Cleaner Docker packaging
- Accuracy comparisons against upstream diffusers

Before opening a pull request:

1. Keep generated artifacts out of git: no model weights, ONNX files, TensorRT
   engines, or output images except curated README media.
2. Run syntax checks for modified Python and shell scripts.
3. Document hardware, JetPack, TensorRT version, resolution, step count, and
   memory/cache settings for performance claims.
4. Do not claim support for devices that were not tested.

For large behavioral changes, include a short note in `docs/TRT_STATUS.md`.
