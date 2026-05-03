#!/usr/bin/env python3
"""Compatibility entrypoint for the FP8 ComfyUI-to-Diffusers conversion."""

import os
import runpy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(SCRIPT_DIR, "convert_fp8_stream.py"), run_name="__main__")
