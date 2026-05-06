#!/usr/bin/env python3
"""HTTP wrapper for the no-PyTorch TensorRT Z-Image runtime."""

import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


DEFAULT_PROMPT = (
    "A cute orange tabby cat sitting on a sunny windowsill, "
    "soft natural lighting, photorealistic, high detail"
)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/z-image-api/uploads"))
RESOLUTION = int(os.environ.get("RESOLUTION", "384"))

app = FastAPI(title="Z-Image Turbo TensorRT API", version="0.1.0")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
request_lock = Lock()


class GenerateJsonRequest(BaseModel):
    prompt: str = Field(default=DEFAULT_PROMPT)
    image_path: Optional[str] = Field(default=None, description="Path visible inside the container.")
    num_steps: int = Field(default=4, ge=1, le=50)
    strength: float = Field(default=0.6, ge=0.0, le=1.0)
    seed: int = Field(default=42)
    output_name: Optional[str] = Field(default=None)


def output_path(output_name: Optional[str]) -> Path:
    if output_name:
        name = Path(output_name).name
        if not name.lower().endswith(".png"):
            name += ".png"
    else:
        name = f"z-image-{uuid.uuid4().hex}.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def save_upload(upload: UploadFile) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "input.png").suffix or ".png"
    path = UPLOAD_DIR / f"input-{uuid.uuid4().hex}{suffix}"
    with path.open("wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return str(path)


def generate_image(
    *,
    prompt: str,
    init_image: Optional[str],
    num_steps: int,
    strength: float,
    seed: int,
    output_name: Optional[str],
):
    if init_image and not os.path.exists(init_image):
        raise FileNotFoundError(f"Input image not found: {init_image}")

    out_path = output_path(output_name)
    mode = "img2img" if init_image else "text2img"
    start = time.time()
    env = os.environ.copy()
    env.update(
        {
            "PROMPT": prompt,
            "NUM_STEPS": str(num_steps),
            "STRENGTH": str(strength),
            "SEED": str(seed),
            "OUTPUT_PATH": str(out_path),
            "INIT_IMAGE": init_image or "",
            "IMG2IMG_ENCODE_ONLY": "0",
        }
    )
    with request_lock:
        proc = subprocess.run(
            ["python3", "-u", "/workspace/pipeline_trt_no_torch.py"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    elapsed = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or f"pipeline exited {proc.returncode}").strip())
    if not out_path.exists():
        raise RuntimeError(f"Pipeline finished but output image was not created: {out_path}")
    trt_seconds = None
    match = re.search(r"Total TRT: ([0-9.]+)s", proc.stdout)
    if match:
        trt_seconds = float(match.group(1))
    return {
        "success": True,
        "mode": mode,
        "resolution": RESOLUTION,
        "num_steps": num_steps,
        "strength": strength if init_image else None,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 3),
        "trt_seconds": trt_seconds,
        "image_path": str(out_path),
        "image_url": f"/outputs/{out_path.name}",
    }


@app.get("/health")
def health():
    return {"success": True, "status": "ok", "resolution": RESOLUTION}


@app.post("/generate")
def generate_multipart(
    prompt: str = Form(default=DEFAULT_PROMPT),
    image: Optional[UploadFile] = File(default=None),
    num_steps: int = Form(default=4),
    strength: float = Form(default=0.6),
    seed: int = Form(default=42),
    output_name: Optional[str] = Form(default=None),
):
    init_image = None
    try:
        if image is not None:
            init_image = save_upload(image)
        result = generate_image(
            prompt=prompt,
            init_image=init_image,
            num_steps=num_steps,
            strength=strength,
            seed=seed,
            output_name=output_name,
        )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)


@app.post("/generate_json")
def generate_json(request: GenerateJsonRequest):
    try:
        result = generate_image(
            prompt=request.prompt,
            init_image=request.image_path,
            num_steps=request.num_steps,
            strength=request.strength,
            seed=request.seed,
            output_name=request.output_name,
        )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)
