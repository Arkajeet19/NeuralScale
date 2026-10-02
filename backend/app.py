"""
FastAPI backend for the mini-DLSS demo.

Loads BOTH trained checkpoints (plain DIV2K model + GAN-finetuned-on-game-
screenshots model) once at startup, and serves a single endpoint that
returns bicubic, plain-model, and GAN-model outputs together -- powering
the React frontend's 3-way comparison viewer.

Run from the project root:
    uvicorn backend.app:app --reload --port 8000

Then run the React frontend's dev server (see frontend/README or `npm run dev`).
"""
import os
import sys
import io
import time
import base64

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import EDSRBaseline

SCALE = 4
PLAIN_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pth")
GAN_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gan_generator.pth")
MAX_INPUT_DIMENSION = 512  # safety cap -- a 512px LR image becomes a 2048px output at 4x

app = FastAPI(title="Mini-DLSS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {"plain": None, "gan": None}


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    m = EDSRBaseline(scale=SCALE, num_blocks=16, channels=64).to(device)
    ckpt = torch.load(path, map_location=device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


@app.on_event("startup")
def load_models():
    models["plain"] = load_checkpoint(PLAIN_CHECKPOINT_PATH)
    models["gan"] = load_checkpoint(GAN_CHECKPOINT_PATH)
    for name, m in models.items():
        status = "loaded" if m is not None else "MISSING -- that output will error"
        print(f"[{name}] {status}")
    print(f"Running on {device}")


def image_to_base64(img_array_uint8):
    img = Image.fromarray(img_array_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_inference(model, lr_tensor):
    if device.type == "cuda":
        torch.cuda.synchronize()  # wait for any prior queued GPU work before starting the timer
    t0 = time.time()
    with torch.no_grad():
        sr_tensor = model(lr_tensor).clamp(0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize()  # GPU ops are async -- this blocks until compute actually finishes,
                                        # otherwise we'd just be timing how fast the kernel launch returned
    elapsed_ms = (time.time() - t0) * 1000
    sr_array = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return sr_array, elapsed_ms


@app.get("/health")
def health():
    return {
        "status": "ok",
        "plain_loaded": models["plain"] is not None,
        "gan_loaded": models["gan"] is not None,
        "device": str(device),
    }


@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    if models["plain"] is None or models["gan"] is None:
        raise HTTPException(
            status_code=503,
            detail="One or both model checkpoints are missing -- check models/best.pth and models/gan_generator.pth"
        )

    contents = await file.read()
    try:
        lr_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image")

    w, h = lr_img.size
    if max(w, h) > MAX_INPUT_DIMENSION:
        ratio = MAX_INPUT_DIMENSION / max(w, h)
        lr_img = lr_img.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC)
        w, h = lr_img.size

    lr_array = np.array(lr_img)
    lr_tensor = torch.from_numpy(lr_array / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)

    plain_array, plain_ms = run_inference(models["plain"], lr_tensor)
    gan_array, gan_ms = run_inference(models["gan"], lr_tensor)

    t0 = time.time()
    bicubic_img = lr_img.resize((w * SCALE, h * SCALE), Image.BICUBIC)
    bicubic_array = np.array(bicubic_img)
    bicubic_ms = (time.time() - t0) * 1000

    return JSONResponse({
        "bicubic_b64": image_to_base64(bicubic_array),
        "plain_b64": image_to_base64(plain_array),
        "gan_b64": image_to_base64(gan_array),
        "original_size": [w, h],
        "output_size": [w * SCALE, h * SCALE],
        "timing_ms": {
            "bicubic": round(bicubic_ms, 1),
            "plain": round(plain_ms, 1),
            "gan": round(gan_ms, 1),
        },
        "device": str(device),
    })
