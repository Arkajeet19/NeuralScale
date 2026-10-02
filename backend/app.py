"""
FastAPI backend for the mini-DLSS demo.

Loads the trained model once at startup, then serves a single endpoint that
takes an uploaded low-res image and returns both the model's upscaled output
and a bicubic-upscaled version (for the frontend's before/after comparison).

Run from the project root:
    uvicorn backend.app:app --reload --port 8000

Then open frontend/index.html in a browser (it talks to http://localhost:8000).
"""
import os
import sys
import io
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
# Switched to the GAN-finetuned generator. Point back at "best.pth" to
# serve the plain DIV2K-trained model instead.
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gan_generator.pth")
MAX_INPUT_DIMENSION = 512  # safety cap -- a 512px LR image becomes a 2048px output at 4x,
                            # which is already a lot of VRAM; bigger uploads get downscaled first

app = FastAPI(title="Mini-DLSS API")

# allow the frontend (opened as a local file or on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"WARNING: no checkpoint found at {CHECKPOINT_PATH}. "
              f"/upscale will fail until training produces models/best.pth")
        return
    model = EDSRBaseline(scale=SCALE, num_blocks=16, channels=64).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded on {device} (checkpoint epoch {ckpt['epoch']+1})")


def image_to_base64(img_array_uint8):
    img = Image.fromarray(img_array_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": str(device)}


@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded -- is models/best.pth present?")

    contents = await file.read()
    try:
        lr_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image")

    # downscale oversized uploads so inference stays within VRAM limits
    w, h = lr_img.size
    if max(w, h) > MAX_INPUT_DIMENSION:
        ratio = MAX_INPUT_DIMENSION / max(w, h)
        lr_img = lr_img.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC)
        w, h = lr_img.size

    lr_array = np.array(lr_img)

    # ---- model inference ----
    lr_tensor = torch.from_numpy(lr_array / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
    with torch.no_grad():
        sr_tensor = model(lr_tensor).clamp(0, 1)
    sr_array = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

    # ---- bicubic comparison ----
    bicubic_img = lr_img.resize((w * SCALE, h * SCALE), Image.BICUBIC)
    bicubic_array = np.array(bicubic_img)

    return JSONResponse({
        "original_b64": image_to_base64(lr_array),
        "bicubic_b64": image_to_base64(bicubic_array),
        "model_b64": image_to_base64(sr_array),
        "original_size": [w, h],
        "output_size": [w * SCALE, h * SCALE],
    })