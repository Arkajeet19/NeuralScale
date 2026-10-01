# Mini-DLSS: 4x Image Super-Resolution

A smaller-scale take on NVIDIA DLSS — a CNN that reconstructs high-resolution
detail from a low-resolution image, trained from scratch on DIV2K and served
through a full-stack demo with a live before/after comparison.

![demo slider placeholder](outputs/eval_samples/compare_00.png)

## Results

Evaluated on the DIV2K validation set (100 images), full-resolution, Y-channel
PSNR/SSIM with border cropping — the standard protocol used in SR papers, not
the small-patch RGB metric used for quick training-time monitoring.

| Method  | PSNR (dB) | SSIM   |
|---------|-----------|--------|
| Bicubic | 28.10     | 0.7858 |
| **Model (ours)** | **30.00** | **0.8400** |

**+1.90 dB / +0.054 SSIM over bicubic interpolation**, trained on roughly
1/6th the iterations typical full EDSR-baseline training recipes use.

## What this is

An EDSR-baseline architecture (16 residual blocks, 64 channels, ~1.4M
parameters, no batch norm, sub-pixel convolution upsampling) trained from
scratch on the DIV2K dataset for 4x single-image super-resolution, with a
FastAPI backend and a browser-based demo for visually comparing the model's
output against plain bicubic upscaling.

## Why this project

Medical-imaging CNNs and sentiment classifiers are common beginner deep
learning projects. This one is a smaller, buildable version of a problem
real production systems (NVIDIA DLSS, game engine upscalers) solve at a much
larger scale — reconstructing plausible high-frequency detail that isn't
present in the low-resolution input, rather than just classifying an image.

## Architecture

- **Residual blocks**: the network learns the *difference* between the
  low-res input and high-res target, not the whole image from scratch —
  easier to train, faster to converge.
- **No BatchNorm**: removing it (per the EDSR paper) avoids normalizing away
  pixel-intensity information the model needs to preserve, and cuts memory
  usage meaningfully.
- **Sub-pixel convolution (PixelShuffle)** for upsampling instead of
  transposed convolution, avoiding checkerboard artifacts — the same
  technique real-time upscalers use because it's cheap and artifact-free.
- **L1 loss**, not MSE — MSE over-penalizes large errors and biases the
  model toward blurry, "safe" outputs; L1 produces sharper results.

## A real bug I found and fixed along the way

The first training run plateaued well *below* the bicubic baseline after 60
epochs. Debugging traced it to `res_scale=0.1` in the residual blocks — a
stabilization trick the EDSR paper uses only for its much deeper 32-block/
256-channel model, not the 16-block baseline this project uses. With it
dampened to 10%, the residual signal through every block was artificially
throttled, and the LR schedule's decay compounded the problem by cutting the
learning rate before the (already slow) model had learned enough. Setting
`res_scale=1.0` to match the actual EDSR-baseline config, combined with a
`repeat` factor fixing an undersized effective epoch (the dataset was
yielding only ~800 patches per epoch instead of a properly-sized pass over
the data), fixed convergence and produced the results above.

## Project structure

```
mini-dlss/
├── data/raw/              # DIV2K dataset (not included -- see Setup)
├── src/
│   ├── download_data.py   # downloads + extracts DIV2K
│   ├── dataset.py         # paired LR/HR patch dataset with caching + augmentation
│   ├── model.py            # EDSR-baseline architecture
│   ├── train.py             # training loop, checkpointing, TensorBoard logging
│   └── evaluate.py          # Y-channel PSNR/SSIM vs bicubic, full validation images
├── backend/app.py          # FastAPI inference server
├── frontend/index.html     # drag-and-drop demo with comparison slider
├── models/                 # saved checkpoints (not included)
└── outputs/                # evaluation sample images, training logs
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python src/download_data.py    # downloads DIV2K (~5-6 GB)
```

## Training

```bash
python src/train.py
```

Trains for 60 epochs (~12,800 patches/epoch at 48px LR / 192px HR patches,
batch size 16), checkpointing every epoch to `models/latest.pth` and saving
`models/best.pth` whenever validation PSNR improves. Safe to interrupt and
resume — training automatically picks up from the last checkpoint.

## Evaluation

```bash
python src/evaluate.py
```

Runs full-image inference on all 100 DIV2K validation images, reports
PSNR/SSIM for both the model and a bicubic baseline (Y-channel, standard SR
evaluation protocol), and saves side-by-side comparison images to
`outputs/eval_samples/`.

## Demo

```bash
uvicorn backend.app:app --reload --port 8000
```

Then open `frontend/index.html` in a browser. Drag any image onto the page
to see a live 4x super-resolution comparison against bicubic upscaling.

## Hardware

Trained on a single NVIDIA RTX 3050 (8GB VRAM). Full 60-epoch training run
takes roughly 1.5-2 hours once images are cached in memory (the first epoch
is slower due to one-time disk reads from the dataset).

## Tech stack

PyTorch · FastAPI · HTML/CSS/JS (no build step) · DIV2K dataset

## Possible next steps

- **SRGAN / perceptual loss**: add an adversarial + perceptual loss component
  for sharper, more realistic textures than pure PSNR-optimized L1 training
  tends to produce.
- **Video frame interpolation**: a simplified version of DLSS Frame
  Generation, synthesizing in-between frames from consecutive ones.
- **Game-screenshot fine-tuning**: fine-tune on gameplay footage specifically,
  since DIV2K is general photography and real-time upscalers target
  synthetic/rendered content with different texture statistics.
