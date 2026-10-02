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

## Stretch goal: SRGAN fine-tuning on real gameplay footage

The base model above is trained on DIV2K, which is general photography --
not representative of rendered game content (flat shading, anti-aliased
edges, particle effects, UI elements). To close that gap and push toward
the actual DLSS use case:

- **Collected a custom dataset** of ~180 screenshots across 10 AAA titles
  (mixed 768p/1080p/1440p/4K, all genuine native captures or verified
  official press sources -- no re-compressed/resized web images, to avoid
  training on baked-in compression artifacts).
- **Added a discriminator network + VGG19 perceptual loss** (`src/discriminator.py`,
  `src/perceptual_loss.py`) on top of the existing generator, following the
  SRGAN approach: pixel (L1) + perceptual (VGG feature-space) + adversarial
  loss combined, rather than pixel loss alone.
- **Warm-started from the DIV2K-trained checkpoint** rather than training
  the GAN from scratch, since GAN training is notoriously unstable early on
  -- starting from an already-competent generator skips that failure-prone
  phase (`src/train_gan.py`).

**Result**: 32.67 dB / 0.8639 SSIM vs. a 32.80 dB / 0.8667 bicubic baseline
on held-out game screenshots -- essentially tied with bicubic on PSNR/SSIM,
while producing visibly sharper texture detail (clothing, foliage) with no
GAN-typical artifacts on inspection.

This is a known and documented tradeoff in SR research: adversarial/perceptual
training optimizes for *looking* realistic, which doesn't always align with
pixel-accurate PSNR. Training logs showed the discriminator outpacing the
generator partway through fine-tuning (D_loss dropped from ~0.76 to ~0.24
while generator adversarial loss rose from ~1.4 to ~3.7) -- likely a result
of the relatively small fine-tuning set (~162 images) letting the
discriminator effectively memorize "real," which limited how much the
adversarial term could push the output away from an accurate reconstruction.
A deliberately low adversarial-loss weight (0.001) kept this from degrading
output quality, at the cost of a more muted version of the classic SRGAN
"look" than a more aggressive weighting would produce.

## Possible next steps

- **Rebalance the GAN**: lower the discriminator's learning rate or increase
  the adversarial loss weight to counter the imbalance observed above, for
  a more pronounced sharpening effect.
- **Video frame interpolation**: a simplified version of DLSS Frame
  Generation, synthesizing in-between frames from consecutive ones.
- **Larger fine-tuning set**: more collected screenshots would likely reduce
  the discriminator-memorization effect seen above.
