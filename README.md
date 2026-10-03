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

## Stretch goal: real-time performance benchmarking

Quality metrics alone don't tell the whole DLSS story -- real-time upscaling
has to run within a tight per-frame latency budget. To measure actual
deployability rather than just claim it:

- **Exported the model to ONNX** (`src/export_onnx.py`) for comparison
  against PyTorch's standard execution path.
- **Benchmarked 4 backends** (`src/benchmark.py`) across 3 resolution
  presets simulating realistic internal-render -> display-resolution
  upscaling scenarios (240p->960p, 320p->1280p, 480p->1920p): PyTorch FP32,
  PyTorch FP16, ONNX Runtime (CPU), and ONNX Runtime with NVIDIA's
  TensorRT-RTX execution provider.
- **Caught and fixed a benchmarking bug along the way**: an initial pass
  showed ONNX Runtime's CUDA provider performing far worse than PyTorch,
  scaling non-linearly with resolution -- traced to the benchmark copying
  data between CPU and GPU on every single inference call instead of
  keeping tensors GPU-resident, fixed using ONNX Runtime's IO Binding API.

**Results** (480p -> 1920p, RTX 3050 8GB):

| Backend | ms/frame | FPS |
|---|---|---|
| PyTorch FP32 | 125.3 | 8.0 |
| **PyTorch FP16** | **73.4** | **13.6** |
| ONNX Runtime CPU | 2183.0 | 0.5 |
| ONNX Runtime TensorRT-RTX | 120.9 | 8.3 |

(Full results across all three resolutions in `outputs/benchmark_results.csv`.)

**PyTorch FP16 was the fastest backend tested** -- a genuinely useful,
near-free win (same architecture, half the memory bandwidth). TensorRT-RTX,
somewhat counter-intuitively, did not outperform it here. Two caveats
matter for interpreting this fairly: the TensorRT-RTX run used an FP32
ONNX graph (not an apples-to-apples precision comparison against PyTorch's
FP16 result), and this package's execution provider doesn't currently
expose the IO Binding API the plain CUDA provider does, so its measured
time still includes CPU<->GPU transfer overhead the other GPU backends
avoid. Within those constraints, the likely explanation is that TensorRT's
graph-optimization benefits (operator fusion, kernel tuning) matter most
for larger, more complex models -- a lean 16-block, 1.4M-parameter network
may simply have little inefficiency left for it to optimize away.

**None of the tested backends hit conventional real-time thresholds
(30+ FPS) at the largest resolution tested (480p -> 1920p)** -- 13.6 FPS
with FP16 is the honest number. At the smaller presets, PyTorch FP16
does clear 30 FPS (48.4 FPS at 240p->960p, 29.6 FPS at 320p->1280p). This
is a legitimate finding, not a failure: it shows precisely where a
quality-focused model this size sits relative to true real-time deployment,
and what would need to change (model compression, a lighter architecture,
or a fairer FP16 TensorRT comparison) to close the gap.

## Possible next steps

- **Rebalance the GAN**: lower the discriminator's learning rate or increase
  the adversarial loss weight to counter the imbalance observed above, for
  a more pronounced sharpening effect.
- **Video frame interpolation**: a simplified version of DLSS Frame
  Generation, synthesizing in-between frames from consecutive ones.
- **Larger fine-tuning set**: more collected screenshots would likely reduce
  the discriminator-memorization effect seen above.
- **Fairer TensorRT-RTX comparison**: test with FP16 precision enabled
  (rather than the FP32 ONNX graph used above) for an apples-to-apples
  comparison against PyTorch FP16's current win.
- **Model compression**: pruning or a lighter architecture to close the gap
  to real-time (30+ FPS) at larger resolutions.