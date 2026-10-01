"""
Phase 3: Evaluation.

Computes PSNR and SSIM the way SR papers actually report them, so we can
tell for certain whether the model is beating bicubic upscaling:
  - Full validation images (not small training patches)
  - Y-channel only (luminance, via the standard ITU-R BT.601 RGB->YCbCr
    conversion) -- chrominance channels are excluded because the human eye
    is far less sensitive to color detail than brightness detail, and
    nearly every SR paper's reported numbers use this convention. Comparing
    our raw-RGB patch numbers to those published figures was apples-to-oranges.
  - Border-cropped by `scale` pixels, standard practice to avoid edge
    artifacts dominating the metric.

Also saves a handful of side-by-side comparison images (LR / bicubic /
model / ground truth) to outputs/ for a visual sanity check -- numbers can
mislead, pictures usually don't.

Run from the project root:
    py src/evaluate.py
"""
import os
import sys
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2ycbcr

sys.path.append(os.path.dirname(__file__))
from model import EDSRBaseline

DATA_ROOT = r"G:\mini dlss\data\raw"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pth")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "eval_samples")
SCALE = 4
NUM_SAMPLE_IMAGES = 5  # how many comparison images to save


def mod_crop(img, scale):
    """Crop so height/width are exact multiples of scale -- keeps LR*scale == HR exactly."""
    h, w = img.shape[0], img.shape[1]
    h -= h % scale
    w -= w % scale
    return img[:h, :w, ...]


def to_y_channel(img_uint8):
    """img_uint8: HxWx3 RGB, 0-255. Returns the Y (luminance) channel, 0-255 float."""
    ycbcr = rgb2ycbcr(img_uint8)
    return ycbcr[:, :, 0]


def crop_border(img, border):
    if border == 0:
        return img
    return img[border:-border, border:-border, ...]


def bicubic_upscale(lr_img_uint8, scale):
    lr_pil = Image.fromarray(lr_img_uint8)
    w, h = lr_pil.size
    return np.array(lr_pil.resize((w * scale, h * scale), Image.BICUBIC))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = EDSRBaseline(scale=SCALE, num_blocks=16, channels=64).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']+1}, "
          f"recorded val_psnr={ckpt.get('val_psnr', 'n/a')}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hr_dir = os.path.join(DATA_ROOT, "DIV2K_valid_HR")
    lr_dir = os.path.join(DATA_ROOT, "DIV2K_valid_LR_bicubic", "X4")
    hr_files = sorted(os.listdir(hr_dir))
    lr_files = sorted(os.listdir(lr_dir))

    model_psnr_list, model_ssim_list = [], []
    bicubic_psnr_list, bicubic_ssim_list = [], []

    for idx, (hr_name, lr_name) in enumerate(zip(hr_files, lr_files)):
        hr_img = np.array(Image.open(os.path.join(hr_dir, hr_name)).convert("RGB"))
        lr_img = np.array(Image.open(os.path.join(lr_dir, lr_name)).convert("RGB"))

        hr_img = mod_crop(hr_img, SCALE)
        # ensure LR*scale exactly matches the (mod-cropped) HR size
        expected_lr_h, expected_lr_w = hr_img.shape[0] // SCALE, hr_img.shape[1] // SCALE
        lr_img = lr_img[:expected_lr_h, :expected_lr_w, :]

        # ---- model inference (full image, fully-convolutional) ----
        lr_tensor = torch.from_numpy(lr_img / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp(0, 1)
        sr_img = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

        # ---- bicubic baseline ----
        bicubic_img = bicubic_upscale(lr_img, SCALE)

        # ---- metrics on Y channel, border-cropped ----
        hr_y = crop_border(to_y_channel(hr_img), SCALE)
        sr_y = crop_border(to_y_channel(sr_img), SCALE)
        bic_y = crop_border(to_y_channel(bicubic_img), SCALE)

        model_psnr_list.append(peak_signal_noise_ratio(hr_y, sr_y, data_range=255))
        model_ssim_list.append(structural_similarity(hr_y, sr_y, data_range=255))
        bicubic_psnr_list.append(peak_signal_noise_ratio(hr_y, bic_y, data_range=255))
        bicubic_ssim_list.append(structural_similarity(hr_y, bic_y, data_range=255))

        if idx < NUM_SAMPLE_IMAGES:
            comparison = np.concatenate([bicubic_img, sr_img, hr_img], axis=1)
            Image.fromarray(comparison).save(
                os.path.join(OUTPUT_DIR, f"compare_{idx:02d}_{hr_name}")
            )

        print(f"[{idx+1}/{len(hr_files)}] {hr_name}  "
              f"model_psnr={model_psnr_list[-1]:.2f}  bicubic_psnr={bicubic_psnr_list[-1]:.2f}")

    print("\n===== Results (Y-channel, full images, border-cropped) =====")
    print(f"Model:   avg PSNR = {np.mean(model_psnr_list):.2f} dB   avg SSIM = {np.mean(model_ssim_list):.4f}")
    print(f"Bicubic: avg PSNR = {np.mean(bicubic_psnr_list):.2f} dB   avg SSIM = {np.mean(bicubic_ssim_list):.4f}")
    diff = np.mean(model_psnr_list) - np.mean(bicubic_psnr_list)
    print(f"Model {'beats' if diff > 0 else 'loses to'} bicubic by {abs(diff):.2f} dB")
    print(f"\nSample comparison images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()