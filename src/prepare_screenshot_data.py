"""
Takes your collected game screenshots (mixed formats, mixed resolutions,
all in one folder) and builds a DIV2K-style train/valid HR/LR dataset from
them -- same structure prepare_data scripts use, so dataset.py (and
train_gan.py) can load this with zero code changes, just a different
DATA_ROOT/folder name.

What it does:
  1. Converts any .jpg/.jpeg/.jfif to .png
  2. Mod-crops each image so its dimensions divide evenly by the scale
     (same reasoning as evaluate.py's mod_crop -- keeps LR*scale == HR exactly)
  3. Splits images into train/valid (90/10 by default)
  4. Generates the matching LR version of each via bicubic downsampling

Usage:
    py src/prepare_screenshot_data.py "C:\\Users\\ARKAJEET\\Pictures\\Gameplay screenshots"

Output goes to data/game_screenshots/ (train_HR, train_LR_bicubic/X4,
valid_HR, valid_LR_bicubic/X4) alongside your existing data/raw/ DIV2K folders.
"""
import os
import sys
import random
import shutil
from PIL import Image

SCALE = 4
VAL_FRACTION = 0.1   # ~10% held out for validation (so 180 images -> ~162 train / ~18 val)
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "game_screenshots")
SEED = 42


def convert_to_png_inplace(src_path):
    """Returns a path to a .png version of the image, converting if needed."""
    if src_path.lower().endswith(".png"):
        return src_path
    img = Image.open(src_path).convert("RGB")
    png_path = os.path.splitext(src_path)[0] + "_converted.png"
    img.save(png_path, "PNG")
    return png_path


def mod_crop_and_save(img, out_path, scale):
    w, h = img.size
    w -= w % scale
    h -= h % scale
    img.crop((0, 0, w, h)).save(out_path, "PNG")
    return w, h


def main():
    if len(sys.argv) != 2:
        print('Usage: py src/prepare_screenshot_data.py "path\\to\\your\\screenshots"')
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"ERROR: {input_dir} is not a folder")
        sys.exit(1)

    valid_exts = (".png", ".jpg", ".jpeg", ".jfif")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    print(f"Found {len(files)} image(s) in {input_dir}")

    random.seed(SEED)
    random.shuffle(files)
    n_val = max(1, int(len(files) * VAL_FRACTION))
    val_files = set(files[:n_val])
    train_files = files[n_val:]
    print(f"Split: {len(train_files)} train, {len(val_files)} validation")

    for split_name, split_files in [("train", train_files), ("valid", val_files)]:
        hr_dir = os.path.join(OUTPUT_ROOT, f"{split_name}_HR")
        lr_dir = os.path.join(OUTPUT_ROOT, f"{split_name}_LR_bicubic", "X4")
        os.makedirs(hr_dir, exist_ok=True)
        os.makedirs(lr_dir, exist_ok=True)

        for i, fname in enumerate(split_files):
            src_path = os.path.join(input_dir, fname)
            png_path = convert_to_png_inplace(src_path)

            img = Image.open(png_path).convert("RGB")
            out_name = f"{split_name}_{i:04d}.png"
            hr_out_path = os.path.join(hr_dir, out_name)
            w, h = mod_crop_and_save(img, hr_out_path, SCALE)

            # bicubic downsample for the LR pair
            lr_img = Image.open(hr_out_path).resize((w // SCALE, h // SCALE), Image.BICUBIC)
            lr_out_path = os.path.join(lr_dir, out_name)
            lr_img.save(lr_out_path, "PNG")

            # clean up any temp converted file
            if png_path != src_path and os.path.exists(png_path):
                os.remove(png_path)

        print(f"  {split_name}: {len(split_files)} HR/LR pairs written to "
              f"{hr_dir} / {lr_dir}")

    print(f"\nDone. Dataset ready at {OUTPUT_ROOT}")
    print("Use this as DATA_ROOT in train_gan.py (point hr_dir/lr_dir at "
          "game_screenshots/train_HR and game_screenshots/train_LR_bicubic/X4) "
          "to fine-tune on these instead of DIV2K.")


if __name__ == "__main__":
    main()
