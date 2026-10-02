"""
Batch-converts all .jpg/.jpeg files in a folder to .png.

Note: this does NOT remove JPEG compression artifacts already baked into
the image -- it just repackages the (already-compressed) pixel data
losslessly from this point forward. It won't restore detail that JPEG
compression already discarded.

Usage:
    py src/convert_to_png.py "G:\\mini dlss\\data\\game_screenshots\\hr"
"""
import os
import sys
from PIL import Image


def convert_folder(folder):
    converted = 0
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".jfif")):
            src_path = os.path.join(folder, fname)
            dst_path = os.path.join(folder, os.path.splitext(fname)[0] + ".png")
            img = Image.open(src_path).convert("RGB")
            img.save(dst_path, "PNG")
            converted += 1
            print(f"  {fname} -> {os.path.basename(dst_path)}")
    print(f"\nConverted {converted} file(s).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: py src/convert_to_png.py <folder_path>")
        sys.exit(1)
    convert_folder(sys.argv[1])
