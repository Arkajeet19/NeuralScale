"""
Downloads the DIV2K dataset (train + validation, HR + official bicubic 4x LR pairs).

Run this on your own machine (Windows), NOT in a sandbox:
    python src/download_data.py

Total download size: ~5-6 GB. Uses the official DIV2K CVL ETH Zurich mirror.
If the download is slow/unstable, you can instead manually download the same
zips from https://data.vision.ee.ethz.ch/cvl/DIV2K/ and place them in data/raw/,
then just run this script again — it will skip re-downloading and only extract.
"""
import os
import zipfile
import urllib.request
from tqdm import tqdm

BASE_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

FILES = {
    "DIV2K_train_HR.zip": "DIV2K_train_HR.zip",
    "DIV2K_train_LR_bicubic_X4.zip": "DIV2K_train_LR_bicubic_X4.zip",
    "DIV2K_valid_HR.zip": "DIV2K_valid_HR.zip",
    "DIV2K_valid_LR_bicubic_X4.zip": "DIV2K_valid_LR_bicubic_X4.zip",
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, dest_path):
    if os.path.exists(dest_path):
        print(f"  already downloaded: {os.path.basename(dest_path)}")
        return
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1,
                              desc=os.path.basename(dest_path)) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)


def extract(zip_path, extract_to):
    print(f"  extracting {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    for remote_name, local_name in FILES.items():
        url = BASE_URL + remote_name
        dest = os.path.join(RAW_DIR, local_name)
        print(f"Downloading {remote_name} ...")
        download(url, dest)
        extract(dest, RAW_DIR)

    print("\nDone. Expected structure under data/raw/:")
    print("  DIV2K_train_HR/               (800 HR images)")
    print("  DIV2K_train_LR_bicubic/X4/    (800 LR images, already 4x downsampled)")
    print("  DIV2K_valid_HR/               (100 HR images)")
    print("  DIV2K_valid_LR_bicubic/X4/    (100 LR images)")


if __name__ == "__main__":
    main()
