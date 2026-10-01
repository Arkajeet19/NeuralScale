"""
Dataset for 4x super-resolution training on DIV2K.

v2 changes from the first version:
1. In-memory caching: the first version re-read and re-decoded full
   2K-resolution PNGs from disk on EVERY iteration, which is brutal on an
   HDD (mechanical seek time) and was the real cause of ~5.4s/iteration.
   Now each image is decoded once and cached as a numpy array; every
   subsequent access (all remaining epochs) just crops from RAM.
2. `repeat` factor: the original epoch was just 800 patches (one random
   crop per image) -- far too few gradient updates per epoch. `repeat`
   makes each epoch sample `repeat` random crops per image instead of 1,
   giving a properly-sized epoch without changing what data exists.
"""
import os
import random
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class DIV2KPatchDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, scale=4, lr_patch_size=48, train=True, repeat=16):
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.train = train
        self.repeat = repeat if train else 1

        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.png")))

        assert len(self.hr_paths) == len(self.lr_paths), (
            f"Mismatch: {len(self.hr_paths)} HR images vs {len(self.lr_paths)} LR images."
        )
        assert len(self.hr_paths) > 0, f"No images found in {hr_dir}"

        self.num_images = len(self.hr_paths)
        # cache: populated lazily on first access, then reused for free
        self._hr_cache = {}
        self._lr_cache = {}

    def __len__(self):
        return self.num_images * self.repeat

    def _load(self, idx):
        if idx not in self._hr_cache:
            self._hr_cache[idx] = np.array(Image.open(self.hr_paths[idx]).convert("RGB"))
            self._lr_cache[idx] = np.array(Image.open(self.lr_paths[idx]).convert("RGB"))
        return self._hr_cache[idx], self._lr_cache[idx]

    def __getitem__(self, idx):
        real_idx = idx % self.num_images
        hr_arr, lr_arr = self._load(real_idx)

        lr_h, lr_w = lr_arr.shape[0], lr_arr.shape[1]
        ps = self.lr_patch_size

        if self.train:
            x = random.randint(0, max(0, lr_w - ps))
            y = random.randint(0, max(0, lr_h - ps))
        else:
            x = max(0, (lr_w - ps) // 2)
            y = max(0, (lr_h - ps) // 2)

        lr_patch = lr_arr[y:y + ps, x:x + ps, :]
        hs = self.hr_patch_size
        hr_patch = hr_arr[y * self.scale:y * self.scale + hs,
                           x * self.scale:x * self.scale + hs, :]

        lr_img = Image.fromarray(lr_patch)
        hr_img = Image.fromarray(hr_patch)

        if self.train:
            lr_img, hr_img = self._augment(lr_img, hr_img)

        return TF.to_tensor(lr_img), TF.to_tensor(hr_img)

    @staticmethod
    def _augment(lr_patch, hr_patch):
        if random.random() < 0.5:
            lr_patch = TF.hflip(lr_patch)
            hr_patch = TF.hflip(hr_patch)
        if random.random() < 0.5:
            lr_patch = TF.vflip(lr_patch)
            hr_patch = TF.vflip(hr_patch)
        if random.random() < 0.5:
            lr_patch = lr_patch.rotate(90, expand=True)
            hr_patch = hr_patch.rotate(90, expand=True)
        return lr_patch, hr_patch


if __name__ == "__main__":
    root = r"G:\mini dlss\data\raw"
    ds = DIV2KPatchDataset(
        hr_dir=os.path.join(root, "DIV2K_train_HR"),
        lr_dir=os.path.join(root, "DIV2K_train_LR_bicubic", "X4"),
        scale=4, lr_patch_size=48, train=True, repeat=16
    )
    print(f"Dataset size (with repeat): {len(ds)}")
    lr, hr = ds[0]
    print(f"LR patch shape: {lr.shape}  (expect 3x48x48)")
    print(f"HR patch shape: {hr.shape}  (expect 3x192x192)")