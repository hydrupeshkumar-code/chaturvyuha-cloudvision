
import glob
import os
from typing import Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class CloudDataset(Dataset):
    """Cloud detector dataset with GRN input and binary mask target."""

    def __init__(self, image_dir: str, mask_dir: str, augment: bool = False, bands: Tuple[int, int, int] = (1, 2, 3)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.bands = bands

        image_files = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        if not image_files:
            image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))

        mask_lookup = {}
        for ext in ("*.tif", "*.png"):
            for p in glob.glob(os.path.join(mask_dir, ext)):
                mask_lookup[os.path.splitext(os.path.basename(p))[0]] = p

        self.samples = []
        for ip in image_files:
            stem = os.path.splitext(os.path.basename(ip))[0]
            if stem in mask_lookup:
                self.samples.append((ip, mask_lookup[stem]))

        if not self.samples:
            raise RuntimeError(f"No matching image-mask pairs in {image_dir} and {mask_dir}")

    def __len__(self):
        return len(self.samples)

    def _read_image(self, path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            img = src.read(list(self.bands)).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                img[img == nodata] = np.nan

        out = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[0]):
            ch = img[c]
            cmin = np.nanpercentile(ch, 1)
            cmax = np.nanpercentile(ch, 99)
            if cmax - cmin < 1e-8:
                out[c] = 0.0
            else:
                out[c] = np.clip((ch - cmin) / (cmax - cmin), 0.0, 1.0)
        return np.nan_to_num(out)

    def _read_mask(self, path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            mask = src.read(1)
        # Required format uses uint8 0 clear / 255 cloud. Convert to 0/1 float for loss.
        mask = (mask > 127).astype(np.float32)
        return np.expand_dims(mask, axis=0)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.samples[idx]
        image = self._read_image(image_path)
        mask = self._read_mask(mask_path)

        x = torch.from_numpy(image)
        y = torch.from_numpy(mask)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                x = torch.rot90(x, k, [1, 2])
                y = torch.rot90(y, k, [1, 2])

        return x, y