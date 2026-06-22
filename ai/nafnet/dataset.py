import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    # reuse existing helpers from the TF pipeline
    from ai.dsen2cr_liss.dataset import read_image, normalize_image
except Exception:
    # best-effort fallback implementations
    def read_image(path: str) -> np.ndarray:
        import rasterio

        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
            # rasterio returns (bands, H, W) -> convert to HWC
            return np.transpose(arr, (1, 2, 0))

    def normalize_image(img: np.ndarray, clip_min: List[float], clip_max: List[float]) -> np.ndarray:
        img = img.astype(np.float32)
        # per-band clipping and scaling to [0,1]
        out = np.empty_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            mn = clip_min[c] if isinstance(clip_min, (list, tuple)) else clip_min
            mx = clip_max[c] if isinstance(clip_max, (list, tuple)) else clip_max
            out[:, :, c] = np.clip((img[:, :, c] - mn) / (mx - mn + 1e-8), 0.0, 1.0)
        return out


class NAFDataset(Dataset):
    """PyTorch Dataset for GRN (3-channel) paired cloud/clear images.

    Args:
        pairs: list of tuples (cloudy_path, clear_path, mask_path_optional)
        clip_min, clip_max: per-band clip bounds (lists length == channels)
        patch_size: optional crop size (H, W)
        augment: enable simple augmentations
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, Optional[str]]],
        clip_min,
        clip_max,
        patch_size: Optional[Tuple[int, int]] = None,
        augment: bool = True,
    ):
        self.pairs = pairs
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def _apply_augment(self, a: np.ndarray) -> np.ndarray:
        if not self.augment:
            return a
        if random.random() < 0.5:
            a = np.flip(a, axis=0).copy()
        if random.random() < 0.5:
            a = np.flip(a, axis=1).copy()
        k = random.choice([0, 1, 2, 3])
        if k:
            a = np.rot90(a, k).copy()
        return a

    def __getitem__(self, idx: int):
        cloudy_path, clear_path, *rest = self.pairs[idx]

        c = read_image(cloudy_path)
        t = read_image(clear_path)

        # keep only first 3 channels if data has >3
        if c.shape[2] >= 3:
            c = c[:, :, :3]
        if t.shape[2] >= 3:
            t = t[:, :, :3]

        c = normalize_image(c, self.clip_min, self.clip_max)
        t = normalize_image(t, self.clip_min, self.clip_max)

        # optional random crop
        if self.patch_size is not None:
            H, W = c.shape[:2]
            ph, pw = self.patch_size
            if H >= ph and W >= pw:
                i = random.randint(0, H - ph)
                j = random.randint(0, W - pw)
                c = c[i : i + ph, j : j + pw]
                t = t[i : i + ph, j : j + pw]

        c = self._apply_augment(c)
        t = self._apply_augment(t)

        # HWC -> CHW
        c = np.transpose(c, (2, 0, 1)).astype(np.float32)
        t = np.transpose(t, (2, 0, 1)).astype(np.float32)

        return torch.from_numpy(c), torch.from_numpy(t)
