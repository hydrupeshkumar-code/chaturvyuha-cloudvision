import os
from typing import Tuple

import numpy as np
import torch

from .model import NAFNetWrapper
try:
    from ai.dsen2cr_liss.dataset import read_image, normalize_image
except Exception:
    def read_image(path):
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read().astype('float32')
            return np.transpose(arr, (1,2,0))


def infer(model: NAFNetWrapper, input_path: str, clip_min, clip_max, device: str = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    img = read_image(input_path)[:, :, :3]
    img_n = normalize_image(img, clip_min, clip_max)
    x = torch.from_numpy(np.transpose(img_n, (2,0,1))[None]).float().to(device)

    with torch.no_grad():
        p = model(x)
    out = p[0].cpu().numpy()
    out = np.transpose(out, (1,2,0))
    out = np.clip(out, 0.0, 1.0)
    return out
