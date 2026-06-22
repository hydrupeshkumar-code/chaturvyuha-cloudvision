"""Compute joint per-band p1/p99 statistics across cloudy + clear images.
Writes tmp_stats/band_statistics.json with keys 'p1' and 'p99' each a list of floats.

Sampling strategy: sample up to `max_images` per folder and `pixels_per_image` random pixels per band.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import rasterio


def sample_pixels_from_image(path, bands=(1,2,3), pixels=1000):
    with rasterio.open(path) as src:
        # read selected bands (rasterio 1-indexed)
        arr = src.read(bands).astype(np.float32)  # shape (C,H,W)
    C, H, W = arr.shape
    arr = arr.reshape(C, -1)
    total = arr.shape[1]
    if pixels >= total:
        idx = np.arange(total)
    else:
        idx = np.random.choice(total, size=pixels, replace=False)
    return arr[:, idx]


def compute(cloudy_dir, clear_dir, out_path, max_images=500, pixels_per_image=2000, seed=42):
    random.seed(seed)
    pth = Path(out_path)
    pth.parent.mkdir(parents=True, exist_ok=True)

    cfiles = list(Path(cloudy_dir).glob('**/*.tif'))
    tfiles = list(Path(clear_dir).glob('**/*.tif'))
    combined = cfiles + tfiles
    if len(combined) == 0:
        raise SystemExit('No images found')

    if len(combined) > max_images:
        combined = random.sample(combined, max_images)

    samples = []
    for p in combined:
        try:
            s = sample_pixels_from_image(p, bands=(1,2,3), pixels=pixels_per_image)
            samples.append(s)
        except Exception:
            continue

    if len(samples) == 0:
        raise SystemExit('No samples read')

    arr = np.concatenate(samples, axis=1)  # C x N

    p1 = np.percentile(arr, 1, axis=1).tolist()
    p99 = np.percentile(arr, 99, axis=1).tolist()

    out = {'p1': [float(x) for x in p1], 'p99': [float(x) for x in p99]}

    # save to tmp_stats/band_statistics.json and to out_path
    tmp = Path('tmp_stats')
    tmp.mkdir(exist_ok=True)
    tmp_file = tmp / 'band_statistics.json'
    tmp_file.write_text(json.dumps(out, indent=2))

    Path(out_path).write_text(json.dumps(out, indent=2))
    print('Wrote stats to', tmp_file, 'and', out_path)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cloudy', required=True)
    p.add_argument('--clear', required=True)
    p.add_argument('--out', default='checkpoints_nafnet/smoke/joint_band_statistics.json')
    p.add_argument('--max_images', type=int, default=500)
    p.add_argument('--pixels_per_image', type=int, default=2000)
    args = p.parse_args()
    compute(args.cloudy, args.clear, args.out, args.max_images, args.pixels_per_image)
