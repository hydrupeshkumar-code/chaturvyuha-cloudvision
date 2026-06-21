import argparse
from pathlib import Path

import numpy as np
import rasterio


def read_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def save_raster(path: Path, chw: np.ndarray, profile: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=chw.shape[0], dtype=rasterio.float32)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(chw.astype(np.float32))


def save_png(path: Path, hwc01: np.ndarray):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), np.clip(hwc01 * 255.0, 0, 255).astype(np.uint8))


def percentile_vis(chw: np.ndarray):
    hwc = np.transpose(chw, (1, 2, 0)).astype(np.float32)
    out = np.zeros_like(hwc)
    for c in range(hwc.shape[2]):
        ch = hwc[:, :, c]
        p2 = np.percentile(ch, 2)
        p98 = np.percentile(ch, 98)
        if p98 - p2 < 1e-8:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = np.clip((ch - p2) / (p98 - p2), 0.0, 1.0)
    return out


def fuse(original_chw: np.ndarray, reconstruction_chw: np.ndarray, mask: np.ndarray):
    if original_chw.shape != reconstruction_chw.shape:
        raise ValueError(f"original/reconstruction shape mismatch: {original_chw.shape} vs {reconstruction_chw.shape}")

    if mask.ndim == 3:
        mask_hw = mask[0]
    else:
        mask_hw = mask

    if mask_hw.shape != original_chw.shape[1:]:
        raise ValueError(f"mask shape mismatch: {mask_hw.shape} vs {original_chw.shape[1:]}")

    m = np.clip(mask_hw.astype(np.float32) / 255.0, 0.0, 1.0)
    m = np.expand_dims(m, axis=0)

    # Required formula: output = (mask * reconstruction) + ((1-mask) * original)
    fused_chw = (m * reconstruction_chw) + ((1.0 - m) * original_chw)

    diff = np.mean(np.abs(fused_chw - original_chw), axis=0)
    dmin = float(np.percentile(diff, 2))
    dmax = float(np.percentile(diff, 98))
    if dmax - dmin < 1e-8:
        diff_vis = np.zeros_like(diff, dtype=np.float32)
    else:
        diff_vis = np.clip((diff - dmin) / (dmax - dmin), 0.0, 1.0)

    return fused_chw.astype(np.float32), diff_vis.astype(np.float32)


def run(args):
    orig, profile = read_raster(Path(args.original))
    recon, _ = read_raster(Path(args.reconstruction))
    mask, _ = read_raster(Path(args.mask))

    if orig.shape[0] >= 3:
        orig = orig[:3]
    if recon.shape[0] >= 3:
        recon = recon[:3]

    fused_chw, diff_vis = fuse(orig, recon, mask)

    save_raster(Path(args.output_tif), fused_chw, profile)

    fused_vis = percentile_vis(fused_chw)
    save_png(Path(args.output_png), fused_vis)
    save_png(Path(args.output_diff_png), np.repeat(diff_vis[:, :, None], 3, axis=2))

    print(f"Saved fused TIFF: {args.output_tif}")
    print(f"Saved fused PNG: {args.output_png}")
    print(f"Saved difference map: {args.output_diff_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fuse original + NAFNet reconstruction using cloud mask")
    p.add_argument("--original", required=True)
    p.add_argument("--reconstruction", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--output_tif", default="fused.tif")
    p.add_argument("--output_png", default="fused.png")
    p.add_argument("--output_diff_png", default="difference_map.png")
    run(p.parse_args())
