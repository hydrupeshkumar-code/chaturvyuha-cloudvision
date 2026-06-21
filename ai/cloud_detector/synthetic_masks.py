import argparse
import json
import random
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening


def _smooth_noise(h: int, w: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = rng.rand(h, w).astype(np.float32)
    return gaussian_filter(base, sigma=sigma)


def perlin_clouds(h: int, w: int, seed: int) -> np.ndarray:
    n1 = _smooth_noise(h, w, sigma=6.0, seed=seed)
    n2 = _smooth_noise(h, w, sigma=12.0, seed=seed + 31)
    n = 0.65 * n1 + 0.35 * n2
    return n


def fractal_clouds(h: int, w: int, seed: int) -> np.ndarray:
    layers = []
    for i, sigma in enumerate([2.5, 5.0, 10.0, 18.0]):
        layers.append((0.5 ** i) * _smooth_noise(h, w, sigma=sigma, seed=seed + i * 19))
    n = np.sum(layers, axis=0)
    return n


def blob_clouds(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    canvas = np.zeros((h, w), dtype=np.float32)
    n_blobs = rng.randint(12, 28)
    yy, xx = np.mgrid[:h, :w]
    for _ in range(n_blobs):
        cx = rng.randint(0, w)
        cy = rng.randint(0, h)
        rx = rng.randint(max(6, w // 24), max(10, w // 8))
        ry = rng.randint(max(6, h // 24), max(10, h // 8))
        blob = np.exp(-(((xx - cx) ** 2) / (2 * rx * rx) + ((yy - cy) ** 2) / (2 * ry * ry)))
        canvas += blob.astype(np.float32)
    canvas = gaussian_filter(canvas, sigma=2.0)
    return canvas


def generate_mask(h: int, w: int, coverage: float, seed: int) -> np.ndarray:
    # Blend required modes: Perlin + Fractal + random cloud blobs.
    n = 0.4 * perlin_clouds(h, w, seed) + 0.4 * fractal_clouds(h, w, seed + 101) + 0.2 * blob_clouds(h, w, seed + 211)
    n = (n - n.min()) / (n.max() - n.min() + 1e-8)

    thr = np.percentile(n, 100.0 * (1.0 - coverage))
    mask = n >= thr

    # Required smoothing + morphology.
    mask = gaussian_filter(mask.astype(np.float32), sigma=1.4) > 0.5
    mask = binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    mask = binary_opening(mask, structure=np.ones((3, 3), dtype=bool))

    return np.where(mask, 255, 0).astype(np.uint8)


def synthesize_cloudy(clear_chw: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    mask = (mask_u8.astype(np.float32) / 255.0)[None, ...]
    clear = clear_chw.astype(np.float32)
    cloudy = clear.copy()

    per_band_scale = np.array([0.82, 0.90, 1.00], dtype=np.float32)[:, None, None]
    max_val = np.maximum(np.percentile(clear, 99, axis=(1, 2), keepdims=True), 1.0)
    cloud_val = per_band_scale * max_val

    # Thin cloud veil + bright cloud core.
    veil = gaussian_filter(mask[0], sigma=5.0)[None, ...]
    veil = 0.25 + 0.75 * (veil / (veil.max() + 1e-8))
    cloudy = clear * (1.0 - veil * mask) + cloud_val * (veil * mask)

    return cloudy.astype(clear_chw.dtype)


def _save_png(arr_hwc: np.ndarray, out_path: Path):
    import imageio.v2 as imageio

    out = np.clip(arr_hwc * 255.0, 0, 255).astype(np.uint8)
    imageio.imwrite(str(out_path), out)


def _to_vis(chw: np.ndarray) -> np.ndarray:
    hwc = np.transpose(chw, (1, 2, 0)).astype(np.float32)
    out = np.zeros_like(hwc)
    for c in range(hwc.shape[2]):
        ch = hwc[:, :, c]
        p2 = np.percentile(ch, 2)
        p98 = np.percentile(ch, 98)
        if p98 - p2 < 1e-8:
            out[:, :, c] = 0
        else:
            out[:, :, c] = np.clip((ch - p2) / (p98 - p2), 0, 1)
    return out


def generate_examples(split_manifest: Path, out_dir: Path, num_examples: int, seed: int):
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    val_pairs = manifest.get("val_pairs", [])
    if not val_pairs:
        raise RuntimeError("No val_pairs in split manifest")

    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    cloudy_dir = out_dir / "cloudy_examples"
    masks_dir.mkdir(parents=True, exist_ok=True)
    cloudy_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    idxs = rng.sample(range(len(val_pairs)), min(num_examples, len(val_pairs)))

    coverage_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    for i, vi in enumerate(idxs):
        pair = val_pairs[vi]
        p = Path(pair["target_path"])
        if not p.exists():
            p = Path(str(p).replace("chaturvyuha-cloudvision\\", ""))
        with rasterio.open(p) as src:
            clear = src.read([1, 2, 3])
            profile = src.profile.copy()

        _, h, w = clear.shape
        cov = coverage_levels[i % len(coverage_levels)]
        mask = generate_mask(h, w, coverage=cov, seed=seed + i * 13)
        cloudy = synthesize_cloudy(clear, mask)

        mask_path = masks_dir / f"example_{i:03d}.tif"
        mask_png = masks_dir / f"example_{i:03d}.png"
        cloudy_path = cloudy_dir / f"example_{i:03d}.tif"
        cloudy_png = cloudy_dir / f"example_{i:03d}.png"

        mprof = profile.copy()
        mprof.update(count=1, dtype=rasterio.uint8)
        with rasterio.open(mask_path, "w", **mprof) as dst:
            dst.write(mask, 1)

        cprof = profile.copy()
        cprof.update(count=3)
        with rasterio.open(cloudy_path, "w", **cprof) as dst:
            dst.write(cloudy)

        _save_png(np.repeat((mask.astype(np.float32) / 255.0)[:, :, None], 3, axis=2), mask_png)
        _save_png(_to_vis(cloudy), cloudy_png)


def main():
    p = argparse.ArgumentParser(description="Synthetic cloud mask generation with required noise modes")
    p.add_argument("--split_manifest", default="checkpoints_nafnet/final_sharpen_finetune/split_manifest.json")
    p.add_argument("--out_dir", default="cloud_mask_examples")
    p.add_argument("--num_examples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    generate_examples(
        split_manifest=Path(args.split_manifest),
        out_dir=Path(args.out_dir),
        num_examples=args.num_examples,
        seed=args.seed,
    )
    print(f"Generated examples at {args.out_dir}")


if __name__ == "__main__":
    main()