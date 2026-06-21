import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from ai.nafnet.model import NAFNetWrapper
from ai.nafnet.dataset import normalize_image


def _load_stats(stats_path: Path):
    stats = {"p1": [0.0, 0.0, 0.0], "p99": [6000.0, 6000.0, 6000.0]}
    if stats_path.exists():
        try:
            j = json.loads(stats_path.read_text(encoding="utf-8"))
            stats["p1"] = j.get("p1", stats["p1"])
            stats["p99"] = j.get("p99", stats["p99"])
        except Exception:
            pass
    return stats


def _to_vis(img_hwc: np.ndarray):
    out = np.zeros_like(img_hwc, dtype=np.float32)
    for c in range(img_hwc.shape[2]):
        ch = img_hwc[:, :, c]
        p2 = np.percentile(ch, 2)
        p98 = np.percentile(ch, 98)
        if p98 - p2 < 1e-8:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = np.clip((ch - p2) / (p98 - p2), 0.0, 1.0)
    return out


def _save_png(img_hwc: np.ndarray, out_path: Path):
    import imageio.v2 as imageio

    x = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)
    imageio.imwrite(str(out_path), x)


def run(args):
    in_path = Path(args.input)
    out_tif = Path(args.output_tif)
    out_png = Path(args.output_png)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    stats = _load_stats(Path(args.stats_json))

    with rasterio.open(in_path) as src:
        raw = src.read().astype(np.float32)
        profile = src.profile.copy()

    if raw.shape[0] < 3:
        raise RuntimeError("Input must contain at least 3 bands (Green, Red, NIR expected)")

    grn = np.transpose(raw[:3], (1, 2, 0))
    grn_norm = normalize_image(grn, stats["p1"], stats["p99"])

    x = torch.from_numpy(np.transpose(grn_norm, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()

    pred_norm = np.clip(np.transpose(pred, (1, 2, 0)), 0.0, 1.0)

    pred_denorm = np.empty_like(pred_norm, dtype=np.float32)
    for c in range(3):
        pred_denorm[:, :, c] = pred_norm[:, :, c] * (stats["p99"][c] - stats["p1"][c]) + stats["p1"][c]

    out_profile = profile.copy()
    out_profile.update(count=3, dtype=rasterio.float32)
    with rasterio.open(out_tif, "w", **out_profile) as dst:
        dst.write(np.transpose(pred_denorm.astype(np.float32), (2, 0, 1)))

    _save_png(_to_vis(pred_norm), out_png)
    print(f"Saved reconstruction TIFF: {out_tif}")
    print(f"Saved reconstruction PNG: {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NAFNet inference-only wrapper (frozen production model)")
    p.add_argument("--input", required=True)
    p.add_argument("--output_tif", default="reconstruction.tif")
    p.add_argument("--output_png", default="reconstruction.png")
    p.add_argument("--checkpoint", default="checkpoints_nafnet/strict_curated_training/best_ssim.pth")
    p.add_argument("--stats_json", default="tmp_stats/band_statistics.json")
    run(p.parse_args())
