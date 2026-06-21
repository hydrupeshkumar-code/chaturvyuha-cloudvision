import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from ai.metrics.compute import compute_all_metrics


def _read3(path: Path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    if arr.shape[0] < 3:
        raise RuntimeError(f"Expected >=3 bands in {path}")
    return arr[:3]


def _vis(chw: np.ndarray):
    hwc = np.transpose(chw, (1, 2, 0)).astype(np.float32)
    out = np.zeros_like(hwc)
    for c in range(hwc.shape[2]):
        ch = hwc[:, :, c]
        p2 = np.percentile(ch, 2)
        p98 = np.percentile(ch, 98)
        out[:, :, c] = 0 if p98 - p2 < 1e-8 else np.clip((ch - p2) / (p98 - p2), 0, 1)
    return out


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cloudy = _read3(Path(args.cloudy))
    reconstruction = _read3(Path(args.reconstruction))
    reference = _read3(Path(args.reference_clear))

    metrics = compute_all_metrics(reference, reconstruction)

    diff = np.mean(np.abs(reconstruction.astype(np.float32) - reference.astype(np.float32)), axis=0)
    d2 = np.percentile(diff, 2)
    d98 = np.percentile(diff, 98)
    diff_vis = np.zeros_like(diff, dtype=np.float32) if d98 - d2 < 1e-8 else np.clip((diff - d2) / (d98 - d2), 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(_vis(cloudy))
    axes[0].set_title("Cloudy Scene")
    axes[1].imshow(_vis(reconstruction))
    axes[1].set_title("NAFNet Reconstruction")
    axes[2].imshow(_vis(reference))
    axes[2].set_title("Reference Clear Scene")
    im = axes[3].imshow(diff_vis, cmap="inferno")
    axes[3].set_title("Difference Map")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_dir / "temporal_comparison.png", dpi=170)
    plt.close(fig)

    temporal_metrics = {
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "rmse": metrics["rmse"],
        "sam": metrics["sam"],
    }
    (out_dir / "temporal_metrics.json").write_text(json.dumps(temporal_metrics, indent=2), encoding="utf-8")

    report_lines = [
        "# Temporal Comparison Report",
        "",
        f"- Cloudy Scene: {args.cloudy}",
        f"- Reconstruction: {args.reconstruction}",
        f"- Reference Clear: {args.reference_clear}",
        "",
        f"- PSNR: {temporal_metrics['psnr']:.6f}",
        f"- SSIM: {temporal_metrics['ssim']:.6f}",
        f"- RMSE: {temporal_metrics['rmse']:.6f}",
        f"- SAM: {temporal_metrics['sam']:.6f}",
    ]
    (out_dir / "comparison_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(temporal_metrics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Temporal comparison for CloudVision")
    p.add_argument("--cloudy", required=True)
    p.add_argument("--reconstruction", required=True)
    p.add_argument("--reference_clear", required=True)
    p.add_argument("--output_dir", default="outputs/temporal")
    run(p.parse_args())
