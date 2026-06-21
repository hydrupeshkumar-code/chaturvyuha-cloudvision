import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from ai.cloud_detector.model import UNet
from ai.fusion.fuse import fuse, percentile_vis, save_png, save_raster
from ai.metrics.compute import compute_all_metrics
from ai.nafnet.dataset import normalize_image
from ai.nafnet.model import NAFNetWrapper


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


def _read3(path: Path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
    if arr.shape[0] < 3:
        raise RuntimeError(f"Expected >=3 bands in {path}")
    return arr[:3], profile


def _predict_mask(unet: UNet, image_chw: np.ndarray, device: str, threshold: float = 0.5):
    # Use robust percentile normalization.
    nimg = np.zeros_like(image_chw, dtype=np.float32)
    for c in range(image_chw.shape[0]):
        ch = image_chw[c]
        p2 = np.percentile(ch, 2)
        p98 = np.percentile(ch, 98)
        nimg[c] = 0 if p98 - p2 < 1e-8 else np.clip((ch - p2) / (p98 - p2), 0, 1)

    x = torch.from_numpy(nimg).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = unet(x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = np.where(probs >= threshold, 255, 0).astype(np.uint8)
    return mask


def _predict_reconstruction(nafnet: NAFNetWrapper, image_chw: np.ndarray, stats: dict, device: str):
    hwc = np.transpose(image_chw, (1, 2, 0))
    n = normalize_image(hwc, stats["p1"], stats["p99"])
    x = torch.from_numpy(np.transpose(n, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = nafnet(x)[0].cpu().numpy()

    pred_norm = np.clip(np.transpose(pred, (1, 2, 0)), 0.0, 1.0)
    pred_denorm = np.empty_like(pred_norm, dtype=np.float32)
    for c in range(3):
        pred_denorm[:, :, c] = pred_norm[:, :, c] * (stats["p99"][c] - stats["p1"][c]) + stats["p1"][c]
    return np.transpose(pred_denorm, (2, 0, 1)).astype(np.float32), pred_norm


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Input Image
    orig_chw, profile = _read3(Path(args.input_image))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # U-Net Detection
    unet = UNet(in_channels=3, out_channels=1).to(device)
    u_state = torch.load(args.unet_checkpoint, map_location=device)
    u_state = u_state["model_state_dict"] if isinstance(u_state, dict) and "model_state_dict" in u_state else u_state
    unet.load_state_dict(u_state, strict=False)
    unet.eval()

    mask_u8 = _predict_mask(unet, orig_chw, device=device, threshold=args.mask_threshold)

    # NAFNet Reconstruction (frozen production model)
    nafnet = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    n_state = torch.load(args.nafnet_checkpoint, map_location=device)
    n_state = n_state["state_dict"] if isinstance(n_state, dict) and "state_dict" in n_state else n_state
    nafnet.load_state_dict(n_state, strict=False)
    nafnet.eval()

    stats = _load_stats(Path(args.stats_json))
    recon_chw, recon_norm_hwc = _predict_reconstruction(nafnet, orig_chw, stats=stats, device=device)

    # Fusion Engine
    fused_chw, diff_vis = fuse(orig_chw, recon_chw, mask_u8)

    # Save required outputs
    save_png(out_dir / "original.png", percentile_vis(orig_chw))
    save_png(out_dir / "cloud_mask.png", np.repeat((mask_u8.astype(np.float32) / 255.0)[:, :, None], 3, axis=2))
    save_png(out_dir / "reconstruction.png", recon_norm_hwc)
    save_png(out_dir / "fused.png", percentile_vis(fused_chw))
    save_png(out_dir / "difference_map.png", np.repeat(diff_vis[:, :, None], 3, axis=2))

    save_raster(out_dir / "fused.tif", fused_chw, profile)

    # Metrics Engine
    metrics = {}
    if args.reference_clear:
        ref_chw, _ = _read3(Path(args.reference_clear))
        gt_mask = None
        if args.gt_mask:
            with rasterio.open(args.gt_mask) as src:
                gt_mask = src.read(1)
        metrics = compute_all_metrics(ref_chw, fused_chw, pred_mask=mask_u8, gt_mask=gt_mask)
    else:
        # Keep mask statistics even without reference clear.
        metrics = compute_all_metrics(orig_chw, fused_chw, pred_mask=mask_u8, gt_mask=None)

    quality = metrics.get("quality_flags", {})

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "quality_flags.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    print(f"Pipeline complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CloudVision master pipeline")
    p.add_argument("--input_image", required=True)
    p.add_argument("--reference_clear", default=None)
    p.add_argument("--gt_mask", default=None)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--unet_checkpoint", default="checkpoints_unet_cloud/best_iou.pth")
    p.add_argument("--nafnet_checkpoint", default="checkpoints_nafnet/strict_curated_training/best_ssim.pth")
    p.add_argument("--stats_json", default="tmp_stats/band_statistics.json")
    p.add_argument("--mask_threshold", type=float, default=0.5)
    run(p.parse_args())
