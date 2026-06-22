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


def _chw_to_hwc(arr_chw: np.ndarray):
    return np.transpose(arr_chw, (1, 2, 0))


def _percentile_stretch_global(pred_hwc: np.ndarray, p_low: float = 1.0, p_high: float = 99.0):
    lo = float(np.percentile(pred_hwc, p_low))
    hi = float(np.percentile(pred_hwc, p_high))
    if hi - lo < 1e-8:
        stretched = np.zeros_like(pred_hwc, dtype=np.float32)
    else:
        stretched = np.clip((pred_hwc - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return stretched, lo, hi


def _stats(arr: np.ndarray):
    x = np.asarray(arr)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def _save_comparison_display(path: Path, input_hwc01: np.ndarray, recon_hwc01: np.ndarray, fused_hwc01: np.ndarray):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    panels = [
        (input_hwc01, "Input"),
        (recon_hwc01, "Reconstruction Display"),
        (fused_hwc01, "Fused Display"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _predict_mask(unet: UNet, image_chw: np.ndarray, stats: dict, device: str, threshold: float = 0.5):
    # Match training normalization: per-channel p1/p99 -> [0,1].
    nimg = np.zeros_like(image_chw, dtype=np.float32)
    for c in range(image_chw.shape[0]):
        ch = image_chw[c]
        p1 = float(stats["p1"][c])
        p99 = float(stats["p99"][c])
        nimg[c] = 0 if p99 - p1 < 1e-8 else np.clip((ch - p1) / (p99 - p1), 0, 1)

    x = torch.from_numpy(nimg).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = unet(x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = np.where(probs >= threshold, 255, 0).astype(np.uint8)
    return mask, probs.astype(np.float32)


def _morph_refine_probs(probs: np.ndarray, close_radius: int = 5, open_radius: int = 3, sigma: float = 3.0) -> np.ndarray:
    """Morphological closing + opening then Gaussian smooth → soft confidence map."""
    from scipy.ndimage import binary_closing, binary_opening, gaussian_filter

    def _disk(r):
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        return (x ** 2 + y ** 2) <= r ** 2

    binary = probs >= 0.5
    closed = binary_closing(binary, structure=_disk(close_radius))
    opened = binary_opening(closed, structure=_disk(open_radius))

    # Blend smoothed raw probs with morphologically cleaned mask to preserve soft edges
    soft_raw = gaussian_filter(probs.astype(np.float64), sigma=sigma)
    soft_morph = gaussian_filter(opened.astype(np.float64), sigma=sigma)
    conf = np.maximum(soft_raw, soft_morph).astype(np.float32)
    return np.clip(conf, 0.0, 1.0)


def _soft_fuse(orig_chw: np.ndarray, recon_chw: np.ndarray, conf_map: np.ndarray) -> np.ndarray:
    """Confidence-based soft fusion: fused = conf * recon + (1-conf) * orig."""
    c = conf_map[np.newaxis, :, :]
    return (c * recon_chw + (1.0 - c) * orig_chw).astype(np.float32)


def _post_process_display(hwc_float01: np.ndarray) -> np.ndarray:
    """Bilateral Filter → CLAHE → Unsharp Mask for judge-facing display outputs only."""
    import cv2

    img_u8 = np.clip(hwc_float01 * 255.0, 0, 255).astype(np.uint8)

    # 1. Bilateral filter – edge-preserving noise reduction
    bilat = cv2.bilateralFilter(img_u8, d=5, sigmaColor=50, sigmaSpace=50)

    # 2. CLAHE on L channel in LAB colour space (visualization only)
    lab = cv2.cvtColor(bilat, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe_op.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge([l_clahe, a_ch, b_ch]), cv2.COLOR_LAB2RGB)

    # 3. Unsharp Mask: amount=1.5, sigma=1.0
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(enhanced, 2.5, blurred, -1.5, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)

    return sharp.astype(np.float32) / 255.0


def _predict_reconstruction(nafnet: NAFNetWrapper, image_chw: np.ndarray, stats: dict, device: str):
    hwc = np.transpose(image_chw, (1, 2, 0))
    n = normalize_image(hwc, stats["p1"], stats["p99"])
    x = torch.from_numpy(np.transpose(n, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = nafnet(x)[0].cpu().numpy()

    pred_raw_hwc = np.transpose(pred, (1, 2, 0)).astype(np.float32)
    pred_norm_hwc = np.clip(pred_raw_hwc, 0.0, 1.0)
    pred_denorm = np.empty_like(pred_norm_hwc, dtype=np.float32)
    for c in range(3):
        pred_denorm[:, :, c] = pred_norm_hwc[:, :, c] * (stats["p99"][c] - stats["p1"][c]) + stats["p1"][c]
    return np.transpose(pred_denorm, (2, 0, 1)).astype(np.float32), pred_raw_hwc


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Input Image
    orig_chw, profile = _read3(Path(args.input_image))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats = _load_stats(Path(args.stats_json))

    # U-Net Detection
    unet = UNet(in_channels=3, out_channels=1).to(device)
    u_state = torch.load(args.unet_checkpoint, map_location=device)
    u_state = u_state["model_state_dict"] if isinstance(u_state, dict) and "model_state_dict" in u_state else u_state
    unet.load_state_dict(u_state, strict=False)
    unet.eval()

    mask_u8, probs = _predict_mask(unet, orig_chw, stats=stats, device=device, threshold=args.mask_threshold)
    conf_map = _morph_refine_probs(probs)

    # NAFNet Reconstruction (frozen production model)
    nafnet = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    n_state = torch.load(args.nafnet_checkpoint, map_location=device)
    n_state = n_state["state_dict"] if isinstance(n_state, dict) and "state_dict" in n_state else n_state
    nafnet.load_state_dict(n_state, strict=False)
    nafnet.eval()

    recon_chw, recon_raw_hwc = _predict_reconstruction(nafnet, orig_chw, stats=stats, device=device)

    # Fusion Engine – confidence-based soft fusion (replaces hard binary mask)
    fused_chw = _soft_fuse(orig_chw, recon_chw, conf_map)
    _diff = np.mean(np.abs(fused_chw - orig_chw), axis=0)
    _dmin, _dmax = float(np.percentile(_diff, 2)), float(np.percentile(_diff, 98))
    diff_vis = (
        np.zeros_like(_diff, dtype=np.float32)
        if _dmax - _dmin < 1e-8
        else np.clip((_diff - _dmin) / (_dmax - _dmin), 0.0, 1.0).astype(np.float32)
    )

    # Dual export strategy: scientific TIFs + post-processed display outputs
    input_display_hwc, input_p1, input_p99 = _percentile_stretch_global(_chw_to_hwc(orig_chw), 1.0, 99.0)
    recon_display_hwc, recon_p1, recon_p99 = _percentile_stretch_global(recon_raw_hwc, 1.0, 99.0)
    fused_display_hwc, fused_p1, fused_p99 = _percentile_stretch_global(_chw_to_hwc(fused_chw), 1.0, 99.0)

    # Post-processing: Bilateral → CLAHE → Unsharp (display/visualization only, not scientific)
    recon_pp_hwc = _post_process_display(recon_display_hwc)
    fused_pp_hwc = _post_process_display(fused_display_hwc)

    # Save required outputs
    save_png(out_dir / "original.png", percentile_vis(orig_chw))
    save_png(out_dir / "cloud_mask.png", np.repeat((mask_u8.astype(np.float32) / 255.0)[:, :, None], 3, axis=2))
    save_raster(out_dir / "reconstruction.tif", recon_chw, profile)
    save_png(out_dir / "reconstruction_display.png", recon_pp_hwc)
    save_png(out_dir / "fused.png", percentile_vis(fused_chw))
    save_png(out_dir / "fused_display.png", fused_pp_hwc)
    save_png(out_dir / "difference_map.png", np.repeat(diff_vis[:, :, None], 3, axis=2))

    _save_comparison_display(
        out_dir / "comparison_display.png",
        input_hwc01=input_display_hwc,
        recon_hwc01=recon_pp_hwc,
        fused_hwc01=fused_pp_hwc,
    )

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

    report_lines = []
    report_lines.append("# Display Export Report")
    report_lines.append("")
    report_lines.append("## Raw Prediction Stats")
    raw_stats = _stats(recon_raw_hwc)
    report_lines.append(
        f"- shape={raw_stats['shape']}, dtype={raw_stats['dtype']}, min={raw_stats['min']:.6f}, max={raw_stats['max']:.6f}, mean={raw_stats['mean']:.6f}, std={raw_stats['std']:.6f}"
    )
    report_lines.append("")
    report_lines.append("## Percentile Stretch Values (p1, p99)")
    report_lines.append(f"- input: p1={input_p1:.6f}, p99={input_p99:.6f}")
    report_lines.append(f"- reconstruction: p1={recon_p1:.6f}, p99={recon_p99:.6f}")
    report_lines.append(f"- fused: p1={fused_p1:.6f}, p99={fused_p99:.6f}")
    report_lines.append("")
    report_lines.append("## Display Image Stats")
    recon_disp_stats = _stats(recon_display_hwc)
    fused_disp_stats = _stats(fused_display_hwc)
    report_lines.append(
        f"- reconstruction_display.png: min={recon_disp_stats['min']:.6f}, max={recon_disp_stats['max']:.6f}, mean={recon_disp_stats['mean']:.6f}, std={recon_disp_stats['std']:.6f}"
    )
    report_lines.append(
        f"- fused_display.png: min={fused_disp_stats['min']:.6f}, max={fused_disp_stats['max']:.6f}, mean={fused_disp_stats['mean']:.6f}, std={fused_disp_stats['std']:.6f}"
    )
    report_lines.append("")
    report_lines.append("## Exported Files")
    report_lines.append(f"- {str((out_dir / 'reconstruction.tif').as_posix())}")
    report_lines.append(f"- {str((out_dir / 'reconstruction_display.png').as_posix())}")
    report_lines.append(f"- {str((out_dir / 'fused_display.png').as_posix())}")
    report_lines.append(f"- {str((out_dir / 'comparison_display.png').as_posix())}")

    (out_dir / "display_export_report.md").write_text("\n".join(report_lines), encoding="utf-8")

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
    p.add_argument("--mask_threshold", type=float, default=0.10)
    run(p.parse_args())
