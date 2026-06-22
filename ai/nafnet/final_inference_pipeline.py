"""Final NAFNet inference pipeline for curated strict validation samples.

Pipeline:
Cloudy TIFF -> Cloud Detector -> NAFNet -> soft fusion -> TIFF + polished visualization

Outputs per sample:
- reconstructed TIFF (metadata-preserving profile)
- fused TIFF (soft confidence fusion)
- cloud mask, confidence, and mask overlay PNGs
- post-processed display PNGs
- side-by-side comparison PNG
- reconstruction_gallery.html
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import rasterio
import torch

from ai.cloud_detector.model import UNet
from .dataset import normalize_image
from .model import NAFNetWrapper


def _load_stats(path: Path):
    stats = {"p1": [0.0, 0.0, 0.0], "p99": [6000.0, 6000.0, 6000.0]}
    if path.exists():
        try:
            j = json.loads(path.read_text(encoding="utf-8"))
            if "p1" in j and "p99" in j:
                stats["p1"] = j["p1"]
                stats["p99"] = j["p99"]
        except Exception:
            pass
    return stats


def _resolve_pair_path(path_text: str, workspace_root: Path):
    p = Path(path_text)
    if p.exists():
        return p
    p2 = workspace_root / path_text
    if p2.exists():
        return p2
    if path_text.startswith("chaturvyuha-cloudvision/") or path_text.startswith("chaturvyuha-cloudvision\\"):
        stripped = path_text.replace("\\", "/").split("/", 1)[1]
        p3 = workspace_root / stripped
        if p3.exists():
            return p3
    return p2


def _percentile_stretch_global(img_hwc, p_low: float = 1.0, p_high: float = 99.0):
    img = img_hwc.astype(np.float32)
    lo = float(np.percentile(img, p_low))
    hi = float(np.percentile(img, p_high))
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.float32), lo, hi
    stretched = np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return stretched, lo, hi


def _percentile_stretch(img_hwc):
    img = img_hwc.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        p2 = float(np.percentile(ch, 2))
        p98 = float(np.percentile(ch, 98))
        if p98 - p2 < 1e-8:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = np.clip((ch - p2) / (p98 - p2), 0.0, 1.0)
    return out


def _save_png(arr_hwc, path: Path):
    import imageio.v2 as imageio

    x = np.clip(arr_hwc * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), x)


def _load_three_band_tiff(path: Path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
    if arr.shape[0] < 3:
        raise ValueError(f"Expected >=3 bands in {path}, got {arr.shape[0]}")
    return np.transpose(arr[:3], (1, 2, 0)), profile


def _denormalize(norm_hwc, p1, p99):
    out = np.empty_like(norm_hwc, dtype=np.float32)
    for c in range(norm_hwc.shape[2]):
        out[:, :, c] = norm_hwc[:, :, c] * (p99[c] - p1[c]) + p1[c]
    return out


def _write_reconstructed_tiff(pred_hwc, cloudy_profile, out_path: Path):
    prof = dict(cloudy_profile)
    prof.update(
        {
            "count": 3,
            "dtype": "float32",
            "nodata": None,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(np.transpose(pred_hwc.astype(np.float32), (2, 0, 1)))


def _predict_mask(unet: UNet, image_hwc: np.ndarray, stats: dict, device: str, threshold: float = 0.5):
    nimg = np.zeros_like(image_hwc, dtype=np.float32)
    for c in range(image_hwc.shape[2]):
        p1 = float(stats["p1"][c])
        p99 = float(stats["p99"][c])
        ch = image_hwc[:, :, c]
        nimg[:, :, c] = 0.0 if p99 - p1 < 1e-8 else np.clip((ch - p1) / (p99 - p1), 0.0, 1.0)

    x = torch.from_numpy(np.transpose(nimg, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = unet(x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = np.where(probs >= threshold, 255, 0).astype(np.uint8)
    return mask, probs.astype(np.float32)


def _morph_refine_probs(probs: np.ndarray, close_radius: int = 5, open_radius: int = 3, sigma: float = 3.0) -> np.ndarray:
    from scipy.ndimage import binary_closing, binary_opening, gaussian_filter

    def _disk(r):
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        return (x ** 2 + y ** 2) <= r ** 2

    binary = probs >= 0.5
    closed = binary_closing(binary, structure=_disk(close_radius))
    opened = binary_opening(closed, structure=_disk(open_radius))

    soft_raw = gaussian_filter(probs.astype(np.float64), sigma=sigma)
    soft_morph = gaussian_filter(opened.astype(np.float64), sigma=sigma)
    conf = np.maximum(soft_raw, soft_morph).astype(np.float32)
    return np.clip(conf, 0.0, 1.0)


def _soft_fuse(orig_chw: np.ndarray, recon_chw: np.ndarray, conf_map: np.ndarray) -> np.ndarray:
    c = conf_map[np.newaxis, :, :]
    return (c * recon_chw + (1.0 - c) * orig_chw).astype(np.float32)


def _post_process_display(hwc_float01: np.ndarray) -> np.ndarray:
    import cv2

    img_u8 = np.clip(hwc_float01 * 255.0, 0, 255).astype(np.uint8)
    bilat = cv2.bilateralFilter(img_u8, d=5, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(bilat, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe_op.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge([l_clahe, a_ch, b_ch]), cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(enhanced, 2.5, blurred, -1.5, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp.astype(np.float32) / 255.0


def _overlay_mask(base_hwc: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    mask_rgb = np.zeros_like(base_hwc, dtype=np.float32)
    mask_rgb[:, :, 0] = mask_u8.astype(np.float32) / 255.0
    return np.clip(base_hwc * (1.0 - alpha) + mask_rgb * alpha, 0.0, 1.0)


def _save_comparison_display(path: Path, input_hwc01: np.ndarray, recon_hwc01: np.ndarray, fused_hwc01: np.ndarray):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        (input_hwc01, "Input Cloudy"),
        (recon_hwc01, "Reconstruction"),
        (fused_hwc01, "Fused Soft"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_gallery(rows, out_html: Path):
    css = """
body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 20px; background: #f7f7f4; color: #1a1a1a; }
h1 { margin-bottom: 6px; }
p.meta { margin-top: 0; color: #444; }
.grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
.card { background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
.row { display: grid; gap: 8px; grid-template-columns: repeat(3, minmax(120px, 1fr)); }
.row img { width: 100%; border-radius: 6px; border: 1px solid #ccc; image-rendering: auto; }
.cap { font-size: 12px; color: #333; margin: 4px 0 0; text-align: center; }
.small { font-size: 12px; color: #555; margin-top: 6px; word-break: break-word; }
@media (max-width: 900px) { .row { grid-template-columns: repeat(2, minmax(120px, 1fr)); } }
"""

    parts = []
    parts.append("<!doctype html>")
    parts.append("<html><head><meta charset='utf-8'><title>Reconstruction Gallery</title>")
    parts.append(f"<style>{css}</style></head><body>")
    parts.append("<h1>Final NAFNet Reconstruction Gallery</h1>")
    parts.append("<p class='meta'>Validation examples show cloudy input, mask overlay, prediction, soft fusion, target, and difference.</p>")
    parts.append("<div class='grid'>")

    for r in rows:
        parts.append("<div class='card'>")
        parts.append(f"<div><strong>{r['sample_id']}</strong></div>")
        parts.append("<div class='row'>")
        for label, key in [
            ("Input Cloudy", "cloudy_png"),
            ("Mask Overlay", "mask_overlay_png"),
            ("Prediction", "pred_png"),
            ("Fused Soft", "fused_png"),
            ("Target", "target_png"),
            ("Difference Map", "diff_png"),
        ]:
            parts.append("<div>")
            parts.append(f"<img src='{r[key]}' alt='{label}'>")
            parts.append(f"<div class='cap'>{label}</div>")
            parts.append("</div>")
        parts.append("</div>")
        parts.append(
            f"<div class='small'>Cloudy: {r['cloudy_path']}<br>Target: {r['target_path']}<br>Reconstructed TIFF: {r['recon_tiff']}</div>"
        )
        parts.append("</div>")

    parts.append("</div></body></html>")
    out_html.write_text("\n".join(parts), encoding="utf-8")


def run(args):
    workspace_root = Path(args.workspace_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.split_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_pairs = manifest.get("val_pairs", [])
    if not val_pairs:
        raise RuntimeError("No val_pairs found in split manifest.")

    rng = random.Random(args.seed)
    k = min(args.num_samples, len(val_pairs))
    idxs = rng.sample(range(len(val_pairs)), k)

    stats = _load_stats(Path(args.stats_json))
    p1 = stats["p1"]
    p99 = stats["p99"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    unet = UNet(in_channels=3, out_channels=1).to(device)
    u_state = torch.load(args.unet_checkpoint, map_location=device)
    if isinstance(u_state, dict) and "model_state_dict" in u_state:
        u_state = u_state["model_state_dict"]
    unet.load_state_dict(u_state, strict=False)
    unet.eval()

    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    rows = []

    for out_idx, vi in enumerate(idxs):
        rec = val_pairs[vi]
        cloudy_path = _resolve_pair_path(rec["cloudy_path"], workspace_root)
        target_path = _resolve_pair_path(rec["target_path"], workspace_root)

        cloudy_raw, cloudy_profile = _load_three_band_tiff(cloudy_path)
        target_raw, _ = _load_three_band_tiff(target_path)

        cloudy_norm = normalize_image(cloudy_raw, p1, p99)
        target_norm = normalize_image(target_raw, p1, p99)

        x = torch.from_numpy(np.transpose(cloudy_norm, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x)[0].cpu().numpy()

        pred_norm = np.clip(np.transpose(pred, (1, 2, 0)), 0.0, 1.0)
        pred_denorm = _denormalize(pred_norm, p1, p99)

        mask_u8, probs = _predict_mask(unet, cloudy_raw, stats, device, args.mask_threshold)
        conf_map = _morph_refine_probs(probs, close_radius=args.close_radius, open_radius=args.open_radius, sigma=args.sigma)

        orig_chw = np.transpose(cloudy_raw, (2, 0, 1))
        recon_chw = np.transpose(pred_denorm, (2, 0, 1))
        fused_chw = _soft_fuse(orig_chw, recon_chw, conf_map)
        fused_hwc = np.transpose(fused_chw, (1, 2, 0))

        fused_vis, _, _ = _percentile_stretch_global(fused_hwc)
        recon_vis, _, _ = _percentile_stretch_global(pred_denorm)
        cloudy_vis, _, _ = _percentile_stretch_global(cloudy_raw)
        target_vis, _, _ = _percentile_stretch_global(target_raw)

        recon_pp = _post_process_display(recon_vis)
        fused_pp = _post_process_display(fused_vis)
        cloudy_overlay = _overlay_mask(cloudy_vis, mask_u8, alpha=0.30)

        diff = np.mean(np.abs(fused_vis - target_vis), axis=2, keepdims=True)
        diff_vis = _percentile_stretch(np.repeat(diff, 3, axis=2))

        sample_id = f"sample_{out_idx:02d}_validx_{vi}"
        sample_dir = out_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        recon_tiff = sample_dir / "reconstruction.tif"
        fused_tiff = sample_dir / "fused.tif"
        _write_reconstructed_tiff(pred_denorm, cloudy_profile, recon_tiff)
        _write_reconstructed_tiff(fused_hwc, cloudy_profile, fused_tiff)

        sample_outputs = {
            "cloudy_png": sample_dir / "input_cloudy.png",
            "mask_png": sample_dir / "cloud_mask.png",
            "mask_confidence_png": sample_dir / "cloud_mask_confidence.png",
            "mask_overlay_png": sample_dir / "mask_overlay.png",
            "prediction_png": sample_dir / "prediction.png",
            "fused_png": sample_dir / "fused_soft.png",
            "target_png": sample_dir / "target.png",
            "diff_png": sample_dir / "difference_map.png",
            "comparison_png": sample_dir / "comparison.png",
            "reconstruction_display_png": sample_dir / "reconstruction_display.png",
            "fused_display_png": sample_dir / "fused_display.png",
        }

        _save_png(cloudy_vis, sample_outputs["cloudy_png"])
        _save_png(np.repeat(mask_u8.astype(np.float32) / 255.0, 3, axis=2), sample_outputs["mask_png"])
        _save_png(np.repeat(probs[:, :, None], 3, axis=2), sample_outputs["mask_confidence_png"])
        _save_png(cloudy_overlay, sample_outputs["mask_overlay_png"])
        _save_png(recon_pp, sample_outputs["reconstruction_display_png"])
        _save_png(fused_pp, sample_outputs["fused_display_png"])
        _save_png(recon_vis, sample_outputs["prediction_png"])
        _save_png(fused_vis, sample_outputs["fused_png"])
        _save_png(target_vis, sample_outputs["target_png"])
        _save_png(diff_vis, sample_outputs["diff_png"])
        _save_png(np.concatenate([cloudy_vis, recon_vis, fused_vis, target_vis], axis=1), sample_outputs["comparison_png"])

        _save_comparison_display(
            sample_dir / "comparison_display.png",
            input_hwc01=cloudy_vis,
            recon_hwc01=recon_pp,
            fused_hwc01=fused_pp,
        )

        rows.append(
            {
                "sample_id": sample_id,
                "cloudy_path": str(cloudy_path).replace("\\", "/"),
                "target_path": str(target_path).replace("\\", "/"),
                "recon_tiff": str(recon_tiff).replace("\\", "/"),
                "cloudy_png": str(sample_outputs["cloudy_png"].relative_to(out_dir)).replace("\\", "/"),
                "mask_overlay_png": str(sample_outputs["mask_overlay_png"].relative_to(out_dir)).replace("\\", "/"),
                "pred_png": str(sample_outputs["prediction_png"].relative_to(out_dir)).replace("\\", "/"),
                "fused_png": str(sample_outputs["fused_png"].relative_to(out_dir)).replace("\\", "/"),
                "target_png": str(sample_outputs["target_png"].relative_to(out_dir)).replace("\\", "/"),
                "diff_png": str(sample_outputs["diff_png"].relative_to(out_dir)).replace("\\", "/"),
            }
        )

        print(f"[{out_idx + 1}/{k}] {sample_id} done")

    _build_gallery(rows, out_dir / "reconstruction_gallery.html")

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "unet_checkpoint": str(args.unet_checkpoint),
                "split_manifest": str(args.split_manifest),
                "stats_json": str(args.stats_json),
                "num_samples": k,
                "seed": args.seed,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Inference complete.")
    print("Gallery:", out_dir / "reconstruction_gallery.html")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints_nafnet/strict_curated_training/best_ssim.pth")
    p.add_argument("--unet_checkpoint", default="checkpoints_unet_cloud/best_iou.pth")
    p.add_argument("--split_manifest", default="checkpoints_nafnet/strict_curated_training/split_manifest.json")
    p.add_argument("--stats_json", default="tmp_stats/band_statistics.json")
    p.add_argument("--workspace_root", default=".")
    p.add_argument("--out_dir", default="checkpoints_nafnet/strict_curated_training/inference_val50")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_threshold", type=float, default=0.5)
    p.add_argument("--close_radius", type=int, default=5)
    p.add_argument("--open_radius", type=int, default=3)
    p.add_argument("--sigma", type=float, default=3.0)
    raise SystemExit(run(p.parse_args()))
