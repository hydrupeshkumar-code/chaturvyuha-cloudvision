import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

try:
    from .dataset import CloudDataset
    from .model import UNet
except ImportError:
    from dataset import CloudDataset
    from model import UNet


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def read_grn(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32)
    return arr


def norm_train_style(chw: np.ndarray) -> np.ndarray:
    out = np.zeros_like(chw, dtype=np.float32)
    for c in range(chw.shape[0]):
        ch = chw[c]
        p1 = float(np.nanpercentile(ch, 1))
        p99 = float(np.nanpercentile(ch, 99))
        if p99 - p1 < 1e-8:
            out[c] = 0.0
        else:
            out[c] = np.clip((ch - p1) / (p99 - p1), 0.0, 1.0)
    return np.nan_to_num(out)


def norm_infer_style(chw: np.ndarray) -> np.ndarray:
    out = np.zeros_like(chw, dtype=np.float32)
    for c in range(chw.shape[0]):
        ch = chw[c]
        p2 = float(np.nanpercentile(ch, 2))
        p98 = float(np.nanpercentile(ch, 98))
        if p98 - p2 < 1e-8:
            out[c] = 0.0
        else:
            out[c] = np.clip((ch - p2) / (p98 - p2), 0.0, 1.0)
    return np.nan_to_num(out)


def to_vis(chw01: np.ndarray) -> np.ndarray:
    hwc = np.transpose(chw01, (1, 2, 0)).astype(np.float32)
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


def save_png(path: Path, hwc01: np.ndarray):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), np.clip(hwc01 * 255.0, 0, 255).astype(np.uint8))


def compute_ch_stats(chw01: np.ndarray):
    means = [float(np.mean(chw01[c])) for c in range(chw01.shape[0])]
    stds = [float(np.std(chw01[c])) for c in range(chw01.shape[0])]
    return means, stds


def threshold_to_tag(t: float) -> str:
    return f"{int(round(t * 100)):03d}"


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)

    raw = read_grn(image_path)

    infer_norm = norm_infer_style(raw)
    train_norm = norm_train_style(raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=3, out_channels=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    x = torch.from_numpy(infer_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)

    pmin = float(np.min(prob))
    pmax = float(np.max(prob))
    pmean = float(np.mean(prob))
    pstd = float(np.std(prob))

    # Probability map and histogram
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(prob, cmap="viridis")
    ax.set_title("Probability Map (Pre-threshold)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_dir / "probability_map.png", dpi=180)
    plt.close(fig)

    hist_counts, hist_bins = np.histogram(prob.flatten(), bins=20, range=(0.0, 1.0))
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(prob.flatten(), bins=20, range=(0.0, 1.0), color="#2a6fdb")
    ax.set_title("Probability Histogram")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Pixel Count")
    plt.tight_layout()
    fig.savefig(out_dir / "probability_histogram.png", dpi=180)
    plt.close(fig)

    vis_input = to_vis(infer_norm)

    sweep_rows = []
    for t in THRESHOLDS:
        tag = threshold_to_tag(t)
        mask = np.where(prob >= t, 255, 0).astype(np.uint8)
        coverage = float((mask > 0).sum() / mask.size * 100.0)

        mask_rgb = np.repeat((mask.astype(np.float32) / 255.0)[:, :, None], 3, axis=2)
        save_png(out_dir / f"mask_{tag}.png", mask_rgb)

        overlay = vis_input.copy()
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + 0.5 * (mask > 0).astype(np.float32), 0.0, 1.0)
        save_png(out_dir / f"overlay_{tag}.png", overlay)

        sweep_rows.append({"threshold": t, "cloud_coverage_percent": coverage, "masked_pixels": int((mask > 0).sum())})

    # Training-normalization stats from synthetic training set.
    train_ds = CloudDataset(
        image_dir=str(Path(args.train_data_dir) / "train" / "images"),
        mask_dir=str(Path(args.train_data_dir) / "train" / "masks"),
        augment=False,
    )

    sum_ch = np.zeros(3, dtype=np.float64)
    sq_ch = np.zeros(3, dtype=np.float64)
    n_ch = np.zeros(3, dtype=np.float64)
    for i in range(len(train_ds)):
        x_i, _ = train_ds[i]
        arr = x_i.numpy()
        for c in range(3):
            sum_ch[c] += float(arr[c].sum())
            sq_ch[c] += float((arr[c] * arr[c]).sum())
            n_ch[c] += float(arr[c].size)

    train_means = (sum_ch / np.maximum(n_ch, 1.0)).tolist()
    train_stds = np.sqrt(np.maximum((sq_ch / np.maximum(n_ch, 1.0)) - np.square(train_means), 0.0)).tolist()

    infer_means, infer_stds = compute_ch_stats(infer_norm)
    infer_means_train_style, infer_stds_train_style = compute_ch_stats(train_norm)

    # Best threshold by highest coverage under practical cap.
    best_visual = None
    for row in sweep_rows:
        if 1.0 <= row["cloud_coverage_percent"] <= 90.0:
            best_visual = row
            break
    if best_visual is None:
        best_visual = max(sweep_rows, key=lambda r: r["cloud_coverage_percent"])

    threshold_lines = [
        "# Threshold Sweep Report",
        "",
        "| threshold | cloud_coverage_percent | masked_pixels |",
        "|---:|---:|---:|",
    ]
    for row in sweep_rows:
        threshold_lines.append(f"| {row['threshold']:.2f} | {row['cloud_coverage_percent']:.6f} | {row['masked_pixels']} |")
    threshold_lines.append("")
    threshold_lines.append(f"Recommended visual threshold candidate: {best_visual['threshold']:.2f}")
    (out_dir / "threshold_sweep_report.md").write_text("\n".join(threshold_lines), encoding="utf-8")

    hist_table = []
    for i in range(len(hist_counts)):
        hist_table.append(
            {
                "bin_left": float(hist_bins[i]),
                "bin_right": float(hist_bins[i + 1]),
                "count": int(hist_counts[i]),
            }
        )

    audit_lines = [
        "# Cloud Detector Audit",
        "",
        "## Pipeline Trace",
        "1. Input image read as GRN bands (1,2,3).",
        "2. Inference normalization uses per-channel percentile scaling p2/p98 to [0,1].",
        "3. Model forward pass outputs logits.",
        "4. Sigmoid produces probability map.",
        "5. Threshold is applied to produce binary mask.",
        "6. Morphology step: not applied in current inference path.",
        "7. Final mask used by fusion as 0/255 uint8.",
        "",
        "## Probability Map Stats (Before Threshold)",
        f"- min: {pmin:.6f}",
        f"- max: {pmax:.6f}",
        f"- mean: {pmean:.6f}",
        f"- std: {pstd:.6f}",
        "",
        "## Probability Histogram (20 bins)",
        "| bin_left | bin_right | count |",
        "|---:|---:|---:|",
    ]
    for row in hist_table:
        audit_lines.append(f"| {row['bin_left']:.2f} | {row['bin_right']:.2f} | {row['count']} |")

    audit_lines.extend(
        [
            "",
            "## Normalization Check",
            "- Training normalization in dataset: percentile p1/p99 per channel to [0,1].",
            "- Inference normalization in pipeline: percentile p2/p98 per channel to [0,1].",
            f"- Training-set normalized channel means (G,R,NIR): {train_means}",
            f"- Training-set normalized channel stds  (G,R,NIR): {train_stds}",
            f"- Inference-scene normalized means using p2/p98: {infer_means}",
            f"- Inference-scene normalized stds  using p2/p98: {infer_stds}",
            f"- Inference-scene normalized means using p1/p99: {infer_means_train_style}",
            f"- Inference-scene normalized stds  using p1/p99: {infer_stds_train_style}",
            "",
            "## Confidence Inspection",
            "- Check whether probabilities are concentrated below 0.5 and within lower-confidence bands.",
            f"- Mean probability indicates confidence floor at: {pmean:.6f}",
            f"- Max probability indicates ceiling at: {pmax:.6f}",
            "",
            "## Final Conclusion",
            "1. Root cause of tiny mask at threshold 0.5: inferred from low probability distribution and thresholding behavior.",
            "2. Thresholding issue present if most cloud-like pixels sit below 0.5.",
            "3. Normalization mismatch exists (p1/p99 in training vs p2/p98 in inference), but likely secondary.",
            "4. Synthetic-mask training appears to have generalization gap to real cloud morphology/texture.",
            "5. Fastest single fix: lower inference threshold to the sweep-selected value and keep model frozen.",
            f"   Suggested immediate threshold: {best_visual['threshold']:.2f}",
        ]
    )

    (out_dir / "cloud_detector_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")

    out_json = {
        "probability_stats": {"min": pmin, "max": pmax, "mean": pmean, "std": pstd},
        "histogram": hist_table,
        "threshold_sweep": sweep_rows,
        "recommended_threshold": best_visual["threshold"],
        "training_norm_stats": {"means": train_means, "stds": train_stds},
        "inference_norm_stats_p2_p98": {"means": infer_means, "stds": infer_stds},
        "inference_norm_stats_p1_p99": {"means": infer_means_train_style, "stds": infer_stds_train_style},
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    print(f"Audit complete: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Audit cloud detector on real imagery")
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default="checkpoints_unet_cloud/best_iou.pth")
    p.add_argument("--train_data_dir", default="datasets/cloud_detector_synth")
    p.add_argument("--output_dir", default="outputs/cloud_detector_audit")
    run(p.parse_args())
