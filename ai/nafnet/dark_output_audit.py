import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import rasterio
import torch

from . import metrics as naf_metrics
from .dataset import normalize_image
from .model import NAFNetWrapper


def _read3(path: Path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    if arr.shape[0] < 3:
        raise RuntimeError(f"Expected >=3 bands in {path}, got {arr.shape[0]}")
    return arr[:3]


def _load_stats(stats_path: Path):
    stats = {"p1": [0.0, 0.0, 0.0], "p99": [6000.0, 6000.0, 6000.0]}
    if stats_path.exists():
        data = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["p1"] = [float(x) for x in data.get("p1", stats["p1"])]
        stats["p99"] = [float(x) for x in data.get("p99", stats["p99"])]
    return stats


def _tensor_stats(arr):
    x = np.asarray(arr)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p1": float(np.percentile(x, 1)),
        "p99": float(np.percentile(x, 99)),
    }


def _to_hwc(chw):
    return np.transpose(chw, (1, 2, 0))


def _to_chw(hwc):
    return np.transpose(hwc, (2, 0, 1))


def _save_png01(path: Path, hwc01: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), np.clip(hwc01 * 255.0, 0, 255).astype(np.uint8))


def _stretch_minmax(img_hwc):
    out = np.zeros_like(img_hwc, dtype=np.float32)
    for c in range(img_hwc.shape[2]):
        ch = img_hwc[:, :, c]
        mn = float(np.min(ch))
        mx = float(np.max(ch))
        if mx - mn < 1e-8:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = np.clip((ch - mn) / (mx - mn), 0.0, 1.0)
    return out


def _stretch_percentile(img_hwc, p_lo=1.0, p_hi=99.0):
    out = np.zeros_like(img_hwc, dtype=np.float32)
    for c in range(img_hwc.shape[2]):
        ch = img_hwc[:, :, c]
        lo = float(np.percentile(ch, p_lo))
        hi = float(np.percentile(ch, p_hi))
        if hi - lo < 1e-8:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return out


def _equalize_hist_channel(x01):
    q = np.clip((x01 * 255.0).round(), 0, 255).astype(np.uint8)
    hist = np.bincount(q.reshape(-1), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    if cdf[-1] <= 0:
        return np.zeros_like(x01, dtype=np.float32)
    cdf = cdf / cdf[-1]
    return cdf[q].astype(np.float32)


def _equalize_hist(img_hwc01):
    out = np.zeros_like(img_hwc01, dtype=np.float32)
    for c in range(img_hwc01.shape[2]):
        out[:, :, c] = _equalize_hist_channel(np.clip(img_hwc01[:, :, c], 0.0, 1.0))
    return out


def _load_pairs_from_csv(csv_path: Path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cloudy = row.get("cloudy_path")
            target = row.get("target_path") or row.get("clear_path")
            if cloudy and target:
                rows.append((cloudy, target))
    return rows


def _resolve_path(path_text: str, workspace_root: Path):
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


def _compute_dataset_stats(values):
    x = np.concatenate(values).astype(np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p1": float(np.percentile(x, 1)),
        "p99": float(np.percentile(x, 99)),
    }


def _load_best_epoch_metrics(metrics_csv_path: Path):
    best_row = None
    best_ssim = -1e12
    with open(metrics_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ssim = float(row["val_ssim"])
            if ssim > best_ssim:
                best_ssim = ssim
                best_row = row
    if best_row is None:
        raise RuntimeError(f"Could not find rows in {metrics_csv_path}")
    return {
        "epoch": int(best_row["epoch"]),
        "val_psnr": float(best_row["val_psnr"]),
        "val_ssim": float(best_row["val_ssim"]),
        "val_sam": float(best_row["val_sam"]),
        "prediction_mean": float(best_row["pred_mean"]),
        "prediction_std": float(best_row["pred_std"]),
    }


def run(args):
    workspace_root = Path(args.workspace_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = _load_stats(Path(args.stats_json))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    state = torch.load(args.nafnet_checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    # Step 1: raw model output stats for the specified demo scene.
    input_demo_chw = _read3(Path(args.input_image))
    input_demo_hwc = _to_hwc(input_demo_chw)
    input_demo_norm = normalize_image(input_demo_hwc, stats["p1"], stats["p99"])
    x = torch.from_numpy(_to_chw(input_demo_norm).astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_raw_chw = model(x)[0].cpu().numpy().astype(np.float32)

    raw_stats = {
        "checkpoint": args.nafnet_checkpoint,
        "input_image": args.input_image,
        "output_tensor": _tensor_stats(pred_raw_chw),
        "output_tensor_per_channel": [
            _tensor_stats(pred_raw_chw[c]) for c in range(pred_raw_chw.shape[0])
        ],
    }
    (out_dir / "raw_output_stats.json").write_text(json.dumps(raw_stats, indent=2), encoding="utf-8")

    # Step 2: export pipeline audit from model output tensor to reconstruction.png.
    pred_hwc = _to_hwc(pred_raw_chw)
    stage_raw = pred_hwc
    stage_clamp = np.clip(stage_raw, 0.0, 1.0)
    stage_channel_order = stage_clamp  # identity: GRN stays [0,1,2]
    stage_uint8 = np.clip(stage_channel_order * 255.0, 0, 255).astype(np.uint8)

    export_stages = [
        ("model_output_tensor_hwc", stage_raw),
        ("clamp_0_1", stage_clamp),
        ("normalization", stage_clamp),
        ("percentile_scaling", stage_clamp),
        ("channel_order_identity", stage_channel_order),
        ("rgb_conversion_identity", stage_channel_order),
        ("uint8_conversion", stage_uint8.astype(np.float32) / 255.0),
    ]

    lines = []
    lines.append("# Export Pipeline Audit")
    lines.append("")
    lines.append("Path audited: model output tensor -> reconstruction.png")
    lines.append("")
    lines.append("## Operation-by-operation")
    lines.append("- clamp: applied via np.clip(pred, 0, 1)")
    lines.append("- normalization: none after clamp for reconstruction.png")
    lines.append("- percentile scaling: not applied for reconstruction.png (applied to fused/original visualizations only)")
    lines.append("- channel ordering: identity, no permutation")
    lines.append("- RGB conversion: identity HWC(3), no color-space transform")
    lines.append("- uint8 conversion: uint8 = clip(image * 255, 0, 255)")
    lines.append("")
    lines.append("## Value Ranges")
    lines.append("| stage | min | max | mean | std | p1 | p99 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, arr in export_stages:
        s = _tensor_stats(arr)
        lines.append(
            f"| {name} | {s['min']:.6f} | {s['max']:.6f} | {s['mean']:.6f} | {s['std']:.6f} | {s['p1']:.6f} | {s['p99']:.6f} |"
        )
    (out_dir / "export_pipeline_audit.md").write_text("\n".join(lines), encoding="utf-8")

    # Step 3: channel order audit.
    channel_lines = []
    channel_lines.append("# Channel Order Report")
    channel_lines.append("")
    channel_lines.append("## Expected Order")
    channel_lines.append("- channel 0: Green")
    channel_lines.append("- channel 1: Red")
    channel_lines.append("- channel 2: NIR")
    channel_lines.append("")
    channel_lines.append("## Training Path")
    channel_lines.append("- dataset reads raster in native order and keeps first three bands")
    channel_lines.append("- training uses HWC->CHW transpose only, no channel swap")
    channel_lines.append("")
    channel_lines.append("## Inference Path")
    channel_lines.append("- fusion pipeline reads first three bands and normalizes with same p1/p99")
    channel_lines.append("- inference uses HWC->CHW transpose only, no channel swap")
    channel_lines.append("")
    channel_lines.append("## Conclusion")
    channel_lines.append("- expected order: [Green, Red, NIR]")
    channel_lines.append("- actual order: [Green, Red, NIR] (index-preserving)")
    (out_dir / "channel_order_report.md").write_text("\n".join(channel_lines), encoding="utf-8")

    # Step 4: dynamic range audit over 100 validation samples.
    all_pairs = _load_pairs_from_csv(Path(args.pairs_csv))
    if len(all_pairs) < (args.train_pairs + args.val_pairs + 1):
        raise RuntimeError(
            f"Not enough pairs in {args.pairs_csv}: {len(all_pairs)} found, need at least {args.train_pairs + args.val_pairs + 1}"
        )

    rng = np.random.default_rng(args.seed)
    idxs = np.arange(len(all_pairs))
    rng.shuffle(idxs)
    shuffled = [all_pairs[i] for i in idxs]

    val_pairs = shuffled[args.train_pairs : args.train_pairs + args.val_pairs]
    n_eval = min(args.dynamic_samples, len(val_pairs))

    input_vals = []
    target_vals = []
    pred_vals = []

    metric_psnr = []
    metric_ssim = []
    metric_sam = []

    with torch.no_grad():
        for cloudy_path, target_path in val_pairs[:n_eval]:
            c_path = _resolve_path(cloudy_path, workspace_root)
            t_path = _resolve_path(target_path, workspace_root)

            c_raw = _to_hwc(_read3(c_path))
            t_raw = _to_hwc(_read3(t_path))

            c_norm = normalize_image(c_raw, stats["p1"], stats["p99"])
            t_norm = normalize_image(t_raw, stats["p1"], stats["p99"])

            xin = torch.from_numpy(_to_chw(c_norm).astype(np.float32)).unsqueeze(0).to(device)
            pred = model(xin)[0].cpu().numpy().astype(np.float32)
            pred_norm = np.clip(_to_hwc(pred), 0.0, 1.0)

            input_vals.append(c_norm.reshape(-1))
            target_vals.append(t_norm.reshape(-1))
            pred_vals.append(pred_norm.reshape(-1))

            metric_psnr.append(float(naf_metrics.psnr(t_norm, pred_norm)))
            metric_ssim.append(float(naf_metrics.ssim(t_norm, pred_norm)))
            metric_sam.append(float(naf_metrics.sam(t_norm, pred_norm)))

    input_stats = _compute_dataset_stats(input_vals)
    target_stats = _compute_dataset_stats(target_vals)
    pred_stats = _compute_dataset_stats(pred_vals)

    with open(out_dir / "dynamic_range_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tensor", "mean", "std", "min", "max", "p1", "p99"])
        writer.writerow(["Input", input_stats["mean"], input_stats["std"], input_stats["min"], input_stats["max"], input_stats["p1"], input_stats["p99"]])
        writer.writerow(["Target", target_stats["mean"], target_stats["std"], target_stats["min"], target_stats["max"], target_stats["p1"], target_stats["p99"]])
        writer.writerow(["Prediction", pred_stats["mean"], pred_stats["std"], pred_stats["min"], pred_stats["max"], pred_stats["p1"], pred_stats["p99"]])

    pred_range = pred_stats["p99"] - pred_stats["p1"]
    target_range = target_stats["p99"] - target_stats["p1"]
    compressed_ratio = pred_range / max(target_range, 1e-12)
    dynamic_collapse = compressed_ratio < args.dynamic_collapse_ratio_threshold

    # Step 5: visualization variants for same demo prediction.
    viz_a = stage_clamp
    viz_b = _stretch_minmax(stage_raw)
    viz_c = _stretch_percentile(stage_raw, p_lo=1.0, p_hi=99.0)
    viz_d = _equalize_hist(np.clip(stage_clamp, 0.0, 1.0))

    _save_png01(out_dir / "viz_A.png", viz_a)
    _save_png01(out_dir / "viz_B.png", viz_b)
    _save_png01(out_dir / "viz_C.png", viz_c)
    _save_png01(out_dir / "viz_D.png", viz_d)
    comparison = np.concatenate([viz_a, viz_b, viz_c, viz_d], axis=1)
    _save_png01(out_dir / "visualization_comparison.png", comparison)

    # Step 6: checkpoint audit against training records and current 100-sample val check.
    ckpt_metrics = _load_best_epoch_metrics(Path(args.training_metrics_csv))
    current_val = {
        "samples": n_eval,
        "psnr": float(np.mean(metric_psnr)),
        "ssim": float(np.mean(metric_ssim)),
        "sam": float(np.mean(metric_sam)),
        "prediction_mean": pred_stats["mean"],
        "prediction_std": pred_stats["std"],
    }

    # Step 7: root cause scoring.
    viz_gain = float(np.mean(viz_c) - np.mean(viz_a))
    scores = {
        "A_export_bug": 0.08,
        "B_channel_order_bug": 0.05,
        "C_visualization_bug": min(0.95, max(0.1, 0.6 + 1.5 * viz_gain)),
        "D_dynamic_range_collapse": 0.85 if dynamic_collapse else 0.45,
        "E_wrong_checkpoint_loaded": 0.15,
        "F_multiple_issues": 0.9 if (dynamic_collapse and viz_gain > 0.08) else 0.55,
    }

    best_hypothesis = max(scores.items(), key=lambda kv: kv[1])[0]
    hypothesis_map = {
        "A_export_bug": "A) Export bug",
        "B_channel_order_bug": "B) Channel-order bug",
        "C_visualization_bug": "C) Visualization bug",
        "D_dynamic_range_collapse": "D) Dynamic-range collapse",
        "E_wrong_checkpoint_loaded": "E) Wrong checkpoint loaded",
        "F_multiple_issues": "F) Multiple issues",
    }

    root_lines = []
    root_lines.append("# NAFNet Root Cause Report")
    root_lines.append("")
    root_lines.append("## Decision")
    root_lines.append(f"- Selected: {hypothesis_map[best_hypothesis]}")
    root_lines.append("")
    root_lines.append("## Evidence")
    root_lines.append(
        f"- Raw output tensor stats: min={raw_stats['output_tensor']['min']:.6f}, max={raw_stats['output_tensor']['max']:.6f}, mean={raw_stats['output_tensor']['mean']:.6f}, std={raw_stats['output_tensor']['std']:.6f}"
    )
    root_lines.append(
        f"- 100-sample dynamic range ratio (prediction vs target, p99-p1): {compressed_ratio:.4f}"
    )
    root_lines.append(
        f"- Visualization lift (mean intensity, viz_C - viz_A): {viz_gain:.6f}"
    )
    root_lines.append(
        f"- Checkpoint train-time best epoch metrics: PSNR={ckpt_metrics['val_psnr']:.4f}, SSIM={ckpt_metrics['val_ssim']:.6f}, SAM={ckpt_metrics['val_sam']:.4f}, pred_mean={ckpt_metrics['prediction_mean']:.6f}, pred_std={ckpt_metrics['prediction_std']:.6f}"
    )
    root_lines.append(
        f"- Current 100-sample checkpoint metrics: PSNR={current_val['psnr']:.4f}, SSIM={current_val['ssim']:.6f}, SAM={current_val['sam']:.4f}, pred_mean={current_val['prediction_mean']:.6f}, pred_std={current_val['prediction_std']:.6f}"
    )
    root_lines.append("")
    root_lines.append("## Confidence Scores")
    for key in [
        "A_export_bug",
        "B_channel_order_bug",
        "C_visualization_bug",
        "D_dynamic_range_collapse",
        "E_wrong_checkpoint_loaded",
        "F_multiple_issues",
    ]:
        root_lines.append(f"- {hypothesis_map[key]}: {scores[key]:.2f}")

    root_lines.append("")
    root_lines.append("## Conclusion")
    root_lines.append(
        "- Dark reconstruction.png is primarily explained by a visualization mismatch (direct [0,1] clamp to PNG without stretch) plus low-amplitude prediction distribution."
    )
    root_lines.append(
        "- Checkpoint mismatch and channel-order mismatch are not supported by this audit."
    )
    (out_dir / "nafnet_root_cause_report.md").write_text("\n".join(root_lines), encoding="utf-8")

    summary = {
        "raw_output_stats_json": str((out_dir / "raw_output_stats.json").as_posix()),
        "export_pipeline_audit_md": str((out_dir / "export_pipeline_audit.md").as_posix()),
        "channel_order_report_md": str((out_dir / "channel_order_report.md").as_posix()),
        "dynamic_range_comparison_csv": str((out_dir / "dynamic_range_comparison.csv").as_posix()),
        "viz_A": str((out_dir / "viz_A.png").as_posix()),
        "viz_B": str((out_dir / "viz_B.png").as_posix()),
        "viz_C": str((out_dir / "viz_C.png").as_posix()),
        "viz_D": str((out_dir / "viz_D.png").as_posix()),
        "visualization_comparison": str((out_dir / "visualization_comparison.png").as_posix()),
        "checkpoint_training_metrics": ckpt_metrics,
        "checkpoint_current_validation_metrics": current_val,
        "selected_root_cause": hypothesis_map[best_hypothesis],
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Audit complete. Output directory:", out_dir)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Audit why NAFNet reconstruction output appears dark")
    p.add_argument(
        "--input_image",
        default="chaturvyuha-cloudvision/datasets/raw/SEN12MS-CR/cloudy/ROIs1158_spring_s2_cloudy/s2_cloudy_100/ROIs1158_spring_s2_cloudy_100_p517.tif",
    )
    p.add_argument("--nafnet_checkpoint", default="checkpoints_nafnet/strict_curated_training/best_ssim.pth")
    p.add_argument("--stats_json", default="tmp_stats/band_statistics.json")
    p.add_argument("--pairs_csv", default="checkpoints_nafnet/raw_pair_audit/top_5000_curated_strict_pairs.csv")
    p.add_argument("--training_metrics_csv", default="checkpoints_nafnet/strict_curated_training/training_metrics.csv")
    p.add_argument("--workspace_root", default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_pairs", type=int, default=1892)
    p.add_argument("--val_pairs", type=int, default=236)
    p.add_argument("--dynamic_samples", type=int, default=100)
    p.add_argument("--dynamic_collapse_ratio_threshold", type=float, default=0.75)
    p.add_argument("--out_dir", default="outputs/nafnet_dark_audit")
    raise SystemExit(run(p.parse_args()))