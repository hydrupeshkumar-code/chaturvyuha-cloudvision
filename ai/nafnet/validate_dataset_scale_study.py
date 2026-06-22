#!/usr/bin/env python
"""Validation helpers for dataset-scale study pre-launch checks."""

import argparse
import collections
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim

from ai.nafnet.dataset import NAFDataset
from ai.nafnet import metrics as naf_metrics
from ai.nafnet.select_top_strict_pairs import run as select_pairs_run


BASELINE_CHECKPOINT = Path("checkpoints_nafnet/strict_curated_training/best_ssim.pth")
RAW_RANKING_CSV = Path("checkpoints_nafnet/raw_pair_audit/raw_pair_ranking.csv")
STATS_JSON = Path("tmp_stats/band_statistics.json")
EXPECTED_BASELINE = {
    "psnr": (35.03, 0.2),
    "ssim": (0.9015, 0.005),
    "sam": (4.86, 0.3),
}


def _pair_key(cloudy, clear):
    return (Path(cloudy).as_posix(), Path(clear).as_posix())


def _load_pairs(csv_path: Path):
    pairs = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cloudy = row.get("cloudy_path")
            target = row.get("target_path") or row.get("clear_path")
            if cloudy and target:
                pairs.append((cloudy, target))
    return pairs


def _load_raw_rankings(ranking_csv: Path):
    rows = []
    with open(ranking_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _build_raw_index(raw_rows):
    index = {}
    for row in raw_rows:
        cloudy = row.get("cloudy_path", "")
        clear = row.get("clear_path", "")
        index[_pair_key(cloudy, clear)] = row
    return index


def _is_strict_match(raw_row):
    if raw_row is None:
        return False
    return (
        bool(raw_row.get("roi", ""))
        and raw_row.get("cloudy_season", "") == raw_row.get("clear_season", "")
        and raw_row.get("cloudy_scene_id", "") == raw_row.get("clear_scene_id", "")
        and bool(raw_row.get("patch_id", ""))
    )


def _valid_file_paths(pair):
    cloudy, target = pair
    return Path(cloudy).exists() and Path(target).exists()


def _summary_stats(values):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _save_markdown(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_quality_histograms(dataset_stats, out_path: Path):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    metrics = ["psnr", "ssim", "sam"]
    titles = ["PSNR Distribution", "SSIM Distribution", "SAM Distribution"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, metric in enumerate(metrics):
        ax = axs[i]
        for idx, stats in enumerate(dataset_stats):
            values = stats["values"][metric]
            if len(values) == 0:
                continue
            ax.hist(values, bins=40, alpha=0.45, label=f"{stats['dataset_size']}", color=colors[idx % len(colors)])
        ax.set_title(titles[i])
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _evaluate_pair_quality(pairs, stats):
    ds = NAFDataset(pairs, stats["p1"], stats["p99"], patch_size=None, augment=False)
    values = {"psnr": [], "ssim": [], "sam": []}
    for x, y in ds:
        cloudy = np.transpose(x.numpy(), (1, 2, 0))
        target = np.transpose(y.numpy(), (1, 2, 0))
        values["psnr"].append(naf_metrics.psnr(target, cloudy))
        values["ssim"].append(naf_metrics.ssim(target, cloudy))
        values["sam"].append(naf_metrics.sam(target, cloudy))
    return values


def _evaluate_checkpoint(model, data_loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            pred = model(x)[0].cpu().numpy()
            pred = np.clip(np.transpose(pred, (1, 2, 0)), 0.0, 1.0)
            tgt = np.transpose(y[0].numpy(), (1, 2, 0))
            rows.append({
                "psnr": naf_metrics.psnr(tgt, pred),
                "ssim": naf_metrics.ssim(tgt, pred),
                "sam": naf_metrics.sam(tgt, pred),
            })
    if not rows:
        return {"psnr": 0.0, "ssim": 0.0, "sam": 0.0}
    return {
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "sam": float(np.mean([r["sam"] for r in rows])),
    }


def _assert_baseline_reproduction(metrics, tolerance):
    failing = []
    for key, (target, tol) in tolerance.items():
        actual = metrics.get(key)
        if actual is None:
            failing.append(key)
            continue
        if abs(actual - target) > tol:
            failing.append(f"{key} {actual:.4f} outside {target:.4f} ± {tol:.4f}")
    return failing


def _validate_pair_integrity(selected_pairs, raw_rows, exp_dir: Path):
    raw_index = _build_raw_index(raw_rows)
    strict_count = 0
    invalid_file_count = 0
    duplicate_count = 0
    seen_cloudy = set()
    seen_clear = set()
    mismatch_samples = []

    for cloudy, target in selected_pairs:
        key = _pair_key(cloudy, target)
        raw_row = raw_index.get(key)
        if not _valid_file_paths((cloudy, target)):
            invalid_file_count += 1
        if cloudy in seen_cloudy or target in seen_clear:
            duplicate_count += 1
        seen_cloudy.add(cloudy)
        seen_clear.add(target)
        if raw_row is not None and _is_strict_match(raw_row):
            strict_count += 1
        else:
            mismatch_samples.append({
                "cloudy": cloudy,
                "target": target,
                "has_raw_row": raw_row is not None,
                "same_roi": bool(raw_row and raw_row.get("roi", "")),
                "same_season": bool(raw_row and raw_row.get("cloudy_season") == raw_row.get("clear_season")),
                "same_scene_id": bool(raw_row and raw_row.get("cloudy_scene_id") == raw_row.get("clear_scene_id")),
                "same_patch_id": bool(raw_row and raw_row.get("patch_id", "")),
            })

    lines = ["# Pair Integrity Report", "", f"Selected strict pair count: {len(selected_pairs)}", f"Strict metadata matches: {strict_count}", f"Duplicate cloudy/clear pairs: {duplicate_count}", f"Invalid file paths: {invalid_file_count}", ""]
    if mismatch_samples:
        lines.append("## Mismatched or incomplete strict metadata examples")
        lines.append("")
        for sample in mismatch_samples[:10]:
            lines.append(f"- cloudy: {sample['cloudy']}")
            lines.append(f"  target: {sample['target']}")
            lines.append(f"  raw row found: {sample['has_raw_row']}")
            lines.append(f"  same ROI: {sample['same_roi']}")
            lines.append(f"  same season: {sample['same_season']}")
            lines.append(f"  same scene_id: {sample['same_scene_id']}")
            lines.append(f"  same patch_id: {sample['same_patch_id']}")
            lines.append("")
    report_path = exp_dir / "pair_integrity_report.md"
    _save_markdown(report_path, lines)
    if strict_count != len(selected_pairs) or duplicate_count != 0 or invalid_file_count != 0:
        raise RuntimeError(f"Pair integrity failed for {exp_dir.name}: strict_count={strict_count}, duplicate_count={duplicate_count}, invalid_file_count={invalid_file_count}")
    return {
        "strict_pair_count": strict_count,
        "duplicate_count": duplicate_count,
        "invalid_file_count": invalid_file_count,
        "report_path": report_path,
    }


def _evaluate_baseline_reproduction(model, val_pairs, stats, device, exp_dir: Path):
    val_ds = NAFDataset(val_pairs, stats["p1"], stats["p99"], patch_size=None, augment=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    metrics = _evaluate_checkpoint(model, val_loader, device)
    lines = ["# Baseline Reproduction Report", "", f"Validation pair count: {len(val_pairs)}", "", "## Metrics", "", f"PSNR: {metrics['psnr']:.4f}", f"SSIM: {metrics['ssim']:.4f}", f"SAM: {metrics['sam']:.4f}", "", "## Expected", "", f"PSNR ≈ {EXPECTED_BASELINE['psnr'][0]:.4f} ± {EXPECTED_BASELINE['psnr'][1]:.4f}", f"SSIM ≈ {EXPECTED_BASELINE['ssim'][0]:.4f} ± {EXPECTED_BASELINE['ssim'][1]:.4f}", f"SAM ≈ {EXPECTED_BASELINE['sam'][0]:.4f} ± {EXPECTED_BASELINE['sam'][1]:.4f}", ""]
    failing = _assert_baseline_reproduction(metrics, EXPECTED_BASELINE)
    if failing:
        lines.append("## Validation Result")
        lines.append("")
        lines.append("Baseline reproduction failed with the following deviations:")
        lines.extend([f"- {msg}" for msg in failing])
        lines.append("")
    else:
        lines.append("## Validation Result")
        lines.append("")
        lines.append("Baseline reproduction succeeded within expected tolerances.")
        lines.append("")
    report_path = exp_dir / "baseline_reproduction_report.md"
    _save_markdown(report_path, lines)
    if failing:
        raise RuntimeError(f"Baseline reproduction failed for {exp_dir.name}: {failing}")
    return metrics


def run_validation_checks(output_dir: Path, experiment_sizes, stats, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _load_raw_rankings(RAW_RANKING_CSV)
    if not raw_rows:
        raise RuntimeError(f"Raw ranking CSV is empty: {RAW_RANKING_CSV}")
    summaries = []
    hist_data = []

    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {BASELINE_CHECKPOINT}")

    for size in experiment_sizes:
        exp_dir = output_dir / f"experiment_{size}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        selection_dir = exp_dir / "selection"
        selection_dir.mkdir(parents=True, exist_ok=True)
        selection_args = argparse.Namespace(
            ranking_csv=str(RAW_RANKING_CSV),
            out_dir=str(selection_dir),
            top_n=size,
            min_ssim=0.0,
            min_psnr=0.0,
            max_sam=100.0,
            train_ratio=0.9,
            val_ratio=0.1,
            test_ratio=0.0,
            seed=42,
        )
        print(f"Selecting top {size} strict pairs for validation...")
        select_pairs_run(selection_args)

        selected_pairs = _load_pairs(selection_dir / f"top_{size}_strict_pairs.csv")
        train_pairs = _load_pairs(selection_dir / f"top_{size}_strict_train_split.csv")
        val_pairs = _load_pairs(selection_dir / f"top_{size}_strict_val_split.csv")

        integrity = _validate_pair_integrity(selected_pairs, raw_rows, exp_dir)
        quality_values = _evaluate_pair_quality(selected_pairs, stats)
        quality_stats = {
            metric: _summary_stats(quality_values[metric])
            for metric in ["psnr", "ssim", "sam"]
        }
        report_lines = ["# Dataset Quality Report", "", f"Experiment {size}", "", "## Selected Dataset Metrics", ""]
        for metric, stats_map in quality_stats.items():
            report_lines.append(f"### {metric.upper()}")
            report_lines.append(f"- mean: {stats_map['mean']:.4f}")
            report_lines.append(f"- median: {stats_map['median']:.4f}")
            report_lines.append(f"- p10: {stats_map['p10']:.4f}")
            report_lines.append(f"- p90: {stats_map['p90']:.4f}")
            report_lines.append("")
        _save_markdown(exp_dir / "dataset_quality_report.md", report_lines)

        model = torch.load(BASELINE_CHECKPOINT, map_location=device)
        from ai.nafnet.model import NAFNetWrapper
        if isinstance(model, dict) and "state_dict" in model:
            model = model["state_dict"]
        naf_model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
        naf_model.load_state_dict(model, strict=False)
        baseline_metrics = _evaluate_baseline_reproduction(naf_model, val_pairs, stats, device, exp_dir)

        hist_data.append({
            "dataset_size": size,
            "values": quality_values,
        })
        summaries.append({
            "dataset_size": size,
            "quality_stats": quality_stats,
            "baseline_metrics": baseline_metrics,
            "integrity": integrity,
            "train_count": len(train_pairs),
            "val_count": len(val_pairs),
        })

    hist_path = output_dir / "quality_histograms.png"
    _plot_quality_histograms(hist_data, hist_path)

    root_lines = ["# Dataset Scale Study Validation Summary", "", f"Output directory: {output_dir}", "", "## Experiment Summaries", ""]
    for s in summaries:
        root_lines.append(f"### {s['dataset_size']} pairs")
        root_lines.append(f"- train count: {s['train_count']}")
        root_lines.append(f"- val count: {s['val_count']}")
        root_lines.append(f"- strict pairs: {s['integrity']['strict_pair_count']}")
        root_lines.append(f"- duplicate count: {s['integrity']['duplicate_count']}")
        root_lines.append(f"- invalid file count: {s['integrity']['invalid_file_count']}")
        root_lines.append(f"- baseline PSNR: {s['baseline_metrics']['psnr']:.4f}")
        root_lines.append(f"- baseline SSIM: {s['baseline_metrics']['ssim']:.4f}")
        root_lines.append(f"- baseline SAM: {s['baseline_metrics']['sam']:.4f}")
        root_lines.append("")
    _save_markdown(output_dir / "dataset_scale_study_report.md", root_lines)
    _save_markdown(output_dir / "pair_integrity_report.md", ["# Pair Integrity Summary", "", "See per-experiment reports in each experiment folder."])
    _save_markdown(output_dir / "baseline_reproduction_report.md", ["# Baseline Reproduction Summary", "", "See per-experiment reports in each experiment folder."])

    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pre-launch validation checks for dataset scale study.")
    parser.add_argument("--output_dir", type=str, default="checkpoints_nafnet/dataset_scale_study", help="Target root for validation outputs")
    parser.add_argument("--experiment_sizes", type=int, nargs="+", default=[5000, 10000, 15000], help="Experiment dataset sizes to validate")
    args = parser.parse_args()

    stats = {"p1": [0.0, 0.0, 0.0], "p99": [6000.0, 6000.0, 6000.0]}
    stats_path = STATS_JSON
    if stats_path.exists():
        try:
            data = json.loads(stats_path.read_text())
            stats["p1"] = data.get("p1", stats["p1"])
            stats["p99"] = data.get("p99", stats["p99"])
        except Exception:
            pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not RAW_RANKING_CSV.exists():
        raise FileNotFoundError(f"Ranking CSV not found: {RAW_RANKING_CSV}")
    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {BASELINE_CHECKPOINT}")

    summaries = run_validation_checks(Path(args.output_dir), args.experiment_sizes, stats, device)
    print(f"Validation completed for {len(summaries)} experiment sizes. Outputs written to {args.output_dir}")
