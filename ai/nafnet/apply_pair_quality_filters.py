"""Apply pair-quality filters to the full SEN12MS-CR paired dataset.

Outputs:
- filtered_pairs.csv
- filtered_dataset_report.md
- retained_examples/ (visual panels)
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from .dataset import read_image, normalize_image
from . import metrics as naf_metrics


def _discover_pairs(cloudy_dir: Path, clear_dir: Path):
    def _norm(stem: str):
        return stem.replace("_cloudy", "")

    cfiles = {p.name: p for p in cloudy_dir.glob("**/*.tif")}
    tfiles = {p.name: p for p in clear_dir.glob("**/*.tif")}
    cmap = {_norm(Path(n).stem): p for n, p in cfiles.items()}
    tmap = {_norm(Path(n).stem): p for n, p in tfiles.items()}
    keys = sorted(set(cmap.keys()) & set(tmap.keys()))
    return [(k, str(cmap[k]), str(tmap[k])) for k in keys]


def _sobel_mag(img_hwc: np.ndarray):
    mags = []
    for c in range(img_hwc.shape[2]):
        x = img_hwc[:, :, c]
        gx = np.zeros_like(x)
        gy = np.zeros_like(x)
        gx[:, 1:-1] = (x[:, 2:] - x[:, :-2]) * 0.5
        gy[1:-1, :] = (x[2:, :] - x[:-2, :]) * 0.5
        mags.append(np.sqrt(gx * gx + gy * gy + 1e-8))
    return np.stack(mags, axis=2)


def _edge_similarity(a_hwc: np.ndarray, b_hwc: np.ndarray):
    ma = _sobel_mag(a_hwc).reshape(-1)
    mb = _sobel_mag(b_hwc).reshape(-1)
    da = np.linalg.norm(ma)
    db = np.linalg.norm(mb)
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float(np.clip(np.dot(ma, mb) / (da * db), -1.0, 1.0))


def _hist_distance(a_hwc: np.ndarray, b_hwc: np.ndarray, bins=64):
    dists = []
    for c in range(a_hwc.shape[2]):
        ha, _ = np.histogram(a_hwc[:, :, c], bins=bins, range=(0.0, 1.0), density=False)
        hb, _ = np.histogram(b_hwc[:, :, c], bins=bins, range=(0.0, 1.0), density=False)
        ha = ha.astype(np.float64)
        hb = hb.astype(np.float64)
        if ha.sum() <= 0 or hb.sum() <= 0:
            dists.append(1.0)
            continue
        ha /= ha.sum()
        hb /= hb.sum()
        bc = np.sum(np.sqrt(ha * hb))
        dists.append(float(np.sqrt(max(0.0, 1.0 - bc))))
    return float(np.mean(dists))


def _veg_similarity(a_hwc: np.ndarray, b_hwc: np.ndarray):
    na = a_hwc[:, :, 2]
    nb = b_hwc[:, :, 2]
    ma = float(np.mean(na))
    mb = float(np.mean(nb))
    sa = float(np.std(na))
    sb = float(np.std(nb))
    mean_diff = abs(ma - mb)
    std_diff = abs(sa - sb)
    sim = max(0.0, 1.0 - 0.5 * (mean_diff + std_diff))
    return sim, ma, mb, sa, sb


def _compute_metrics(c_hwc: np.ndarray, t_hwc: np.ndarray):
    psnr = float(naf_metrics.psnr(t_hwc, c_hwc))
    ssim = float(naf_metrics.ssim(t_hwc, c_hwc))
    sam = float(naf_metrics.sam(t_hwc, c_hwc))
    bdiff = float(abs(np.mean(c_hwc) - np.mean(t_hwc)))
    hdist = _hist_distance(c_hwc, t_hwc)
    esim = _edge_similarity(c_hwc, t_hwc)
    vsim, nir_mc, nir_mt, nir_sc, nir_st = _veg_similarity(c_hwc, t_hwc)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "sam": sam,
        "brightness_diff_abs": bdiff,
        "hist_distance": hdist,
        "edge_similarity": esim,
        "veg_similarity": vsim,
        "nir_mean_cloudy": nir_mc,
        "nir_mean_target": nir_mt,
        "nir_std_cloudy": nir_sc,
        "nir_std_target": nir_st,
    }


def _save_panel(path: Path, c_hwc: np.ndarray, t_hwc: np.ndarray):
    import imageio.v2 as imageio

    diff = np.mean(np.abs(c_hwc - t_hwc), axis=2, keepdims=True)
    scale = float(np.percentile(diff, 98)) + 1e-8
    diff_rgb = np.repeat(np.clip(diff / scale, 0.0, 1.0), 3, axis=2)
    panel = np.concatenate([c_hwc, t_hwc, diff_rgb], axis=1)
    imageio.imwrite(str(path), np.clip(panel * 255.0, 0, 255).astype(np.uint8))


def _mean(rows, key):
    if not rows:
        return None
    return float(np.mean([r[key] for r in rows]))


def run(args):
    cloudy_dir = Path(args.cloudy)
    clear_dir = Path(args.clear)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"p1": [0.0, 0.0, 0.0], "p99": [6000.0, 6000.0, 6000.0]}
    sp = Path("tmp_stats/band_statistics.json")
    if sp.exists():
        try:
            j = json.loads(sp.read_text())
            stats["p1"] = j["p1"]
            stats["p99"] = j["p99"]
        except Exception:
            pass

    pairs = _discover_pairs(cloudy_dir, clear_dir)
    total = len(pairs)
    if total == 0:
        raise RuntimeError("No matched pairs found")

    retained = []
    all_rows = []

    for i, (pid, cp, tp) in enumerate(pairs, start=1):
        c = read_image(cp)
        t = read_image(tp)
        c = c[:, :, :3] if c.shape[2] >= 3 else c
        t = t[:, :, :3] if t.shape[2] >= 3 else t
        c = normalize_image(c, stats["p1"], stats["p99"])
        t = normalize_image(t, stats["p1"], stats["p99"])

        m = _compute_metrics(c, t)
        row = {
            "pair_id": pid,
            "cloudy_path": cp,
            "target_path": tp,
            **m,
        }
        all_rows.append(row)

        keep = (
            m["ssim"] >= args.min_ssim
            and m["psnr"] >= args.min_psnr
            and m["brightness_diff_abs"] <= args.max_brightness_diff
            and m["hist_distance"] <= args.max_hist_distance
            and m["edge_similarity"] >= args.min_edge_similarity
            and m["veg_similarity"] >= args.min_veg_similarity
        )
        if keep:
            retained.append(row)

        if i % 250 == 0 or i == total:
            print(f"processed {i}/{total} pairs, retained={len(retained)}")

    retained_sorted = sorted(retained, key=lambda r: (r["ssim"], r["psnr"]), reverse=True)

    # Save filtered training list
    filtered_csv = out_dir / "filtered_pairs.csv"
    with open(filtered_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pair_id",
                "cloudy_path",
                "target_path",
                "psnr",
                "ssim",
                "sam",
                "brightness_diff_abs",
                "hist_distance",
                "edge_similarity",
                "veg_similarity",
            ]
        )
        for r in retained_sorted:
            w.writerow(
                [
                    r["pair_id"],
                    r["cloudy_path"],
                    r["target_path"],
                    r["psnr"],
                    r["ssim"],
                    r["sam"],
                    r["brightness_diff_abs"],
                    r["hist_distance"],
                    r["edge_similarity"],
                    r["veg_similarity"],
                ]
            )

    # Save all metrics (optional debug trace)
    with open(out_dir / "all_pairs_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Visual examples from retained subset
    vis_dir = out_dir / "retained_examples"
    vis_dir.mkdir(parents=True, exist_ok=True)
    ex_count = min(args.examples, len(retained_sorted))
    if ex_count > 0:
        rng = random.Random(args.seed)
        sample_rows = rng.sample(retained_sorted, ex_count)
        for k, r in enumerate(sample_rows, start=1):
            c = normalize_image(read_image(r["cloudy_path"])[:, :, :3], stats["p1"], stats["p99"])
            t = normalize_image(read_image(r["target_path"])[:, :, :3], stats["p1"], stats["p99"])
            _save_panel(vis_dir / f"retained_{k:02d}_{r['pair_id']}.png", c, t)

    retained_pct = (100.0 * len(retained_sorted) / total) if total > 0 else 0.0

    retained_avg = {
        "psnr": _mean(retained_sorted, "psnr"),
        "ssim": _mean(retained_sorted, "ssim"),
        "sam": _mean(retained_sorted, "sam"),
    }

    # Expected achievable metrics estimate for NAFNet on filtered subset.
    # Heuristic: training should approach a large fraction of pair-consistency metrics.
    # We report a conservative range anchored to retained-set quality.
    if len(retained_sorted) > 0:
        psnr_vals = np.array([r["psnr"] for r in retained_sorted], dtype=np.float64)
        ssim_vals = np.array([r["ssim"] for r in retained_sorted], dtype=np.float64)
        sam_vals = np.array([r["sam"] for r in retained_sorted], dtype=np.float64)

        expected = {
            "psnr_range_db": [
                float(np.percentile(psnr_vals, 25) - 1.0),
                float(np.percentile(psnr_vals, 50) + 0.5),
            ],
            "ssim_range": [
                float(np.percentile(ssim_vals, 25) - 0.03),
                float(min(0.98, np.percentile(ssim_vals, 50) + 0.02)),
            ],
            "sam_range_deg": [
                float(max(0.0, np.percentile(sam_vals, 50) - 1.0)),
                float(np.percentile(sam_vals, 75) + 0.5),
            ],
            "method": "Heuristic estimate from retained pair-quality distribution; no model training performed.",
        }
    else:
        expected = {
            "psnr_range_db": [None, None],
            "ssim_range": [None, None],
            "sam_range_deg": [None, None],
            "method": "No retained samples; estimate unavailable.",
        }

    summary = {
        "original_pairs": total,
        "retained_pairs": len(retained_sorted),
        "retention_percent": retained_pct,
        "retained_average_metrics": retained_avg,
        "filters": {
            "ssim_min": args.min_ssim,
            "psnr_min": args.min_psnr,
            "brightness_diff_abs_max": args.max_brightness_diff,
            "hist_distance_max": args.max_hist_distance,
            "edge_similarity_min": args.min_edge_similarity,
            "veg_similarity_min": args.min_veg_similarity,
        },
        "expected_nafnet_metrics_on_filtered_subset": expected,
        "filtered_pairs_csv": str(filtered_csv),
        "retained_examples_dir": str(vis_dir),
    }

    (out_dir / "filtered_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = []
    report.append("# Filtered Dataset Report")
    report.append("")
    report.append("## Filtering Criteria")
    report.append(f"- SSIM(cloudy,target) >= {args.min_ssim}")
    report.append(f"- PSNR(cloudy,target) >= {args.min_psnr} dB")
    report.append(f"- Brightness difference <= {args.max_brightness_diff}")
    report.append(f"- Histogram distance <= {args.max_hist_distance}")
    report.append(f"- Edge similarity >= {args.min_edge_similarity}")
    report.append(f"- Vegetation similarity >= {args.min_veg_similarity}")
    report.append("")
    report.append("## Dataset Counts")
    report.append(f"- Total original pairs: {total}")
    report.append(f"- Total retained pairs: {len(retained_sorted)}")
    report.append(f"- Retention percentage: {retained_pct:.2f}%")
    report.append("")
    report.append("## Retained Subset Quality")
    report.append(f"- Average PSNR: {retained_avg['psnr']}")
    report.append(f"- Average SSIM: {retained_avg['ssim']}")
    report.append(f"- Average SAM: {retained_avg['sam']}")
    report.append("")
    report.append("## Output Files")
    report.append("- Filtered training list: filtered_pairs.csv")
    report.append("- Per-pair trace: all_pairs_metrics.csv")
    report.append("- Example retained visualizations: retained_examples/")
    report.append("")
    report.append("## Expected Achievable Metrics (NAFNet on filtered subset)")
    report.append("- No training performed; estimates are distribution-based.")
    report.append(
        f"- Expected PSNR range: {expected['psnr_range_db'][0]} to {expected['psnr_range_db'][1]} dB"
    )
    report.append(
        f"- Expected SSIM range: {expected['ssim_range'][0]} to {expected['ssim_range'][1]}"
    )
    report.append(
        f"- Expected SAM range: {expected['sam_range_deg'][0]} to {expected['sam_range_deg'][1]} deg"
    )
    report.append(f"- Method: {expected['method']}")

    (out_dir / "filtered_dataset_report.md").write_text("\n".join(report), encoding="utf-8")

    print("Saved:", out_dir / "filtered_pairs.csv")
    print("Saved:", out_dir / "filtered_dataset_report.md")
    print("Saved:", out_dir / "filtered_dataset_summary.json")
    print("Saved visuals:", out_dir / "retained_examples")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cloudy", default="chaturvyuha-cloudvision/datasets/processed/cloudy")
    p.add_argument("--clear", default="chaturvyuha-cloudvision/datasets/processed/clear")
    p.add_argument("--out_dir", default="checkpoints_nafnet/filtered_dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--examples", type=int, default=20)

    # Requested thresholds
    p.add_argument("--min_ssim", type=float, default=0.65)
    p.add_argument("--min_psnr", type=float, default=24.0)
    p.add_argument("--max_brightness_diff", type=float, default=0.08)
    p.add_argument("--max_hist_distance", type=float, default=0.28)
    p.add_argument("--min_edge_similarity", type=float, default=0.70)
    p.add_argument("--min_veg_similarity", type=float, default=0.80)

    args = p.parse_args()
    raise SystemExit(run(args))
