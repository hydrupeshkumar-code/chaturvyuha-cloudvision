"""Paired-dataset quality audit for SEN12MS-CR cloudy/clear pairs.

Computes alignment metrics on random paired samples and generates:
- pair_quality_report.md
- best/worst ranking CSVs (top/bottom 50)
- visual examples for highly aligned and poorly aligned pairs
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
    # gradient magnitude per channel; numpy-only to avoid extra dependencies
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
    # Mean Bhattacharyya distance across channels: 0=identical hist, 1=very different
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
        d = float(np.sqrt(max(0.0, 1.0 - bc)))
        dists.append(d)
    return float(np.mean(dists))


def _nir_similarity(a_hwc: np.ndarray, b_hwc: np.ndarray):
    # LISS-IV mapping in docs: channel 2 (0-indexed) is NIR
    na = a_hwc[:, :, 2]
    nb = b_hwc[:, :, 2]
    ma = float(np.mean(na))
    mb = float(np.mean(nb))
    sa = float(np.std(na))
    sb = float(np.std(nb))
    mean_diff = abs(ma - mb)
    std_diff = abs(sa - sb)
    # Convert to similarity in [0,1] with simple bounded transform.
    sim = max(0.0, 1.0 - 0.5 * (mean_diff + std_diff))
    return sim, ma, mb, sa, sb


def _compute_metrics(c_hwc: np.ndarray, t_hwc: np.ndarray):
    psnr = float(naf_metrics.psnr(t_hwc, c_hwc))
    ssim = float(naf_metrics.ssim(t_hwc, c_hwc))
    sam = float(naf_metrics.sam(t_hwc, c_hwc))
    bdiff = float(np.mean(c_hwc) - np.mean(t_hwc))
    hdist = _hist_distance(c_hwc, t_hwc)
    esim = _edge_similarity(c_hwc, t_hwc)
    vsim, nir_mc, nir_mt, nir_sc, nir_st = _nir_similarity(c_hwc, t_hwc)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "sam": sam,
        "brightness_diff": bdiff,
        "hist_distance": hdist,
        "edge_similarity": esim,
        "veg_similarity": vsim,
        "nir_mean_cloudy": nir_mc,
        "nir_mean_target": nir_mt,
        "nir_std_cloudy": nir_sc,
        "nir_std_target": nir_st,
    }


def _save_panel(out_path: Path, cloudy_hwc: np.ndarray, target_hwc: np.ndarray):
    import imageio.v2 as imageio

    diff = np.mean(np.abs(cloudy_hwc - target_hwc), axis=2, keepdims=True)
    diff_rgb = np.repeat(np.clip(diff / (np.percentile(diff, 98) + 1e-8), 0.0, 1.0), 3, axis=2)
    panel = np.concatenate([cloudy_hwc, target_hwc, diff_rgb], axis=1)
    imageio.imwrite(str(out_path), np.clip(panel * 255.0, 0, 255).astype(np.uint8))


def _rank(rows):
    # Higher is better alignment.
    psnr = np.array([r["psnr"] for r in rows], dtype=np.float64)
    ssim = np.array([r["ssim"] for r in rows], dtype=np.float64)
    sam = np.array([r["sam"] for r in rows], dtype=np.float64)
    bdiff = np.array([abs(r["brightness_diff"]) for r in rows], dtype=np.float64)
    hdist = np.array([r["hist_distance"] for r in rows], dtype=np.float64)
    esim = np.array([r["edge_similarity"] for r in rows], dtype=np.float64)
    vsim = np.array([r["veg_similarity"] for r in rows], dtype=np.float64)

    def _norm(x):
        mn = float(np.min(x))
        mx = float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    score = (
        0.22 * _norm(psnr)
        + 0.22 * _norm(ssim)
        + 0.16 * (1.0 - _norm(sam))
        + 0.12 * (1.0 - _norm(bdiff))
        + 0.10 * (1.0 - _norm(hdist))
        + 0.10 * _norm(esim)
        + 0.08 * _norm(vsim)
    )
    for i, r in enumerate(rows):
        r["alignment_score"] = float(score[i])
    return rows


def _summary_stats(rows, key):
    v = np.array([r[key] for r in rows], dtype=np.float64)
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p10": float(np.percentile(v, 10)),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
    }


def _write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


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

    all_pairs = _discover_pairs(cloudy_dir, clear_dir)
    if len(all_pairs) == 0:
        raise RuntimeError("No matched cloudy-clear pairs found")

    rng = random.Random(args.seed)
    sample_n = min(args.sample_size, len(all_pairs))
    sampled = rng.sample(all_pairs, sample_n)

    rows = []
    for i, (pid, cp, tp) in enumerate(sampled, start=1):
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
        rows.append(row)
        if i % 50 == 0:
            print(f"processed {i}/{sample_n} pairs")

    rows = _rank(rows)
    rows_sorted = sorted(rows, key=lambda r: r["alignment_score"], reverse=True)
    best50 = rows_sorted[:50]
    worst50 = rows_sorted[-50:]

    _write_csv(out_dir / "pair_metrics_all.csv", rows_sorted)
    _write_csv(out_dir / "best_50_aligned_pairs.csv", best50)
    _write_csv(out_dir / "worst_50_aligned_pairs.csv", worst50)

    # Visual examples
    best_vis_dir = out_dir / "visual_examples" / "highly_aligned"
    worst_vis_dir = out_dir / "visual_examples" / "poorly_aligned"
    best_vis_dir.mkdir(parents=True, exist_ok=True)
    worst_vis_dir.mkdir(parents=True, exist_ok=True)

    for rank, r in enumerate(best50[:20], start=1):
        c = normalize_image(read_image(r["cloudy_path"])[:, :, :3], stats["p1"], stats["p99"])
        t = normalize_image(read_image(r["target_path"])[:, :, :3], stats["p1"], stats["p99"])
        _save_panel(best_vis_dir / f"best_{rank:02d}_{r['pair_id']}.png", c, t)

    for rank, r in enumerate(worst50[:20], start=1):
        c = normalize_image(read_image(r["cloudy_path"])[:, :, :3], stats["p1"], stats["p99"])
        t = normalize_image(read_image(r["target_path"])[:, :, :3], stats["p1"], stats["p99"])
        _save_panel(worst_vis_dir / f"worst_{rank:02d}_{r['pair_id']}.png", c, t)

    # Theoretical upper-bound estimates using pair consistency distribution
    psnr_vals = np.array([r["psnr"] for r in rows_sorted], dtype=np.float64)
    ssim_vals = np.array([r["ssim"] for r in rows_sorted], dtype=np.float64)
    top10_n = max(1, int(0.10 * len(rows_sorted)))
    psnr_top10_mean = float(np.mean(np.sort(psnr_vals)[-top10_n:]))
    ssim_top10_mean = float(np.mean(np.sort(ssim_vals)[-top10_n:]))
    psnr_p90 = float(np.percentile(psnr_vals, 90))
    ssim_p90 = float(np.percentile(ssim_vals, 90))

    # Consistency flags (non-overlapping bins from multiple indicators)
    severe = []
    moderate = []
    for r in rows_sorted:
        severe_flags = 0
        severe_flags += int(r["ssim"] < 0.45)
        severe_flags += int(r["psnr"] < 16.0)
        severe_flags += int(abs(r["brightness_diff"]) > 0.15)
        severe_flags += int(r["hist_distance"] > 0.85)
        severe_flags += int(r["edge_similarity"] < 0.45)
        severe_flags += int(r["veg_similarity"] < 0.70)

        if severe_flags >= 3:
            severe.append(r)
            continue

        moderate_flags = 0
        moderate_flags += int(r["ssim"] < 0.60)
        moderate_flags += int(r["psnr"] < 22.0)
        moderate_flags += int(abs(r["brightness_diff"]) > 0.10)
        moderate_flags += int(r["hist_distance"] > 0.70)
        moderate_flags += int(r["edge_similarity"] < 0.55)
        moderate_flags += int(r["veg_similarity"] < 0.80)
        if moderate_flags >= 2:
            moderate.append(r)
    severe_frac = len(severe) / len(rows_sorted)
    moderate_frac = len(moderate) / len(rows_sorted)

    temporal_consistent = severe_frac < 0.20
    seasonal_large = (
        float(np.mean([abs(r["brightness_diff"]) for r in rows_sorted])) > 0.10
        or float(np.mean([r["hist_distance"] for r in rows_sorted])) > 0.70
        or float(np.mean([r["veg_similarity"] for r in rows_sorted])) < 0.85
    )
    many_fundamental_diff = severe_frac > 0.30

    suitable_direct_i2i = (severe_frac < 0.25) and (np.mean(ssim_vals) > 0.60)

    # filtering recommendations
    filter_rules = [
        "SSIM(cloudy,target) >= 0.65",
        "PSNR(cloudy,target) >= 24 dB",
        "|mean_brightness_diff| <= 0.08",
        "hist_distance <= 0.28",
        "edge_similarity >= 0.70",
        "vegetation_similarity(NIR) >= 0.80",
    ]

    summary = {
        "sampled_pairs": len(rows_sorted),
        "metrics": {
            "psnr": _summary_stats(rows_sorted, "psnr"),
            "ssim": _summary_stats(rows_sorted, "ssim"),
            "sam": _summary_stats(rows_sorted, "sam"),
            "brightness_diff": _summary_stats(rows_sorted, "brightness_diff"),
            "hist_distance": _summary_stats(rows_sorted, "hist_distance"),
            "edge_similarity": _summary_stats(rows_sorted, "edge_similarity"),
            "veg_similarity": _summary_stats(rows_sorted, "veg_similarity"),
        },
        "upper_bound_estimate": {
            "psnr_p90": psnr_p90,
            "ssim_p90": ssim_p90,
            "psnr_top10_mean": psnr_top10_mean,
            "ssim_top10_mean": ssim_top10_mean,
            "note": "Estimated from cloudy-clear alignment quality; acts as practical ceiling for direct one-to-one cloud reconstruction under temporal inconsistency.",
        },
        "diagnosis": {
            "temporally_consistent": temporal_consistent,
            "large_seasonal_differences_present": seasonal_large,
            "many_targets_fundamentally_different": many_fundamental_diff,
            "suitable_for_direct_image_to_image": suitable_direct_i2i,
            "severe_inconsistency_fraction": severe_frac,
            "moderate_inconsistency_fraction": moderate_frac,
        },
        "recommended_filtering_criteria": filter_rules,
    }

    (out_dir / "pair_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Pair Quality Report (SEN12MS-CR)")
    lines.append("")
    lines.append(f"Sampled random pairs: {len(rows_sorted)}")
    lines.append(f"Cloudy dir: {cloudy_dir}")
    lines.append(f"Clear dir: {clear_dir}")
    lines.append("")
    lines.append("## Aggregate Metrics")
    for k in ["psnr", "ssim", "sam", "brightness_diff", "hist_distance", "edge_similarity", "veg_similarity"]:
        s = summary["metrics"][k]
        lines.append(
            f"- {k}: mean={s['mean']:.6f}, std={s['std']:.6f}, p10/p50/p90={s['p10']:.6f}/{s['p50']:.6f}/{s['p90']:.6f}"
        )

    lines.append("")
    lines.append("## Rankings")
    lines.append("- Best 50 aligned pairs: `best_50_aligned_pairs.csv`")
    lines.append("- Worst 50 aligned pairs: `worst_50_aligned_pairs.csv`")

    lines.append("")
    lines.append("## Visual Examples")
    lines.append("- Highly aligned examples: `visual_examples/highly_aligned/`")
    lines.append("- Poorly aligned examples: `visual_examples/poorly_aligned/`")
    lines.append("- Panel format: cloudy | clear target | abs difference heatmap")

    lines.append("")
    lines.append("## Questions A-D")
    lines.append(f"- A) Temporally consistent? {summary['diagnosis']['temporally_consistent']}")
    lines.append(f"- B) Large seasonal differences present? {summary['diagnosis']['large_seasonal_differences_present']}")
    lines.append(f"- C) Many targets fundamentally different from inputs? {summary['diagnosis']['many_targets_fundamentally_different']}")
    lines.append(f"- D) Suitable for direct image-to-image cloud reconstruction? {summary['diagnosis']['suitable_for_direct_image_to_image']}")
    lines.append(
        f"- Severe inconsistency fraction: {100.0 * summary['diagnosis']['severe_inconsistency_fraction']:.2f}%"
    )
    lines.append(
        f"- Moderate inconsistency fraction: {100.0 * summary['diagnosis']['moderate_inconsistency_fraction']:.2f}%"
    )

    lines.append("")
    lines.append("## Theoretical Upper Bound Estimate")
    ub = summary["upper_bound_estimate"]
    lines.append(f"- PSNR practical upper bound (top-10% mean): {ub['psnr_top10_mean']:.4f} dB")
    lines.append(f"- SSIM practical upper bound (top-10% mean): {ub['ssim_top10_mean']:.4f}")
    lines.append(f"- PSNR optimistic percentile bound (p90): {ub['psnr_p90']:.4f} dB")
    lines.append(f"- SSIM optimistic percentile bound (p90): {ub['ssim_p90']:.4f}")
    lines.append(f"- Note: {ub['note']}")

    lines.append("")
    lines.append("## Recommended Filtering Criteria")
    for rule in filter_rules:
        lines.append(f"- {rule}")

    (out_dir / "pair_quality_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Saved:", out_dir / "pair_quality_report.md")
    print("Saved:", out_dir / "best_50_aligned_pairs.csv")
    print("Saved:", out_dir / "worst_50_aligned_pairs.csv")
    print("Saved:", out_dir / "pair_metrics_all.csv")
    print("Saved visuals:", out_dir / "visual_examples")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cloudy", default="chaturvyuha-cloudvision/datasets/processed/cloudy")
    p.add_argument("--clear", default="chaturvyuha-cloudvision/datasets/processed/clear")
    p.add_argument("--sample_size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="checkpoints_nafnet/pair_quality_audit")
    args = p.parse_args()
    raise SystemExit(run(args))
