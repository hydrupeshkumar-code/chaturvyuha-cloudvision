"""Audit raw SEN12MS-CR archive and rank all cloudy-clear candidate pairs."""

import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np

from .dataset import read_image, normalize_image
from . import metrics as naf_metrics


FILE_RE = re.compile(r"^(ROIs\d+)_([a-zA-Z]+)_s2(?:_cloudy)?_(\d+)_p(\d+)\.tif$", re.IGNORECASE)


def _parse_file(path: Path):
    m = FILE_RE.match(path.name)
    if not m:
        return None
    roi = m.group(1)
    season = m.group(2).lower()
    scene_id = m.group(3)
    patch_id = m.group(4)
    return {
        "roi": roi,
        "season": season,
        "scene_id": scene_id,
        "patch_id": patch_id,
    }


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
    return sim


def _downsample_hwc(a: np.ndarray, max_side: int):
    h, w = a.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return a
    step = max(1, int(np.ceil(m / max_side)))
    return a[::step, ::step, :]


def _compute_metrics(c_hwc: np.ndarray, t_hwc: np.ndarray, metric_max_side: int):
    c_eval = _downsample_hwc(c_hwc, metric_max_side)
    t_eval = _downsample_hwc(t_hwc, metric_max_side)
    return {
        "psnr": float(naf_metrics.psnr(t_eval, c_eval)),
        "ssim": float(naf_metrics.ssim(t_eval, c_eval)),
        "sam": float(naf_metrics.sam(t_eval, c_eval)),
        "brightness_diff": float(np.mean(c_eval) - np.mean(t_eval)),
        "hist_distance": _hist_distance(c_eval, t_eval),
        "edge_similarity": _edge_similarity(c_eval, t_eval),
        "veg_similarity": _veg_similarity(c_eval, t_eval),
    }


def _score_rows(rows):
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


def run(args):
    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"starting raw pair audit at root={root}", flush=True)

    seasons = ["spring", "summer", "fall", "winter"]
    available_seasons = [s for s in seasons if (root / s).exists()]

    cloudy_root = root / "cloudy"
    if not cloudy_root.exists():
        raise RuntimeError(f"cloudy folder not found: {cloudy_root}")

    cloudy_files = sorted(cloudy_root.glob("**/*.tif"))
    clear_files = []
    for s in available_seasons:
        clear_files.extend(sorted((root / s).glob("**/*.tif")))
    print(f"discovered cloudy_files={len(cloudy_files)}, clear_files={len(clear_files)}", flush=True)

    if not cloudy_files:
        raise RuntimeError("No cloudy tif files found")
    if not clear_files:
        raise RuntimeError("No clear tif files found in seasonal folders")

    # Index clear files by (roi, patch_id) to generate all candidate pairs per cloudy scene.
    clear_index = {}
    skipped_clear = 0
    for p in clear_files:
        meta = _parse_file(p)
        if meta is None:
            skipped_clear += 1
            continue
        key = (meta["roi"], meta["patch_id"])
        clear_index.setdefault(key, []).append((str(p), meta))
    print(f"indexed clear keys={len(clear_index)}, skipped_clear={skipped_clear}", flush=True)

    stats = {"p1": [0.0, 0.0, 0.0], "p99": [6000.0, 6000.0, 6000.0]}
    sp = Path("tmp_stats/band_statistics.json")
    if sp.exists():
        try:
            j = json.loads(sp.read_text())
            stats["p1"] = j["p1"]
            stats["p99"] = j["p99"]
        except Exception:
            pass

    rows = []
    skipped_cloudy = 0
    missing_clear_match = 0

    # Bounded cache to avoid memory blow-up on full raw archive.
    cache = {}
    cache_order = []
    cache_limit = 256

    def _load_norm(path_str):
        if path_str in cache:
            return cache[path_str]
        arr = read_image(path_str)
        arr = arr[:, :, :3] if arr.shape[2] >= 3 else arr
        arr = normalize_image(arr, stats["p1"], stats["p99"])
        cache[path_str] = arr
        cache_order.append(path_str)
        if len(cache_order) > cache_limit:
            old = cache_order.pop(0)
            cache.pop(old, None)
        return arr

    processed_cloudy = 0
    iter_cloudy = cloudy_files
    if args.limit_cloudy and args.limit_cloudy > 0:
        iter_cloudy = cloudy_files[: args.limit_cloudy]
        print(f"limiting cloudy scan to first {len(iter_cloudy)} files", flush=True)
    for cp in iter_cloudy:
        cmeta = _parse_file(cp)
        if cmeta is None:
            skipped_cloudy += 1
            continue
        key = (cmeta["roi"], cmeta["patch_id"])
        candidates = clear_index.get(key, [])
        if not candidates:
            missing_clear_match += 1
            continue

        cloudy_norm = _load_norm(str(cp))
        processed_cloudy += 1
        for tp, tmeta in candidates:
            clear_norm = _load_norm(tp)
            m = _compute_metrics(cloudy_norm, clear_norm, args.metric_max_side)
            rows.append(
                {
                    "cloudy_path": str(cp),
                    "clear_path": tp,
                    "roi": cmeta["roi"],
                    "cloudy_season": cmeta["season"],
                    "clear_season": tmeta["season"],
                    "cloudy_scene_id": cmeta["scene_id"],
                    "clear_scene_id": tmeta["scene_id"],
                    "patch_id": cmeta["patch_id"],
                    **m,
                }
            )

        if processed_cloudy % 200 == 0:
            print(f"processed cloudy scenes: {processed_cloudy}/{len(iter_cloudy)}; candidate pairs={len(rows)}", flush=True)
        if processed_cloudy % 2000 == 0:
            cache.clear()
            cache_order.clear()

    if not rows:
        raise RuntimeError("No candidate cloudy-clear pairs generated")

    _score_rows(rows)
    ranked = sorted(rows, key=lambda r: r["alignment_score"], reverse=True)

    # Write ranking CSV
    ranking_csv = out_dir / "raw_pair_ranking.csv"
    with open(ranking_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
        w.writeheader()
        for r in ranked:
            w.writerow(r)

    ssim_vals = np.array([r["ssim"] for r in ranked], dtype=np.float64)
    psnr_vals = np.array([r["psnr"] for r in ranked], dtype=np.float64)

    counts = {
        "ssim_gt_0_6": int(np.sum(ssim_vals > 0.6)),
        "ssim_gt_0_7": int(np.sum(ssim_vals > 0.7)),
        "ssim_gt_0_8": int(np.sum(ssim_vals > 0.8)),
        "psnr_gt_20": int(np.sum(psnr_vals > 20.0)),
        "psnr_gt_25": int(np.sum(psnr_vals > 25.0)),
        "psnr_gt_30": int(np.sum(psnr_vals > 30.0)),
    }

    top100 = min(100, len(ranked))
    top500 = min(500, len(ranked))
    top1000 = min(1000, len(ranked))

    # Save top lists as helper files for easy reuse
    for n in [100, 500, 1000]:
        nn = min(n, len(ranked))
        with open(out_dir / f"top_{n}_pairs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
            w.writeheader()
            for r in ranked[:nn]:
                w.writerow(r)

    summary = {
        "dataset_root": str(root),
        "available_seasons": available_seasons,
        "cloudy_files": len(cloudy_files),
        "clear_files": len(clear_files),
        "skipped_cloudy_unparsed": skipped_cloudy,
        "skipped_clear_unparsed": skipped_clear,
        "missing_clear_match_for_cloudy": missing_clear_match,
        "total_candidate_pairs": len(ranked),
        "metric_max_side": args.metric_max_side,
        "threshold_counts": counts,
        "top_counts": {"top100": top100, "top500": top500, "top1000": top1000},
        "raw_pair_ranking_csv": str(ranking_csv),
    }
    (out_dir / "raw_dataset_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Raw Dataset Quality Report")
    lines.append("")
    lines.append(f"Dataset root: {root}")
    lines.append(f"Available clear seasons found: {', '.join(available_seasons) if available_seasons else 'none'}")
    lines.append(f"Cloudy folder scanned: {cloudy_root}")
    lines.append("")
    lines.append("## Candidate Generation")
    lines.append(f"- Total cloudy scenes scanned: {len(cloudy_files)}")
    lines.append(f"- Total clear scenes scanned: {len(clear_files)}")
    lines.append(f"- Total number of possible pairs: {len(ranked)}")
    lines.append(f"- Metric evaluation max image side: {args.metric_max_side} pixels")
    lines.append(f"- Cloudy files skipped (name parse): {skipped_cloudy}")
    lines.append(f"- Clear files skipped (name parse): {skipped_clear}")
    lines.append(f"- Cloudy files with no corresponding clear candidate: {missing_clear_match}")
    lines.append("")
    lines.append("## Threshold Counts")
    lines.append(f"- Pairs with SSIM > 0.6: {counts['ssim_gt_0_6']}")
    lines.append(f"- Pairs with SSIM > 0.7: {counts['ssim_gt_0_7']}")
    lines.append(f"- Pairs with SSIM > 0.8: {counts['ssim_gt_0_8']}")
    lines.append(f"- Pairs with PSNR > 20: {counts['psnr_gt_20']}")
    lines.append(f"- Pairs with PSNR > 25: {counts['psnr_gt_25']}")
    lines.append(f"- Pairs with PSNR > 30: {counts['psnr_gt_30']}")
    lines.append("")
    lines.append("## Top Ranked Sets")
    lines.append(f"- Top 100 pairs exported: top_100_pairs.csv ({top100} rows)")
    lines.append(f"- Top 500 pairs exported: top_500_pairs.csv ({top500} rows)")
    lines.append(f"- Top 1000 pairs exported: top_1000_pairs.csv ({top1000} rows)")
    lines.append("")
    lines.append("## Main Output")
    lines.append("- raw_pair_ranking.csv (all ranked candidate pairs)")

    (out_dir / "raw_dataset_quality_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Saved:", out_dir / "raw_pair_ranking.csv")
    print("Saved:", out_dir / "raw_dataset_quality_report.md")
    print("Saved:", out_dir / "raw_dataset_quality_summary.json")
    print("Saved:", out_dir / "top_100_pairs.csv")
    print("Saved:", out_dir / "top_500_pairs.csv")
    print("Saved:", out_dir / "top_1000_pairs.csv")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default="chaturvyuha-cloudvision/datasets/raw/SEN12MS-CR")
    p.add_argument("--out_dir", default="checkpoints_nafnet/raw_pair_audit")
    p.add_argument("--limit_cloudy", type=int, default=0)
    p.add_argument("--metric_max_side", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    raise SystemExit(run(args))
