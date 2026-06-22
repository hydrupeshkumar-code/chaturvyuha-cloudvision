"""Validate top-ranked raw pairs and curate physically meaningful one-to-one matches."""

import argparse
import csv
import re
from pathlib import Path


FILE_RE = re.compile(r"^(ROIs\d+)_([a-zA-Z]+)_s2(?:_cloudy)?_(\d+)_p(\d+)\.tif$", re.IGNORECASE)


def _parse_name(path_str: str):
    p = Path(path_str)
    m = FILE_RE.match(p.name)
    if not m:
        return None
    return {
        "roi": m.group(1),
        "season": m.group(2).lower(),
        "scene_id": m.group(3),
        "patch_id": m.group(4),
    }


def run(args):
    ranking_csv = Path(args.ranking_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ranking_csv.exists():
        raise FileNotFoundError(f"ranking csv not found: {ranking_csv}")

    rows = []
    with open(ranking_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= args.top_k:
                break
            rows.append(row)

    if not rows:
        raise RuntimeError("No rows loaded from ranking csv")

    stats = {
        "top_k": len(rows),
        "cloudy_exists": 0,
        "clear_exists": 0,
        "both_exist": 0,
        "parsed_ok": 0,
        "strict_scene_match": 0,
        "same_patch_cross_scene": 0,
        "metadata_mismatch": 0,
    }

    curated = []
    seen_cloudy = set()
    seen_clear = set()
    seen_curated_pairs = set()

    for row in rows:
        cp = Path(row["cloudy_path"])
        tp = Path(row["clear_path"])

        c_exists = cp.exists()
        t_exists = tp.exists()
        if c_exists:
            stats["cloudy_exists"] += 1
            seen_cloudy.add(str(cp))
        if t_exists:
            stats["clear_exists"] += 1
            seen_clear.add(str(tp))
        if c_exists and t_exists:
            stats["both_exist"] += 1

        cmeta = _parse_name(str(cp))
        tmeta = _parse_name(str(tp))
        if cmeta is None or tmeta is None:
            continue
        stats["parsed_ok"] += 1

        same_roi = cmeta["roi"] == tmeta["roi"]
        same_patch = cmeta["patch_id"] == tmeta["patch_id"]
        same_season = cmeta["season"] == tmeta["season"]
        same_scene = cmeta["scene_id"] == tmeta["scene_id"]

        if same_roi and same_patch and same_season and same_scene:
            stats["strict_scene_match"] += 1
            key = (str(cp), str(tp))
            if key not in seen_curated_pairs:
                seen_curated_pairs.add(key)
                curated.append(row)
        elif same_roi and same_patch:
            stats["same_patch_cross_scene"] += 1
        else:
            stats["metadata_mismatch"] += 1

    # Write curated strict pairs (train-ready, one-to-one scene matches only)
    curated_csv = out_dir / f"top_{args.top_k}_curated_strict_pairs.csv"
    with open(curated_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in curated:
            w.writerow(r)

    # Also write simplified train list
    train_list_csv = out_dir / f"top_{args.top_k}_curated_train_list.csv"
    with open(train_list_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cloudy_path", "target_path"])
        for r in curated:
            w.writerow([r["cloudy_path"], r["clear_path"]])

    strict_ratio = 100.0 * stats["strict_scene_match"] / max(1, stats["top_k"])

    report_lines = []
    report_lines.append("# Top Ranked Pair Validation Report")
    report_lines.append("")
    report_lines.append(f"Top-K analyzed: {stats['top_k']}")
    report_lines.append(f"Cloudy files existing: {stats['cloudy_exists']}")
    report_lines.append(f"Clear files existing: {stats['clear_exists']}")
    report_lines.append(f"Rows where both files exist: {stats['both_exist']}")
    report_lines.append(f"Rows with parseable naming metadata: {stats['parsed_ok']}")
    report_lines.append("")
    report_lines.append("## Pair Validity")
    report_lines.append(f"- Strict physically meaningful matches (same ROI+season+scene_id+patch): {stats['strict_scene_match']}")
    report_lines.append(f"- Same ROI+patch but cross-scene combinations: {stats['same_patch_cross_scene']}")
    report_lines.append(f"- Metadata mismatch rows: {stats['metadata_mismatch']}")
    report_lines.append(f"- Strict-match ratio in top-K: {strict_ratio:.2f}%")
    report_lines.append("")
    report_lines.append("## Curated Outputs")
    report_lines.append(f"- Curated strict-pair ranking CSV: {curated_csv}")
    report_lines.append(f"- Curated train list CSV: {train_list_csv}")

    if stats["strict_scene_match"] < stats["top_k"]:
        report_lines.append("")
        report_lines.append("## Recommendation")
        report_lines.append("- Do not train directly on top-K ranked combinations.")
        report_lines.append("- Train only on curated strict one-to-one pairs from the generated curated CSV.")

    report_path = out_dir / f"top_{args.top_k}_validation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Saved:", curated_csv)
    print("Saved:", train_list_csv)
    print("Saved:", report_path)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ranking_csv", default="checkpoints_nafnet/raw_pair_audit/raw_pair_ranking.csv")
    p.add_argument("--top_k", type=int, default=5000)
    p.add_argument("--out_dir", default="checkpoints_nafnet/raw_pair_audit")
    args = p.parse_args()
    raise SystemExit(run(args))
