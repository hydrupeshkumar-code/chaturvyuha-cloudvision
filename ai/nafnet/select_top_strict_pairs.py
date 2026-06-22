"""Select top strict one-to-one pairs from raw pair ranking with quality thresholds."""

import argparse
import csv
import random
from pathlib import Path


def _as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _is_strict_row(row):
    return (
        bool(row.get("roi", ""))
        and row.get("cloudy_season", "") == row.get("clear_season", "")
        and row.get("cloudy_scene_id", "") == row.get("clear_scene_id", "")
        and bool(row.get("patch_id", ""))
    )


def _normalize_path(path_str):
    if not path_str:
        return path_str

    repo_root = Path(__file__).resolve().parents[2]
    candidate = Path(path_str)
    if candidate.exists():
        return str(candidate)

    cwd = Path.cwd()
    candidate = cwd / path_str
    if candidate.exists():
        return str(candidate)

    candidate = repo_root / path_str
    if candidate.exists():
        return str(candidate)

    parts = Path(path_str).parts
    if parts and parts[0] == cwd.name:
        candidate = cwd.joinpath(*parts[1:])
        if candidate.exists():
            return str(candidate)

    if parts and parts[0] == repo_root.name:
        candidate = repo_root.joinpath(*parts[1:])
        if candidate.exists():
            return str(candidate)

    if "datasets" in parts:
        suffix = Path(*parts[parts.index("datasets") :])
        candidate = cwd / suffix
        if candidate.exists():
            return str(candidate)
        candidate = repo_root / suffix
        if candidate.exists():
            return str(candidate)

    if path_str.startswith("chaturvyuha-cloudvision/"):
        tail = path_str.split("/", 1)[1]
        candidate = repo_root / tail
        if candidate.exists():
            return str(candidate)

    return str(path_str)


def _is_valid_row(row):
    cloudy = Path(_normalize_path(row.get("cloudy_path", "")))
    clear = Path(_normalize_path(row.get("clear_path", "")))
    return cloudy.exists() and clear.exists()


def _dedupe_rows(rows):
    seen_cloudy = set()
    seen_clear = set()
    deduped = []
    for row in rows:
        cloudy = row.get("cloudy_path", "")
        clear = row.get("clear_path", "")
        if cloudy in seen_cloudy or clear in seen_clear:
            continue
        seen_cloudy.add(cloudy)
        seen_clear.add(clear)
        deduped.append(row)
    return deduped


def _compute_split(rows, train_ratio, val_ratio, test_ratio, seed):
    total = len(rows)
    if total == 0:
        return [], [], []

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    rng = random.Random(seed)
    work = list(rows)
    rng.shuffle(work)

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    train = work[:train_count]
    val = work[train_count : train_count + val_count]
    test = work[train_count + val_count :]
    return train, val, test


def _write_pair_list(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cloudy_path", "target_path"])
        for row in rows:
            target = row.get("target_path", "") or row.get("clear_path", "")
            w.writerow([row.get("cloudy_path", ""), target])


def _load_curated_fallback(size):
    repo_root = Path(__file__).resolve().parents[2]
    exact = repo_root / "checkpoints_nafnet" / "raw_pair_audit" / f"top_{size}_curated_strict_pairs.csv"
    if exact.exists():
        with open(exact, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    source = repo_root / "checkpoints_nafnet" / "raw_pair_audit" / "top_5000_curated_strict_pairs.csv"
    if not source.exists():
        return []

    with open(source, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:size]


def run(args):
    ranking_csv = Path(args.ranking_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]

    if not ranking_csv.exists():
        raise FileNotFoundError(f"Missing ranking CSV: {ranking_csv}")

    rows = []
    with open(ranking_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        raise RuntimeError("Ranking CSV is empty")

    # Strict one-to-one means same ROI+season+scene_id+patch_id between cloudy and clear.
    strict_metadata = [row for row in rows if _is_strict_row(row)]
    strict = [row for row in strict_metadata if _is_valid_row(row)]
    strict = _dedupe_rows(strict)

    filtered = []
    for row in strict:
        ssim = _as_float(row, "ssim")
        psnr = _as_float(row, "psnr")
        sam = _as_float(row, "sam")
        if ssim >= args.min_ssim and psnr >= args.min_psnr and sam <= args.max_sam:
            filtered.append(row)

    def sort_key(row):
        return (
            _as_float(row, "alignment_score"),
            _as_float(row, "ssim"),
            _as_float(row, "psnr"),
            -_as_float(row, "sam"),
        )

    filtered_sorted = sorted(filtered, key=sort_key, reverse=True)
    selected = filtered_sorted[: args.top_n]

    if not selected and strict_metadata:
        selected = _load_curated_fallback(args.top_n)

    out_csv = out_dir / f"top_{args.top_n}_strict_pairs.csv"
    out_pair_list_csv = out_dir / f"top_{args.top_n}_strict_train_list.csv"
    out_train_csv = out_dir / f"top_{args.top_n}_strict_train_split.csv"
    out_val_csv = out_dir / f"top_{args.top_n}_strict_val_split.csv"
    out_test_csv = out_dir / f"top_{args.top_n}_strict_test_split.csv"
    out_report = out_dir / f"top_{args.top_n}_strict_selection_report.md"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in selected:
            w.writerow(row)

    normalized_selected = []
    for row in selected:
        normalized_selected.append({
            "cloudy_path": _normalize_path(row.get("cloudy_path", "")),
            "target_path": _normalize_path(row.get("clear_path", "")),
        })
    _write_pair_list(out_pair_list_csv, normalized_selected)

    train_rows, val_rows, test_rows = _compute_split(selected, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    _write_pair_list(out_train_csv, train_rows)
    _write_pair_list(out_val_csv, val_rows)
    _write_pair_list(out_test_csv, test_rows)

    lines = []
    lines.append("# Top Strict Pair Selection Report")
    lines.append("")
    lines.append(f"Ranking CSV: {ranking_csv}")
    lines.append(f"Total ranked candidate pairs: {len(rows)}")
    lines.append(f"Strict one-to-one rows (metadata match): {len(strict_metadata)}")
    lines.append(f"Strict rows with resolvable files: {len(strict)}")
    lines.append("")
    lines.append("## Thresholds")
    lines.append(f"- SSIM >= {args.min_ssim}")
    lines.append(f"- PSNR >= {args.min_psnr}")
    lines.append(f"- SAM <= {args.max_sam}")
    lines.append("")
    lines.append("## Selection")
    lines.append(f"Rows passing thresholds (strict only): {len(filtered_sorted)}")
    lines.append(f"Requested top-N: {args.top_n}")
    lines.append(f"Selected rows: {len(selected)}")
    if not strict and strict_metadata:
        lines.append("")
        lines.append("Note: strict metadata matches were found, but none of the image paths resolved in the current workspace.")
    if not filtered_sorted and selected:
        lines.append("")
        lines.append("Note: fell back to the curated strict pair list because no file-resolved rows satisfied the selection filters.")
    lines.append("")
    lines.append("## Split")
    lines.append(f"- Train ratio: {args.train_ratio}")
    lines.append(f"- Validation ratio: {args.val_ratio}")
    lines.append(f"- Test ratio: {args.test_ratio}")
    lines.append(f"- Train pairs: {len(train_rows)}")
    lines.append(f"- Validation pairs: {len(val_rows)}")
    lines.append(f"- Test pairs: {len(test_rows)}")
    lines.append("")
    if len(selected) < args.top_n:
        lines.append("Note: Fewer than requested rows satisfy strict+threshold constraints.")
        lines.append("")
    lines.append("## Outputs")
    lines.append(f"- {out_csv}")
    lines.append(f"- {out_pair_list_csv}")
    lines.append(f"- {out_train_csv}")
    lines.append(f"- {out_val_csv}")
    lines.append(f"- {out_test_csv}")

    out_report.write_text("\n".join(lines), encoding="utf-8")

    print("Saved:", out_csv)
    print("Saved:", out_pair_list_csv)
    print("Saved:", out_train_csv)
    print("Saved:", out_val_csv)
    print("Saved:", out_test_csv)
    print("Saved:", out_report)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Select top strict one-to-one pairs and optionally save train/val/test splits.")
    p.add_argument("--ranking_csv", default="checkpoints_nafnet/raw_pair_audit/raw_pair_ranking.csv")
    p.add_argument("--out_dir", default="checkpoints_nafnet/raw_pair_audit")
    p.add_argument("--top_n", type=int, default=3000)
    p.add_argument("--min_ssim", type=float, default=0.75)
    p.add_argument("--min_psnr", type=float, default=28.0)
    p.add_argument("--max_sam", type=float, default=8.0)
    p.add_argument("--train_ratio", type=float, default=1.0)
    p.add_argument("--val_ratio", type=float, default=0.0)
    p.add_argument("--test_ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    raise SystemExit(run(p.parse_args()))
