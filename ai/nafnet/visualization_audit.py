"""Audit image export path for input/prediction/target/comparison PNGs."""

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .dataset import NAFDataset
from .exp_run import save_png
from .model import NAFNetWrapper
from .smoke_test import find_pairs


def _stats(name, arr_hwc):
    return {
        "name": name,
        "min": float(np.min(arr_hwc)),
        "max": float(np.max(arr_hwc)),
        "mean": float(np.mean(arr_hwc)),
    }


def _range_kind(arr_hwc):
    mn = float(np.min(arr_hwc))
    mx = float(np.max(arr_hwc))
    if mn >= 0.0 and mx <= 1.0:
        return "normalized_[0,1]"
    if mn >= -1.0 and mx <= 1.0:
        return "normalized_[-1,1]"
    return "raw_DN_or_other"


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = out_dir / "audit_sample"
    audit_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.cloudy, args.clear, max_pairs=args.max_pairs)
    if len(pairs) < (args.train_pairs + 1):
        raise RuntimeError("Not enough pairs for requested train/val split")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(pairs)
    train_pairs = pairs[: args.train_pairs]
    val_pairs = pairs[args.train_pairs : args.train_pairs + args.val_pairs]
    if len(val_pairs) == 0:
        raise RuntimeError("Validation split is empty")

    stats_file = Path("tmp_stats/band_statistics.json")
    clip_min = [0.0, 0.0, 0.0]
    clip_max = [6000.0, 6000.0, 6000.0]
    if stats_file.exists():
        j = json.loads(stats_file.read_text())
        if "p1" in j and "p99" in j:
            clip_min = j["p1"]
            clip_max = j["p99"]

    val_ds = NAFDataset(val_pairs, clip_min, clip_max, patch_size=None, augment=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
    model.eval()

    c, t = val_ds[args.sample_index]
    x = c.unsqueeze(0).to(device)
    with torch.no_grad():
        p = model(x)[0].detach().cpu().numpy()

    input_hwc = np.transpose(c.numpy(), (1, 2, 0))
    pred_hwc = np.transpose(p, (1, 2, 0))
    pred_hwc = np.clip(pred_hwc, 0.0, 1.0)
    target_hwc = np.transpose(t.numpy(), (1, 2, 0))
    comparison_hwc = np.concatenate([input_hwc, pred_hwc, target_hwc], axis=1)

    pre_export = {
        "input": _stats("input", input_hwc),
        "prediction": _stats("prediction", pred_hwc),
        "target": _stats("target", target_hwc),
        "value_kind": {
            "input": _range_kind(input_hwc),
            "prediction": _range_kind(pred_hwc),
            "target": _range_kind(target_hwc),
        },
    }

    input_png = audit_dir / "input.png"
    pred_png = audit_dir / "prediction.png"
    target_png = audit_dir / "target.png"
    comp_png = audit_dir / "comparison.png"

    save_png(input_hwc, str(input_png))
    save_png(pred_hwc, str(pred_png))
    save_png(target_hwc, str(target_png))
    save_png(comparison_hwc, str(comp_png))

    def _png_stats(path):
        arr = imageio.imread(path)
        arr = arr.astype(np.float32) / 255.0
        return _stats(path.name, arr)

    post_export = {
        "input_png": _png_stats(input_png),
        "prediction_png": _png_stats(pred_png),
        "target_png": _png_stats(target_png),
        "comparison_png": _png_stats(comp_png),
    }

    report = {
        "train_pairs_count": len(train_pairs),
        "val_pairs_count": len(val_pairs),
        "sample_index": args.sample_index,
        "checkpoint_used": str(ckpt_path),
        "export_code_path": {
            "save_function": "ai/nafnet/exp_run.py::save_png",
            "evaluate_function": "ai/nafnet/exp_run.py::_evaluate",
            "display_normalization": "identical for input/prediction/target: uint8 = clip(arr*255,0,255)",
        },
        "pre_export_tensor_stats": pre_export,
        "post_export_png_stats": post_export,
        "comparison": {
            "input_mean_delta": float(post_export["input_png"]["mean"] - pre_export["input"]["mean"]),
            "prediction_mean_delta": float(post_export["prediction_png"]["mean"] - pre_export["prediction"]["mean"]),
            "target_mean_delta": float(post_export["target_png"]["mean"] - pre_export["target"]["mean"]),
        },
    }

    json_path = out_dir / "visualization_audit.json"
    md_path = out_dir / "visualization_audit.md"
    json_path.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append("# Visualization Audit")
    lines.append("")
    lines.append("## Export Code Path")
    lines.append(f"- save function: `{report['export_code_path']['save_function']}`")
    lines.append(f"- evaluate function: `{report['export_code_path']['evaluate_function']}`")
    lines.append(f"- display normalization: {report['export_code_path']['display_normalization']}")
    lines.append("")
    lines.append("## Pre-Export Tensor Stats")
    for key in ["input", "prediction", "target"]:
        s = pre_export[key]
        lines.append(
            f"- {key}: min={s['min']:.6f}, max={s['max']:.6f}, mean={s['mean']:.6f}, kind={pre_export['value_kind'][key]}"
        )
    lines.append("")
    lines.append("## Post-Export PNG Stats (after reading PNG/255)")
    for key in ["input_png", "prediction_png", "target_png", "comparison_png"]:
        s = post_export[key]
        lines.append(f"- {key}: min={s['min']:.6f}, max={s['max']:.6f}, mean={s['mean']:.6f}")
    lines.append("")
    lines.append("## Tensor vs PNG Appearance")
    lines.append(
        f"- input mean delta (png - tensor): {report['comparison']['input_mean_delta']:.6f}"
    )
    lines.append(
        f"- prediction mean delta (png - tensor): {report['comparison']['prediction_mean_delta']:.6f}"
    )
    lines.append(
        f"- target mean delta (png - tensor): {report['comparison']['target_mean_delta']:.6f}"
    )

    md_path.write_text("\n".join(lines))

    print("Pre-export stats:")
    print(json.dumps(pre_export, indent=2))
    print("Wrote:", json_path)
    print("Wrote:", md_path)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cloudy", required=True)
    p.add_argument("--clear", required=True)
    p.add_argument("--checkpoint", default="checkpoints_nafnet/full_training/nafnet_epoch_1.pt")
    p.add_argument("--max_pairs", type=int, default=250)
    p.add_argument("--train_pairs", type=int, default=200)
    p.add_argument("--val_pairs", type=int, default=50)
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="checkpoints_nafnet/full_training")
    args = p.parse_args()
    raise SystemExit(run(args))
