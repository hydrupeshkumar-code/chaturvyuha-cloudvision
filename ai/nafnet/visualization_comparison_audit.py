"""Visualization scaling audit for NAFNet outputs.

For 20 random validation samples:
- compute per-channel min/max/mean/std before PNG export
- generate visualization variants for one representative sample:
  A) current visualization
  B) min-max stretch per image
  C) percentile stretch (2%-98%)
  D) histogram equalization
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .dataset import NAFDataset
from .model import NAFNetWrapper
from .smoke_test import find_pairs


def _channel_stats(hwc):
    stats = []
    for c in range(hwc.shape[2]):
        x = hwc[:, :, c]
        stats.append(
            {
                "channel": c,
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
            }
        )
    return stats


def _to_uint8(img):
    y = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return y


def _stretch_minmax(img):
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        x = img[:, :, c]
        mn = float(np.min(x))
        mx = float(np.max(x))
        if mx - mn < 1e-8:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = (x - mn) / (mx - mn)
    return np.clip(out, 0.0, 1.0)


def _stretch_percentile(img, p_low=2.0, p_high=98.0):
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        x = img[:, :, c]
        lo = float(np.percentile(x, p_low))
        hi = float(np.percentile(x, p_high))
        if hi - lo < 1e-8:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = (x - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _equalize_hist_channel(x):
    # x in [0,1]
    q = np.clip((x * 255.0).round(), 0, 255).astype(np.uint8)
    hist = np.bincount(q.reshape(-1), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    if cdf[-1] <= 0:
        return np.zeros_like(x, dtype=np.float32)
    cdf = cdf / cdf[-1]
    return cdf[q].astype(np.float32)


def _equalize_hist(img):
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        out[:, :, c] = _equalize_hist_channel(np.clip(img[:, :, c], 0.0, 1.0))
    return np.clip(out, 0.0, 1.0)


def _write_variant_panel(input_img, pred_img, target_img, out_path, mode):
    import imageio.v2 as imageio

    if mode == "current":
        a = np.clip(input_img, 0.0, 1.0)
        b = np.clip(pred_img, 0.0, 1.0)
        c = np.clip(target_img, 0.0, 1.0)
    elif mode == "minmax":
        a = _stretch_minmax(input_img)
        b = _stretch_minmax(pred_img)
        c = _stretch_minmax(target_img)
    elif mode == "percentile":
        a = _stretch_percentile(input_img, 2.0, 98.0)
        b = _stretch_percentile(pred_img, 2.0, 98.0)
        c = _stretch_percentile(target_img, 2.0, 98.0)
    elif mode == "equalized":
        a = _equalize_hist(input_img)
        b = _equalize_hist(pred_img)
        c = _equalize_hist(target_img)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    panel = np.concatenate([a, b, c], axis=1)
    imageio.imwrite(str(out_path), _to_uint8(panel))


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.cloudy, args.clear, max_pairs=args.max_pairs)
    if len(pairs) < (args.train_pairs + args.val_pairs):
        raise RuntimeError("Not enough pairs for train/val split")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    val_pairs = pairs[args.train_pairs : args.train_pairs + args.val_pairs]
    if len(val_pairs) < args.samples:
        raise RuntimeError("Validation pool smaller than requested audit sample count")

    stats_file = Path("tmp_stats/band_statistics.json")
    clip_min = [0.0, 0.0, 0.0]
    clip_max = [6000.0, 6000.0, 6000.0]
    if stats_file.exists():
        j = json.loads(stats_file.read_text())
        if "p1" in j and "p99" in j:
            clip_min = j["p1"]
            clip_max = j["p99"]

    ds = NAFDataset(val_pairs, clip_min, clip_max, patch_size=None, augment=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # select 20 random validation indices
    idxs = list(range(len(val_pairs)))
    rng.shuffle(idxs)
    idxs = idxs[: args.samples]

    rows = []
    representative = None

    for k, idx in enumerate(idxs):
        c, t = ds[idx]
        with torch.no_grad():
            p = model(c.unsqueeze(0).to(device))[0].cpu().numpy()
        inp = np.transpose(c.numpy(), (1, 2, 0))
        pred = np.clip(np.transpose(p, (1, 2, 0)), 0.0, 1.0)
        tgt = np.transpose(t.numpy(), (1, 2, 0))

        rows.append(
            {
                "sample_idx": int(idx),
                "pair": val_pairs[idx],
                "input": _channel_stats(inp),
                "prediction": _channel_stats(pred),
                "target": _channel_stats(tgt),
            }
        )

        if representative is None:
            representative = (inp, pred, tgt, idx)

    # generate requested 4 visualizations for one representative sample
    assert representative is not None
    inp, pred, tgt, rep_idx = representative
    _write_variant_panel(inp, pred, tgt, out_dir / "vis_current.png", "current")
    _write_variant_panel(inp, pred, tgt, out_dir / "vis_minmax.png", "minmax")
    _write_variant_panel(inp, pred, tgt, out_dir / "vis_percentile.png", "percentile")
    _write_variant_panel(inp, pred, tgt, out_dir / "vis_equalized.png", "equalized")

    # aggregate means across 20 samples
    def _aggregate(obj_key):
        out = []
        for c in range(3):
            mins, maxs, means, stds = [], [], [], []
            for r in rows:
                s = r[obj_key][c]
                mins.append(s["min"])
                maxs.append(s["max"])
                means.append(s["mean"])
                stds.append(s["std"])
            out.append(
                {
                    "channel": c,
                    "min_mean": float(np.mean(mins)),
                    "max_mean": float(np.mean(maxs)),
                    "mean_mean": float(np.mean(means)),
                    "std_mean": float(np.mean(stds)),
                }
            )
        return out

    summary = {
        "samples": args.samples,
        "representative_sample_idx": int(rep_idx),
        "checkpoint": args.checkpoint,
        "tensor_range_classification": {
            "input": "normalized_[0,1]",
            "prediction": "normalized_[0,1]",
            "target": "normalized_[0,1]",
        },
        "aggregates": {
            "input": _aggregate("input"),
            "prediction": _aggregate("prediction"),
            "target": _aggregate("target"),
        },
        "rows": rows,
    }

    (out_dir / "visualization_comparison_report.json").write_text(json.dumps(summary, indent=2))

    # markdown report with per-sample stats
    lines = []
    lines.append("# Visualization Comparison Report")
    lines.append("")
    lines.append(f"Samples analyzed: {args.samples}")
    lines.append(f"Checkpoint: `{args.checkpoint}`")
    lines.append(f"Representative sample idx for panels: {rep_idx}")
    lines.append("")
    lines.append("## Export-time Tensor Interpretation")
    lines.append("- input: normalized [0,1]")
    lines.append("- prediction: normalized [0,1]")
    lines.append("- target: normalized [0,1]")
    lines.append("")
    lines.append("## Aggregate Channel Statistics (mean over 20 samples)")
    for key in ["input", "prediction", "target"]:
        lines.append(f"### {key}")
        for s in summary["aggregates"][key]:
            lines.append(
                f"- ch{s['channel']}: min={s['min_mean']:.6f}, max={s['max_mean']:.6f}, mean={s['mean_mean']:.6f}, std={s['std_mean']:.6f}"
            )
    lines.append("")
    lines.append("## Per-sample Channel Stats")
    for r in rows:
        lines.append(f"### sample_idx={r['sample_idx']}")
        for key in ["input", "prediction", "target"]:
            lines.append(f"- {key}:")
            for s in r[key]:
                lines.append(
                    f"  - ch{s['channel']}: min={s['min']:.6f}, max={s['max']:.6f}, mean={s['mean']:.6f}, std={s['std']:.6f}"
                )

    # diagnosis
    pred_mean = np.mean([c['mean_mean'] for c in summary['aggregates']['prediction']])
    tgt_mean = np.mean([c['mean_mean'] for c in summary['aggregates']['target']])
    if pred_mean < 0.5 * tgt_mean:
        verdict = "Dark appearance primarily reflects low-intensity prediction tensors (actual reconstruction behavior)."
    else:
        verdict = "Dark appearance is partly influenced by display scaling; tensor intensity gap is moderate."
    lines.append("")
    lines.append("## Diagnosis")
    lines.append(f"- {verdict}")
    lines.append("- Compare `vis_current.png` vs stretched variants to separate tensor-intensity effects from visualization scaling.")

    (out_dir / "visualization_comparison_report.md").write_text("\n".join(lines))

    print("Representative sample index:", rep_idx)
    print("Saved:", out_dir / "visualization_comparison_report.md")
    print("Saved:", out_dir / "vis_current.png")
    print("Saved:", out_dir / "vis_minmax.png")
    print("Saved:", out_dir / "vis_percentile.png")
    print("Saved:", out_dir / "vis_equalized.png")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cloudy", required=True)
    p.add_argument("--clear", required=True)
    p.add_argument("--checkpoint", default="checkpoints_nafnet/full_dataset_training/best_ssim.pth")
    p.add_argument("--max_pairs", type=int, default=5891)
    p.add_argument("--train_pairs", type=int, default=5000)
    p.add_argument("--val_pairs", type=int, default=891)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="checkpoints_nafnet/full_dataset_training")
    args = p.parse_args()
    raise SystemExit(run(args))
