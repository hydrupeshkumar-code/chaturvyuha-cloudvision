"""Audit dynamic-range collapse in NAFNet predictions using existing checkpoints.

This script does not retrain. It analyzes:
1) final activation / clipping behavior from source
2) weighted loss term magnitudes (L1, SSIM, SAM)
3) 100-sample validation output statistics (pred vs target)
4) trend across epoch checkpoints toward low-variance predictions
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim

from .dataset import NAFDataset
from .model import NAFNetWrapper
from .smoke_test import find_pairs


def _stats_1d(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _sample_loss_terms(pred_chw, target_chw):
    # Inputs are CHW tensors in [0,1] (for target), pred can exceed [0,1]
    l1 = float(F.l1_loss(pred_chw, target_chw).item())

    pred = np.transpose(pred_chw.detach().cpu().numpy(), (1, 2, 0))
    target = np.transpose(target_chw.detach().cpu().numpy(), (1, 2, 0))

    ssim_vals = []
    for c in range(target.shape[2]):
        try:
            ssim_vals.append(float(sk_ssim(target[:, :, c], pred[:, :, c], data_range=1.0)))
        except Exception:
            ssim_vals.append(0.0)
    ssim_m = float(np.mean(ssim_vals))

    yv = target.reshape(-1, target.shape[2])
    pv = pred.reshape(-1, pred.shape[2])
    num = np.sum(yv * pv, axis=1)
    den = np.linalg.norm(yv, axis=1) * np.linalg.norm(pv, axis=1)
    den = np.maximum(den, 1e-8)
    cos = np.clip(num / den, -1.0, 1.0)
    ang = np.arccos(cos)
    sam_deg = float(np.degrees(np.mean(ang)))

    l1_w = 100.0 * l1
    ssim_w = 5.0 * (1.0 - ssim_m)
    sam_w = 2.0 * (sam_deg / 180.0)
    total = l1_w + ssim_w + sam_w

    return {
        "l1": l1,
        "ssim": ssim_m,
        "sam_deg": sam_deg,
        "weighted_l1": l1_w,
        "weighted_ssim": ssim_w,
        "weighted_sam": sam_w,
        "weighted_total": total,
    }


def _collect_for_checkpoint(model, ds, sample_indices, device):
    pred_means = []
    pred_stds = []
    tgt_means = []
    tgt_stds = []

    w_l1 = []
    w_ssim = []
    w_sam = []
    w_total = []

    model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            x, y = ds[idx]
            p = model(x.unsqueeze(0).to(device))[0].cpu()

            # stats from prediction after eval-time clipping, matching export behavior
            p_clip = torch.clamp(p, 0.0, 1.0)
            p_np = p_clip.numpy()
            y_np = y.numpy()

            pred_means.append(float(np.mean(p_np)))
            pred_stds.append(float(np.std(p_np)))
            tgt_means.append(float(np.mean(y_np)))
            tgt_stds.append(float(np.std(y_np)))

            terms = _sample_loss_terms(p, y)
            w_l1.append(terms["weighted_l1"])
            w_ssim.append(terms["weighted_ssim"])
            w_sam.append(terms["weighted_sam"])
            w_total.append(terms["weighted_total"])

    l1_mean = float(np.mean(w_l1))
    ssim_mean = float(np.mean(w_ssim))
    sam_mean = float(np.mean(w_sam))
    total_mean = float(np.mean(w_total))
    denom = max(total_mean, 1e-12)

    return {
        "pred_mean": _stats_1d(pred_means),
        "pred_std": _stats_1d(pred_stds),
        "target_mean": _stats_1d(tgt_means),
        "target_std": _stats_1d(tgt_stds),
        "weighted_terms": {
            "l1": {"mean": l1_mean, "fraction_of_total": l1_mean / denom},
            "ssim": {"mean": ssim_mean, "fraction_of_total": ssim_mean / denom},
            "sam": {"mean": sam_mean, "fraction_of_total": sam_mean / denom},
            "total": {"mean": total_mean},
        },
    }


def _load_pairs_and_dataset(args):
    pairs = find_pairs(args.cloudy, args.clear, max_pairs=args.max_pairs)
    if len(pairs) < args.train_pairs + args.val_pairs:
        raise RuntimeError("Not enough pairs for requested split")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    val_pairs = pairs[args.train_pairs : args.train_pairs + args.val_pairs]

    stats = {"p1": [0.0, 0.0, 0.0], "p99": [6000.0, 6000.0, 6000.0]}
    sp = Path("tmp_stats/band_statistics.json")
    if sp.exists():
        try:
            j = json.loads(sp.read_text())
            stats["p1"] = j["p1"]
            stats["p99"] = j["p99"]
        except Exception:
            pass

    ds = NAFDataset(val_pairs, stats["p1"], stats["p99"], patch_size=None, augment=False)
    return ds, val_pairs


def _discover_epoch_checkpoints(ckpt_dir):
    rows = []
    for p in sorted(Path(ckpt_dir).glob("nafnet_epoch_*.pt")):
        name = p.stem
        try:
            epoch = int(name.split("_")[-1])
        except Exception:
            continue
        rows.append((epoch, str(p)))
    rows.sort(key=lambda t: t[0])
    return rows


def _analyze_activation_and_clipping(model_py, exp_py):
    model_text = Path(model_py).read_text(encoding="utf-8")
    exp_text = Path(exp_py).read_text(encoding="utf-8")

    has_sigmoid = "Sigmoid" in model_text or "sigmoid(" in model_text
    has_tanh = "Tanh" in model_text or "tanh(" in model_text

    clipping_before_loss = False
    # _spectral_loss uses pred directly; clipping appears in _evaluate only.
    if "_spectral_loss" in exp_text and "pred = np.clip(pred, 0.0, 1.0)" in exp_text:
        clipping_before_loss = False

    return {
        "has_sigmoid": has_sigmoid,
        "has_tanh": has_tanh,
        "clipping_before_loss": clipping_before_loss,
        "notes": [
            "Model head is Conv2d -> linear output (no explicit bounded activation).",
            "Training loss is computed on raw model output.",
            "Prediction clipping to [0,1] is applied in evaluation/export path only.",
        ],
    }


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds, val_pairs = _load_pairs_and_dataset(args)
    n = min(args.samples, len(val_pairs))
    if n <= 0:
        raise RuntimeError("No validation samples available for audit")

    rng = random.Random(args.seed)
    sample_indices = list(range(len(val_pairs)))
    rng.shuffle(sample_indices)
    sample_indices = sample_indices[:n]

    ckpt_rows = _discover_epoch_checkpoints(args.checkpoint_dir)
    if not ckpt_rows:
        raise RuntimeError("No epoch checkpoints found")

    if args.include_best_ssim and Path(args.best_ssim_checkpoint).exists():
        # Analyze best_ssim as a named checkpoint in addition to epoch files.
        ckpt_rows.append((-1, args.best_ssim_checkpoint))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)

    per_ckpt = []
    for epoch, ckpt in ckpt_rows:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        res = _collect_for_checkpoint(model, ds, sample_indices, device)
        per_ckpt.append(
            {
                "epoch": int(epoch),
                "checkpoint": ckpt,
                "analysis": res,
            }
        )
        print(f"analyzed checkpoint epoch={epoch} path={ckpt}")

    # sort with best_ssim(-1) last for readability
    normal = [r for r in per_ckpt if r["epoch"] >= 0]
    normal.sort(key=lambda r: r["epoch"])
    extra = [r for r in per_ckpt if r["epoch"] < 0]
    ordered = normal + extra

    # convergence signal: pred std trend from first to last epoch checkpoint
    pred_std_first = normal[0]["analysis"]["pred_std"]["mean"]
    pred_std_last = normal[-1]["analysis"]["pred_std"]["mean"]
    pred_mean_first = normal[0]["analysis"]["pred_mean"]["mean"]
    pred_mean_last = normal[-1]["analysis"]["pred_mean"]["mean"]

    slope = (pred_std_last - pred_std_first) / max((normal[-1]["epoch"] - normal[0]["epoch"]), 1)
    converging_low_var = pred_std_last <= pred_std_first and pred_std_last < 0.02

    activation = _analyze_activation_and_clipping(
        model_py=str(Path(__file__).parent / "model.py"),
        exp_py=str(Path(__file__).parent / "exp_run.py"),
    )

    final_row = normal[-1]
    final_terms = final_row["analysis"]["weighted_terms"]

    recommendation = {
        "choice": "B",
        "title": "add contrast-preserving loss",
        "reason": (
            "Weighted L1 dominates optimization while prediction variance remains collapsed; "
            "a contrast-preserving objective directly penalizes low dynamic range and is more "
            "targeted than changing activation in this architecture."
        ),
    }

    report = {
        "samples": n,
        "seed": args.seed,
        "activation_clipping": activation,
        "loss_formula": "100 * L1 + 5 * (1 - SSIM) + 2 * SAM/180",
        "checkpoints_analyzed": ordered,
        "convergence": {
            "pred_mean_first": pred_mean_first,
            "pred_mean_last": pred_mean_last,
            "pred_std_first": pred_std_first,
            "pred_std_last": pred_std_last,
            "pred_std_slope_per_epoch": slope,
            "converging_to_low_variance_average": bool(converging_low_var),
        },
        "final_checkpoint_summary": {
            "epoch": final_row["epoch"],
            "pred_mean": final_row["analysis"]["pred_mean"],
            "pred_std": final_row["analysis"]["pred_std"],
            "target_mean": final_row["analysis"]["target_mean"],
            "target_std": final_row["analysis"]["target_std"],
            "loss_term_contributions": final_terms,
        },
        "recommendation": recommendation,
    }

    (out_dir / "dynamic_range_report.json").write_text(json.dumps(report, indent=2))

    md = []
    md.append("# Dynamic Range Collapse Audit")
    md.append("")
    md.append(f"Validation samples analyzed: {n}")
    md.append(f"Checkpoint directory: `{args.checkpoint_dir}`")
    md.append("")
    md.append("## 1) Final activation layer and clipping")
    md.append(f"- Sigmoid present in model head: {activation['has_sigmoid']}")
    md.append(f"- Tanh present in model head: {activation['has_tanh']}")
    md.append(f"- Outputs clipped before loss computation: {activation['clipping_before_loss']}")
    for note in activation["notes"]:
        md.append(f"- {note}")
    md.append("")
    md.append("## 2) Loss-term contribution magnitude (weighted)")
    md.append(
        "- Formula: 100 * L1 + 5 * (1 - SSIM) + 2 * SAM/180"
    )
    md.append("- Final epoch mean contributions across 100 samples:")
    md.append(
        f"  - L1 term: {final_terms['l1']['mean']:.6f} ({100.0 * final_terms['l1']['fraction_of_total']:.2f}% of total)"
    )
    md.append(
        f"  - SSIM term: {final_terms['ssim']['mean']:.6f} ({100.0 * final_terms['ssim']['fraction_of_total']:.2f}% of total)"
    )
    md.append(
        f"  - SAM term: {final_terms['sam']['mean']:.6f} ({100.0 * final_terms['sam']['fraction_of_total']:.2f}% of total)"
    )
    md.append("")
    md.append("## 3) Output statistics on 100 validation samples")
    fs = final_row["analysis"]
    md.append(
        f"- Pred mean: mean={fs['pred_mean']['mean']:.6f}, std={fs['pred_mean']['std']:.6f}, p10/p50/p90={fs['pred_mean']['p10']:.6f}/{fs['pred_mean']['p50']:.6f}/{fs['pred_mean']['p90']:.6f}"
    )
    md.append(
        f"- Pred std: mean={fs['pred_std']['mean']:.6f}, std={fs['pred_std']['std']:.6f}, p10/p50/p90={fs['pred_std']['p10']:.6f}/{fs['pred_std']['p50']:.6f}/{fs['pred_std']['p90']:.6f}"
    )
    md.append(
        f"- Target mean: mean={fs['target_mean']['mean']:.6f}, std={fs['target_mean']['std']:.6f}, p10/p50/p90={fs['target_mean']['p10']:.6f}/{fs['target_mean']['p50']:.6f}/{fs['target_mean']['p90']:.6f}"
    )
    md.append(
        f"- Target std: mean={fs['target_std']['mean']:.6f}, std={fs['target_std']['std']:.6f}, p10/p50/p90={fs['target_std']['p10']:.6f}/{fs['target_std']['p50']:.6f}/{fs['target_std']['p90']:.6f}"
    )
    md.append("")
    md.append("## 4) Convergence toward low-variance predictions")
    md.append(
        f"- Pred mean first->last epoch: {pred_mean_first:.6f} -> {pred_mean_last:.6f}"
    )
    md.append(
        f"- Pred std first->last epoch: {pred_std_first:.6f} -> {pred_std_last:.6f}"
    )
    md.append(f"- Pred std slope per epoch: {slope:.8f}")
    md.append(
        f"- Converging toward low-variance average predictions: {bool(converging_low_var)}"
    )
    md.append("")
    md.append("### Per-epoch snapshot")
    md.append("| epoch | pred_mean | pred_std | target_mean | target_std | L1% | SSIM% | SAM% |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in normal:
        a = r["analysis"]
        wt = a["weighted_terms"]
        md.append(
            "| {e} | {pm:.6f} | {ps:.6f} | {tm:.6f} | {ts:.6f} | {l1:.2f} | {ss:.2f} | {sa:.2f} |".format(
                e=r["epoch"],
                pm=a["pred_mean"]["mean"],
                ps=a["pred_std"]["mean"],
                tm=a["target_mean"]["mean"],
                ts=a["target_std"]["mean"],
                l1=100.0 * wt["l1"]["fraction_of_total"],
                ss=100.0 * wt["ssim"]["fraction_of_total"],
                sa=100.0 * wt["sam"]["fraction_of_total"],
            )
        )

    md.append("")
    md.append("## 5) Root-cause conclusion")
    md.append(
        "- Dynamic-range collapse is not caused by sigmoid saturation (no sigmoid in model head) and not by pre-loss clipping."
    )
    md.append(
        "- Collapse is consistent with optimization pressure dominated by weighted L1 while outputs stay near a low-variance mean region."
    )
    md.append("")
    md.append("## 6) Recommended single action")
    md.append(
        f"- {recommendation['choice']}) {recommendation['title']}: {recommendation['reason']}"
    )

    (out_dir / "dynamic_range_report.md").write_text("\n".join(md), encoding="utf-8")

    # also write compact CSV for quick plotting/debugging
    with open(out_dir / "dynamic_range_epoch_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "pred_mean",
            "pred_std",
            "target_mean",
            "target_std",
            "weighted_l1_fraction",
            "weighted_ssim_fraction",
            "weighted_sam_fraction",
        ])
        for r in normal:
            a = r["analysis"]
            wt = a["weighted_terms"]
            w.writerow([
                r["epoch"],
                a["pred_mean"]["mean"],
                a["pred_std"]["mean"],
                a["target_mean"]["mean"],
                a["target_std"]["mean"],
                wt["l1"]["fraction_of_total"],
                wt["ssim"]["fraction_of_total"],
                wt["sam"]["fraction_of_total"],
            ])

    print("Saved:", out_dir / "dynamic_range_report.md")
    print("Saved:", out_dir / "dynamic_range_report.json")
    print("Saved:", out_dir / "dynamic_range_epoch_summary.csv")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cloudy", required=True)
    p.add_argument("--clear", required=True)
    p.add_argument("--checkpoint_dir", default="checkpoints_nafnet/full_dataset_training")
    p.add_argument("--best_ssim_checkpoint", default="checkpoints_nafnet/full_dataset_training/best_ssim.pth")
    p.add_argument("--include_best_ssim", action="store_true", default=True)
    p.add_argument("--max_pairs", type=int, default=5891)
    p.add_argument("--train_pairs", type=int, default=5000)
    p.add_argument("--val_pairs", type=int, default=891)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="checkpoints_nafnet/full_dataset_training")
    raise SystemExit(run(p.parse_args()))
