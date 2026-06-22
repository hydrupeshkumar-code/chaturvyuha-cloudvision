"""Contrast-aware NAFNet fine-tuning experiment.

Starts from an existing checkpoint (best_ssim.pth), trains for a small number of epochs,
and compares baseline vs fine-tuned behavior for dynamic range and quality metrics.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as sk_ssim

from .dataset import NAFDataset
from .model import NAFNetWrapper
from .smoke_test import find_pairs
from . import metrics as naf_metrics


def save_png(arr_hwc, path):
    import imageio.v2 as imageio

    out = np.clip(arr_hwc * 255.0, 0, 255).astype(np.uint8)
    imageio.imwrite(str(path), out)


def _sobel_kernels(channels, device, dtype):
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device) / 8.0
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device) / 8.0
    kx = kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return kx, ky


def _edge_loss(pred, target):
    c = pred.shape[1]
    kx, ky = _sobel_kernels(c, pred.device, pred.dtype)
    gx_p = F.conv2d(pred, kx, padding=1, groups=c)
    gy_p = F.conv2d(pred, ky, padding=1, groups=c)
    gx_t = F.conv2d(target, kx, padding=1, groups=c)
    gy_t = F.conv2d(target, ky, padding=1, groups=c)
    mag_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p + 1e-8)
    mag_t = torch.sqrt(gx_t * gx_t + gy_t * gy_t + 1e-8)
    return F.l1_loss(mag_p, mag_t)


def _batch_ssim_sam(pred, target):
    p_np = pred.detach().cpu().numpy()
    t_np = target.detach().cpu().numpy()
    ssim_vals = []
    sam_vals = []
    for i in range(p_np.shape[0]):
        pp = np.transpose(p_np[i], (1, 2, 0))
        tt = np.transpose(t_np[i], (1, 2, 0))
        try:
            ssim_vals.append(float(np.mean([sk_ssim(tt[:, :, c], pp[:, :, c], data_range=1.0) for c in range(tt.shape[2])])))
        except Exception:
            ssim_vals.append(0.0)

        yv = tt.reshape(-1, tt.shape[2])
        pv = pp.reshape(-1, pp.shape[2])
        num = np.sum(yv * pv, axis=1)
        den = np.linalg.norm(yv, axis=1) * np.linalg.norm(pv, axis=1)
        den = np.maximum(den, 1e-8)
        cos = np.clip(num / den, -1.0, 1.0)
        ang = np.arccos(cos)
        sam_vals.append(float(np.degrees(np.mean(ang))))

    return float(np.mean(ssim_vals)), float(np.mean(sam_vals))


def _contrast_loss(pred, target):
    # 50 * L1 + 20 * (1 - SSIM) + 10 * EdgeLoss + 2 * (SAM / 180)
    l1 = F.l1_loss(pred, target)
    edge = _edge_loss(pred, target)
    ssim_m, sam_deg = _batch_ssim_sam(pred, target)

    weighted_l1 = 50.0 * l1
    weighted_ssim = 20.0 * (1.0 - ssim_m)
    weighted_edge = 10.0 * edge
    weighted_sam = 2.0 * (sam_deg / 180.0)
    total = weighted_l1 + weighted_ssim + weighted_edge + weighted_sam

    return {
        "total": total,
        "l1": float(l1.detach().cpu().item()),
        "ssim": float(ssim_m),
        "sam_deg": float(sam_deg),
        "edge": float(edge.detach().cpu().item()),
        "weighted_l1": float(weighted_l1.detach().cpu().item()),
        "weighted_ssim": float(weighted_ssim),
        "weighted_edge": float(weighted_edge.detach().cpu().item()),
        "weighted_sam": float(weighted_sam),
    }


def _evaluate(model, val_loader, device):
    model.eval()
    rows = []
    pred_means = []
    pred_stds = []
    pred_mins = []
    pred_maxs = []
    tgt_means = []
    tgt_stds = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            pred = model(x)[0].cpu().numpy()
            pred = np.transpose(pred, (1, 2, 0))
            pred = np.clip(pred, 0.0, 1.0)
            tgt = np.transpose(y[0].numpy(), (1, 2, 0))

            rows.append(
                {
                    "psnr": float(naf_metrics.psnr(tgt, pred)),
                    "ssim": float(naf_metrics.ssim(tgt, pred)),
                    "sam": float(naf_metrics.sam(tgt, pred)),
                }
            )

            pred_means.append(float(np.mean(pred)))
            pred_stds.append(float(np.std(pred)))
            pred_mins.append(float(np.min(pred)))
            pred_maxs.append(float(np.max(pred)))

            tgt_means.append(float(np.mean(tgt)))
            tgt_stds.append(float(np.std(tgt)))

    def _mean(key):
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "psnr": _mean("psnr"),
        "ssim": _mean("ssim"),
        "sam": _mean("sam"),
        "pred_stats": {
            "mean": float(np.mean(pred_means)),
            "std": float(np.mean(pred_stds)),
            "min": float(np.mean(pred_mins)),
            "max": float(np.mean(pred_maxs)),
        },
        "target_stats": {
            "mean": float(np.mean(tgt_means)),
            "std": float(np.mean(tgt_stds)),
        },
    }
    return summary


def _save_side_by_side_20(baseline_model, tuned_model, ds, indices, device, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_model.eval()
    tuned_model.eval()
    with torch.no_grad():
        for rank, idx in enumerate(indices):
            x, y = ds[idx]
            inp = np.transpose(x.numpy(), (1, 2, 0))
            tgt = np.transpose(y.numpy(), (1, 2, 0))
            b = baseline_model(x.unsqueeze(0).to(device))[0].cpu().numpy()
            t = tuned_model(x.unsqueeze(0).to(device))[0].cpu().numpy()
            b = np.clip(np.transpose(b, (1, 2, 0)), 0.0, 1.0)
            t = np.clip(np.transpose(t, (1, 2, 0)), 0.0, 1.0)

            panel = np.concatenate([inp, b, t, tgt], axis=1)
            save_png(panel, out_dir / f"comparison_{rank:02d}_idx_{idx}.png")


def _load_stats():
    stats = {"p1": [0.0, 0.0, 0.0], "p99": [6000.0, 6000.0, 6000.0]}
    sp = Path("tmp_stats/band_statistics.json")
    if sp.exists():
        try:
            j = json.loads(sp.read_text())
            stats["p1"] = j["p1"]
            stats["p99"] = j["p99"]
        except Exception:
            pass
    return stats


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.cloudy, args.clear, max_pairs=args.max_pairs)
    if len(pairs) < args.train_pairs + args.val_pairs:
        raise RuntimeError("Not enough pairs for requested split")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    train_pairs = pairs[: args.train_pairs]
    val_pairs = pairs[args.train_pairs : args.train_pairs + args.val_pairs]

    stats = _load_stats()
    train_ds = NAFDataset(train_pairs, stats["p1"], stats["p99"], patch_size=(128, 128), augment=True)
    val_ds = NAFDataset(val_pairs, stats["p1"], stats["p99"], patch_size=None, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    baseline_model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)

    state = torch.load(args.init_checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    baseline_model.load_state_dict(state, strict=False)

    baseline_summary = _evaluate(baseline_model, val_loader, device)

    # fixed 20 validation indices for visual comparison
    vis_rng = random.Random(args.seed + 1000)
    vis_indices = list(range(len(val_ds)))
    vis_rng.shuffle(vis_indices)
    vis_indices = vis_indices[:20]

    opt = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None

    history = []
    best_ssim = -1e9
    best_epoch = -1
    best_path = out_dir / "best_ssim.pth"

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_rows = []
        for i, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    p = model(x)
                    loss_row = _contrast_loss(p, y)
                    loss = loss_row["total"]
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                p = model(x)
                loss_row = _contrast_loss(p, y)
                loss = loss_row["total"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            row = {
                "total": float(loss.detach().cpu().item()),
                "l1": loss_row["l1"],
                "ssim": loss_row["ssim"],
                "sam_deg": loss_row["sam_deg"],
                "edge": loss_row["edge"],
                "weighted_l1": loss_row["weighted_l1"],
                "weighted_ssim": loss_row["weighted_ssim"],
                "weighted_edge": loss_row["weighted_edge"],
                "weighted_sam": loss_row["weighted_sam"],
            }
            epoch_rows.append(row)

            if i % 100 == 0:
                print(
                    "epoch {ep} batch {b}: total={t:.6f} l1={l1:.6f} ssim={ss:.6f} sam={sa:.6f} edge={ed:.6f}".format(
                        ep=epoch,
                        b=i,
                        t=row["total"],
                        l1=row["l1"],
                        ss=row["ssim"],
                        sa=row["sam_deg"],
                        ed=row["edge"],
                    )
                )

        def _avg(k):
            return float(np.mean([r[k] for r in epoch_rows]))

        val_summary = _evaluate(model, val_loader, device)
        epoch_summary = {
            "epoch": epoch,
            "train_loss_total": _avg("total"),
            "train_l1": _avg("l1"),
            "train_ssim": _avg("ssim"),
            "train_sam_deg": _avg("sam_deg"),
            "train_edge": _avg("edge"),
            "train_weighted_l1": _avg("weighted_l1"),
            "train_weighted_ssim": _avg("weighted_ssim"),
            "train_weighted_edge": _avg("weighted_edge"),
            "train_weighted_sam": _avg("weighted_sam"),
            "val": val_summary,
        }
        history.append(epoch_summary)

        ckpt_path = out_dir / f"contrast_epoch_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        torch.save(model.state_dict(), out_dir / "last_epoch.pth")

        if val_summary["ssim"] > best_ssim:
            best_ssim = val_summary["ssim"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(
            "epoch {ep} done: val_psnr={ps:.4f} val_ssim={ss:.4f} val_sam={sa:.4f} pred_mean={pm:.5f} pred_std={pd:.5f}".format(
                ep=epoch,
                ps=val_summary["psnr"],
                ss=val_summary["ssim"],
                sa=val_summary["sam"],
                pm=val_summary["pred_stats"]["mean"],
                pd=val_summary["pred_stats"]["std"],
            )
        )

    # load best tuned checkpoint for final comparison artifacts
    best_state = torch.load(best_path, map_location=device)
    tuned_best_model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    tuned_best_model.load_state_dict(best_state, strict=False)

    tuned_summary = _evaluate(tuned_best_model, val_loader, device)
    _save_side_by_side_20(
        baseline_model=baseline_model,
        tuned_model=tuned_best_model,
        ds=val_ds,
        indices=vis_indices,
        device=device,
        out_dir=out_dir / "side_by_side_20",
    )

    # structured csv for component logging
    with open(out_dir / "loss_components_per_epoch.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss_total",
                "train_l1",
                "train_ssim",
                "train_sam_deg",
                "train_edge",
                "train_weighted_l1",
                "train_weighted_ssim",
                "train_weighted_edge",
                "train_weighted_sam",
                "val_psnr",
                "val_ssim",
                "val_sam",
                "val_pred_mean",
                "val_pred_std",
                "val_pred_min",
                "val_pred_max",
            ]
        )
        for row in history:
            v = row["val"]
            writer.writerow(
                [
                    row["epoch"],
                    row["train_loss_total"],
                    row["train_l1"],
                    row["train_ssim"],
                    row["train_sam_deg"],
                    row["train_edge"],
                    row["train_weighted_l1"],
                    row["train_weighted_ssim"],
                    row["train_weighted_edge"],
                    row["train_weighted_sam"],
                    v["psnr"],
                    v["ssim"],
                    v["sam"],
                    v["pred_stats"]["mean"],
                    v["pred_stats"]["std"],
                    v["pred_stats"]["min"],
                    v["pred_stats"]["max"],
                ]
            )

    comparison = {
        "baseline": {
            "psnr": baseline_summary["psnr"],
            "ssim": baseline_summary["ssim"],
            "sam": baseline_summary["sam"],
            "pred_mean": baseline_summary["pred_stats"]["mean"],
            "pred_std": baseline_summary["pred_stats"]["std"],
        },
        "contrast_best_after_5_epochs": {
            "psnr": tuned_summary["psnr"],
            "ssim": tuned_summary["ssim"],
            "sam": tuned_summary["sam"],
            "pred_mean": tuned_summary["pred_stats"]["mean"],
            "pred_std": tuned_summary["pred_stats"]["std"],
        },
        "delta": {
            "psnr": tuned_summary["psnr"] - baseline_summary["psnr"],
            "ssim": tuned_summary["ssim"] - baseline_summary["ssim"],
            "sam": tuned_summary["sam"] - baseline_summary["sam"],
            "pred_mean": tuned_summary["pred_stats"]["mean"] - baseline_summary["pred_stats"]["mean"],
            "pred_std": tuned_summary["pred_stats"]["std"] - baseline_summary["pred_stats"]["std"],
        },
    }

    with open(out_dir / "contrast_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "loss": "50*L1 + 20*(1-SSIM) + 10*Edge + 2*(SAM/180)",
                    "init_checkpoint": args.init_checkpoint,
                    "train_pairs": len(train_pairs),
                    "val_pairs": len(val_pairs),
                },
                "baseline": baseline_summary,
                "best_epoch": best_epoch,
                "best_ssim": best_ssim,
                "final_best": tuned_summary,
                "comparison": comparison,
                "time_sec": time.time() - start,
            },
            f,
            indent=2,
        )

    report_lines = []
    report_lines.append("# Contrast Experiment Report")
    report_lines.append("")
    report_lines.append("## Setup")
    report_lines.append(f"- Initialized from: `{args.init_checkpoint}`")
    report_lines.append(f"- Output dir: `{args.out_dir}`")
    report_lines.append(f"- Epochs trained: {args.epochs}")
    report_lines.append("- Loss: 50*L1 + 20*(1-SSIM) + 10*EdgeLoss + 2*(SAM/180)")
    report_lines.append("- EdgeLoss: Sobel gradient magnitude L1")
    report_lines.append("")
    report_lines.append("## Metric Comparison (baseline vs contrast)")
    report_lines.append("| metric | baseline | contrast_best | delta |")
    report_lines.append("|---|---:|---:|---:|")
    report_lines.append("| PSNR | {b:.6f} | {n:.6f} | {d:.6f} |".format(b=comparison["baseline"]["psnr"], n=comparison["contrast_best_after_5_epochs"]["psnr"], d=comparison["delta"]["psnr"]))
    report_lines.append("| SSIM | {b:.6f} | {n:.6f} | {d:.6f} |".format(b=comparison["baseline"]["ssim"], n=comparison["contrast_best_after_5_epochs"]["ssim"], d=comparison["delta"]["ssim"]))
    report_lines.append("| SAM | {b:.6f} | {n:.6f} | {d:.6f} |".format(b=comparison["baseline"]["sam"], n=comparison["contrast_best_after_5_epochs"]["sam"], d=comparison["delta"]["sam"]))
    report_lines.append("| pred mean | {b:.6f} | {n:.6f} | {d:.6f} |".format(b=comparison["baseline"]["pred_mean"], n=comparison["contrast_best_after_5_epochs"]["pred_mean"], d=comparison["delta"]["pred_mean"]))
    report_lines.append("| pred std | {b:.6f} | {n:.6f} | {d:.6f} |".format(b=comparison["baseline"]["pred_std"], n=comparison["contrast_best_after_5_epochs"]["pred_std"], d=comparison["delta"]["pred_std"]))
    report_lines.append("")
    report_lines.append("## Prediction/Target Stats (contrast best)")
    report_lines.append("- Prediction: mean={m:.6f}, std={s:.6f}, min={mn:.6f}, max={mx:.6f}".format(
        m=tuned_summary["pred_stats"]["mean"],
        s=tuned_summary["pred_stats"]["std"],
        mn=tuned_summary["pred_stats"]["min"],
        mx=tuned_summary["pred_stats"]["max"],
    ))
    report_lines.append("- Target: mean={m:.6f}, std={s:.6f}".format(
        m=tuned_summary["target_stats"]["mean"],
        s=tuned_summary["target_stats"]["std"],
    ))
    report_lines.append("")
    report_lines.append("## Training Loss Components")
    report_lines.append("- See `loss_components_per_epoch.csv` for L1, SSIM, SAM, EdgeLoss and weighted terms per epoch.")
    report_lines.append("")
    report_lines.append("## Visual Comparison")
    report_lines.append("- 20 side-by-side files saved in `side_by_side_20/` as: input | baseline_pred | contrast_pred | target")
    report_lines.append("")
    report_lines.append("## Success-Criteria Check")
    report_lines.append("- pred std approaches target std: {ok}".format(ok=tuned_summary["pred_stats"]["std"] > baseline_summary["pred_stats"]["std"]))
    report_lines.append("- pred mean approaches target mean: {ok}".format(ok=abs(tuned_summary["pred_stats"]["mean"] - tuned_summary["target_stats"]["mean"]) < abs(baseline_summary["pred_stats"]["mean"] - tuned_summary["target_stats"]["mean"])))
    report_lines.append("- no dark low-variance collapse: {ok}".format(ok=tuned_summary["pred_stats"]["std"] > baseline_summary["pred_stats"]["std"] and tuned_summary["pred_stats"]["mean"] > baseline_summary["pred_stats"]["mean"]))

    (out_dir / "contrast_experiment_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("Saved:", out_dir / "contrast_experiment_report.md")
    print("Saved:", out_dir / "contrast_experiment_summary.json")
    print("Saved:", out_dir / "loss_components_per_epoch.csv")
    print("Saved side-by-side visuals in:", out_dir / "side_by_side_20")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cloudy", required=True)
    p.add_argument("--clear", required=True)
    p.add_argument("--init_checkpoint", default="checkpoints_nafnet/full_dataset_training/best_ssim.pth")
    p.add_argument("--out_dir", default="checkpoints_nafnet/contrast_experiment")
    p.add_argument("--max_pairs", type=int, default=5891)
    p.add_argument("--train_pairs", type=int, default=5000)
    p.add_argument("--val_pairs", type=int, default=891)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    raise SystemExit(run(args))
