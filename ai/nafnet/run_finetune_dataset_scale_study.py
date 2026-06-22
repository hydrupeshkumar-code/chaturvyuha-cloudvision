#!/usr/bin/env python
"""
Run dataset-scale study experiments for validated NAFNet reconstruction.

This runner executes controlled experiments for 5,000 / 10,000 / 15,000 strict curated pairs,
starting from the preserved `best_ssim.pth` baseline checkpoint.

Outputs per experiment:
  experiment_5000/
    best_ssim.pth
    best_psnr.pth
    metrics_per_epoch.csv
    validation_report.md
    visual_grid_5000.png

And summary:
  dataset_scale_study.md
"""

import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(root))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as sk_ssim
from PIL import Image

from ai.nafnet.dataset import NAFDataset
from ai.nafnet.model import NAFNetWrapper
from ai.nafnet import metrics as naf_metrics
from ai.nafnet.select_top_strict_pairs import run as select_pairs_run

_matplotlib_available = None
plt = None


def _ensure_matplotlib():
    global _matplotlib_available, plt
    if _matplotlib_available is not None:
        return _matplotlib_available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt_module
        plt = plt_module
        _matplotlib_available = True
    except Exception as exc:
        print(f"Warning: matplotlib unavailable, skipping plots: {exc}")
        _matplotlib_available = False
    return _matplotlib_available


BASELINE_CHECKPOINT = Path("checkpoints_nafnet/strict_curated_training/best_ssim.pth")
RAW_RANKING_CSV = Path("checkpoints_nafnet/raw_pair_audit/raw_pair_ranking.csv")
CURATED_STRICT_DIR = Path("checkpoints_nafnet/raw_pair_audit")
CURATED_STRICT_SOURCE = CURATED_STRICT_DIR / "top_5000_curated_strict_pairs.csv"
STATS_JSON = Path("tmp_stats/band_statistics.json")
BASELINE_METRICS = {
    "psnr": 35.03,
    "ssim": 0.9015,
    "rmse": 0.0205,
    "sam": 4.86,
}
TERMINATION_THRESHOLDS = {
    "ssim": 0.85,
    "psnr": 32.0,
    "sam": 7.5,
}

EPOCH1_SUCCESS_GATE = {
    "ssim": 0.86,
    "psnr": 33.0,
    "sam": 7.0,
}

EXPERIMENT_SIZES = [5000, 10000, 15000]
MAX_EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-6
LOSS_WEIGHTS = {"l1": 0.5, "ssim": 0.3, "edge": 0.2}
SEED = 42


def _as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _load_fallback_curated_pairs(size):
    fallback_csv = CURATED_STRICT_DIR / f"top_{size}_curated_strict_pairs.csv"
    if fallback_csv.exists():
        pairs = _load_pairs(fallback_csv)
        return pairs if pairs else None

    if not CURATED_STRICT_SOURCE.exists():
        return None

    pairs = _load_pairs(CURATED_STRICT_SOURCE)
    if len(pairs) < size:
        return None
    pairs = pairs[:size]
    return pairs if pairs else None


def _save_png(arr_hwc, path: Path):
    img = np.clip(arr_hwc * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(str(path))


def _composite_loss(pred, target):
    l1_loss = F.l1_loss(pred, target)

    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    ssim_vals = []
    n = pred_np.shape[0]
    for i in range(n):
        pp = np.transpose(pred_np[i], (1, 2, 0))
        yy = np.transpose(tgt_np[i], (1, 2, 0))
        try:
            ssim_vals.append(np.mean([sk_ssim(yy[:, :, c], pp[:, :, c], data_range=1.0) for c in range(yy.shape[2])]))
        except Exception:
            ssim_vals.append(0.0)
    ssim_loss = 1.0 - float(np.mean(ssim_vals)) if ssim_vals else 1.0
    ssim_loss = torch.tensor(ssim_loss, device=pred.device, dtype=pred.dtype)

    c = pred.shape[1]
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3) / 8.0
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx_pred = F.conv2d(pred, kx, padding=1, groups=c)
    gy_pred = F.conv2d(pred, ky, padding=1, groups=c)
    gx_tgt = F.conv2d(target, kx, padding=1, groups=c)
    gy_tgt = F.conv2d(target, ky, padding=1, groups=c)
    mag_pred = torch.sqrt(gx_pred * gx_pred + gy_pred * gy_pred + 1e-8)
    mag_tgt = torch.sqrt(gx_tgt * gx_tgt + gy_tgt * gy_tgt + 1e-8)
    edge_loss = F.l1_loss(mag_pred, mag_tgt)

    loss = LOSS_WEIGHTS["l1"] * l1_loss + LOSS_WEIGHTS["ssim"] * ssim_loss + LOSS_WEIGHTS["edge"] * edge_loss
    return loss, {"l1": float(l1_loss.detach().cpu().item()), "ssim": float(ssim_loss.detach().cpu().item()), "edge": float(edge_loss.detach().cpu().item())}


def _eval_checkpoint(model, val_loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            p = model(x)[0].cpu().numpy()
            p = np.clip(np.transpose(p, (1, 2, 0)), 0.0, 1.0)
            tgt = np.transpose(y[0].numpy(), (1, 2, 0))
            rows.append({
                "psnr": naf_metrics.psnr(tgt, p),
                "ssim": naf_metrics.ssim(tgt, p),
                "sam": naf_metrics.sam(tgt, p),
            })
    return {
        "psnr": float(np.mean([r["psnr"] for r in rows])) if rows else 0.0,
        "ssim": float(np.mean([r["ssim"] for r in rows])) if rows else 0.0,
        "sam": float(np.mean([r["sam"] for r in rows])) if rows else 0.0,
    }


def _save_visual_audit(model, val_ds, device, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        x, y = val_ds[0]
        inp = np.transpose(x.numpy(), (1, 2, 0))
        tgt = np.transpose(y.numpy(), (1, 2, 0))
        pred = model(x.unsqueeze(0).to(device))[0].cpu().numpy()
        pred = np.clip(np.transpose(pred, (1, 2, 0)), 0.0, 1.0)
        _save_png(inp, out_dir / "input.png")
        _save_png(pred, out_dir / "prediction.png")
        _save_png(tgt, out_dir / "target.png")
        comp = np.concatenate([inp, pred, tgt], axis=1)
        _save_png(comp, out_dir / "comparison.png")
    return inp, pred, tgt


def _plot_visual_grid(metrics_history, out_path, title):
    if not _ensure_matplotlib():
        print(f"Skipping plot generation because matplotlib is not available: {out_path}")
        return
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sizes = [h["dataset_size"] for h in metrics_history]
    ssims = [h["best_ssim"] for h in metrics_history]
    psnrs = [h["best_psnr"] for h in metrics_history]
    sams = [h["best_sam"] for h in metrics_history]
    axs[0].plot(sizes, ssims, marker="o")
    axs[0].set_title("Best SSIM")
    axs[0].set_xlabel("Dataset Size")
    axs[0].grid(True)
    axs[1].plot(sizes, psnrs, marker="o")
    axs[1].set_title("Best PSNR")
    axs[1].set_xlabel("Dataset Size")
    axs[1].grid(True)
    axs[2].plot(sizes, sams, marker="o")
    axs[2].set_title("Best SAM")
    axs[2].set_xlabel("Dataset Size")
    axs[2].grid(True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_experiment(size, out_root, stats, device, epochs, batch_size, lr):
    exp_dir = out_root / f"experiment_{size}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    select_out = exp_dir / "selection"
    select_out.mkdir(parents=True, exist_ok=True)
    select_args = argparse.Namespace(
        ranking_csv=str(RAW_RANKING_CSV),
        out_dir=str(select_out),
        top_n=size,
        min_ssim=0.0,
        min_psnr=0.0,
        max_sam=100.0,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=SEED,
    )
    print(f"Selecting top {size} strict pairs...")
    select_pairs_run(select_args)

    pair_csv = select_out / f"top_{size}_strict_train_list.csv"
    if not pair_csv.exists():
        fallback_pairs = _load_fallback_curated_pairs(size)
        if fallback_pairs is None:
            raise FileNotFoundError(f"Expected selection CSV not found: {pair_csv}")
        print(f"Selection CSV missing, using curated fallback source: {CURATED_STRICT_SOURCE}")

    pairs = _load_pairs(pair_csv)
    if len(pairs) != size:
        fallback_pairs = _load_fallback_curated_pairs(size)
        if fallback_pairs is not None and len(fallback_pairs) == size:
            pairs = fallback_pairs
            print(f"Loaded {len(pairs)} curated fallback pairs for experiment {size}")
        else:
            raise RuntimeError(f"Expected {size} pairs but loaded {len(pairs)}")

    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    state = torch.load(BASELINE_CHECKPOINT, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    # Use a small validation split for regression detection, but keep most data for training
    val_count = max(1, int(len(pairs) * 0.1))
    train_pairs = pairs[:-val_count]
    val_pairs = pairs[-val_count:]
    if len(train_pairs) == 0:
        raise RuntimeError("Not enough pairs for training after validation split")

    val_ds = NAFDataset(val_pairs, stats["p1"], stats["p99"], patch_size=None, augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    baseline_eval = _eval_checkpoint(model, val_loader, device)

    train_ds = NAFDataset(train_pairs, stats["p1"], stats["p99"], patch_size=(128, 128), augment=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None

    best_ssim = baseline_eval["ssim"]
    best_psnr = baseline_eval["psnr"]
    best_sam = baseline_eval["sam"]
    best_ssim_epoch = None
    best_psnr_epoch = None
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    p = model(x)
                    loss, comps = _composite_loss(p, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                p = model(x)
                loss, comps = _composite_loss(p, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        eval_metrics = _eval_checkpoint(model, val_loader, device)

        if eval_metrics["ssim"] < TERMINATION_THRESHOLDS["ssim"] or eval_metrics["psnr"] < TERMINATION_THRESHOLDS["psnr"] or eval_metrics["sam"] > TERMINATION_THRESHOLDS["sam"]:
            status = "REGRESSION_DETECTED"
            print(f"Termination triggered at epoch {epoch} for experiment {size}:", eval_metrics)
            history.append({"epoch": epoch, "train_loss": avg_train_loss, **eval_metrics, "status": status})
            break
        if epoch == 1 and (
            eval_metrics["ssim"] <= EPOCH1_SUCCESS_GATE["ssim"]
            or eval_metrics["psnr"] <= EPOCH1_SUCCESS_GATE["psnr"]
            or eval_metrics["sam"] >= EPOCH1_SUCCESS_GATE["sam"]
        ):
            status = "EPOCH1_GATE_FAILED"
            print(f"Epoch-1 success gate failed for experiment {size}:", eval_metrics)
            history.append({"epoch": epoch, "train_loss": avg_train_loss, **eval_metrics, "status": status})
            break
        else:
            status = "OK"

        if eval_metrics["ssim"] > best_ssim:
            best_ssim = eval_metrics["ssim"]
            best_ssim_epoch = epoch
            torch.save(model.state_dict(), exp_dir / "best_ssim.pth")
        if eval_metrics["psnr"] > best_psnr:
            best_psnr = eval_metrics["psnr"]
            best_psnr_epoch = epoch
            torch.save(model.state_dict(), exp_dir / "best_psnr.pth")
        if eval_metrics["sam"] < best_sam:
            best_sam = eval_metrics["sam"]

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_psnr": eval_metrics["psnr"],
            "val_ssim": eval_metrics["ssim"],
            "val_sam": eval_metrics["sam"],
            "status": status,
        })

        print(f"Experiment {size} epoch {epoch}: PSNR={eval_metrics['psnr']:.4f}, SSIM={eval_metrics['ssim']:.4f}, SAM={eval_metrics['sam']:.4f}")

    total_time = time.time() - start_time
    csv_path = exp_dir / "metrics_per_epoch.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    summary = {
        "dataset_size": size,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "best_sam": best_sam,
        "best_psnr_epoch": best_psnr_epoch,
        "best_ssim_epoch": best_ssim_epoch,
        "baseline_psnr": baseline_eval["psnr"],
        "baseline_ssim": baseline_eval["ssim"],
        "baseline_sam": baseline_eval["sam"],
        "total_time_sec": total_time,
        "status": status,
    }

    report_path = exp_dir / "validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Experiment {size} Validation Report\n\n")
        f.write(f"Dataset size: {size}\n")
        f.write(f"Baseline PSNR: {baseline_eval['psnr']:.4f}\n")
        f.write(f"Baseline SSIM: {baseline_eval['ssim']:.4f}\n")
        f.write(f"Baseline SAM: {baseline_eval['sam']:.4f}\n")
        f.write("\n")
        f.write(f"Best PSNR: {best_psnr:.4f} {f'(epoch {best_psnr_epoch})' if best_psnr_epoch is not None else '(baseline)'}\n")
        f.write(f"Best SSIM: {best_ssim:.4f} {f'(epoch {best_ssim_epoch})' if best_ssim_epoch is not None else '(baseline)'}\n")
        f.write(f"Best SAM: {best_sam:.4f}\n")
        f.write(f"Total training time (sec): {total_time:.1f}\n")
        f.write(f"Final experiment status: {status}\n")

    _, pred, tgt = _save_visual_audit(model, val_ds, device, exp_dir)
    grid_path = out_root / f"visual_grid_{size}.png"
    _plot_visual_grid([summary], grid_path, f"Dataset Scale Study {size}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run dataset-scale study experiments for NAFNet.")
    parser.add_argument("--output_dir", type=str, default="checkpoints_nafnet/dataset_scale_study", help="Experiment root output directory")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Maximum epochs per experiment")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=LR, help="AdamW learning rate")
    parser.add_argument("--experiment_sizes", type=int, nargs="+", default=EXPERIMENT_SIZES, help="Experiment dataset sizes to run")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {BASELINE_CHECKPOINT}")
    if not RAW_RANKING_CSV.exists():
        raise FileNotFoundError(f"Ranking CSV not found: {RAW_RANKING_CSV}")

    stats = {"p1": [0.0, 0.0, 0.0], "p99": [6000.0, 6000.0, 6000.0]}
    if STATS_JSON.exists():
        try:
            with open(STATS_JSON, "r", encoding="utf-8") as f:
                j = json.load(f)
                stats["p1"] = j.get("p1", stats["p1"])
                stats["p99"] = j.get("p99", stats["p99"])
        except Exception:
            print(f"Warning: could not read stats JSON, using defaults: {STATS_JSON}")
    else:
        print(f"Warning: stats JSON not found, using default normalization stats: {STATS_JSON}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _set_seed(SEED)

    summaries = []
    for size in args.experiment_sizes:
        summary = _run_experiment(size, out_root, stats, device, args.epochs, args.batch_size, args.lr)
        summaries.append(summary)

    # Add a combined summary plot for final comparison.
    if summaries:
        _plot_visual_grid(summaries, out_root / "dataset_scale_study_grid.png", "Dataset Scale Study Summary")

    summary_path = out_root / "dataset_scale_study.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Scale Study\n\n")
        f.write("| Dataset Size | Best PSNR | Best SSIM | Best SAM | Training Time (sec) | Status |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.write(f"| 2365 (baseline) | {BASELINE_METRICS['psnr']:.2f} | {BASELINE_METRICS['ssim']:.4f} | {BASELINE_METRICS['sam']:.2f} | - | BASELINE |\n")
        for s in summaries:
            f.write(f"| {s['dataset_size']} | {s['best_psnr']:.2f} | {s['best_ssim']:.4f} | {s['best_sam']:.2f} | {s['total_time_sec']:.1f} | {s['status']} |\n")
        f.write("\n## Decision Logic\n")
        best = max([s for s in summaries if s['status'] != 'REGRESSION_DETECTED'], key=lambda x: (x['best_ssim'], -x['best_sam'], x['best_psnr']), default=None)
        if best is None or best['best_ssim'] <= BASELINE_METRICS['ssim']:
            f.write("No experiment beat the validated baseline. Keep original best_ssim.pth.\n")
        else:
            f.write(f"Promote experiment_{best['dataset_size']} as best_ssim_final.pth.\n")

    print(f"Summary written to {summary_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
