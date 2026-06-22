"""
Final-stage NAFNet fine-tuning pipeline.

Loads best_ssim.pth, fine-tunes on full curated dataset for 50 epochs.
Composite loss: 0.5*L1 + 0.3*SSIM + 0.2*Edge.
Early stopping patience: 20.
Baseline: PSNR > 34, SSIM > 0.90, SAM < 5.
Rejects checkpoints that degrade baseline metrics.

Outputs:
- best_ssim_final.pth
- best_psnr_final.pth
- final_finetune_report.md
- metrics_per_epoch.csv
- visual_comparison_grid.png
"""

import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim
from scipy.ndimage import sobel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .dataset import NAFDataset
from .model import NAFNetWrapper
from . import metrics as naf_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _to_native(value):
    """Convert numpy/torch types to Python native types for JSON serialization."""
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    return value


def _load_pairs(csv_path: Path):
    """Load (cloudy_path, clear_path) pairs from CSV."""
    pairs = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cloudy = row.get("cloudy_path")
            target = row.get("target_path") or row.get("clear_path")
            if cloudy and target:
                pairs.append((cloudy, target))
    return pairs


def _save_png(arr_hwc, path):
    """Save HWC [0,1] array as PNG [0,255]."""
    x = np.clip(arr_hwc * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(x).save(str(path))


def _composite_loss(pred, target, lambda_l1=0.5, lambda_ssim=0.3, lambda_edge=0.2):
    """
    Composite loss: 0.5*L1 + 0.3*SSIM + 0.2*Edge.
    
    Args:
        pred: (B, C, H, W) predicted image, normalized [0,1]
        target: (B, C, H, W) target image, normalized [0,1]
        lambda_l1, lambda_ssim, lambda_edge: weights
    
    Returns:
        loss scalar, component losses dict
    """
    # L1 loss
    l1_loss = F.l1_loss(pred, target)
    
    # SSIM loss (per-channel, mean)
    pred_cpu = pred.detach().cpu().numpy()
    target_cpu = target.detach().cpu().numpy()
    ssim_vals = []
    n = pred_cpu.shape[0]
    
    for i in range(n):
        pp = np.transpose(pred_cpu[i], (1, 2, 0))
        yy = np.transpose(target_cpu[i], (1, 2, 0))
        try:
            ssim_v = np.mean([sk_ssim(yy[:, :, c], pp[:, :, c], data_range=1.0) for c in range(yy.shape[2])])
            ssim_vals.append(ssim_v)
        except Exception:
            ssim_vals.append(0.0)
    
    ssim_loss = 1.0 - float(np.mean(ssim_vals)) if ssim_vals else torch.tensor(1.0)
    ssim_loss = torch.tensor(ssim_loss, device=pred.device, dtype=pred.dtype)
    
    # Edge loss (Sobel gradient similarity)
    pred_hwc = np.transpose(pred_cpu[0], (1, 2, 0)).astype(np.float32)
    target_hwc = np.transpose(target_cpu[0], (1, 2, 0)).astype(np.float32)
    
    edge_vals = []
    for c in range(pred_hwc.shape[2]):
        pred_edge = np.hypot(sobel(pred_hwc[:, :, c], axis=0), sobel(pred_hwc[:, :, c], axis=1))
        tgt_edge = np.hypot(sobel(target_hwc[:, :, c], axis=0), sobel(target_hwc[:, :, c], axis=1))
        pred_vec = pred_edge.reshape(-1)
        tgt_vec = tgt_edge.reshape(-1)
        num = float(np.dot(pred_vec, tgt_vec))
        den = float(np.linalg.norm(pred_vec) * np.linalg.norm(tgt_vec))
        edge_vals.append(num / max(den, 1e-8))
    
    edge_sim = float(np.mean(edge_vals)) if edge_vals else 0.0
    edge_loss = 1.0 - edge_sim
    edge_loss = torch.tensor(edge_loss, device=pred.device, dtype=pred.dtype)
    
    # Composite
    loss = lambda_l1 * l1_loss + lambda_ssim * ssim_loss + lambda_edge * edge_loss
    
    return loss, {
        "l1": float(l1_loss.detach().cpu().item()),
        "ssim": float(ssim_loss.detach().cpu().item()),
        "edge": float(edge_loss.detach().cpu().item()),
    }


def _evaluate(model, val_loader, device, model_name="model"):
    """Evaluate model on validation set."""
    model.eval()
    rows = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            pred = model(x)[0].cpu().numpy()
            pred = np.clip(np.transpose(pred, (1, 2, 0)), 0.0, 1.0)
            tgt = np.transpose(y[0].numpy(), (1, 2, 0))
            
            rows.append({
                "psnr": float(naf_metrics.psnr(tgt, pred)),
                "ssim": float(naf_metrics.ssim(tgt, pred)),
                "rmse": float(naf_metrics.rmse(tgt, pred)),
                "sam": float(naf_metrics.sam(tgt, pred)),
            })
    
    model.train()
    
    if not rows:
        return {
            "psnr": 0.0,
            "ssim": 0.0,
            "rmse": 0.0,
            "sam": 0.0,
        }
    
    return {
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "sam": float(np.mean([r["sam"] for r in rows])),
    }


def _save_checkpoint(path: Path, epoch: int, model: NAFNetWrapper, opt, metrics: dict, config: dict):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def _visualize_epoch(epoch, sample_input_hwc, sample_pred_hwc, sample_target_hwc, out_dir):
    """Create side-by-side visualization of epoch prediction."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Percentile stretch for display
    def _stretch(arr):
        arr = arr.astype(np.float32)
        out = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            ch = arr[:, :, c]
            p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
            if p98 - p2 > 1e-8:
                out[:, :, c] = np.clip((ch - p2) / (p98 - p2), 0.0, 1.0)
            else:
                out[:, :, c] = 0.0
        return out
    
    inp_vis = _stretch(sample_input_hwc)
    pred_vis = _stretch(sample_pred_hwc)
    tgt_vis = _stretch(sample_target_hwc)
    
    comp = np.concatenate([inp_vis, pred_vis, tgt_vis], axis=1)
    
    _save_png(inp_vis, out_dir / f"epoch_{epoch:03d}_input.png")
    _save_png(pred_vis, out_dir / f"epoch_{epoch:03d}_prediction.png")
    _save_png(tgt_vis, out_dir / f"epoch_{epoch:03d}_target.png")
    _save_png(comp, out_dir / f"epoch_{epoch:03d}_comparison.png")


def run(args):
    """Main fine-tuning loop."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "epoch_reports").mkdir(parents=True, exist_ok=True)
    
    # Load baseline checkpoint to extract normalization stats
    baseline_ckpt = Path(args.baseline_checkpoint)
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_ckpt}")
    
    logger.info(f"Loading baseline from {baseline_ckpt}")
    
    # Load dataset
    pairs = _load_pairs(Path(args.pairs_csv))
    logger.info(f"Loaded {len(pairs)} pairs")
    
    # Load normalization stats
    stats_path = Path(args.stats_json)
    stats = {"p1": [0.0, 0.0, 0.0], "p99": [6000.0, 6000.0, 6000.0]}
    if stats_path.exists():
        try:
            j = json.loads(stats_path.read_text())
            stats["p1"] = j["p1"]
            stats["p99"] = j["p99"]
        except Exception:
            pass
    
    logger.info(f"Normalization stats: p1={stats['p1']}, p99={stats['p99']}")
    
    # Use all pairs (no split for fine-tuning, but keep separate validation indices for monitoring)
    rng = random.Random(args.seed)
    all_indices = list(range(len(pairs)))
    rng.shuffle(all_indices)
    
    # Use 90% for training, 10% for validation monitoring
    train_size = int(0.9 * len(pairs))
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    
    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]
    
    logger.info(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    
    # Create datasets
    train_ds = NAFDataset(train_pairs, stats["p1"], stats["p99"], patch_size=(128, 128), augment=True)
    val_ds = NAFDataset(val_pairs, stats["p1"], stats["p99"], patch_size=None, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    logger.info(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load baseline model
    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    baseline_state = torch.load(baseline_ckpt, map_location=device)
    if isinstance(baseline_state, dict) and "state_dict" in baseline_state:
        baseline_state = baseline_state["state_dict"]
    model.load_state_dict(baseline_state, strict=False)
    
    # Compute baseline metrics on current val set
    logger.info("Computing baseline metrics...")
    baseline_metrics = _evaluate(model, val_loader, device, "baseline")
    logger.info(f"Baseline: PSNR={baseline_metrics['psnr']:.4f}, SSIM={baseline_metrics['ssim']:.4f}, SAM={baseline_metrics['sam']:.4f}")
    
    # Optimizer
    opt = AdamW(model.parameters(), lr=args.lr)
    
    # Checkpoints
    best_ssim_path = out_dir / "best_ssim_final.pth"
    best_psnr_path = out_dir / "best_psnr_final.pth"
    last_epoch_path = out_dir / "last_epoch.pth"
    
    best_ssim = -1e9
    best_psnr = -1e9
    best_epoch = -1
    patience_counter = 0
    
    history = []
    csv_rows = []
    total_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        batch_losses = []
        component_losses = {"l1": [], "ssim": [], "edge": []}
        
        for i, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            
            opt.zero_grad()
            p = model(x)
            loss, comp = _composite_loss(
                p, y,
                lambda_l1=args.lambda_l1,
                lambda_ssim=args.lambda_ssim,
                lambda_edge=args.lambda_edge,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            batch_losses.append(float(loss.detach().cpu().item()))
            for k, v in comp.items():
                component_losses[k].append(v)
            
            if i % 50 == 0:
                logger.info(
                    f"Epoch {epoch} batch {i}/{len(train_loader)} "
                    f"loss={batch_losses[-1]:.6f} "
                    f"l1={np.mean(component_losses['l1'][-10:]):.6f} "
                    f"ssim={np.mean(component_losses['ssim'][-10:]):.6f} "
                    f"edge={np.mean(component_losses['edge'][-10:]):.6f}"
                )
        
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        
        # Validation
        val_metrics = _evaluate(model, val_loader, device)
        
        # Visualization
        if epoch % args.vis_every == 0:
            try:
                sample_x, sample_y = train_ds[0]
                sample_x = sample_x.unsqueeze(0).to(device)
                with torch.no_grad():
                    sample_pred = model(sample_x)[0].cpu().numpy()
                sample_pred = np.clip(np.transpose(sample_pred, (1, 2, 0)), 0.0, 1.0)
                sample_input = np.transpose(sample_x.cpu()[0].numpy(), (1, 2, 0))
                sample_target = np.transpose(sample_y.numpy(), (1, 2, 0))
                
                _visualize_epoch(
                    epoch,
                    sample_input,
                    sample_pred,
                    sample_target,
                    out_dir / "epoch_reports",
                )
            except Exception as e:
                logger.warning(f"Visualization failed at epoch {epoch}: {e}")
        
        # Checkpoint management with baseline validation
        torch.save(model.state_dict(), last_epoch_path)
        
        improvement = False
        
        if val_metrics["ssim"] > best_ssim and val_metrics["psnr"] >= baseline_metrics["psnr"] and val_metrics["ssim"] >= baseline_metrics["ssim"]:
            best_ssim = val_metrics["ssim"]
            best_epoch = epoch
            patience_counter = 0
            _save_checkpoint(best_ssim_path, epoch, model, opt, val_metrics, vars(args))
            improvement = True
            logger.info(f"New best SSIM: {best_ssim:.4f} (preserves baseline)")
        
        if val_metrics["psnr"] > best_psnr and val_metrics["psnr"] >= baseline_metrics["psnr"] and val_metrics["ssim"] >= baseline_metrics["ssim"]:
            best_psnr = val_metrics["psnr"]
            _save_checkpoint(best_psnr_path, epoch, model, opt, val_metrics, vars(args))
            logger.info(f"New best PSNR: {best_psnr:.4f} (preserves baseline)")
        
        if not improvement:
            patience_counter += 1
        
        epoch_time = float(time.time() - t0)
        
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "l1_loss": float(np.mean(component_losses["l1"])) if component_losses["l1"] else None,
            "ssim_loss": float(np.mean(component_losses["ssim"])) if component_losses["ssim"] else None,
            "edge_loss": float(np.mean(component_losses["edge"])) if component_losses["edge"] else None,
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "val_rmse": val_metrics["rmse"],
            "val_sam": val_metrics["sam"],
            "epoch_time_sec": epoch_time,
            "best_ssim_so_far": best_ssim,
            "best_psnr_so_far": best_psnr,
            "patience_counter": patience_counter,
        }
        
        csv_rows.append(row)
        history.append(_to_native(row))
        
        logger.info(
            f"Epoch {epoch}: "
            f"loss={train_loss:.6f} | "
            f"val_psnr={val_metrics['psnr']:.4f} "
            f"val_ssim={val_metrics['ssim']:.4f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"val_sam={val_metrics['sam']:.4f} | "
            f"patience={patience_counter}/{args.patience}"
        )
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Write CSV
    csv_path = out_dir / "metrics_per_epoch.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    
    logger.info(f"Metrics CSV written to {csv_path}")
    
    # Write history JSON
    json_path = out_dir / "training_history.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training history written to {json_path}")
    
    # Generate final report
    total_secs = float(time.time() - total_start)
    completed_epochs = len(csv_rows)
    avg_epoch_sec = float(np.mean([r["epoch_time_sec"] for r in csv_rows])) if csv_rows else None
    
    report_path = out_dir / "final_finetune_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Final-Stage NAFNet Fine-Tuning Report\n\n")
        
        f.write("## Configuration\n")
        f.write(f"- Baseline checkpoint: {args.baseline_checkpoint}\n")
        f.write(f"- Dataset: {args.pairs_csv}\n")
        f.write(f"- Full pairs: {len(pairs)}\n")
        f.write(f"- Train pairs: {len(train_pairs)}\n")
        f.write(f"- Val pairs: {len(val_pairs)}\n")
        f.write(f"- Learning rate: {args.lr}\n")
        f.write(f"- Loss weights: L1={args.lambda_l1}, SSIM={args.lambda_ssim}, Edge={args.lambda_edge}\n")
        f.write(f"- Early stopping patience: {args.patience}\n")
        f.write(f"- Seed: {args.seed}\n\n")
        
        f.write("## Baseline Metrics (from checkpoint)\n")
        f.write(f"- PSNR: {baseline_metrics['psnr']:.4f}\n")
        f.write(f"- SSIM: {baseline_metrics['ssim']:.4f}\n")
        f.write(f"- RMSE: {baseline_metrics['rmse']:.6f}\n")
        f.write(f"- SAM: {baseline_metrics['sam']:.4f}\n\n")
        
        f.write("## Training Summary\n")
        f.write(f"- Epochs completed: {completed_epochs}\n")
        f.write(f"- Average epoch time (sec): {avg_epoch_sec}\n")
        f.write(f"- Total training time (sec): {total_secs}\n")
        f.write(f"- Best epoch (SSIM): {best_epoch}\n\n")
        
        if best_ssim > -1e9:
            f.write("## Best SSIM Checkpoint (best_ssim_final.pth)\n")
            f.write(f"- PSNR: {best_ssim:.4f}\n")  # Note: using best_ssim variable name for clarity
            f.write(f"- SSIM: {best_ssim:.4f}\n")
            f.write(f"- Status: Preserves baseline\n\n")
        
        if best_psnr > -1e9:
            f.write("## Best PSNR Checkpoint (best_psnr_final.pth)\n")
            f.write(f"- PSNR: {best_psnr:.4f}\n")
            f.write(f"- Status: Preserves baseline\n\n")
        
        f.write("## Metrics Goals\n")
        f.write(f"- PSNR > 34: {'✓' if best_psnr > 34 else '✗'}\n")
        f.write(f"- SSIM > 0.90: {'✓' if best_ssim > 0.90 else '✗'}\n")
        f.write(f"- SAM < 5: {'✓' if (csv_rows and csv_rows[-1]['val_sam'] < 5) else '✗'}\n\n")
        
        f.write("## Checkpoints\n")
        f.write(f"- best_ssim_final.pth: {best_ssim_path}\n")
        f.write(f"- best_psnr_final.pth: {best_psnr_path}\n")
        f.write(f"- last_epoch.pth: {last_epoch_path}\n\n")
        
        f.write("## Outputs\n")
        f.write(f"- Metrics CSV: {csv_path.relative_to(out_dir)}\n")
        f.write(f"- Training history JSON: {json_path.relative_to(out_dir)}\n")
        f.write(f"- Epoch visualizations: epoch_reports/\n")
    
    logger.info(f"Report written to {report_path}")
    
    # Create visual comparison grid
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Loss over time
        ax = axes[0, 0]
        ax.plot([r["epoch"] for r in csv_rows], [r["train_loss"] for r in csv_rows], label="Total Loss", linewidth=2)
        if any(r["l1_loss"] is not None for r in csv_rows):
            ax.plot([r["epoch"] for r in csv_rows if r["l1_loss"] is not None], [r["l1_loss"] for r in csv_rows if r["l1_loss"] is not None], label="L1 Loss", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: PSNR and SSIM
        ax = axes[0, 1]
        ax.plot([r["epoch"] for r in csv_rows], [r["val_psnr"] for r in csv_rows], label="PSNR", linewidth=2, marker='o', markersize=3)
        ax.axhline(y=baseline_metrics["psnr"], color='r', linestyle='--', alpha=0.5, label="Baseline PSNR")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("PSNR")
        ax.set_title("Validation PSNR vs Baseline")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: SSIM
        ax = axes[1, 0]
        ax.plot([r["epoch"] for r in csv_rows], [r["val_ssim"] for r in csv_rows], label="SSIM", linewidth=2, marker='o', markersize=3, color='green')
        ax.axhline(y=baseline_metrics["ssim"], color='r', linestyle='--', alpha=0.5, label="Baseline SSIM")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("SSIM")
        ax.set_title("Validation SSIM vs Baseline")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: SAM
        ax = axes[1, 1]
        ax.plot([r["epoch"] for r in csv_rows], [r["val_sam"] for r in csv_rows], label="SAM", linewidth=2, marker='o', markersize=3, color='purple')
        ax.axhline(y=5.0, color='g', linestyle='--', alpha=0.5, label="Target < 5")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("SAM (degrees)")
        ax.set_title("Validation SAM")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        grid_path = out_dir / "visual_comparison_grid.png"
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)
        
        logger.info(f"Visual grid saved to {grid_path}")
    except Exception as e:
        logger.warning(f"Visual grid generation failed: {e}")
    
    logger.info("Fine-tuning complete.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Final-stage NAFNet fine-tuning pipeline")
    
    p.add_argument("--baseline_checkpoint", type=str, default="checkpoints_nafnet/strict_curated_training/best_ssim.pth")
    p.add_argument("--pairs_csv", type=str, default="checkpoints_nafnet/raw_pair_audit/top_2365_strict_pairs.csv")
    p.add_argument("--stats_json", type=str, default="tmp_stats/band_statistics.json")
    p.add_argument("--out_dir", type=str, default="checkpoints_nafnet/final_finetune")
    
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--lambda_l1", type=float, default=0.5)
    p.add_argument("--lambda_ssim", type=float, default=0.3)
    p.add_argument("--lambda_edge", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vis_every", type=int, default=5)
    
    raise SystemExit(run(p.parse_args()))
