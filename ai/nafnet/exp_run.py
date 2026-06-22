"""Run a NAFNet experiment with spectral-aware loss and SSIM-based early stopping.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from .dataset import NAFDataset
from .model import NAFNetWrapper
from . import metrics as naf_metrics
from skimage.metrics import structural_similarity as sk_ssim


def save_png(arr, path):
    import imageio
    a = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if a.shape[2] == 3:
        imageio.imwrite(path, a)
    else:
        imageio.imwrite(path, a[:, :, 0])


def _to_native(value):
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    return value


def _spectral_loss(pred, target):
    l1 = F.l1_loss(pred, target)
    pred_cpu = pred.detach().cpu().numpy()
    target_cpu = target.detach().cpu().numpy()
    ss = 0.0
    sam_v = 0.0
    n_samples = pred_cpu.shape[0]
    for si in range(n_samples):
        pp = np.transpose(pred_cpu[si], (1, 2, 0))
        yy = np.transpose(target_cpu[si], (1, 2, 0))
        try:
            ss += float(np.mean([sk_ssim(yy[:, :, c], pp[:, :, c], data_range=1.0) for c in range(yy.shape[2])]))
        except Exception:
            ss += 0.0
        yv = yy.reshape(-1, yy.shape[2])
        pv = pp.reshape(-1, pp.shape[2])
        num = np.sum(yv * pv, axis=1)
        den = np.linalg.norm(yv, axis=1) * np.linalg.norm(pv, axis=1)
        den = np.maximum(den, 1e-8)
        cos = np.clip(num / den, -1.0, 1.0)
        ang = np.arccos(cos)
        sam_v += float(np.degrees(np.mean(ang)))
    ss = ss / max(n_samples, 1)
    sam_v = sam_v / max(n_samples, 1)
    loss = 100.0 * l1 + 5.0 * (1.0 - ss) + 2.0 * (sam_v / 180.0)
    return loss, float(l1.detach().cpu().item()), float(ss), float(sam_v)


def _evaluate(model, val_loader, device, demo_dir=None, save_demos=False, max_demos=10):
    model.eval()
    val_metrics = []
    demo_saved = 0
    if save_demos and demo_dir is not None:
        os.makedirs(demo_dir, exist_ok=True)
        print(f"validation demo dir: {demo_dir}")
    with torch.no_grad():
        for idx, (c, t) in enumerate(val_loader):
            print(f"validation sample {idx + 1}/{len(val_loader)}")
            c = c[0]
            t = t[0]
            x = c.unsqueeze(0).to(device)
            pred = model(x)[0].cpu().numpy()
            pred = np.transpose(pred, (1, 2, 0))
            target = np.transpose(t.numpy(), (1, 2, 0))
            cloudy = np.transpose(c.numpy(), (1, 2, 0))
            pred = np.clip(pred, 0.0, 1.0)

            if save_demos and demo_dir is not None and demo_saved < max_demos:
                save_png(cloudy, os.path.join(demo_dir, f'input_{idx}.png'))
                save_png(pred, os.path.join(demo_dir, f'prediction_{idx}.png'))
                save_png(target, os.path.join(demo_dir, f'target_{idx}.png'))
                comp = np.concatenate([cloudy, pred, target], axis=1)
                save_png(comp, os.path.join(demo_dir, f'comparison_{idx}.png'))
                demo_saved += 1
                print(f"saved validation demo set {demo_saved}/{max_demos}")

            try:
                ps = naf_metrics.psnr(target, pred)
            except Exception:
                ps = None
            try:
                ss = naf_metrics.ssim(target, pred)
            except Exception:
                ss = None
            rm = naf_metrics.rmse(target, pred)
            sa = naf_metrics.sam(target, pred)
            val_metrics.append({'psnr': ps, 'ssim': ss, 'rmse': rm, 'sam': sa})

    def _agg(key):
        vals = [m[key] for m in val_metrics if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    metrics = {
        'psnr': _agg('psnr'),
        'ssim': _agg('ssim'),
        'rmse': _agg('rmse'),
        'sam': _agg('sam'),
    }
    print(f"validation complete: psnr={metrics['psnr']}, ssim={metrics['ssim']}, rmse={metrics['rmse']}, sam={metrics['sam']}")
    return metrics, val_metrics, demo_saved


def _write_csv(path, rows, fieldnames):
    import csv

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_curves(history, out_dir):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in history]
    loss = [h['avg_train_loss'] for h in history]
    psnr = [h['val_metrics']['psnr'] for h in history]
    ssim = [h['val_metrics']['ssim'] for h in history]
    sam = [h['val_metrics']['sam'] for h in history]

    plots = [
        ('loss_curve.png', loss, 'Training Loss', 'Loss'),
        ('psnr_curve.png', psnr, 'Validation PSNR', 'PSNR (dB)'),
        ('ssim_curve.png', ssim, 'Validation SSIM', 'SSIM'),
        ('sam_curve.png', sam, 'Validation SAM', 'SAM (deg)'),
    ]
    for filename, values, title, ylabel in plots:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, values, marker='o')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=160)
        plt.close()


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'epoch_reports'), exist_ok=True)

    # find pairs
    from .smoke_test import find_pairs
    pairs = find_pairs(args.cloudy, args.clear, max_pairs=args.max_pairs)
    if len(pairs) == 0:
        print('No pairs found')
        return 1
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    train_pairs = pairs[:args.train_pairs]
    val_pairs = pairs[args.train_pairs: args.train_pairs + args.val_pairs]
    print(f"train_pairs count: {len(train_pairs)}")
    print(f"val_pairs count: {len(val_pairs)}")
    if len(val_pairs) == 0:
        raise RuntimeError("Validation split is empty. Increase max_pairs or reduce train_pairs.")

    # load joint stats
    import json
    stats = {'p1':[0.0,0.0,0.0],'p99':[6000.0,6000.0,6000.0]}
    sp = Path('tmp_stats/band_statistics.json')
    if sp.exists():
        try:
            j = json.loads(sp.read_text())
            stats['p1'] = j['p1']
            stats['p99'] = j['p99']
            print('Loaded joint stats from', sp)
        except Exception:
            pass

    train_ds = NAFDataset(train_pairs, stats['p1'], stats['p99'], patch_size=(128, 128), augment=True)
    val_ds = NAFDataset(val_pairs, stats['p1'], stats['p99'], patch_size=None, augment=False)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device.startswith('cuda') else None

    start_time = time.time()

    best_ssim = -float('inf')
    best_psnr = -float('inf')
    best_sam = float('inf')
    best_epoch = -1
    best_psnr_epoch = -1
    best_sam_epoch = -1
    best_metrics = None
    best_val_metrics = []
    best_demo_saved = 0
    best_ckpt = os.path.join(args.out_dir, 'best_ssim.pth')
    best_psnr_ckpt = os.path.join(args.out_dir, 'best_psnr.pth')
    best_sam_ckpt = os.path.join(args.out_dir, 'best_sam.pth')
    last_ckpt = os.path.join(args.out_dir, 'last_epoch.pth')
    patience_counter = 0
    history = []
    csv_rows = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        batches_processed = 0
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    p = model(x)
                    loss, _, _, _ = _spectral_loss(p, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                p = model(x)
                loss, _, _, _ = _spectral_loss(p, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(float(loss.detach().cpu().item()))
            batches_processed += 1
            print(f"epoch {epoch + 1} batch {batches_processed}: train_loss={losses[-1]:.6f}")
            if args.max_steps and i >= args.max_steps:
                break

        avg_loss = float(np.mean(losses)) if losses else float('nan')
        ckpt = os.path.join(args.out_dir, f'nafnet_epoch_{epoch + 1}.pt')
        torch.save(model.state_dict(), ckpt)

        # always refresh last-epoch checkpoint
        torch.save(model.state_dict(), last_ckpt)

        val_metrics_summary, val_metrics, demo_saved = _evaluate(
            model,
            val_loader,
            device,
            demo_dir=os.path.join(args.out_dir, f'epoch_reports/epoch_{epoch + 1:03d}'),
            save_demos=True,
            max_demos=10,
        )
        print(f"epoch {epoch + 1}: validation metrics written")
        current_ssim = val_metrics_summary['ssim'] if val_metrics_summary['ssim'] is not None else -float('inf')
        history.append({
            'epoch': epoch + 1,
            'avg_train_loss': avg_loss,
            'batches_processed': batches_processed,
            'val_metrics': val_metrics_summary,
            'demo_images_saved': demo_saved,
        })
        print(f"epoch {epoch + 1}: val_ssim={current_ssim:.6f}")

        current_psnr = val_metrics_summary['psnr'] if val_metrics_summary['psnr'] is not None else -float('inf')
        current_sam = val_metrics_summary['sam'] if val_metrics_summary['sam'] is not None else float('inf')

        # save best-by-metric checkpoints
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_psnr_epoch = epoch + 1
            torch.save(model.state_dict(), best_psnr_ckpt)
        if current_sam < best_sam:
            best_sam = current_sam
            best_sam_epoch = epoch + 1
            torch.save(model.state_dict(), best_sam_ckpt)

        improved = current_ssim > best_ssim
        if improved:
            best_ssim = current_ssim
            best_epoch = epoch + 1
            best_metrics = val_metrics_summary
            best_val_metrics = val_metrics
            best_demo_saved = demo_saved
            torch.save(model.state_dict(), best_ckpt)
            patience_counter = 0
        else:
            patience_counter += 1

        csv_rows.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_psnr': val_metrics_summary['psnr'],
            'val_ssim': val_metrics_summary['ssim'],
            'val_rmse': val_metrics_summary['rmse'],
            'val_sam': val_metrics_summary['sam'],
            'batches_processed': batches_processed,
            'demo_images_saved': demo_saved,
            'best_ssim_so_far': best_ssim,
            'best_psnr_so_far': best_psnr,
            'best_sam_so_far': best_sam,
        })

        with open(os.path.join(args.out_dir, 'training_history.json'), 'w') as f:
            json.dump(_to_native(history), f, indent=2)
        _write_csv(
            os.path.join(args.out_dir, 'metrics_per_epoch.csv'),
            _to_native(csv_rows),
            ['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_rmse', 'val_sam', 'batches_processed', 'demo_images_saved', 'best_ssim_so_far', 'best_psnr_so_far', 'best_sam_so_far']
        )

        if patience_counter >= args.patience:
            print(f"early stopping at epoch {epoch + 1} after {args.patience} epochs without SSIM improvement")
            break

    training_time_sec = time.time() - start_time
    _plot_curves(history, args.out_dir)

    def _best(metric_key, higher_better=True):
        valid = [h for h in history if h['val_metrics'].get(metric_key) is not None]
        if not valid:
            return None
        key_fn = (lambda h: h['val_metrics'][metric_key]) if higher_better else (lambda h: -h['val_metrics'][metric_key])
        chosen = max(valid, key=key_fn) if higher_better else min(valid, key=lambda h: h['val_metrics'][metric_key])
        return chosen

    best_psnr_row = _best('psnr', higher_better=True)
    best_sam_row = _best('sam', higher_better=False)
    epoch_comparison = {
        'best_ssim': {'epoch': best_epoch, 'metrics': best_metrics},
        'best_psnr': {'epoch': best_psnr_epoch, 'metrics': best_psnr_row['val_metrics'] if best_psnr_row else None},
        'best_sam': {'epoch': best_sam_epoch, 'metrics': best_sam_row['val_metrics'] if best_sam_row else None},
    }
    with open(os.path.join(args.out_dir, 'epoch_comparison_report.md'), 'w') as f:
        f.write('# Epoch Comparison Report\n')
        f.write(json.dumps(_to_native(epoch_comparison), indent=2))

    report = {
        'model': 'NAFNet',
        'epochs_requested': args.epochs,
        'epochs_completed': len(history),
        'batch_size': args.batch_size,
        'loss': {'l1_weight': 100.0, 'ssim_weight': 5.0, 'sam_weight': 2.0},
        'optimizer': 'AdamW',
        'lr': args.lr,
        'early_stop': {'monitor': 'SSIM', 'patience': args.patience, 'best_epoch': best_epoch},
        'checkpoint': {
            'save_best_ssim': args.save_best_ssim,
            'best_ssim_checkpoint': best_ckpt,
            'best_psnr_checkpoint': best_psnr_ckpt,
            'best_sam_checkpoint': best_sam_ckpt,
            'last_epoch_checkpoint': last_ckpt,
        },
        'train_pairs': len(train_pairs),
        'val_pairs': len(val_pairs),
        'metrics': best_metrics,
        'best_epoch': best_epoch,
        'best_ssim': best_ssim,
        'best_psnr_epoch': best_psnr_epoch,
        'best_psnr': best_psnr,
        'best_sam_epoch': best_sam_epoch,
        'best_sam': best_sam,
        'training_time_sec': training_time_sec,
        'model_size_params': sum(p.numel() for p in model.parameters()),
        'history': history,
        'val_metrics': best_val_metrics,
        'demo_images_saved': best_demo_saved,
        'best_checkpoint': best_ckpt,
        'epoch_comparison_report': os.path.join(args.out_dir, 'epoch_comparison_report.md'),
        'training_history_json': os.path.join(args.out_dir, 'training_history.json'),
        'metrics_per_epoch_csv': os.path.join(args.out_dir, 'metrics_per_epoch.csv'),
    }
    report = _to_native(report)
    with open(os.path.join(args.out_dir, 'exp_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out_dir, 'exp_report.md'), 'w') as f:
        f.write('# Experiment report\n')
        f.write(json.dumps(report, indent=2))
    with open(os.path.join(args.out_dir, 'final_training_report.md'), 'w') as f:
        f.write('# Final Training Report\n')
        f.write(json.dumps(report, indent=2))

    # Dataset-scale report with baseline deltas for subset experiment
    base = {
        'psnr': args.baseline_psnr,
        'ssim': args.baseline_ssim,
        'sam': args.baseline_sam,
    }
    best_rmse = report['metrics']['rmse'] if report.get('metrics') else None
    deltas = {
        'psnr_delta': (report['best_psnr'] - base['psnr']) if report.get('best_psnr') is not None else None,
        'ssim_delta': (report['best_ssim'] - base['ssim']) if report.get('best_ssim') is not None else None,
        'sam_delta': (report['best_sam'] - base['sam']) if report.get('best_sam') is not None else None,
    }
    with open(os.path.join(args.out_dir, 'full_dataset_training_report.md'), 'w') as f:
        f.write('# Full Dataset Training Report\n\n')
        f.write(f"Total train pairs: {len(train_pairs)}\n")
        f.write(f"Total validation pairs: {len(val_pairs)}\n")
        f.write(f"Training duration (sec): {report['training_time_sec']}\n")
        f.write(f"Best epoch (SSIM): {report['best_epoch']}\n")
        f.write(f"Best PSNR: {report['best_psnr']}\n")
        f.write(f"Best SSIM: {report['best_ssim']}\n")
        f.write(f"Best RMSE: {best_rmse}\n")
        f.write(f"Best SAM: {report['best_sam']}\n\n")
        f.write('## Subset Baseline\n')
        f.write(f"- PSNR: {base['psnr']}\n")
        f.write(f"- SSIM: {base['ssim']}\n")
        f.write(f"- SAM: {base['sam']}\n\n")
        f.write('## Metric Deltas (full - subset)\n')
        f.write(f"- PSNR delta: {deltas['psnr_delta']}\n")
        f.write(f"- SSIM delta: {deltas['ssim_delta']}\n")
        f.write(f"- SAM delta: {deltas['sam_delta']}\n\n")
        f.write('## Goal Check\n')
        f.write(f"- PSNR > 38: {report['best_psnr'] is not None and report['best_psnr'] > 38.0}\n")
        f.write(f"- SSIM > 0.90: {report['best_ssim'] is not None and report['best_ssim'] > 0.90}\n")
        f.write(f"- SAM < 8: {report['best_sam'] is not None and report['best_sam'] < 8.0}\n")

    print('Done. Report in', args.out_dir)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cloudy', required=True)
    p.add_argument('--clear', required=True)
    p.add_argument('--max_pairs', type=int, default=250)
    p.add_argument('--train_pairs', type=int, default=200)
    p.add_argument('--val_pairs', type=int, default=50)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--baseline_psnr', type=float, default=37.47)
    p.add_argument('--baseline_ssim', type=float, default=0.8869)
    p.add_argument('--baseline_sam', type=float, default=9.67)
    p.add_argument('--save_best_ssim', action='store_true', default=True)
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--out_dir', default='checkpoints_nafnet/exp')
    args = p.parse_args()
    raise SystemExit(run(args))