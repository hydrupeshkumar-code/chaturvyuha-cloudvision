"""Train NAFNet on curated strict SEN12MS-CR pairs with full reporting.

Requirements covered:
- reproducible 80/10/10 split (seed 42)
- strict dataset checks (existence, normalized range, channel assumptions)
- per-epoch training/validation metrics + LR + epoch time
- per-epoch dynamic-range collapse checks on prediction mean/std
- 10 random validation visualizations per epoch
- best_ssim / best_psnr / best_sam / last checkpoints
- final test-set evaluation report
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
from skimage.metrics import structural_similarity as sk_ssim
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .dataset import NAFDataset
from .model import NAFNetWrapper
from . import metrics as naf_metrics


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


def _save_png(arr_hwc, path):
    import imageio.v2 as imageio

    x = np.clip(arr_hwc * 255.0, 0, 255).astype(np.uint8)
    imageio.imwrite(str(path), x)


def _spectral_loss(pred, target):
    l1 = F.l1_loss(pred, target)
    pred_cpu = pred.detach().cpu().numpy()
    target_cpu = target.detach().cpu().numpy()
    ss = 0.0
    sam_v = 0.0
    n = pred_cpu.shape[0]

    for i in range(n):
        pp = np.transpose(pred_cpu[i], (1, 2, 0))
        yy = np.transpose(target_cpu[i], (1, 2, 0))
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

    ss = ss / max(n, 1)
    sam_v = sam_v / max(n, 1)
    loss = 100.0 * l1 + 5.0 * (1.0 - ss) + 2.0 * (sam_v / 180.0)
    return loss, float(l1.detach().cpu().item()), float(ss), float(sam_v)


def _dataset_checks_and_stats(pairs, clip_min, clip_max):
    missing = []
    for c, t in pairs:
        if not Path(c).exists() or not Path(t).exists():
            missing.append((c, t))

    if missing:
        return {
            "ok": False,
            "missing": missing,
            "range_ok": False,
            "channel_order": "Green,Red,NIR (assumed by band mapping)",
            "mean": None,
            "std": None,
        }

    ds = NAFDataset(pairs, clip_min, clip_max, patch_size=None, augment=False)

    per_channel_sum = np.zeros(3, dtype=np.float64)
    per_channel_sq_sum = np.zeros(3, dtype=np.float64)
    per_channel_count = np.zeros(3, dtype=np.float64)
    range_ok = True

    for i in range(len(ds)):
        x, _ = ds[i]
        arr = x.numpy()  # CHW
        if arr.shape[0] != 3:
            range_ok = False
            break
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if mn < -1e-6 or mx > 1.0 + 1e-6:
            range_ok = False

        for c in range(3):
            ch = arr[c]
            per_channel_sum[c] += float(np.sum(ch))
            per_channel_sq_sum[c] += float(np.sum(ch * ch))
            per_channel_count[c] += float(ch.size)

    means = per_channel_sum / np.maximum(per_channel_count, 1.0)
    vars_ = per_channel_sq_sum / np.maximum(per_channel_count, 1.0) - means * means
    stds = np.sqrt(np.maximum(vars_, 0.0))

    return {
        "ok": range_ok,
        "missing": [],
        "range_ok": range_ok,
        "channel_order": "Green,Red,NIR (assumed by band mapping: channels [0,1,2])",
        "mean": means.tolist(),
        "std": stds.tolist(),
    }


def _evaluate(model, val_loader, val_ds, device, out_epoch_dir, epoch_num, n_visuals=10):
    model.eval()
    rows = []
    pred_means = []
    pred_stds = []

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            pred = model(x)[0].cpu().numpy()
            pred = np.transpose(pred, (1, 2, 0))
            pred = np.clip(pred, 0.0, 1.0)
            tgt = np.transpose(y[0].numpy(), (1, 2, 0))

            rows.append(
                {
                    "psnr": float(naf_metrics.psnr(tgt, pred)),
                    "ssim": float(naf_metrics.ssim(tgt, pred)),
                    "rmse": float(naf_metrics.rmse(tgt, pred)),
                    "sam": float(naf_metrics.sam(tgt, pred)),
                }
            )

            pred_means.append(float(np.mean(pred)))
            pred_stds.append(float(np.std(pred)))

    metrics = {
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "sam": float(np.mean([r["sam"] for r in rows])),
        "pred_mean": float(np.mean(pred_means)),
        "pred_std": float(np.mean(pred_stds)),
    }

    # 10 random validation visualizations per epoch
    out_epoch_dir.mkdir(parents=True, exist_ok=True)
    idxs = list(range(len(val_ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(n_visuals, len(val_ds))]

    with torch.no_grad():
        for k, idx in enumerate(idxs):
            c, t = val_ds[idx]
            pred = model(c.unsqueeze(0).to(device))[0].cpu().numpy()
            pred = np.clip(np.transpose(pred, (1, 2, 0)), 0.0, 1.0)
            inp = np.transpose(c.numpy(), (1, 2, 0))
            tgt = np.transpose(t.numpy(), (1, 2, 0))
            comp = np.concatenate([inp, pred, tgt], axis=1)

            _save_png(inp, out_epoch_dir / f"input_{k:02d}_idx_{idx}.png")
            _save_png(pred, out_epoch_dir / f"prediction_{k:02d}_idx_{idx}.png")
            _save_png(tgt, out_epoch_dir / f"target_{k:02d}_idx_{idx}.png")
            _save_png(comp, out_epoch_dir / f"comparison_{k:02d}_idx_{idx}.png")

    collapse = (metrics["pred_mean"] < 0.05) or (metrics["pred_std"] < 0.01)
    metrics["dynamic_range_collapse_flag"] = collapse
    return metrics


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "epoch_reports").mkdir(parents=True, exist_ok=True)

    pairs = _load_pairs(Path(args.pairs_csv))
    if len(pairs) != args.total_pairs:
        print(f"warning: expected {args.total_pairs} pairs but loaded {len(pairs)}")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    train_pairs = pairs[: args.train_pairs]
    val_pairs = pairs[args.train_pairs : args.train_pairs + args.val_pairs]
    test_pairs = pairs[args.train_pairs + args.val_pairs : args.train_pairs + args.val_pairs + args.test_pairs]

    # normalization stats
    stats = {"p1": [0.0, 0.0, 0.0], "p99": [6000.0, 6000.0, 6000.0]}
    sp = Path("tmp_stats/band_statistics.json")
    if sp.exists():
        try:
            j = json.loads(sp.read_text())
            stats["p1"] = j["p1"]
            stats["p99"] = j["p99"]
        except Exception:
            pass

    # dataset checks
    train_check = _dataset_checks_and_stats(train_pairs, stats["p1"], stats["p99"])
    val_check = _dataset_checks_and_stats(val_pairs, stats["p1"], stats["p99"])
    test_check = _dataset_checks_and_stats(test_pairs, stats["p1"], stats["p99"])

    missing_pairs = train_check["missing"] + val_check["missing"] + test_check["missing"]
    if missing_pairs:
        miss_report = out_dir / "missing_pairs_abort.json"
        miss_report.write_text(json.dumps({"missing_pairs": missing_pairs}, indent=2), encoding="utf-8")
        raise RuntimeError(f"Abort: found missing pairs ({len(missing_pairs)}). See {miss_report}")

    if not train_check["range_ok"] or not val_check["range_ok"] or not test_check["range_ok"]:
        raise RuntimeError("Abort: normalized tensor range check failed (expected [0,1]).")

    train_ds = NAFDataset(train_pairs, stats["p1"], stats["p99"], patch_size=(128, 128), augment=True)
    val_ds = NAFDataset(val_pairs, stats["p1"], stats["p99"], patch_size=None, augment=False)
    test_ds = NAFDataset(test_pairs, stats["p1"], stats["p99"], patch_size=None, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None

    best_ssim = -1e9
    best_psnr = -1e9
    best_sam = 1e9
    best_epoch = -1
    patience_counter = 0

    best_ssim_path = out_dir / "best_ssim.pth"
    best_psnr_path = out_dir / "best_psnr.pth"
    best_sam_path = out_dir / "best_sam.pth"
    last_epoch_path = out_dir / "last_epoch.pth"

    history = []
    csv_rows = []
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        batch_losses = []
        for i, (x, y) in enumerate(train_loader, start=1):
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

            batch_losses.append(float(loss.detach().cpu().item()))
            if i % 50 == 0:
                print(f"epoch {epoch} batch {i}/{len(train_loader)} train_loss={batch_losses[-1]:.6f}")

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_metrics = _evaluate(
            model,
            val_loader,
            val_ds,
            device,
            out_epoch_dir=out_dir / "epoch_reports" / f"epoch_{epoch:03d}",
            epoch_num=epoch,
            n_visuals=10,
        )

        torch.save(model.state_dict(), last_epoch_path)

        if val_metrics["ssim"] > best_ssim:
            best_ssim = val_metrics["ssim"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_ssim_path)
        else:
            patience_counter += 1

        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            torch.save(model.state_dict(), best_psnr_path)

        if val_metrics["sam"] < best_sam:
            best_sam = val_metrics["sam"]
            torch.save(model.state_dict(), best_sam_path)

        lr = float(opt.param_groups[0]["lr"])
        epoch_time = float(time.time() - t0)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "val_rmse": val_metrics["rmse"],
            "val_sam": val_metrics["sam"],
            "learning_rate": lr,
            "epoch_time_sec": epoch_time,
            "pred_mean": val_metrics["pred_mean"],
            "pred_std": val_metrics["pred_std"],
            "dynamic_range_collapse_flag": val_metrics["dynamic_range_collapse_flag"],
            "best_ssim_so_far": best_ssim,
            "best_psnr_so_far": best_psnr,
            "best_sam_so_far": best_sam,
        }
        csv_rows.append(row)
        history.append(_to_native(row))

        _write_csv(
            out_dir / "training_metrics.csv",
            csv_rows,
            [
                "epoch",
                "train_loss",
                "val_psnr",
                "val_ssim",
                "val_rmse",
                "val_sam",
                "learning_rate",
                "epoch_time_sec",
                "pred_mean",
                "pred_std",
                "dynamic_range_collapse_flag",
                "best_ssim_so_far",
                "best_psnr_so_far",
                "best_sam_so_far",
            ],
        )

        with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(_to_native(history), f, indent=2)

        print(
            "epoch {e}: train_loss={tl:.6f} val_psnr={vp:.4f} val_ssim={vs:.4f} val_rmse={vr:.6f} val_sam={vsa:.4f} pred_mean={pm:.5f} pred_std={ps:.5f} collapse={cf}".format(
                e=epoch,
                tl=train_loss,
                vp=val_metrics["psnr"],
                vs=val_metrics["ssim"],
                vr=val_metrics["rmse"],
                vsa=val_metrics["sam"],
                pm=val_metrics["pred_mean"],
                ps=val_metrics["pred_std"],
                cf=val_metrics["dynamic_range_collapse_flag"],
            )
        )

        if patience_counter >= args.patience:
            print(f"early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Final evaluation on test set using best SSIM checkpoint
    best_state = torch.load(best_ssim_path, map_location=device)
    model.load_state_dict(best_state, strict=False)

    model.eval()
    test_rows = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)[0].cpu().numpy()
            pred = np.clip(np.transpose(pred, (1, 2, 0)), 0.0, 1.0)
            tgt = np.transpose(y[0].numpy(), (1, 2, 0))
            test_rows.append(
                {
                    "psnr": float(naf_metrics.psnr(tgt, pred)),
                    "ssim": float(naf_metrics.ssim(tgt, pred)),
                    "rmse": float(naf_metrics.rmse(tgt, pred)),
                    "sam": float(naf_metrics.sam(tgt, pred)),
                }
            )

    final_eval = {
        "psnr": float(np.mean([r["psnr"] for r in test_rows])) if test_rows else None,
        "ssim": float(np.mean([r["ssim"] for r in test_rows])) if test_rows else None,
        "rmse": float(np.mean([r["rmse"] for r in test_rows])) if test_rows else None,
        "sam": float(np.mean([r["sam"] for r in test_rows])) if test_rows else None,
    }

    with open(out_dir / "final_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("# Final Evaluation Report\n\n")
        f.write(f"PSNR: {final_eval['psnr']}\n")
        f.write(f"SSIM: {final_eval['ssim']}\n")
        f.write(f"RMSE: {final_eval['rmse']}\n")
        f.write(f"SAM: {final_eval['sam']}\n")

    collapse_epochs = [r["epoch"] for r in csv_rows if r["dynamic_range_collapse_flag"]]
    train_secs = float(time.time() - total_start)
    completed_epochs = len(csv_rows)
    avg_epoch_sec = float(np.mean([r["epoch_time_sec"] for r in csv_rows])) if csv_rows else None

    with open(out_dir / "training_report.md", "w", encoding="utf-8") as f:
        f.write("# Curated Strict Training Report\n\n")
        f.write("## Split\n")
        f.write(f"- Total pairs loaded: {len(pairs)}\n")
        f.write(f"- Train pairs: {len(train_pairs)}\n")
        f.write(f"- Validation pairs: {len(val_pairs)}\n")
        f.write(f"- Test pairs: {len(test_pairs)}\n")
        f.write(f"- Seed: {args.seed}\n\n")

        f.write("## Dataset Checks\n")
        f.write(f"- All files exist: {len(missing_pairs) == 0}\n")
        f.write(f"- Tensor range [0,1] train/val/test: {train_check['range_ok'] and val_check['range_ok'] and test_check['range_ok']}\n")
        f.write(f"- Channel order: {train_check['channel_order']}\n")
        f.write(f"- Train mean per channel: {train_check['mean']}\n")
        f.write(f"- Train std per channel: {train_check['std']}\n")
        f.write(f"- Val mean per channel: {val_check['mean']}\n")
        f.write(f"- Val std per channel: {val_check['std']}\n\n")

        f.write("## Training\n")
        f.write(f"- Epochs requested: {args.epochs}\n")
        f.write(f"- Epochs completed: {completed_epochs}\n")
        f.write(f"- Average epoch time (sec): {avg_epoch_sec}\n")
        f.write(f"- Total train time (sec): {train_secs}\n")
        f.write(f"- Best epoch (SSIM): {best_epoch}\n")
        f.write(f"- Best SSIM: {best_ssim}\n")
        f.write(f"- Best PSNR: {best_psnr}\n")
        f.write(f"- Best SAM: {best_sam}\n")
        f.write(f"- Dynamic-range collapse flagged epochs: {collapse_epochs}\n\n")

        f.write("## Checkpoints\n")
        f.write(f"- best_ssim.pth: {best_ssim_path}\n")
        f.write(f"- best_psnr.pth: {best_psnr_path}\n")
        f.write(f"- best_sam.pth: {best_sam_path}\n")
        f.write(f"- last_epoch.pth: {last_epoch_path}\n\n")

        f.write("## Final Test Evaluation\n")
        f.write(f"- PSNR: {final_eval['psnr']}\n")
        f.write(f"- SSIM: {final_eval['ssim']}\n")
        f.write(f"- RMSE: {final_eval['rmse']}\n")
        f.write(f"- SAM: {final_eval['sam']}\n\n")

        f.write("## Goal Check\n")
        f.write(f"- PSNR > 35: {final_eval['psnr'] is not None and final_eval['psnr'] > 35.0}\n")
        f.write(f"- SSIM > 0.88: {final_eval['ssim'] is not None and final_eval['ssim'] > 0.88}\n")
        f.write(f"- SAM < 8: {final_eval['sam'] is not None and final_eval['sam'] < 8.0}\n")

    print("Training complete.")
    print("Best checkpoint (SSIM):", best_ssim_path)
    print("Final test metrics:", final_eval)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_csv", default="checkpoints_nafnet/raw_pair_audit/top_2365_strict_pairs.csv")
    p.add_argument("--out_dir", default="checkpoints_nafnet/strict_curated_training")

    p.add_argument("--total_pairs", type=int, default=2365)
    p.add_argument("--train_pairs", type=int, default=1892)
    p.add_argument("--val_pairs", type=int, default=236)
    p.add_argument("--test_pairs", type=int, default=237)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)

    raise SystemExit(run(p.parse_args()))
