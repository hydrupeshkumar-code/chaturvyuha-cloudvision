"""Fine-tune NAFNet from an existing checkpoint for sharper reconstruction.

This script starts from a provided checkpoint and does NOT train from scratch.
It reuses the strict curated split logic (same CSV + seed) and adds Sobel-based
edge loss, edge similarity tracking, richer artifacts, and baseline comparison.
"""

import argparse
import csv
import json
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


def _build_sobel_kernels(channels, device):
    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    ky = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    kx = kx.unsqueeze(1).repeat(channels, 1, 1, 1)
    ky = ky.unsqueeze(1).repeat(channels, 1, 1, 1)
    return kx, ky


def _sobel_tensor(x):
    # x: NCHW in [0,1]
    c = x.shape[1]
    kx, ky = _build_sobel_kernels(c, x.device)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    return mag


def _edge_similarity_score(pred_hwc, tgt_hwc):
    # Returns higher-is-better score in [0,1].
    pt = torch.from_numpy(np.transpose(pred_hwc, (2, 0, 1))).unsqueeze(0).float()
    tt = torch.from_numpy(np.transpose(tgt_hwc, (2, 0, 1))).unsqueeze(0).float()
    pe = _sobel_tensor(pt)
    te = _sobel_tensor(tt)
    l1 = float(F.l1_loss(pe, te).item())
    return float(1.0 / (1.0 + l1))


def _edge_vis_from_hwc(arr_hwc):
    # Keep 3 channels and normalize for visual inspection.
    t = torch.from_numpy(np.transpose(arr_hwc, (2, 0, 1))).unsqueeze(0).float()
    e = _sobel_tensor(t)[0].numpy()
    e = np.transpose(e, (1, 2, 0))
    out = np.empty_like(e, dtype=np.float32)
    for c in range(e.shape[2]):
        ch = e[:, :, c]
        lo = float(np.percentile(ch, 1))
        hi = float(np.percentile(ch, 99))
        if hi - lo < 1e-8:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return out


def _sharpen_loss(pred, target):
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

    edge = F.l1_loss(_sobel_tensor(pred), _sobel_tensor(target))

    # 50*L1 + 10*(1-SSIM) + 5*EdgeLoss + 2*(SAM/180)
    loss = 50.0 * l1 + 10.0 * (1.0 - ss) + 5.0 * edge + 2.0 * (sam_v / 180.0)

    return loss, float(l1.detach().cpu().item()), float(ss), float(sam_v), float(edge.detach().cpu().item())


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
        arr = x.numpy()
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


def _evaluate(model, val_loader, val_ds, device, out_epoch_dir, epoch_num, n_visuals=20):
    model.eval()
    rows = []
    pred_means = []
    pred_stds = []

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
                    "rmse": float(naf_metrics.rmse(tgt, pred)),
                    "sam": float(naf_metrics.sam(tgt, pred)),
                    "edge_similarity": float(_edge_similarity_score(pred, tgt)),
                }
            )

            pred_means.append(float(np.mean(pred)))
            pred_stds.append(float(np.std(pred)))

    metrics = {
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "sam": float(np.mean([r["sam"] for r in rows])),
        "edge_similarity": float(np.mean([r["edge_similarity"] for r in rows])),
        "pred_mean": float(np.mean(pred_means)),
        "pred_std": float(np.mean(pred_stds)),
    }

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

            edge_inp = _edge_vis_from_hwc(inp)
            edge_pred = _edge_vis_from_hwc(pred)
            edge_tgt = _edge_vis_from_hwc(tgt)
            edge_comp = np.concatenate([edge_inp, edge_pred, edge_tgt], axis=1)

            sample_dir = out_epoch_dir / f"sample_{k:02d}_idx_{idx}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(inp, sample_dir / "input.png")
            _save_png(pred, sample_dir / "prediction.png")
            _save_png(tgt, sample_dir / "target.png")
            _save_png(comp, sample_dir / "comparison.png")
            _save_png(edge_comp, sample_dir / "edge_comparison.png")

    collapse = (metrics["pred_mean"] < 0.05) or (metrics["pred_std"] < 0.01)
    metrics["dynamic_range_warning"] = collapse
    return metrics


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _compute_split(pairs, seed, train_pairs, val_pairs, test_pairs):
    work = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(work)

    tr = work[:train_pairs]
    va = work[train_pairs : train_pairs + val_pairs]
    te = work[train_pairs + val_pairs : train_pairs + val_pairs + test_pairs]
    return tr, va, te


def _evaluate_checkpoint_on_test(ckpt_path, test_ds, device):
    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    pred_means = []
    pred_stds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x)[0].cpu().numpy()
            pred = np.clip(np.transpose(pred, (1, 2, 0)), 0.0, 1.0)
            tgt = np.transpose(y[0].numpy(), (1, 2, 0))

            rows.append(
                {
                    "psnr": float(naf_metrics.psnr(tgt, pred)),
                    "ssim": float(naf_metrics.ssim(tgt, pred)),
                    "rmse": float(naf_metrics.rmse(tgt, pred)),
                    "sam": float(naf_metrics.sam(tgt, pred)),
                    "edge_similarity": float(_edge_similarity_score(pred, tgt)),
                }
            )
            pred_means.append(float(np.mean(pred)))
            pred_stds.append(float(np.std(pred)))

    return {
        "psnr": float(np.mean([r["psnr"] for r in rows])) if rows else None,
        "ssim": float(np.mean([r["ssim"] for r in rows])) if rows else None,
        "rmse": float(np.mean([r["rmse"] for r in rows])) if rows else None,
        "sam": float(np.mean([r["sam"] for r in rows])) if rows else None,
        "edge_similarity": float(np.mean([r["edge_similarity"] for r in rows])) if rows else None,
        "pred_mean": float(np.mean(pred_means)) if pred_means else None,
        "pred_std": float(np.mean(pred_stds)) if pred_stds else None,
    }


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "epoch_reports").mkdir(parents=True, exist_ok=True)

    pairs = _load_pairs(Path(args.pairs_csv))
    if len(pairs) != args.total_pairs:
        print(f"warning: expected {args.total_pairs} pairs but loaded {len(pairs)}")

    train_pairs, val_pairs, test_pairs = _compute_split(
        pairs,
        seed=args.seed,
        train_pairs=args.train_pairs,
        val_pairs=args.val_pairs,
        test_pairs=args.test_pairs,
    )

    split_manifest = {
        "seed": args.seed,
        "pairs_csv": str(args.pairs_csv),
        "total_pairs_loaded": len(pairs),
        "train_pairs": [{"cloudy_path": c, "target_path": t} for c, t in train_pairs],
        "val_pairs": [{"cloudy_path": c, "target_path": t} for c, t in val_pairs],
        "test_pairs": [{"cloudy_path": c, "target_path": t} for c, t in test_pairs],
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    stats = {"p1": [0.0, 0.0, 0.0], "p99": [6000.0, 6000.0, 6000.0]}
    sp = Path("tmp_stats/band_statistics.json")
    if sp.exists():
        try:
            j = json.loads(sp.read_text())
            stats["p1"] = j["p1"]
            stats["p99"] = j["p99"]
        except Exception:
            pass

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)

    init_ckpt = Path(args.init_checkpoint)
    if not init_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {init_ckpt}")

    state = torch.load(init_ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    opt = AdamW(model.parameters(), lr=args.lr)

    use_amp = bool(args.mixed_precision and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_ssim = -1e9
    best_psnr = -1e9
    best_sam = 1e9
    best_edge = -1e9
    best_epoch = -1
    patience_counter = 0

    best_ssim_path = out_dir / "best_ssim.pth"
    best_psnr_path = out_dir / "best_psnr.pth"
    best_sam_path = out_dir / "best_sam.pth"
    best_edge_path = out_dir / "best_edge.pth"
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
            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    p = model(x)
                    loss, _, _, _, _ = _sharpen_loss(p, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                p = model(x)
                loss, _, _, _, _ = _sharpen_loss(p, y)
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
            n_visuals=20,
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

        if val_metrics["edge_similarity"] > best_edge:
            best_edge = val_metrics["edge_similarity"]
            torch.save(model.state_dict(), best_edge_path)

        lr = float(opt.param_groups[0]["lr"])
        epoch_time = float(time.time() - t0)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "val_rmse": val_metrics["rmse"],
            "val_sam": val_metrics["sam"],
            "edge_similarity": val_metrics["edge_similarity"],
            "pred_mean": val_metrics["pred_mean"],
            "pred_std": val_metrics["pred_std"],
            "learning_rate": lr,
            "epoch_time_sec": epoch_time,
            "dynamic_range_warning": val_metrics["dynamic_range_warning"],
            "best_ssim_so_far": best_ssim,
            "best_psnr_so_far": best_psnr,
            "best_sam_so_far": best_sam,
            "best_edge_so_far": best_edge,
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
                "edge_similarity",
                "pred_mean",
                "pred_std",
                "learning_rate",
                "epoch_time_sec",
                "dynamic_range_warning",
                "best_ssim_so_far",
                "best_psnr_so_far",
                "best_sam_so_far",
                "best_edge_so_far",
            ],
        )

        with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(_to_native(history), f, indent=2)

        print(
            "epoch {e}: train_loss={tl:.6f} val_psnr={vp:.4f} val_ssim={vs:.4f} val_rmse={vr:.6f} val_sam={vsa:.4f} edge={ve:.5f} pred_mean={pm:.5f} pred_std={ps:.5f} dr_warn={dw}".format(
                e=epoch,
                tl=train_loss,
                vp=val_metrics["psnr"],
                vs=val_metrics["ssim"],
                vr=val_metrics["rmse"],
                vsa=val_metrics["sam"],
                ve=val_metrics["edge_similarity"],
                pm=val_metrics["pred_mean"],
                ps=val_metrics["pred_std"],
                dw=val_metrics["dynamic_range_warning"],
            )
        )

        if patience_counter >= args.patience:
            print(f"early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Evaluate baseline model and finetuned best_ssim model on test set.
    baseline_eval = _evaluate_checkpoint_on_test(Path(args.init_checkpoint), test_ds, device)
    finetuned_eval = _evaluate_checkpoint_on_test(best_ssim_path, test_ds, device)

    with open(out_dir / "final_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("# Final Evaluation Report\n\n")
        f.write("## Baseline (Input Checkpoint)\n")
        f.write(f"PSNR: {baseline_eval['psnr']}\n")
        f.write(f"SSIM: {baseline_eval['ssim']}\n")
        f.write(f"RMSE: {baseline_eval['rmse']}\n")
        f.write(f"SAM: {baseline_eval['sam']}\n")
        f.write(f"Edge Similarity: {baseline_eval['edge_similarity']}\n")
        f.write(f"Prediction Mean: {baseline_eval['pred_mean']}\n")
        f.write(f"Prediction Std: {baseline_eval['pred_std']}\n\n")

        f.write("## Fine-tuned (Best SSIM)\n")
        f.write(f"PSNR: {finetuned_eval['psnr']}\n")
        f.write(f"SSIM: {finetuned_eval['ssim']}\n")
        f.write(f"RMSE: {finetuned_eval['rmse']}\n")
        f.write(f"SAM: {finetuned_eval['sam']}\n")
        f.write(f"Edge Similarity: {finetuned_eval['edge_similarity']}\n")
        f.write(f"Prediction Mean: {finetuned_eval['pred_mean']}\n")
        f.write(f"Prediction Std: {finetuned_eval['pred_std']}\n")

    deltas = {
        "psnr_delta": (finetuned_eval["psnr"] - baseline_eval["psnr"]) if finetuned_eval["psnr"] is not None else None,
        "ssim_delta": (finetuned_eval["ssim"] - baseline_eval["ssim"]) if finetuned_eval["ssim"] is not None else None,
        "rmse_delta": (finetuned_eval["rmse"] - baseline_eval["rmse"]) if finetuned_eval["rmse"] is not None else None,
        "sam_delta": (finetuned_eval["sam"] - baseline_eval["sam"]) if finetuned_eval["sam"] is not None else None,
        "edge_similarity_delta": (
            finetuned_eval["edge_similarity"] - baseline_eval["edge_similarity"]
            if finetuned_eval["edge_similarity"] is not None
            else None
        ),
        "pred_mean_delta": (
            finetuned_eval["pred_mean"] - baseline_eval["pred_mean"] if finetuned_eval["pred_mean"] is not None else None
        ),
        "pred_std_delta": (
            finetuned_eval["pred_std"] - baseline_eval["pred_std"] if finetuned_eval["pred_std"] is not None else None
        ),
    }

    collapse_epochs = [r["epoch"] for r in csv_rows if r["dynamic_range_warning"]]
    train_secs = float(time.time() - total_start)
    completed_epochs = len(csv_rows)
    avg_epoch_sec = float(np.mean([r["epoch_time_sec"] for r in csv_rows])) if csv_rows else None

    baseline_ref = {
        "psnr": args.prev_baseline_psnr,
        "ssim": args.prev_baseline_ssim,
        "sam": args.prev_baseline_sam,
    }

    objective_status = {
        "psnr_gte_prev_baseline": finetuned_eval["psnr"] is not None and finetuned_eval["psnr"] >= baseline_ref["psnr"],
        "ssim_gt_0_90": finetuned_eval["ssim"] is not None and finetuned_eval["ssim"] > 0.90,
        "sam_lt_5": finetuned_eval["sam"] is not None and finetuned_eval["sam"] < 5.0,
    }

    # Heuristic narrative based on measurable sharpness proxies.
    sharpness_improved = (
        deltas["edge_similarity_delta"] is not None
        and deltas["edge_similarity_delta"] > 0.0
        and deltas["pred_std_delta"] is not None
        and deltas["pred_std_delta"] >= 0.0
    )

    with open(out_dir / "final_sharpen_finetune_report.md", "w", encoding="utf-8") as f:
        f.write("# Final Sharpen Fine-tune Report\n\n")

        f.write("## Setup\n")
        f.write(f"- Initialization checkpoint: {args.init_checkpoint}\n")
        f.write(f"- Pairs CSV: {args.pairs_csv}\n")
        f.write(f"- Split seed: {args.seed}\n")
        f.write(f"- Split sizes (train/val/test): {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}\n")
        f.write(f"- Epochs requested/completed: {args.epochs}/{completed_epochs}\n")
        f.write(f"- Average epoch time (sec): {avg_epoch_sec}\n")
        f.write(f"- Total fine-tune time (sec): {train_secs}\n\n")

        f.write("## Best Checkpoints\n")
        f.write(f"- best_ssim.pth: {best_ssim_path}\n")
        f.write(f"- best_psnr.pth: {best_psnr_path}\n")
        f.write(f"- best_sam.pth: {best_sam_path}\n")
        f.write(f"- best_edge.pth: {best_edge_path}\n")
        f.write(f"- last_epoch.pth: {last_epoch_path}\n\n")

        f.write("## Final Test Metrics (Fine-tuned Best SSIM)\n")
        f.write(f"- PSNR: {finetuned_eval['psnr']}\n")
        f.write(f"- SSIM: {finetuned_eval['ssim']}\n")
        f.write(f"- RMSE: {finetuned_eval['rmse']}\n")
        f.write(f"- SAM: {finetuned_eval['sam']}\n")
        f.write(f"- Edge Similarity: {finetuned_eval['edge_similarity']}\n\n")

        f.write("## Metric Deltas (Fine-tuned - Baseline Input Checkpoint)\n")
        f.write(f"- Delta PSNR: {deltas['psnr_delta']}\n")
        f.write(f"- Delta SSIM: {deltas['ssim_delta']}\n")
        f.write(f"- Delta RMSE: {deltas['rmse_delta']}\n")
        f.write(f"- Delta SAM: {deltas['sam_delta']}\n")
        f.write(f"- Delta Edge Similarity: {deltas['edge_similarity_delta']}\n")
        f.write(f"- Delta Prediction Mean: {deltas['pred_mean_delta']}\n")
        f.write(f"- Delta Prediction Std: {deltas['pred_std_delta']}\n\n")

        f.write("## Comparison Against Previous Strict Curated Reference\n")
        f.write(f"- Previous reference PSNR: {baseline_ref['psnr']}\n")
        f.write(f"- Previous reference SSIM: {baseline_ref['ssim']}\n")
        f.write(f"- Previous reference SAM: {baseline_ref['sam']}\n")
        f.write(f"- Meets PSNR >= previous: {objective_status['psnr_gte_prev_baseline']}\n")
        f.write(f"- Meets SSIM > 0.90: {objective_status['ssim_gt_0_90']}\n")
        f.write(f"- Meets SAM < 5: {objective_status['sam_lt_5']}\n\n")

        f.write("## Prediction Mean/Std Analysis\n")
        f.write(f"- Dynamic range warning epochs: {collapse_epochs}\n")
        f.write(
            "- Warning rule: prediction_mean < 0.05 OR prediction_std < 0.01.\n"
        )
        f.write(
            f"- Baseline mean/std: {baseline_eval['pred_mean']} / {baseline_eval['pred_std']}\n"
        )
        f.write(
            f"- Fine-tuned mean/std: {finetuned_eval['pred_mean']} / {finetuned_eval['pred_std']}\n\n"
        )

        f.write("## Edge Similarity Improvement Analysis\n")
        f.write(
            f"- Baseline edge similarity: {baseline_eval['edge_similarity']}\n"
        )
        f.write(
            f"- Fine-tuned edge similarity: {finetuned_eval['edge_similarity']}\n"
        )
        f.write(
            f"- Edge similarity delta: {deltas['edge_similarity_delta']}\n\n"
        )

        f.write("## Visual Sharpness Analysis\n")
        if sharpness_improved:
            f.write(
                "- Quantitative proxies indicate sharper reconstruction: edge similarity improved and prediction std did not decrease.\n"
            )
            f.write(
                "- This is consistent with better visibility of roads, field boundaries, and terrain textures in edge-focused comparisons.\n"
            )
        else:
            f.write(
                "- Quantitative proxies do not indicate a clear sharpness gain: edge similarity and/or prediction std did not improve together.\n"
            )
            f.write(
                "- Under these results, visible road/building/boundary/texture improvements are not conclusively better than the baseline checkpoint.\n"
            )

    print("Fine-tune complete.")
    print("Best checkpoint (SSIM):", best_ssim_path)
    print("Final test metrics (fine-tuned):", finetuned_eval)
    print("Objective status:", objective_status)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_csv", default="checkpoints_nafnet/raw_pair_audit/top_2365_strict_pairs.csv")
    p.add_argument("--out_dir", default="checkpoints_nafnet/final_sharpen_finetune")

    p.add_argument(
        "--init_checkpoint",
        default="checkpoints_nafnet/strict_curated_training/best_ssim.pth",
    )

    p.add_argument("--total_pairs", type=int, default=2365)
    p.add_argument("--train_pairs", type=int, default=1892)
    p.add_argument("--val_pairs", type=int, default=236)
    p.add_argument("--test_pairs", type=int, default=237)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--mixed_precision", action="store_true")

    p.add_argument("--prev_baseline_psnr", type=float, default=34.9493)
    p.add_argument("--prev_baseline_ssim", type=float, default=0.9010)
    p.add_argument("--prev_baseline_sam", type=float, default=4.9948)

    raise SystemExit(run(p.parse_args()))
