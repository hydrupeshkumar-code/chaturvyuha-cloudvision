import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from .dataset import CloudDataset
    from .model import UNet
except ImportError:
    from dataset import CloudDataset
    from model import UNet


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        inter = (probs * targets).sum()
        dice = (2.0 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


def compute_binary_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.int32)
    t = targets.to(torch.int32)

    tp = int(((preds == 1) & (t == 1)).sum().item())
    fp = int(((preds == 1) & (t == 0)).sum().item())
    fn = int(((preds == 0) & (t == 1)).sum().item())
    tn = int(((preds == 0) & (t == 0)).sum().item())

    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate(model, loader, bce, dice, device):
    model.eval()
    losses = []

    tot_tp = tot_fp = tot_fn = tot_tn = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = bce(logits, y) + dice(logits, y)
            losses.append(float(loss.item()))

            m = compute_binary_metrics(logits, y)
            tot_tp += m["tp"]
            tot_fp += m["fp"]
            tot_fn += m["fn"]
            tot_tn += m["tn"]

    eps = 1e-8
    iou = tot_tp / (tot_tp + tot_fp + tot_fn + eps)
    precision = tot_tp / (tot_tp + tot_fp + eps)
    recall = tot_tp / (tot_tp + tot_fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run(args):
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.mixed_precision and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    train_ds = CloudDataset(
        image_dir=str(Path(args.data_dir) / "train" / "images"),
        mask_dir=str(Path(args.data_dir) / "train" / "masks"),
        augment=True,
    )
    val_ds = CloudDataset(
        image_dir=str(Path(args.data_dir) / "val" / "images"),
        mask_dir=str(Path(args.data_dir) / "val" / "masks"),
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet(in_channels=3, out_channels=1).to(device)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    opt = AdamW(model.parameters(), lr=args.lr)

    best_iou = -1.0
    best_f1 = -1.0

    csv_path = out_dir / "training_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_iou",
            "val_precision",
            "val_recall",
            "val_f1",
            "lr",
            "epoch_time_sec",
        ])

    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(x)
                    loss = bce(logits, y) + dice(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = bce(logits, y) + dice(logits, y)
                loss.backward()
                opt.step()

            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val = evaluate(model, val_loader, bce, dice, device)

        epoch_time = float(time.time() - t0)
        lr = float(opt.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_iou": val["iou"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_f1": val["f1"],
            "lr": lr,
            "epoch_time_sec": epoch_time,
        }
        history.append(row)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    train_loss,
                    val["loss"],
                    val["iou"],
                    val["precision"],
                    val["recall"],
                    val["f1"],
                    lr,
                    epoch_time,
                ]
            )

        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, out_dir / "last_epoch.pth")

        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_iou": best_iou}, out_dir / "best_iou.pth")

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_f1": best_f1}, out_dir / "best_f1.pth")

        print(
            "epoch {e}: train_loss={tl:.5f} val_iou={iou:.4f} val_precision={p:.4f} val_recall={r:.4f} val_f1={f1:.4f}".format(
                e=epoch,
                tl=train_loss,
                iou=val["iou"],
                p=val["precision"],
                r=val["recall"],
                f1=val["f1"],
            )
        )

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Targets requested by problem statement.
    target_report = {
        "target_iou_gt_0_75": best_iou > 0.75,
        "target_precision_gt_0_80": any(h["val_precision"] > 0.80 for h in history),
        "target_recall_gt_0_80": any(h["val_recall"] > 0.80 for h in history),
        "target_f1_gt_0_80": best_f1 > 0.80,
        "best_iou": best_iou,
        "best_f1": best_f1,
        "epochs": args.epochs,
    }
    (out_dir / "target_check.json").write_text(json.dumps(target_report, indent=2), encoding="utf-8")
    print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train U-Net cloud detector (BCE + Dice)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", default="checkpoints_unet_cloud")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mixed_precision", action="store_true")
    run(p.parse_args())
