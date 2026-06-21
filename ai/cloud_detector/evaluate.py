import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .dataset import CloudDataset
    from .model import UNet
except ImportError:
    from dataset import CloudDataset
    from model import UNet


def _percentile_vis(chw: np.ndarray) -> np.ndarray:
    hwc = np.transpose(chw, (1, 2, 0)).astype(np.float32)
    out = np.zeros_like(hwc)
    for c in range(hwc.shape[2]):
        ch = hwc[:, :, c]
        p2 = np.percentile(ch, 2)
        p98 = np.percentile(ch, 98)
        out[:, :, c] = 0 if p98 - p2 < 1e-8 else np.clip((ch - p2) / (p98 - p2), 0, 1)
    return out


def evaluate(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = CloudDataset(
        image_dir=str(Path(args.data_dir) / "images"),
        mask_dir=str(Path(args.data_dir) / "masks"),
        augment=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = UNet(in_channels=3, out_channels=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    tp = fp = fn = tn = 0
    overlay_saved = False

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            pred = (probs >= args.threshold).cpu().numpy().astype(np.uint8)[0, 0]
            gt = y.numpy().astype(np.uint8)[0, 0]

            tp += int(np.logical_and(pred == 1, gt == 1).sum())
            fp += int(np.logical_and(pred == 1, gt == 0).sum())
            fn += int(np.logical_and(pred == 0, gt == 1).sum())
            tn += int(np.logical_and(pred == 0, gt == 0).sum())

            if not overlay_saved:
                img = x.cpu().numpy()[0]
                vis = _percentile_vis(img)
                # Overlay prediction in red channel.
                overlay = vis.copy()
                overlay[:, :, 0] = np.clip(overlay[:, :, 0] + 0.5 * pred.astype(np.float32), 0, 1)

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(vis)
                axes[0].set_title("Input")
                axes[0].axis("off")
                axes[1].imshow(gt, cmap="gray")
                axes[1].set_title("GT Mask")
                axes[1].axis("off")
                axes[2].imshow(overlay)
                axes[2].set_title("Prediction Overlay")
                axes[2].axis("off")
                plt.tight_layout()
                fig.savefig(out_dir / "mask_overlay.png", dpi=160)
                plt.close(fig)
                overlay_saved = True

    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    # Confusion matrix image
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Clear", "Pred Cloud"])
    ax.set_yticklabels(["GT Clear", "GT Cloud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=170)
    plt.close(fig)

    metrics = {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": args.threshold,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "# IoU Report",
        "",
        f"- IoU: {iou:.6f}",
        f"- Precision: {precision:.6f}",
        f"- Recall: {recall:.6f}",
        f"- F1: {f1:.6f}",
        "",
        "## Target Check",
        f"- IoU > 0.75: {iou > 0.75}",
        f"- Precision > 0.80: {precision > 0.80}",
        f"- Recall > 0.80: {recall > 0.80}",
        f"- F1 > 0.80: {f1 > 0.80}",
    ]
    (out_dir / "iou_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate U-Net cloud detector")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="checkpoints_unet_cloud/eval")
    p.add_argument("--threshold", type=float, default=0.5)
    evaluate(p.parse_args())
