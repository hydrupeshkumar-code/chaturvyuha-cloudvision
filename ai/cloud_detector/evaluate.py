import os
import json
import argparse
import logging
from typing import Dict

import torch
import numpy as np
from torch.utils.data import DataLoader

try:
    from .model import UNet
    from .dataset import CloudDataset
except ImportError:
    from model import UNet
    from dataset import CloudDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def compute_metrics(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    smooth: float = 1e-6
) -> Dict[str, float]:

    union = tp + fp + fn

    if union == 0:
        return {
            "iou": 1.0,
            "dice": 1.0,
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0
        }

    iou = (tp + smooth) / (union + smooth)

    precision = (
        tp + smooth
    ) / (
        tp + fp + smooth
    )

    recall = (
        tp + smooth
    ) / (
        tp + fn + smooth
    )

    dice = (
        2 * tp + smooth
    ) / (
        2 * tp + fp + fn + smooth
    )

    accuracy = (
        tp + tn
    ) / (
        tp + tn + fp + fn + smooth
    )

    f1 = (
        2 * precision * recall
    ) / (
        precision + recall + smooth
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate U-Net Cloud Detector"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=str
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold"
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    logger.info(
        f"Running evaluation on {device}"
    )

    dataset = CloudDataset(
        image_dir=os.path.join(
            args.data_dir,
            "images"
        ),
        mask_dir=os.path.join(
            args.data_dir,
            "masks"
        ),
        transform=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_filters=32
    )

    logger.info(
        f"Loading checkpoint: {args.checkpoint}"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    if isinstance(checkpoint, dict) and \
       "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:
        model.load_state_dict(
            checkpoint
        )

    model.to(device)
    model.eval()

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    logger.info(
        "Running inference..."
    )

    with torch.no_grad():

        for images, masks in dataloader:

            images = images.to(device)

            logits = model(images)

            probs = torch.sigmoid(
                logits
            )

            preds = (
                probs > args.threshold
            ).cpu().numpy()

            targets = masks.numpy()

            if preds.shape != targets.shape:
                raise ValueError(
                    f"Shape mismatch: "
                    f"{preds.shape} vs "
                    f"{targets.shape}"
                )

            preds = preds.astype(bool)
            targets = targets.astype(bool)

            tp += np.logical_and(
                preds,
                targets
            ).sum()

            fp += np.logical_and(
                preds,
                np.logical_not(targets)
            ).sum()

            fn += np.logical_and(
                np.logical_not(preds),
                targets
            ).sum()

            tn += np.logical_and(
                np.logical_not(preds),
                np.logical_not(targets)
            ).sum()

    results = compute_metrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn
    )

    results["threshold"] = args.threshold

    logger.info("Evaluation Complete")

    for key, value in results.items():
        logger.info(
            f"{key}: {value:.4f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    output_dir = os.path.dirname(
        args.output
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    with open(args.output, "w") as f:
        json.dump(
            results,
            f,
            indent=4
        )

    logger.info(
        f"Saved metrics -> {args.output}"
    )


if __name__ == "__main__":
    main()