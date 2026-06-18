import os
import json
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

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


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (
            probs * targets
        ).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum() +
            targets.sum() +
            self.smooth
        )

        return 1.0 - dice


def validate(
    model,
    loader,
    criterion_bce,
    criterion_dice,
    device
):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            bce = criterion_bce(
                logits,
                masks
            )

            dice = criterion_dice(
                logits,
                masks
            )

            loss = bce + dice

            running_loss += (
                loss.item() *
                images.size(0)
            )

    model.train()

    return running_loss / max(
        1,
        len(loader.dataset)
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True
    )

    parser.add_argument(
        "--save-dir",
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8
    )

    parser.add_argument(
        "--resume",
        default=None
    )

    args = parser.parse_args()

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    with open(
        os.path.join(
            args.save_dir,
            "config.json"
        ),
        "w"
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=4
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        f"Training on {device}"
    )

    scaler = (
        GradScaler()
        if device.type == "cuda"
        else None
    )

    writer = SummaryWriter(
        log_dir=os.path.join(
            args.save_dir,
            "logs"
        )
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_filters=32
    ).to(device)

    criterion_bce = (
        nn.BCEWithLogitsLoss()
    )

    criterion_dice = DiceLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3
        )
    )

    train_dataset = CloudDataset(
        image_dir=os.path.join(
            args.data_dir,
            "train/images"
        ),
        mask_dir=os.path.join(
            args.data_dir,
            "train/masks"
        ),
        transform=True
    )

    val_dataset = CloudDataset(
        image_dir=os.path.join(
            args.data_dir,
            "val/images"
        ),
        mask_dir=os.path.join(
            args.data_dir,
            "val/masks"
        ),
        transform=False
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset empty"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    best_loss = float("inf")
    patience_counter = 0

    history = []

    start_epoch = 1

    if args.resume and os.path.exists(args.resume):

        checkpoint = torch.load(
            args.resume,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        best_loss = checkpoint["best_loss"]

        start_epoch = (
            checkpoint["epoch"] + 1
        )

    for epoch in range(
        start_epoch,
        args.epochs + 1
    ):

        model.train()

        epoch_loss = 0.0

        for images, masks in train_loader:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            if scaler:

                with autocast():

                    logits = model(images)

                    loss = (
                        criterion_bce(
                            logits,
                            masks
                        )
                        +
                        criterion_dice(
                            logits,
                            masks
                        )
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                logits = model(images)

                loss = (
                    criterion_bce(
                        logits,
                        masks
                    )
                    +
                    criterion_dice(
                        logits,
                        masks
                    )
                )

                loss.backward()

                optimizer.step()

            epoch_loss += loss.item()

        train_loss = (
            epoch_loss /
            max(
                1,
                len(train_loader)
            )
        )

        val_loss = validate(
            model,
            val_loader,
            criterion_bce,
            criterion_dice,
            device
        )

        scheduler.step(
            val_loss
        )

        logger.info(
            f"Epoch {epoch} "
            f"| Train {train_loss:.4f} "
            f"| Val {val_loss:.4f}"
        )

        writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch
        )

        writer.add_scalar(
            "Loss/Val",
            val_loss,
            epoch
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        with open(
            os.path.join(
                args.save_dir,
                "training_history.json"
            ),
            "w"
        ) as f:
            json.dump(
                history,
                f,
                indent=4
            )

        torch.save({
            "epoch": epoch,
            "best_loss": best_loss,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict()
        },
        os.path.join(
            args.save_dir,
            "latest.pth"
        ))

        if val_loss < best_loss:

            best_loss = val_loss
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "best_loss": best_loss,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "scheduler_state_dict":
                    scheduler.state_dict()
            },
            os.path.join(
                args.save_dir,
                "best_unet.pth"
            ))

        else:

            patience_counter += 1

            if patience_counter >= args.patience:

                logger.info(
                    "Early stopping triggered."
                )

                break

    writer.close()

    logger.info(
        "Cloud detector training complete."
    )


if __name__ == "__main__":
    main()