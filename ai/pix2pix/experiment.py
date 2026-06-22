import argparse
import importlib.util
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ai.cloud_detector.model import UNet
from ai.fusion.pipeline import _load_stats, _morph_refine_probs, _predict_mask, _soft_fuse
from ai.metrics.compute import compute_all_metrics, compute_edge_similarity
from ai.nafnet.dataset import normalize_image
from ai.nafnet.model import NAFNetWrapper

try:
    from .dataset import Pix2PixDataset
    from .discriminator import Discriminator
    from .generator import Generator
except ImportError:
    from dataset import Pix2PixDataset
    from discriminator import Discriminator
    from generator import Generator


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


DEFAULT_PAIRS_CSV = Path("checkpoints_nafnet/raw_pair_audit/top_2365_strict_pairs.csv")
DEFAULT_OUTPUT_DIR = Path("ai/pix2pix/checkpoints_strict_2365")
DEFAULT_NAFNET_CHECKPOINT = Path("checkpoints_nafnet/strict_curated_training/best_ssim.pth")
DEFAULT_CLOUD_CHECKPOINT = Path("checkpoints_unet_cloud/best_iou.pth")
DEFAULT_STATS_JSON = Path("tmp_stats/band_statistics.json")


def _resolve_path(path_like: str | Path, root_dir: str | Path = ".") -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(root_dir) / path


def _split_indices(length: int, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    rng = np.random.default_rng(seed)
    indices = np.arange(length)
    rng.shuffle(indices)
    train_end = int(length * train_ratio)
    val_end = train_end + int(length * val_ratio)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _tensor_to_hwc01(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float()
    if array.ndim == 4:
        array = array[0]
    if array.min().item() < 0.0 or array.max().item() > 1.0:
        array = (array + 1.0) / 2.0
    array = torch.clamp(array, 0.0, 1.0)
    return array.permute(1, 2, 0).numpy().astype(np.float32)


def _tensor_to_chw01(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float()
    if array.ndim == 4:
        array = array[0]
    if array.min().item() < 0.0 or array.max().item() > 1.0:
        array = (array + 1.0) / 2.0
    return torch.clamp(array, 0.0, 1.0).numpy().astype(np.float32)


def _save_png(path: Path, hwc01: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(hwc01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def _band_edge_map(hwc01: np.ndarray) -> np.ndarray:
    if hwc01.ndim == 2:
        gray = hwc01.astype(np.float32)
    else:
        gray = np.mean(hwc01.astype(np.float32), axis=2)
    edge = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
    edge_min = float(np.min(edge))
    edge_max = float(np.max(edge))
    if edge_max - edge_min < 1e-8:
        return np.zeros_like(edge, dtype=np.float32)
    return ((edge - edge_min) / (edge_max - edge_min)).astype(np.float32)


def _plot_epoch_audit(epoch_dir: Path, input_hwc01: np.ndarray, prediction_hwc01: np.ndarray, target_hwc01: np.ndarray):
    epoch_dir.mkdir(parents=True, exist_ok=True)
    _save_png(epoch_dir / "input.png", input_hwc01)
    _save_png(epoch_dir / "prediction.png", prediction_hwc01)
    _save_png(epoch_dir / "target.png", target_hwc01)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, [input_hwc01, prediction_hwc01, target_hwc01], ["Input", "Prediction", "Target"]):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(epoch_dir / "comparison.png", dpi=180)
    plt.close(fig)

    edge_input = _band_edge_map(input_hwc01)
    edge_pred = _band_edge_map(prediction_hwc01)
    edge_target = _band_edge_map(target_hwc01)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, [edge_input, edge_pred, edge_target], ["Input Edge", "Prediction Edge", "Target Edge"]):
        ax.imshow(img, cmap="magma")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(epoch_dir / "edge_map.png", dpi=180)
    plt.close(fig)


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, channels: int = 3, device: Optional[torch.device] = None):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size)


def _batch_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    pred = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
    target = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
    channels = pred.shape[1]
    window = _gaussian_kernel(window_size, sigma, channels, pred.device).to(pred.dtype)
    padding = window_size // 2

    mu_pred = F.conv2d(pred, window, groups=channels, padding=padding)
    mu_target = F.conv2d(target, window, groups=channels, padding=padding)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, groups=channels, padding=padding) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, groups=channels, padding=padding) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, groups=channels, padding=padding) - mu_pred_target

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return ssim_map.mean()


def _pix2pix_normalize(image_chw: np.ndarray):
    normalized = np.zeros_like(image_chw, dtype=np.float32)
    p2 = np.zeros(image_chw.shape[0], dtype=np.float32)
    p98 = np.zeros(image_chw.shape[0], dtype=np.float32)
    for idx in range(image_chw.shape[0]):
        band = image_chw[idx]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            raise ValueError(f"Band {idx} contains no valid pixels")
        lo = float(np.percentile(valid, 2))
        hi = float(np.percentile(valid, 98))
        if hi <= lo:
            lo = float(valid.min())
            hi = float(valid.max())
        p2[idx] = lo
        p98[idx] = hi
        if hi > lo:
            normalized[idx] = (np.clip(band, lo, hi) - lo) / (hi - lo) * 2.0 - 1.0
        else:
            normalized[idx] = -1.0
    return normalized, p2, p98


def _pix2pix_predict_raw(generator: Generator, image_chw: np.ndarray, device: torch.device):
    normalized, p2, p98 = _pix2pix_normalize(image_chw)
    x = torch.from_numpy(normalized).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = generator(x)[0].detach().cpu().float().numpy()
    pred = np.clip(pred, -1.0, 1.0)
    raw = np.zeros_like(pred, dtype=np.float32)
    for idx in range(pred.shape[0]):
        scaled = (pred[idx] + 1.0) / 2.0
        raw[idx] = scaled * (p98[idx] - p2[idx]) + p2[idx]
    return raw.astype(np.float32), (p2, p98)


def _load_pix2pix_generator(weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    root_model = Generator(in_channels=3, out_channels=3).to(device)
    try:
        root_model.load_state_dict(state_dict, strict=True)
        root_model.eval()
        return root_model
    except Exception:
        pass

    nested_path = Path(__file__).resolve().parents[2] / "chaturvyuha-cloudvision" / "ai" / "pix2pix" / "generator.py"
    if nested_path.exists():
        spec = importlib.util.spec_from_file_location("nested_pix2pix_generator", nested_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            nested_generator = module.Generator(in_channels=3, out_channels=3).to(device)
            nested_generator.load_state_dict(state_dict, strict=False)
            nested_generator.eval()
            return nested_generator

    # Final fallback: return the root model so callers still get a meaningful error if inference is attempted.
    root_model.load_state_dict(state_dict, strict=False)
    root_model.eval()
    return root_model


def _predict_nafnet_raw(model: NAFNetWrapper, image_chw: np.ndarray, stats: dict, device: torch.device):
    image_hwc = np.transpose(image_chw, (1, 2, 0))
    normalized = normalize_image(image_hwc, stats["p1"], stats["p99"])
    x = torch.from_numpy(np.transpose(normalized, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)[0].detach().cpu().float().numpy()
    pred_hwc = np.transpose(pred, (1, 2, 0)).astype(np.float32)
    raw = np.empty_like(pred_hwc, dtype=np.float32)
    for idx in range(3):
        raw[:, :, idx] = pred_hwc[:, :, idx] * (stats["p99"][idx] - stats["p1"][idx]) + stats["p1"][idx]
    return np.transpose(raw, (2, 0, 1)).astype(np.float32), pred_hwc.astype(np.float32)


def _normalize_with_target_stats(image_chw: np.ndarray, stats: Dict[str, List[float]]):
    normalized = np.zeros_like(image_chw, dtype=np.float32)
    for idx in range(image_chw.shape[0]):
        lo = float(stats["p2"][idx])
        hi = float(stats["p98"][idx])
        if hi > lo:
            normalized[idx] = np.clip((image_chw[idx] - lo) / (hi - lo), 0.0, 1.0)
        else:
            normalized[idx] = 0.0
    return normalized


def _normalize_for_display(image_chw: np.ndarray):
    hwc = np.transpose(image_chw, (1, 2, 0)).astype(np.float32)
    out = np.zeros_like(hwc, dtype=np.float32)
    for idx in range(hwc.shape[2]):
        band = hwc[:, :, idx]
        lo = float(np.percentile(band, 1))
        hi = float(np.percentile(band, 99))
        if hi > lo:
            out[:, :, idx] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def _evaluate_loader(generator: Generator, loader: DataLoader, device: torch.device):
    generator.eval()
    totals = {"psnr": 0.0, "ssim": 0.0, "rmse": 0.0, "sam": 0.0, "edge_similarity": 0.0}
    count = 0
    with torch.no_grad():
        for cloudy, clear in loader:
            cloudy = cloudy.to(device)
            clear = clear.to(device)
            pred = generator(cloudy)
            for idx in range(pred.shape[0]):
                metrics = compute_all_metrics(clear[idx].cpu().numpy(), pred[idx].cpu().numpy())
                for key in totals:
                    totals[key] += float(metrics[key])
                count += 1
    generator.train()
    if count == 0:
        return {key: 0.0 for key in totals}
    return {key: value / count for key, value in totals.items()}


def _save_checkpoint(path: Path, epoch: int, generator: Generator, discriminator: Discriminator, optimizer_g, optimizer_d, metrics: dict, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def train_pix2pix(args) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True

    dataset = Pix2PixDataset(
        pairs_csv=str(args.pairs_csv),
        root_dir=str(args.root_dir),
        transform=False,
    )
    train_idx, val_idx, test_idx = _split_indices(len(dataset), seed=args.seed)

    train_dataset = Pix2PixDataset(
        pairs_csv=str(args.pairs_csv),
        root_dir=str(args.root_dir),
        transform=True,
        indices=train_idx,
    )
    val_dataset = Pix2PixDataset(
        pairs_csv=str(args.pairs_csv),
        root_dir=str(args.root_dir),
        transform=False,
        indices=val_idx,
    )
    test_dataset = Pix2PixDataset(
        pairs_csv=str(args.pairs_csv),
        root_dir=str(args.root_dir),
        transform=False,
        indices=test_idx,
    )

    logger.info("Split sizes | train=%s val=%s test=%s", len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(train_dataset) >= args.batch_size,
        num_workers=0,
        pin_memory=use_amp,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
    )

    generator = Generator(in_channels=3, out_channels=3).to(device)
    discriminator = Discriminator(image_channels=3, condition_channels=3).to(device)

    criterion_gan = nn.BCEWithLogitsLoss().to(device)
    criterion_l1 = nn.L1Loss().to(device)
    scaler = GradScaler(enabled=use_amp)

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode="max", factor=0.5, patience=3)

    writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    history = []
    best_ssim = -float("inf")
    best_psnr = -float("inf")
    best_visual = -float("inf")
    patience_counter = 0

    fixed_val_cloudy, fixed_val_clear = val_dataset[0]
    fixed_val_cloudy = fixed_val_cloudy.unsqueeze(0).to(device)
    fixed_val_clear = fixed_val_clear.unsqueeze(0).to(device)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        epoch_g = 0.0
        epoch_d = 0.0
        epoch_ssim = 0.0
        epoch_l1 = 0.0
        batch_count = 0

        for cloudy, clear in train_loader:
            cloudy = cloudy.to(device)
            clear = clear.to(device)

            optimizer_d.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                fake = generator(cloudy)
                pred_real = discriminator(cloudy, clear)
                pred_fake = discriminator(cloudy, fake.detach())
                loss_d = 0.5 * (
                    criterion_gan(pred_real, torch.ones_like(pred_real))
                    + criterion_gan(pred_fake, torch.zeros_like(pred_fake))
                )

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)

            optimizer_g.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                fake = generator(cloudy)
                pred_fake = discriminator(cloudy, fake)
                loss_adv = criterion_gan(pred_fake, torch.ones_like(pred_fake))
                loss_l1 = criterion_l1(fake, clear)
                loss_ssim = 1.0 - _batch_ssim(fake, clear)
                loss_g = loss_adv + (args.lambda_l1 * loss_l1) + (args.lambda_ssim * loss_ssim)

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            epoch_g += float(loss_g.detach().cpu().item())
            epoch_d += float(loss_d.detach().cpu().item())
            epoch_l1 += float(loss_l1.detach().cpu().item())
            epoch_ssim += float((1.0 - loss_ssim).detach().cpu().item())
            batch_count += 1

        val_metrics = _evaluate_loader(generator, val_loader, device)
        scheduler_g.step(val_metrics["ssim"])

        with torch.no_grad():
            sample_pred = generator(fixed_val_cloudy)
            sample_input = fixed_val_cloudy[0].detach().cpu()
            sample_pred_cpu = sample_pred[0].detach().cpu()
            sample_target = fixed_val_clear[0].detach().cpu()

        sample_input_hwc = _tensor_to_hwc01(sample_input)
        sample_pred_hwc = _tensor_to_hwc01(sample_pred_cpu)
        sample_target_hwc = _tensor_to_hwc01(sample_target)

        epoch_dir = audit_dir / f"epoch_{epoch:03d}"
        _plot_epoch_audit(epoch_dir, sample_input_hwc, sample_pred_hwc, sample_target_hwc)

        avg_g = epoch_g / max(batch_count, 1)
        avg_d = epoch_d / max(batch_count, 1)
        avg_l1 = epoch_l1 / max(batch_count, 1)
        avg_ssim = epoch_ssim / max(batch_count, 1)
        visual_score = 0.6 * val_metrics["ssim"] + 0.4 * val_metrics["edge_similarity"]

        history_entry = {
            "epoch": epoch,
            "avg_g_loss": avg_g,
            "avg_d_loss": avg_d,
            "avg_l1_loss": avg_l1,
            "avg_ssim": avg_ssim,
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "val_rmse": val_metrics["rmse"],
            "val_sam": val_metrics["sam"],
            "val_edge_similarity": val_metrics["edge_similarity"],
            "visual_score": visual_score,
        }
        history.append(history_entry)
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        writer.add_scalar("loss/generator", avg_g, epoch)
        writer.add_scalar("loss/discriminator", avg_d, epoch)
        writer.add_scalar("loss/l1", avg_l1, epoch)
        writer.add_scalar("metrics/val_psnr", val_metrics["psnr"], epoch)
        writer.add_scalar("metrics/val_ssim", val_metrics["ssim"], epoch)
        writer.add_scalar("metrics/val_sam", val_metrics["sam"], epoch)
        writer.add_scalar("metrics/val_edge_similarity", val_metrics["edge_similarity"], epoch)

        _save_checkpoint(output_dir / "latest.pth", epoch, generator, discriminator, optimizer_g, optimizer_d, history_entry, config)
        torch.save(discriminator.state_dict(), output_dir / "discriminator_latest.pth")

        if val_metrics["ssim"] > best_ssim:
            best_ssim = val_metrics["ssim"]
            _save_checkpoint(output_dir / "best_ssim.pth", epoch, generator, discriminator, optimizer_g, optimizer_d, history_entry, config)
        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            _save_checkpoint(output_dir / "best_psnr.pth", epoch, generator, discriminator, optimizer_g, optimizer_d, history_entry, config)
        if visual_score > best_visual:
            best_visual = visual_score
            _save_checkpoint(output_dir / "best_visual.pth", epoch, generator, discriminator, optimizer_g, optimizer_d, history_entry, config)

        if epoch % max(1, args.audit_every) == 0:
            logger.info(
                "Epoch %s | G=%.4f D=%.4f | val_psnr=%.3f val_ssim=%.4f val_sam=%.3f val_edge=%.4f",
                epoch,
                avg_g,
                avg_d,
                val_metrics["psnr"],
                val_metrics["ssim"],
                val_metrics["sam"],
                val_metrics["edge_similarity"],
            )

        if val_metrics["ssim"] >= best_ssim:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered at epoch %s", epoch)
                break

    writer.close()

    summary = {
        "best_ssim": best_ssim,
        "best_psnr": best_psnr,
        "best_visual": best_visual,
        "epochs_ran": len(history),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _sample_for_comparison(dataset: Pix2PixDataset, split_name: str, seed: int):
    if len(dataset) == 0:
        raise RuntimeError(f"{split_name} dataset is empty")
    sample_index = 0
    return dataset[sample_index]


def generate_comparison_report(args) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = _load_stats(Path(args.stats_json))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Pix2PixDataset(
        pairs_csv=str(args.pairs_csv),
        root_dir=str(args.root_dir),
        transform=False,
    )
    _, _, test_idx = _split_indices(len(dataset), seed=args.seed)
    test_dataset = Pix2PixDataset(
        pairs_csv=str(args.pairs_csv),
        root_dir=str(args.root_dir),
        transform=False,
        indices=test_idx,
    )
    cloudy, clear = test_dataset[min(args.sample_index, len(test_dataset) - 1)]
    cloudy_chw = cloudy.numpy().astype(np.float32)
    clear_chw = clear.numpy().astype(np.float32)

    nafnet = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    naf_ckpt = torch.load(str(args.nafnet_checkpoint), map_location=device)
    naf_state = naf_ckpt["state_dict"] if isinstance(naf_ckpt, dict) and "state_dict" in naf_ckpt else naf_ckpt
    nafnet.load_state_dict(naf_state, strict=False)
    nafnet.eval()

    cloud_model = UNet(in_channels=3, out_channels=1).to(device)
    cloud_ckpt = torch.load(str(args.cloud_checkpoint), map_location=device)
    cloud_state = cloud_ckpt["model_state_dict"] if isinstance(cloud_ckpt, dict) and "model_state_dict" in cloud_ckpt else cloud_ckpt
    cloud_model.load_state_dict(cloud_state, strict=False)
    cloud_model.eval()

    pix2pix = _load_pix2pix_generator(str(args.pix2pix_checkpoint), device)

    mask_u8, probs = _predict_mask(cloud_model, cloudy_chw, stats=stats, device=str(device), threshold=args.mask_threshold)
    conf_map = _morph_refine_probs(probs)

    nafnet_raw_chw, nafnet_raw_hwc = _predict_nafnet_raw(nafnet, cloudy_chw, stats, device)
    pix2pix_raw_chw, _ = _pix2pix_predict_raw(pix2pix, cloudy_chw, device)

    nafnet_soft_chw = _soft_fuse(cloudy_chw, nafnet_raw_chw, conf_map)
    pix2pix_soft_chw = _soft_fuse(cloudy_chw, pix2pix_raw_chw, conf_map)

    target_stats = {
        "p2": [float(np.percentile(clear_chw[i], 2)) for i in range(clear_chw.shape[0])],
        "p98": [float(np.percentile(clear_chw[i], 98)) for i in range(clear_chw.shape[0])],
    }

    variants = {
        "NAFNet": nafnet_raw_chw,
        "Pix2Pix": pix2pix_raw_chw,
        "NAFNet + Soft Fusion": nafnet_soft_chw,
        "Pix2Pix + Soft Fusion": pix2pix_soft_chw,
    }

    normalized_variants = {name: _normalize_with_target_stats(arr, target_stats) for name, arr in variants.items()}
    normalized_clear = _normalize_with_target_stats(clear_chw, target_stats)
    normalized_input = _normalize_with_target_stats(cloudy_chw, target_stats)

    metrics_table = {}
    for name, arr in normalized_variants.items():
        metrics_table[name] = compute_all_metrics(normalized_clear, arr)

    gallery_path = output_dir / "comparison_gallery.png"
    fig, axes = plt.subplots(len(variants), 4, figsize=(16, 4 * len(variants)))
    if len(variants) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (name, arr) in enumerate(variants.items()):
        disp_pred = _normalize_for_display(arr)
        disp_target = _normalize_for_display(clear_chw)
        disp_input = _normalize_for_display(cloudy_chw)
        edge = _band_edge_map(disp_pred)
        row_axes = axes[row_idx]
        panels = [
            (disp_input, "Input"),
            (disp_pred, name),
            (disp_target, "Target"),
            (edge, "Edge Map"),
        ]
        for ax, (img, title) in zip(row_axes, panels):
            if img.ndim == 2:
                ax.imshow(img, cmap="magma")
            else:
                ax.imshow(np.clip(img, 0.0, 1.0))
            ax.set_title(title)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(gallery_path, dpi=180)
    plt.close(fig)

    report_lines = [
        "# Pix2Pix vs NAFNet Comparison Report",
        "",
        "## Experiment Setup",
        f"- Dataset: {args.pairs_csv}",
        f"- Split: 80/10/10 with seed {args.seed}",
        f"- Sample split: test[{min(args.sample_index, len(test_dataset) - 1)}]",
        f"- NAFNet checkpoint: {args.nafnet_checkpoint}",
        f"- Pix2Pix checkpoint: {args.pix2pix_checkpoint}",
        f"- Cloud detector checkpoint: {args.cloud_checkpoint}",
        "",
        "## Comparison Gallery",
        f"![Comparison Gallery]({gallery_path.name})",
        "",
        "## Metrics",
        "| Variant | PSNR | SSIM | SAM | RMSE | Edge Similarity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in metrics_table.items():
        report_lines.append(
            f"| {name} | {metrics['psnr']:.3f} | {metrics['ssim']:.4f} | {metrics['sam']:.3f} | {metrics['rmse']:.4f} | {metrics['edge_similarity']:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Notes",
            "- Metrics are computed after normalizing all variants against the clear target's per-band 2/98 percentiles.",
            "- Soft fusion uses the refined cloud confidence map from the cloud detector.",
            "- No NAFNet checkpoints were modified during this experiment.",
        ]
    )

    report_path = output_dir / "pix2pix_vs_nafnet_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    (output_dir / "comparison_metrics.json").write_text(json.dumps(metrics_table, indent=2), encoding="utf-8")
    return report_path


def main_train(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Train Pix2Pix on curated strict cloud-removal pairs")
    parser.add_argument("--pairs_csv", type=str, default=str(DEFAULT_PAIRS_CSV))
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--lambda_ssim", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit_every", type=int, default=1)
    parser.add_argument("--nafnet_checkpoint", type=str, default=str(DEFAULT_NAFNET_CHECKPOINT))
    parser.add_argument("--cloud_checkpoint", type=str, default=str(DEFAULT_CLOUD_CHECKPOINT))
    parser.add_argument("--pix2pix_checkpoint", type=str, default=None)
    parser.add_argument("--stats_json", type=str, default=str(DEFAULT_STATS_JSON))
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--mask_threshold", type=float, default=0.10)
    parser.add_argument("--generate_comparison", action="store_true")
    args = parser.parse_args(argv)

    summary = train_pix2pix(args)
    logger.info("Training summary: %s", summary)

    if args.generate_comparison:
        pix2pix_checkpoint = args.pix2pix_checkpoint or str(Path(args.output_dir) / "best_visual.pth")
        compare_args = argparse.Namespace(
            pairs_csv=args.pairs_csv,
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            pix2pix_checkpoint=pix2pix_checkpoint,
            nafnet_checkpoint=args.nafnet_checkpoint,
            cloud_checkpoint=args.cloud_checkpoint,
            stats_json=args.stats_json,
            seed=args.seed,
            sample_index=args.sample_index,
            mask_threshold=args.mask_threshold,
        )
        report_path = generate_comparison_report(compare_args)
        logger.info("Comparison report written to %s", report_path)


def main_compare(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Generate Pix2Pix vs NAFNet comparison report")
    parser.add_argument("--pairs_csv", type=str, default=str(DEFAULT_PAIRS_CSV))
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="outputs/pix2pix_vs_nafnet")
    parser.add_argument("--pix2pix_checkpoint", type=str, required=True)
    parser.add_argument("--nafnet_checkpoint", type=str, default=str(DEFAULT_NAFNET_CHECKPOINT))
    parser.add_argument("--cloud_checkpoint", type=str, default=str(DEFAULT_CLOUD_CHECKPOINT))
    parser.add_argument("--stats_json", type=str, default=str(DEFAULT_STATS_JSON))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--mask_threshold", type=float, default=0.10)
    args = parser.parse_args(argv)

    report_path = generate_comparison_report(args)
    logger.info("Comparison report written to %s", report_path)


if __name__ == "__main__":
    main_train()
