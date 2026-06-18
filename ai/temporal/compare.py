import os
import json
import logging
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Any
from skimage.metrics import structural_similarity as ssim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def compute_spectral_angle_mapper(
    recon: np.ndarray,
    hist: np.ndarray
) -> float:
    """
    Computes Spectral Angle Mapper (SAM) in degrees.
    Lower is better.
    """

    c, h, w = recon.shape

    recon_flat = recon.reshape(c, -1).T
    hist_flat = hist.reshape(c, -1).T

    dot_product = np.sum(recon_flat * hist_flat, axis=1)

    norm_recon = np.linalg.norm(recon_flat, axis=1)
    norm_hist = np.linalg.norm(hist_flat, axis=1)

    denominator = norm_recon * norm_hist

    valid = denominator > 1e-8

    if not np.any(valid):
        return 0.0

    cos_theta = np.ones_like(dot_product)

    cos_theta[valid] = (
        dot_product[valid] /
        denominator[valid]
    )

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0
    )

    angles = np.degrees(
        np.arccos(cos_theta)
    )

    return float(
        np.mean(
            angles[valid]
        )
    )


def compute_temporal_similarity(
    recon: np.ndarray,
    hist: np.ndarray
) -> float:
    """
    Computes SSIM between reconstruction and
    historical reference image.
    """

    recon_ch = np.transpose(
        recon,
        (1, 2, 0)
    )

    hist_ch = np.transpose(
        hist,
        (1, 2, 0)
    )

    data_range = (
        max(
            float(recon_ch.max()),
            float(hist_ch.max())
        )
        -
        min(
            float(recon_ch.min()),
            float(hist_ch.min())
        )
    )

    if data_range <= 0:
        data_range = 1.0

    min_dim = min(
        hist_ch.shape[0],
        hist_ch.shape[1]
    )

    win_size = 7

    if min_dim < 7:
        win_size = max(
            3,
            (min_dim // 2) * 2 + 1
        )

    return float(
        ssim(
            hist_ch,
            recon_ch,
            data_range=data_range,
            channel_axis=-1,
            win_size=win_size
        )
    )


def compute_temporal_difference(
    recon: np.ndarray,
    hist: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Computes temporal difference map and
    change percentage.
    """

    diff = np.abs(
        recon.astype(np.float32)
        -
        hist.astype(np.float32)
    )

    diff_map = np.mean(
        diff,
        axis=0
    )

    threshold = max(
        np.percentile(diff_map, 75),
        np.mean(diff_map) + np.std(diff_map)
    )

    changed_pixels = np.sum(
        diff_map > threshold
    )

    change_percentage = (
        changed_pixels /
        diff_map.size
    ) * 100.0

    return diff_map, float(change_percentage)


def save_temporal_visualization(
    recon: np.ndarray,
    hist: np.ndarray,
    diff_map: np.ndarray,
    output_png: str
) -> None:
    """
    Save side-by-side temporal comparison.
    """

    os.makedirs(
        os.path.dirname(output_png),
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    def prep_viz(img):

        img = np.transpose(
            img,
            (1, 2, 0)
        )

        max_val = img.max()

        if max_val > 0:
            img = img / max_val

        return np.clip(
            img,
            0,
            1
        )

    axes[0].imshow(
        prep_viz(recon)
    )

    axes[0].set_title(
        "Reconstructed Scene"
    )

    axes[0].axis("off")

    axes[1].imshow(
        prep_viz(hist)
    )

    axes[1].set_title(
        "Historical Reference"
    )

    axes[1].axis("off")

    im = axes[2].imshow(
        diff_map,
        cmap="hot"
    )

    axes[2].set_title(
        "Temporal Difference"
    )

    axes[2].axis("off")

    fig.colorbar(
        im,
        ax=axes[2],
        fraction=0.046,
        pad=0.04
    )

    plt.tight_layout()

    try:
        plt.savefig(
            output_png,
            dpi=150
        )
    finally:
        plt.close(fig)

    logger.info(
        f"Visualization saved to {output_png}"
    )


def compare_temporal(
    recon_path: str,
    hist_path: str,
    out_dir: str
) -> Dict[str, Any]:
    """
    Full temporal comparison workflow.
    """

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    with rasterio.open(recon_path) as src:

        recon = src.read().astype(
            np.float32
        )

        recon_crs = src.crs
        recon_transform = src.transform
        profile = src.profile

    with rasterio.open(hist_path) as src:

        hist = src.read().astype(
            np.float32
        )

        hist_crs = src.crs
        hist_transform = src.transform

    if recon_crs != hist_crs:
        raise ValueError(
            f"CRS mismatch: "
            f"{recon_crs} vs {hist_crs}"
        )

    if recon_transform != hist_transform:
        logger.warning(
            "Spatial transform mismatch detected."
        )

    if recon.shape != hist.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{recon.shape} vs {hist.shape}"
        )

    logger.info(
        "Computing temporal metrics..."
    )

    ssim_val = compute_temporal_similarity(
        recon,
        hist
    )

    sam_val = compute_spectral_angle_mapper(
        recon,
        hist
    )

    diff_map, change_pct = (
        compute_temporal_difference(
            recon,
            hist
        )
    )

    if ssim_val >= 0.90:
        quality = "PASS"
    elif ssim_val >= 0.75:
        quality = "MARGINAL"
    else:
        quality = "FAIL"

    temporal_confidence = round(
        (ssim_val * 100.0)
        *
        (
            1.0 -
            min(change_pct, 100.0) / 100.0
        ),
        2
    )

    png_path = os.path.join(
        out_dir,
        "temporal_diff.png"
    )

    save_temporal_visualization(
        recon,
        hist,
        diff_map,
        png_path
    )

    diff_profile = profile.copy()

    diff_profile.update(
        count=1,
        dtype=rasterio.float32
    )

    diff_tif = os.path.join(
        out_dir,
        "temporal_diff.tif"
    )

    with rasterio.open(
        diff_tif,
        "w",
        **diff_profile
    ) as dst:
        dst.write(
            diff_map.astype(np.float32),
            1
        )

    metrics = {
        "ssim": round(ssim_val, 4),
        "sam_degrees": round(sam_val, 2),
        "mean_diff": round(
            float(np.mean(diff_map)),
            4
        ),
        "change_percentage": round(
            change_pct,
            2
        ),
        "temporal_confidence": temporal_confidence,
        "quality_flag": quality
    }

    metrics_path = os.path.join(
        out_dir,
        "temporal_metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=2
        )

    logger.info(
        f"Temporal analysis complete | "
        f"SSIM={ssim_val:.4f} | "
        f"SAM={sam_val:.2f}° | "
        f"Quality={quality}"
    )

    return metrics