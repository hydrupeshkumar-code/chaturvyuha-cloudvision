import os
import json
import argparse
import logging
from typing import Tuple, Dict, Any

import numpy as np
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fuse_images(
    original_image: np.ndarray,
    reconstructed_image: np.ndarray,
    cloud_mask: np.ndarray,
    alpha: float = 1.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Mask-guided fusion of reconstructed pixels into original imagery.
    """

    if original_image.shape != reconstructed_image.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"original={original_image.shape}, "
            f"reconstructed={reconstructed_image.shape}"
        )

    mask_2d = cloud_mask[0] if cloud_mask.ndim == 3 else cloud_mask

    if mask_2d.shape != original_image.shape[1:]:
        raise ValueError(
            f"Mask shape mismatch: "
            f"{mask_2d.shape} vs {original_image.shape[1:]}"
        )

    cloud_idx = mask_2d > 0

    fused_image = original_image.copy()

    for band in range(fused_image.shape[0]):
        fused_image[band, cloud_idx] = (
            alpha * reconstructed_image[band, cloud_idx]
            + (1.0 - alpha) * original_image[band, cloud_idx]
        )

    pixel_diff = np.abs(
        fused_image.astype(np.float32)
        - original_image.astype(np.float32)
    )

    cloud_pixels = int(np.sum(cloud_idx))
    total_pixels = int(mask_2d.size)

    stats = {
        "cloud_pixels": cloud_pixels,
        "changed_pixels": cloud_pixels,
        "cloud_percentage": round(
            (cloud_pixels / total_pixels) * 100,
            2
        ),
        "total_pixels": total_pixels,
        "fusion_alpha": float(alpha),

        # Radiometric metrics
        "mean_change": float(pixel_diff.mean()),
        "max_change": float(pixel_diff.max()),
        "std_change": float(pixel_diff.std())
    }

    return fused_image, stats


def generate_difference_map(
    original_image: np.ndarray,
    fused_image: np.ndarray
) -> np.ndarray:
    """
    Generates normalized difference heatmap.
    """

    diff = np.abs(
        fused_image.astype(np.float32)
        - original_image.astype(np.float32)
    )

    diff_map = np.mean(diff, axis=0)

    if diff_map.max() > 0:
        diff_map = diff_map / diff_map.max()

    diff_map = np.nan_to_num(diff_map)

    return diff_map.astype(np.float32)


def save_outputs(
    image: np.ndarray,
    profile: dict,
    output_path: str,
    is_single_channel: bool = False
):
    """
    Save GeoTIFF while preserving metadata.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    out_profile = profile.copy()

    out_profile.update(
        compress="lzw"
    )

    if is_single_channel:

        out_profile.update(
            count=1,
            dtype=rasterio.float32
        )

        if image.ndim == 2:
            image = np.expand_dims(
                image,
                axis=0
            )

    else:

        dtype = np.dtype(profile["dtype"])

        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            image = np.clip(
                image,
                info.min,
                info.max
            )

        image = image.astype(dtype)

        out_profile.update(
            count=image.shape[0],
            dtype=dtype
        )

    with rasterio.open(
        output_path,
        "w",
        **out_profile
    ) as dst:

        dst.write(image)

        dst.update_tags(
            generated_by="ChaturVyuha CloudVision AI",
            fusion_method="mask_guided_alpha_blending"
        )


def save_fusion_stats(
    stats: Dict[str, Any],
    output_json: str
):
    """
    Save fusion metadata.
    """

    os.makedirs(
        os.path.dirname(output_json),
        exist_ok=True
    )

    with open(output_json, "w") as f:
        json.dump(stats, f, indent=4)


def validate_geospatial_alignment(
    orig_meta,
    recon_meta,
    mask_meta
):
    """
    Ensure all datasets are spatially aligned.
    """

    if orig_meta["crs"] != recon_meta["crs"]:
        raise ValueError(
            "CRS mismatch between original and reconstruction."
        )

    if orig_meta["crs"] != mask_meta["crs"]:
        raise ValueError(
            "CRS mismatch between original and mask."
        )

    if orig_meta["transform"] != recon_meta["transform"]:
        raise ValueError(
            "Transform mismatch between original and reconstruction."
        )

    if orig_meta["transform"] != mask_meta["transform"]:
        raise ValueError(
            "Transform mismatch between original and mask."
        )


def main():

    parser = argparse.ArgumentParser(
        description="Fuse Cloud Detection + Pix2Pix Reconstruction"
    )

    parser.add_argument(
        "--original",
        required=True
    )

    parser.add_argument(
        "--recon",
        required=True
    )

    parser.add_argument(
        "--mask",
        required=True
    )

    parser.add_argument(
        "--out-fused",
        required=True
    )

    parser.add_argument(
        "--out-diff",
        required=True
    )

    parser.add_argument(
        "--out-stats",
        required=True
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Fusion blending factor (0.0-1.0)"
    )

    args = parser.parse_args()

    try:

        with rasterio.open(args.original) as src:
            orig_img = src.read()
            profile = src.profile

            orig_meta = {
                "crs": src.crs,
                "transform": src.transform
            }

        with rasterio.open(args.recon) as src:
            recon_img = src.read()

            recon_meta = {
                "crs": src.crs,
                "transform": src.transform
            }

        with rasterio.open(args.mask) as src:
            mask_img = src.read()

            mask_meta = {
                "crs": src.crs,
                "transform": src.transform
            }

        validate_geospatial_alignment(
            orig_meta,
            recon_meta,
            mask_meta
        )

        if orig_img.shape != recon_img.shape:
            raise ValueError(
                f"Shape mismatch:\n"
                f"Original: {orig_img.shape}\n"
                f"Recon   : {recon_img.shape}"
            )

        fused_img, stats = fuse_images(
            orig_img,
            recon_img,
            mask_img,
            alpha=args.alpha
        )

        diff_map = generate_difference_map(
            orig_img,
            fused_img
        )

        save_outputs(
            fused_img,
            profile,
            args.out_fused
        )

        save_outputs(
            diff_map,
            profile,
            args.out_diff,
            is_single_channel=True
        )

        save_fusion_stats(
            stats,
            args.out_stats
        )

        logger.info(
            f"Fusion completed successfully:\n"
            f"{json.dumps(stats, indent=2)}"
        )

    except Exception as e:
        logger.exception(
            f"Fusion failed: {str(e)}"
        )
        raise


if __name__ == "__main__":
    main()