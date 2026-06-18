import os
import glob
import argparse
import logging
import numpy as np
import rasterio

from skimage.morphology import binary_dilation, disk
from scipy.ndimage import gaussian_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def generate_cloud_mask(
    h,
    w,
    coverage=0.35,
    tolerance=0.03,
    seed=None
):
    """
    Generate cloud mask close to requested coverage.
    """

    if seed is not None:
        np.random.seed(seed)

    for _ in range(20):

        mask = np.random.rand(h, w) < 0.02

        iterations = np.random.randint(3, 7)

        for _ in range(iterations):
            radius = np.random.randint(5, 20)
            mask = binary_dilation(mask, disk(radius))

        current = mask.mean()

        if abs(current - coverage) <= tolerance:
            break

    return mask.astype(np.uint8)


def generate_opacity_map(mask):
    """
    Create semi-transparent cloud regions.
    """

    opacity = gaussian_filter(
        mask.astype(np.float32),
        sigma=8
    )

    opacity = opacity / (
        opacity.max() + 1e-8
    )

    opacity = 0.3 + 0.7 * opacity

    return opacity


def generate_shadow_map(
    mask,
    shadow_strength=0.15
):
    """
    Create cloud shadows.
    """

    shadow = np.roll(mask, 15, axis=0)
    shadow = np.roll(shadow, 15, axis=1)

    shadow = gaussian_filter(
        shadow.astype(np.float32),
        sigma=6
    )

    shadow = shadow / (
        shadow.max() + 1e-8
    )

    shadow *= shadow_strength

    return shadow


def apply_clouds(
    image,
    mask,
    opacity,
    shadow
):
    """
    Spectrally-aware cloud simulation.
    """

    image = image.astype(np.float32)

    cloudy = image.copy()

    n_bands = image.shape[0]

    dtype_max = (
        np.iinfo(image.dtype).max
        if np.issubdtype(image.dtype, np.integer)
        else float(np.max(image))
    )

    for b in range(n_bands):

        if b == 0:
            cloud_reflectance = 0.85
        elif b == 1:
            cloud_reflectance = 0.90
        else:
            cloud_reflectance = 1.00

        cloud_value = (
            cloud_reflectance *
            dtype_max
        )

        cloudy[b] = (
            image[b] * (1.0 - opacity)
            +
            cloud_value * opacity
        )

        cloudy[b] = (
            cloudy[b] *
            (1.0 - shadow)
        )

    cloudy = np.clip(
        cloudy,
        0,
        dtype_max
    )

    return cloudy.astype(image.dtype)


def process_directory(
    clear_dir,
    out_cloudy,
    out_mask,
    coverage,
    seed,
    shadow_strength
):

    os.makedirs(out_cloudy, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    files = sorted(
        glob.glob(
            os.path.join(clear_dir, "*.tif")
        )
    )

    logger.info(
        f"Found {len(files)} files."
    )

    for idx, filepath in enumerate(files):

        filename = os.path.basename(
            filepath
        )

        with rasterio.open(filepath) as src:

            image = src.read()
            profile = src.profile

            _, h, w = image.shape

        local_seed = (
            seed + idx
            if seed is not None
            else None
        )

        mask = generate_cloud_mask(
            h,
            w,
            coverage=coverage,
            seed=local_seed
        )

        opacity = generate_opacity_map(
            mask
        )

        shadow = generate_shadow_map(
            mask,
            shadow_strength
        )

        cloudy = apply_clouds(
            image,
            mask,
            opacity,
            shadow
        )

        cloudy_path = os.path.join(
            out_cloudy,
            filename
        )

        with rasterio.open(
            cloudy_path,
            "w",
            **profile
        ) as dst:
            dst.write(cloudy)

        mask_profile = profile.copy()

        mask_profile.update(
            count=1,
            dtype=rasterio.uint8
        )

        mask_path = os.path.join(
            out_mask,
            filename
        )

        with rasterio.open(
            mask_path,
            "w",
            **mask_profile
        ) as dst:
            dst.write(mask, 1)

        logger.info(
            f"Generated {filename}"
        )

    logger.info(
        "Synthetic dataset generation complete."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clear-dir",
        required=True
    )

    parser.add_argument(
        "--out-cloudy",
        required=True
    )

    parser.add_argument(
        "--out-mask",
        required=True
    )

    parser.add_argument(
        "--coverage",
        type=float,
        default=0.35
    )

    parser.add_argument(
        "--shadow-strength",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    process_directory(
        args.clear_dir,
        args.out_cloudy,
        args.out_mask,
        args.coverage,
        args.seed,
        args.shadow_strength
    )