import os
import glob
import logging
from typing import Tuple

import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio

logger = logging.getLogger(__name__)


class Pix2PixDataset(Dataset):
    """
    Pix2Pix Dataset for Cloud Removal.

    Features:
    - Explicit cloudy/clear filename matching
    - CRS validation
    - GeoTransform validation
    - Shape validation
    - Dtype validation
    - NoData handling
    - GAN normalization [-1, 1]
    - Paired augmentations
    """

    def __init__(
        self,
        cloudy_dir: str,
        clear_dir: str,
        transform: bool = False,
        bands: Tuple[int, ...] = (1, 2, 3)
    ):
        self.cloudy_dir = cloudy_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.bands = bands

        cloudy_files = sorted(
            glob.glob(os.path.join(cloudy_dir, "*.tif"))
        )

        clear_files = sorted(
            glob.glob(os.path.join(clear_dir, "*.tif"))
        )

        if len(cloudy_files) == 0:
            raise RuntimeError(
                f"No TIFF files found in {cloudy_dir}"
            )

        cloudy_lookup = {
            os.path.splitext(
                os.path.basename(p)
            )[0]: p
            for p in cloudy_files
        }

        clear_lookup = {
            os.path.splitext(
                os.path.basename(p)
            )[0]: p
            for p in clear_files
        }

        self.samples = []

        for name in cloudy_lookup:

            if name not in clear_lookup:
                raise RuntimeError(
                    f"Missing clear image for: {name}"
                )

            self.samples.append(
                (
                    cloudy_lookup[name],
                    clear_lookup[name]
                )
            )

        logger.info(
            f"Loaded {len(self.samples)} cloudy-clear pairs."
        )

    def __len__(self):
        return len(self.samples)

    def _normalize_for_gan(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Robust normalization to [-1, 1].

        Uses 2-98 percentile clipping to reduce
        influence of outliers.
        """

        image = np.nan_to_num(image)

        norm_image = np.zeros_like(
            image,
            dtype=np.float32
        )

        for i in range(image.shape[0]):

            band = image[i]

            p2 = np.percentile(band, 2)
            p98 = np.percentile(band, 98)

            if p98 <= p2:
                p2 = band.min()
                p98 = band.max()

            if p98 > p2:

                band = np.clip(
                    band,
                    p2,
                    p98
                )

                band = (
                    band - p2
                ) / (
                    p98 - p2
                )

                norm_image[i] = (
                    band * 2.0
                ) - 1.0

            else:
                norm_image[i] = -1.0

        return norm_image

    def __getitem__(self, idx):

        cloudy_path, clear_path = self.samples[idx]

        try:

            with rasterio.open(cloudy_path) as src:

                cloudy_img = src.read(
                    list(self.bands)
                ).astype(np.float32)

                cloudy_crs = src.crs
                cloudy_transform = src.transform
                cloudy_dtype = src.dtypes[0]
                cloudy_nodata = src.nodata

                if cloudy_nodata is not None:
                    cloudy_img[
                        cloudy_img == cloudy_nodata
                    ] = np.nan

            with rasterio.open(clear_path) as src:

                clear_img = src.read(
                    list(self.bands)
                ).astype(np.float32)

                clear_crs = src.crs
                clear_transform = src.transform
                clear_dtype = src.dtypes[0]
                clear_nodata = src.nodata

                if clear_nodata is not None:
                    clear_img[
                        clear_img == clear_nodata
                    ] = np.nan

            # CRS validation

            if cloudy_crs != clear_crs:
                raise ValueError(
                    f"CRS mismatch\n"
                    f"Cloudy: {cloudy_crs}\n"
                    f"Clear : {clear_crs}"
                )

            # Transform validation

            if cloudy_transform != clear_transform:
                raise ValueError(
                    "GeoTransform mismatch"
                )

            # Shape validation

            if cloudy_img.shape != clear_img.shape:
                raise ValueError(
                    f"Shape mismatch\n"
                    f"{cloudy_img.shape}\n"
                    f"{clear_img.shape}"
                )

            # Dtype validation

            if cloudy_dtype != clear_dtype:
                logger.warning(
                    f"Dtype mismatch: "
                    f"{cloudy_dtype} vs {clear_dtype}"
                )

            # GAN normalization

            cloudy_img = self._normalize_for_gan(
                cloudy_img
            )

            clear_img = self._normalize_for_gan(
                clear_img
            )

            cloudy_tensor = torch.from_numpy(
                cloudy_img
            )

            clear_tensor = torch.from_numpy(
                clear_img
            )

            # Augmentations

            if self.transform:

                if torch.rand(1).item() > 0.5:
                    cloudy_tensor = torch.flip(
                        cloudy_tensor,
                        dims=[2]
                    )

                    clear_tensor = torch.flip(
                        clear_tensor,
                        dims=[2]
                    )

                if torch.rand(1).item() > 0.5:
                    cloudy_tensor = torch.flip(
                        cloudy_tensor,
                        dims=[1]
                    )

                    clear_tensor = torch.flip(
                        clear_tensor,
                        dims=[1]
                    )

                k = torch.randint(
                    0,
                    4,
                    (1,)
                ).item()

                if k > 0:

                    cloudy_tensor = torch.rot90(
                        cloudy_tensor,
                        k,
                        [1, 2]
                    )

                    clear_tensor = torch.rot90(
                        clear_tensor,
                        k,
                        [1, 2]
                    )

            return (
                cloudy_tensor,
                clear_tensor
            )

        except Exception as e:

            logger.error(
                f"Failed loading sample {idx}\n"
                f"Cloudy: {cloudy_path}\n"
                f"Clear : {clear_path}\n"
                f"Error : {str(e)}"
            )

            raise RuntimeError(
                "Pix2Pix dataset corruption detected."
            ) from e