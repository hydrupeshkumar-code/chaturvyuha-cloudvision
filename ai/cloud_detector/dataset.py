
import os
import glob
import logging
from typing import Tuple

import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio

logger = logging.getLogger(__name__)


class CloudDataset(Dataset):
    """
    PyTorch Dataset for cloud detection.

    Features:
    - Explicit image/mask filename matching
    - CRS validation
    - Transform validation
    - Shape validation
    - NoData handling
    - Data augmentation
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: bool = False,
        bands: Tuple[int, ...] = (1, 2, 3)
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.bands = bands

        image_files = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

        if len(image_files) == 0:
            raise RuntimeError(f"No TIFF files found in {image_dir}")

        mask_lookup = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in mask_files
        }

        self.samples = []

        for img_path in image_files:
            name = os.path.splitext(
                os.path.basename(img_path)
            )[0]

            if name not in mask_lookup:
                raise RuntimeError(
                    f"Mask missing for image: {name}"
                )

            self.samples.append(
                (img_path, mask_lookup[name])
            )

        logger.info(
            f"Loaded {len(self.samples)} image-mask pairs."
        )

    def __len__(self):
        return len(self.samples)

    def _normalize(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Min-max normalization per band.
        """

        norm_image = np.zeros_like(
            image,
            dtype=np.float32
        )

        for i in range(image.shape[0]):
            band = image[i]

            band_min = np.nanmin(band)
            band_max = np.nanmax(band)

            if (band_max - band_min) > 0:
                norm_image[i] = (
                    band - band_min
                ) / (band_max - band_min)

        return np.nan_to_num(norm_image)

    def __getitem__(self, idx):

        img_path, mask_path = self.samples[idx]

        try:

            with rasterio.open(img_path) as img_src:

                image = img_src.read(
                    list(self.bands)
                ).astype(np.float32)

                image_crs = img_src.crs
                image_transform = img_src.transform
                image_nodata = img_src.nodata

                if image_nodata is not None:
                    image[
                        image == image_nodata
                    ] = np.nan

            with rasterio.open(mask_path) as mask_src:

                mask = mask_src.read(
                    1
                ).astype(np.float32)

                mask_crs = mask_src.crs
                mask_transform = mask_src.transform

            # CRS validation
            if image_crs != mask_crs:
                raise ValueError(
                    f"CRS mismatch:\n"
                    f"Image: {image_crs}\n"
                    f"Mask : {mask_crs}"
                )

            # Spatial transform validation
            if image_transform != mask_transform:
                raise ValueError(
                    f"Transform mismatch:\n"
                    f"Image: {image_transform}\n"
                    f"Mask : {mask_transform}"
                )

            # Shape validation
            if image.shape[1:] != mask.shape:
                raise ValueError(
                    f"Shape mismatch:\n"
                    f"Image: {image.shape}\n"
                    f"Mask : {mask.shape}"
                )

            # Binary mask enforcement
            mask = (mask > 0).astype(
                np.float32
            )

            image = self._normalize(image)

            mask = np.expand_dims(
                mask,
                axis=0
            )

            image_tensor = torch.from_numpy(
                image
            )

            mask_tensor = torch.from_numpy(
                mask
            )

            # Data augmentation
            if self.transform:

                if torch.rand(1).item() > 0.5:
                    image_tensor = torch.flip(
                        image_tensor,
                        dims=[2]
                    )
                    mask_tensor = torch.flip(
                        mask_tensor,
                        dims=[2]
                    )

                if torch.rand(1).item() > 0.5:
                    image_tensor = torch.flip(
                        image_tensor,
                        dims=[1]
                    )
                    mask_tensor = torch.flip(
                        mask_tensor,
                        dims=[1]
                    )

                k = torch.randint(
                    0,
                    4,
                    (1,)
                ).item()

                if k > 0:
                    image_tensor = torch.rot90(
                        image_tensor,
                        k,
                        [1, 2]
                    )

                    mask_tensor = torch.rot90(
                        mask_tensor,
                        k,
                        [1, 2]
                    )

            return image_tensor, mask_tensor

        except Exception as e:

            logger.error(
                f"Failed loading sample {idx}\n"
                f"Image: {img_path}\n"
                f"Mask : {mask_path}\n"
                f"Error: {str(e)}"
            )

            raise RuntimeError(
                f"Dataset corruption detected."
            ) from e