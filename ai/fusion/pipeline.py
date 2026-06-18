import os
import json
import time
import logging
from typing import Dict, Any, Optional

import numpy as np
import rasterio
import torch

try:
    from ai.pix2pix.inference import load_generator, process_patch
    from ai.fusion.fuse import (
        fuse_images,
        generate_difference_map,
        save_outputs,
        save_fusion_stats
    )
    from ai.metrics.compute import compute_all_metrics
except ImportError:
    from inference import load_generator, process_patch
    from fuse import (
        fuse_images,
        generate_difference_map,
        save_outputs,
        save_fusion_stats
    )
    from compute import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class FusionPipeline:
    """
    End-to-end CloudVision Fusion Pipeline

    Steps:
    1. Pix2Pix Reconstruction
    2. Fusion
    3. Difference Map
    4. Metrics
    5. Reporting
    """

    PIPELINE_VERSION = "1.0.0"

    def __init__(self, model_path: str):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"Loading Pix2Pix Generator on {self.device}"
        )

        self.model_path = model_path
        self.model = load_generator(
            model_path,
            self.device
        )

    def validate_alignment(
        self,
        image_shape,
        mask_shape,
        image_crs,
        mask_crs,
        image_transform,
        mask_transform
    ):
        """
        Ensure image and mask are perfectly aligned.
        """

        if image_crs != mask_crs:
            raise ValueError(
                f"CRS mismatch:\n"
                f"Image: {image_crs}\n"
                f"Mask : {mask_crs}"
            )

        if image_transform != mask_transform:
            raise ValueError(
                "Spatial transform mismatch."
            )

        if image_shape[-2:] != mask_shape[-2:]:
            raise ValueError(
                f"Shape mismatch:\n"
                f"Image: {image_shape}\n"
                f"Mask : {mask_shape}"
            )

    def run(
        self,
        image_path: str,
        mask_path: str,
        output_dir: str,
        alpha: float = 1.0,
        reference_path: Optional[str] = None
    ) -> Dict[str, Any]:

        start_time = time.time()

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        fused_path = os.path.join(
            output_dir,
            "fused.tif"
        )

        recon_path = os.path.join(
            output_dir,
            "reconstruction.tif"
        )

        diff_path = os.path.join(
            output_dir,
            "diff_map.tif"
        )

        stats_path = os.path.join(
            output_dir,
            "fusion_stats.json"
        )

        metrics_path = os.path.join(
            output_dir,
            "metrics.json"
        )

        manifest_path = os.path.join(
            output_dir,
            "run_manifest.json"
        )

        try:

            logger.info(
                f"Loading image: {image_path}"
            )

            with rasterio.open(image_path) as src:

                image = src.read()

                if image.shape[0] < 3:
                    raise ValueError(
                        "Input image must contain "
                        "at least 3 bands."
                    )

                image = image[:3].astype(np.float32)

                profile = src.profile

                image_crs = src.crs
                image_transform = src.transform

            logger.info(
                f"Loading mask: {mask_path}"
            )

            with rasterio.open(mask_path) as src:

                mask = src.read()

                mask_crs = src.crs
                mask_transform = src.transform

            self.validate_alignment(
                image.shape,
                mask.shape,
                image_crs,
                mask_crs,
                image_transform,
                mask_transform
            )

            logger.info(
                "Running Pix2Pix reconstruction..."
            )

            reconstruction = process_patch(
                self.model,
                image,
                self.device
            )

            if reconstruction.shape != image.shape:
                raise ValueError(
                    f"Reconstruction shape mismatch:\n"
                    f"{reconstruction.shape} vs {image.shape}"
                )

            save_outputs(
                reconstruction,
                profile,
                recon_path
            )

            logger.info(
                f"Running fusion alpha={alpha}"
            )

            fused_image, stats = fuse_images(
                image,
                reconstruction,
                mask,
                alpha=alpha
            )

            save_outputs(
                fused_image,
                profile,
                fused_path
            )

            logger.info(
                "Generating difference map..."
            )

            diff_map = generate_difference_map(
                image,
                fused_image
            )

            save_outputs(
                diff_map,
                profile,
                diff_path,
                is_single_channel=True
            )

            processing_time = round(
                time.time() - start_time,
                2
            )

            stats["processing_time_seconds"] = processing_time
            stats["pipeline_version"] = self.PIPELINE_VERSION
            stats["execution_device"] = str(self.device)
            stats["model_type"] = "Pix2Pix"

            save_fusion_stats(
                stats,
                stats_path
            )

            metrics = {}

            if reference_path:

                logger.info(
                    f"Loading reference image: "
                    f"{reference_path}"
                )

                with rasterio.open(
                    reference_path
                ) as src:

                    reference = src.read()

                    if reference.shape[0] >= 3:
                        reference = reference[:3]

                    reference = reference.astype(
                        np.float32
                    )

                if reference.shape != fused_image.shape:
                    raise ValueError(
                        "Reference image shape mismatch."
                    )

                raw_metrics = compute_all_metrics(
                    reference,
                    fused_image
                )

                metrics = {
                    k: float(v)
                    if isinstance(
                        v,
                        (
                            np.integer,
                            np.floating
                        )
                    )
                    else v
                    for k, v in raw_metrics.items()
                }

            with open(
                metrics_path,
                "w"
            ) as f:
                json.dump(
                    metrics,
                    f,
                    indent=4
                )

            manifest = {
                "pipeline_version":
                    self.PIPELINE_VERSION,
                "input_image":
                    image_path,
                "mask":
                    mask_path,
                "reference":
                    reference_path,
                "model":
                    self.model_path,
                "alpha":
                    alpha,
                "device":
                    str(self.device),
                "processing_time":
                    processing_time
            }

            with open(
                manifest_path,
                "w"
            ) as f:
                json.dump(
                    manifest,
                    f,
                    indent=4
                )

            logger.info(
                "Pipeline completed successfully."
            )

            return {
                "fused_path": fused_path,
                "reconstruction_path": recon_path,
                "diff_path": diff_path,
                "stats_path": stats_path,
                "metrics_path": metrics_path,
                "manifest_path": manifest_path,
                "fusion_stats": stats,
                "metrics": metrics
            }

        except Exception as e:

            logger.exception(
                f"Pipeline failed: {str(e)}"
            )

            raise


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Fusion Pipeline CLI"
    )

    parser.add_argument(
        "--img",
        required=True
    )

    parser.add_argument(
        "--mask",
        required=True
    )

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--out",
        required=True
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--ref",
        default=None
    )

    args = parser.parse_args()

    pipeline = FusionPipeline(
        args.model
    )

    pipeline.run(
        image_path=args.img,
        mask_path=args.mask,
        output_dir=args.out,
        alpha=args.alpha,
        reference_path=args.ref
    )