import os
import argparse
import logging

import numpy as np
import rasterio
import torch

# Hackathon-friendly import fallback
try:
    from .generator import Generator
except ImportError:
    from generator import Generator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_generator(
    weights_path: str,
    device: torch.device
) -> Generator:
    """
    Load Generator checkpoint safely.
    """

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}"
        )

    model = Generator(
        in_channels=3,
        out_channels=3
    )

    checkpoint = torch.load(
        weights_path,
        map_location=device
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info(
        f"Loaded Generator from {weights_path}"
    )

    return model


def process_patch(
    model: Generator,
    image_array: np.ndarray,
    device: torch.device
) -> np.ndarray:
    """
    Runs Pix2Pix inference on an image patch.

    Pipeline:
    1. NoData cleanup
    2. 2-98 percentile normalization
    3. Scale to [-1,1]
    4. Generator inference
    5. Tanh clipping
    6. Denormalization
    """

    image_array = np.nan_to_num(
        image_array.astype(np.float32)
    )

    c, h, w = image_array.shape

    norm_image = np.zeros_like(
        image_array,
        dtype=np.float32
    )

    p2_vals = np.zeros(c, dtype=np.float32)
    p98_vals = np.zeros(c, dtype=np.float32)

    for i in range(c):

        band = image_array[i]

        valid_pixels = band[
            np.isfinite(band)
        ]

        if valid_pixels.size == 0:
            raise ValueError(
                f"Band {i} contains no valid pixels."
            )

        p2 = np.percentile(
            valid_pixels,
            2
        )

        p98 = np.percentile(
            valid_pixels,
            98
        )

        if p98 <= p2:
            p98 = valid_pixels.max()

        p2_vals[i] = p2
        p98_vals[i] = p98

        band = np.clip(
            band,
            p2,
            p98
        )

        if (p98 - p2) > 0:
            normalized = (
                band - p2
            ) / (p98 - p2)

            norm_image[i] = (
                normalized * 2.0
            ) - 1.0
        else:
            norm_image[i] = -1.0

    tensor = torch.from_numpy(
        norm_image
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        generated_tensor = model(tensor)

    if generated_tensor.shape[2:] != tensor.shape[2:]:
        raise RuntimeError(
            f"Generator output shape mismatch: "
            f"{generated_tensor.shape} vs "
            f"{tensor.shape}"
        )

    generated_array = (
        generated_tensor
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # Tanh safety
    generated_array = np.clip(
        generated_array,
        -1.0,
        1.0
    )

    denorm_image = np.zeros_like(
        generated_array,
        dtype=np.float32
    )

    for i in range(c):

        scaled_01 = (
            generated_array[i] + 1.0
        ) / 2.0

        denorm_image[i] = (
            scaled_01 *
            (p98_vals[i] - p2_vals[i])
        ) + p2_vals[i]

    return denorm_image


def main():

    parser = argparse.ArgumentParser(
        description="Run Pix2Pix Inference"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input cloudy TIFF"
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="generator_best.pth"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output reconstructed TIFF"
    )

    args = parser.parse_args()

    torch.set_num_threads(
        max(
            1,
            os.cpu_count() // 2
        )
    )

    device = torch.device("cpu")

    model = load_generator(
        args.weights,
        device
    )

    logger.info(
        f"Reading image: {args.input}"
    )

    with rasterio.open(args.input) as src:

        if src.count < 3:
            raise ValueError(
                f"Expected at least 3 bands, "
                f"found {src.count}"
            )

        input_image = src.read(
            [1, 2, 3]
        ).astype(np.float32)

        profile = src.profile

        if src.crs is None:
            logger.warning(
                "Input image has no CRS."
            )

    logger.info(
        "Running Generator..."
    )

    output_image = process_patch(
        model,
        input_image,
        device
    )

    target_dtype = np.dtype(
        profile["dtype"]
    )

    if np.issubdtype(
        target_dtype,
        np.integer
    ):
        info = np.iinfo(target_dtype)

        output_image = np.clip(
            output_image,
            info.min,
            info.max
        )

    output_image = output_image.astype(
        target_dtype
    )

    profile.update(
        count=3
    )

    if profile.get("crs") is None:
        logger.warning(
            "Output image has no CRS metadata."
        )

    out_dir = os.path.dirname(
        args.output
    )

    if out_dir:
        os.makedirs(
            out_dir,
            exist_ok=True
        )

    with rasterio.open(
        args.output,
        "w",
        **profile
    ) as dst:
        dst.write(output_image)

    logger.info(
        f"Saved reconstruction: {args.output}"
    )


if __name__ == "__main__":
    main()