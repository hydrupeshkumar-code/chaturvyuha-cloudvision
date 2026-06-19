from backend.services.model_loader import get_pipeline


def run_reconstruction(
    image_path: str,
    mask_path: str,
    output_dir: str
):

    pipeline = get_pipeline()

    result = pipeline.run(
        image_path=image_path,
        mask_path=mask_path,
        output_dir=output_dir
    )

    return result