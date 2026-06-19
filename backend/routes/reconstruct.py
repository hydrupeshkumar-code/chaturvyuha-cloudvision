from fastapi import APIRouter, UploadFile, File
import os

from backend.services.inference_service import (
    run_reconstruction
)

router = APIRouter()


@router.post("/reconstruct")
async def reconstruct_image(
    file: UploadFile = File(...)
):

    upload_path = f"backend/uploads/{file.filename}"

    with open(upload_path, "wb") as buffer:
        buffer.write(
            await file.read()
        )

    result = run_reconstruction(
        image_path=upload_path,
        mask_path="backend/outputs/masks/mask.tif",
        output_dir="backend/outputs/reconstructed"
    )

    return {
        "reconstruction_path":
            result["reconstruction_path"],
        "diff_path":
            result["diff_path"],
        "status": "complete"
    }