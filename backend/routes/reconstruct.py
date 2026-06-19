from fastapi import APIRouter, UploadFile, File, HTTPException
import os

from backend.services.inference_service import run_reconstruction

router = APIRouter()

UPLOAD_DIR = "backend/uploads"
OUTPUT_DIR = "backend/outputs/reconstructed"


@router.post("/reconstruct")
async def reconstruct_image(
    file: UploadFile = File(...)
):

    if not file.filename.endswith(
        (".tif", ".tiff", ".jpg", ".jpeg", ".png")
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type"
        )

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    upload_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(upload_path, "wb") as buffer:
        buffer.write(
            await file.read()
        )

    mask_path = "backend/outputs/masks/mask.tif"

    if not os.path.exists(mask_path):
        raise HTTPException(
            status_code=404,
            detail="Cloud mask not found. Run detection first."
        )

    result = run_reconstruction(
        image_path=upload_path,
        mask_path=mask_path,
        output_dir=OUTPUT_DIR
    )

    return {
        "reconstruction_path":
            result["reconstruction_path"],

        "diff_path":
            result["diff_path"],

        "status":
            "complete"
    }