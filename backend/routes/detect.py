from fastapi import APIRouter, UploadFile, File, HTTPException
import os

from backend.services.cloud_detection_service import detect_clouds

router = APIRouter(tags=["Cloud Detection"])

UPLOAD_DIR = "backend/uploads"


@router.post("/detect")
async def detect_clouds_route(
    file: UploadFile = File(...)
):

    if not file.filename.endswith(
        (".tif", ".tiff", ".jpg", ".jpeg", ".png")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only TIFF, JPG, JPEG and PNG files are allowed"
        )

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    image_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(image_path, "wb") as buffer:
        buffer.write(
            await file.read()
        )

    result = detect_clouds(
        image_path
    )

    return result