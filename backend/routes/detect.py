from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(tags=["Cloud Detection"])

@router.post("/detect")
async def detect_clouds(file: UploadFile = File(...)):

    if not file.filename.endswith(
        (".tif", ".tiff", ".jpg", ".jpeg", ".png")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only TIFF, JPG, JPEG and PNG files are allowed"
        )

    return {
        "file_id": "abc123",
        "mask_url": "/outputs/masks/mask_001.png",
        "cloud_coverage_pct": 42.5,
        "status": "complete"
    }