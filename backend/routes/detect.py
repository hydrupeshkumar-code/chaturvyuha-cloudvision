from fastapi import APIRouter, UploadFile, File, HTTPException
router = APIRouter()
@router.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    if not file.filename.endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only TIFF, JPG, JPEG, and PNG files are allowed.")
    
    detected_objects = ["object1", "object2", "object3"]  
    
    return {
    "file_id": "abc123",
    "mask_url": "/outputs/masks/mask_001.png",
    "cloud_coverage_pct": 42.5,
    "status": "complete"
}