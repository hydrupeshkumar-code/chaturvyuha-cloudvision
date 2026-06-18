from fastapi import APIRouter, UploadFile, File, HTTPException
router = APIRouter()
@router.post("/metrics")
async def calculate_metrics(file: UploadFile = File(...)):
    if not file.filename.endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only TIFF, JPG, JPEG, and PNG files are allowed.")
    
    metrics = {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.90
    }
    
    return {
    "psnr_db": 28.4,
    "ssim": 0.91,
    "rmse_normalized": 0.04,
    "sam_degrees": 2.1
}