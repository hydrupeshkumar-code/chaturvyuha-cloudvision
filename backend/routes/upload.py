from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()
@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only TIFF, JPG, JPEG, and PNG files are allowed.")
    
    return {
    "file_id": "abc123",
    "filename": file.filename,
    "status": "uploaded"
}