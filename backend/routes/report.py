from fastapi import APIRouter, UploadFile, File, HTTPException
router = APIRouter()
@router.post("/report")
async def generate_report(file: UploadFile = File(...)):
    if not file.filename.endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only TIF, TIFF, JPG, JPEG, and PNG files are allowed.")
    
    report_url = "http://example.com/report.pdf"
    
    return {
    "file_id": "abc123",
    "report_url": "/outputs/reports/report_001.pdf",
    "status": "generated"
}