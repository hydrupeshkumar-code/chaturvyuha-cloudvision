from fastapi import APIRouter, HTTPException
import json
import os

from backend.services.pdf_service import generate_pdf_report

router = APIRouter(tags=["Report"])

METRICS_FILE = "backend/outputs/reconstructed/metrics.json"
REPORT_FILE = "backend/outputs/reports/cloudvision_report.pdf"


@router.get("/report")
async def generate_report():

    if not os.path.exists(METRICS_FILE):
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found"
        )

    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)

    os.makedirs(
        "backend/outputs/reports",
        exist_ok=True
    )

    generate_pdf_report(
        REPORT_FILE,
        metrics
    )

    return {
        "report_url": REPORT_FILE,
        "status": "generated"
    }