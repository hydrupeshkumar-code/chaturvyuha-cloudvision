from fastapi import APIRouter

router = APIRouter(tags=["Report"])

@router.get("/report")
async def generate_report():

    return {
        "report_url":
            "/outputs/reports/report_001.pdf",
        "status": "generated"
    }