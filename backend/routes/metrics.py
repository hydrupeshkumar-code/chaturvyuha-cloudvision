from fastapi import APIRouter

router = APIRouter()


@router.get("/metrics")
async def get_metrics():

    return {
        "psnr": 31.2,
        "ssim": 0.91,
        "rmse": 0.03,
        "sam": 3.2,
        "quality_score": 92,
        "quality_flags": {
            "overall": "PASS"
        }
    }