from fastapi import APIRouter, HTTPException
import json
import os

router = APIRouter()

METRICS_FILE = (
    "backend/outputs/reconstructed/metrics.json"
)


@router.get("/metrics")
async def get_metrics():

    if not os.path.exists(METRICS_FILE):
        raise HTTPException(
            status_code=404,
            detail="Metrics not generated yet"
        )

    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)

    return metrics