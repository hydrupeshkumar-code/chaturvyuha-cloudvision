from fastapi import FastAPI

from backend.routes.upload import router as upload_router
from backend.routes.detect import router as detect_router
from backend.routes.reconstruct import router as reconstruct_router
from backend.routes.metrics import router as metrics_router
from backend.routes.report import router as report_router

app = FastAPI(title="ChaturVyuha CloudVision API")


@app.get("/health")
def health():
    return {"status": "healthy"}


app.include_router(upload_router)
app.include_router(detect_router)
app.include_router(reconstruct_router)
app.include_router(metrics_router)
app.include_router(report_router)