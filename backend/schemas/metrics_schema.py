from pydantic import BaseModel

class MetricsSchema(BaseModel):
    psnr: float
    ssim: float
    rmse: float
    sam: float
    quality_score: int
    quality_flags: dict