from pydantic import BaseModel

class MetricsResponse(BaseModel):
    psnr: float
    ssim: float
    rmse: float
    sam: float
    quality_flag: str