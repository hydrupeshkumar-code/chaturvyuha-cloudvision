from pydantic import BaseModel

class DetectSchema(BaseModel):
    file_id: str
    mask_url: str
    cloud_coverage_pct: float
    status: str