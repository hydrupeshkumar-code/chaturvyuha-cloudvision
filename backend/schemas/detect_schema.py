from pydantic import BaseModel

from backend.schemas.metrics_schema import MetricsSchema

class DetectSchema(BaseModel):
    detection_path: str
    status: str
    metrics: MetricsSchema
    