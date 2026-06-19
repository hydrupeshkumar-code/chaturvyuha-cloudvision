from pydantic import BaseModel

class ReportSchema(BaseModel):
    report_url: str
    quality_score: int
    quality_flags: dict
    