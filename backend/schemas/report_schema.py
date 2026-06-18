from pydantic import BaseModel

class ReportSchema(BaseModel):
    report_url: str
    s