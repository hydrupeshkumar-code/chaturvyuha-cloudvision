from pydantic import BaseModel

class ReconstructSchema(BaseModel):
    reconstruction_path: str
    diff_path: str
    status: str