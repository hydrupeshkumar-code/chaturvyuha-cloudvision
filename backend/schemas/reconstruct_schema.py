from pydantic import BaseModel

class ReconstructSchema(BaseModel):
    file_id: str
    reconstructed_url: str
    status: str