from pydantic import BaseModel

class UploadSchema(BaseModel):
    file: bytes
    file_id: str
    file_name: str
    file_type: str