from pydantic import BaseModel

class OCR_Output(BaseModel):
    bbox: list[list[int]]
    text: str
    score: float

class OCR_Response(BaseModel):
    data: list[OCR_Output]
    status_code: int

class img_output(BaseModel):
    text: str

class img_response(BaseModel):
    data: img_output  # Thay vì list[img_output]
    status_code: int

