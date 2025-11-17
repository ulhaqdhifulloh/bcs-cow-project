from pydantic import BaseModel


class PredictionResponse(BaseModel):
  bcs: float
  confidence: float | None = None

