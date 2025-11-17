from fastapi import FastAPI

from . import inference


app = FastAPI(title="BCS Cow API", version="0.1.0")


@app.get("/health")
def health_check() -> dict:
  return {"status": "ok"}


@app.post("/api/v1/predict")
async def predict_bcs():
  # TODO: implement image upload & prediction logic via inference module
  return {"message": "BCS prediction endpoint (to be implemented)"}

