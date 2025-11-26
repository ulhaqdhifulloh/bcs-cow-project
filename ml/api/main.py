import os
from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BCS Cow Classification API",
    description="API for classifying Body Condition Score (BCS) of dairy cows using YOLOv8.",
    version="1.2.0"
)

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "../models/production/20251125_202253/best.pt") 
model = None

# Class names mapping 
CLASS_NAMES = {
    0: '3.25',
    1: '3.5',
    2: '3.75',
    3: '4.0',
    4: '4.25'
}

@app.on_event("startup")
async def startup_event():
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found at {MODEL_PATH}. Please ensure the model is mounted or available.")
        else:
            model = YOLO(MODEL_PATH)
            logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.get("/")
async def root():
    return {"message": "BCS Cow Classification API is running. Use /predict to classify images."}

@app.get("/health")
async def health_check():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "reason": "Model not loaded"})
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Inference
        results = model(image)
        
        # Process results
        result = results[0]
        probs = result.probs
        top1_index = probs.top1
        top1_conf = probs.top1conf.item()
        
        predicted_class = CLASS_NAMES.get(top1_index, str(top1_index))
        
        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": float(top1_conf),
            "all_probabilities": {CLASS_NAMES.get(i, str(i)): float(p) for i, p in enumerate(probs.data.tolist())}
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
