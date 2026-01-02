import os
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
    description="API for classifying Body Condition Score (BCS) of dairy cows using YOLOv8. Includes image validation to ensure only cow rear images are processed.",
    version="1.3.0"
)

# Model paths
# MODEL_PATH = os.getenv("MODEL_PATH", "../models/production/20251125_202253/best.pt") 
BCS_MODEL_PATH = os.getenv("BCS_MODEL_PATH", "../models/yolov8_finetune_finetune/weights/best.pt")
VALIDATOR_MODEL_PATH = os.getenv("VALIDATOR_MODEL_PATH", "../models/cow_rear_validator/weights/best.pt")

# Models
bcs_model = None
validator_model = None

# Validation settings
VALIDATOR_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to accept as cow_rear

# Class names mapping for BCS
CLASS_NAMES = {
    0: '3.25',
    1: '3.5',
    2: '3.75',
    3: '4.0',
    4: '4.25'
}

@app.on_event("startup")
async def startup_event():
    global bcs_model, validator_model
    
    # Load BCS model
    try:
        logger.info(f"Loading BCS model from {BCS_MODEL_PATH}...")
        if not os.path.exists(BCS_MODEL_PATH):
            logger.warning(f"BCS model file not found at {BCS_MODEL_PATH}. Please ensure the model is mounted or available.")
        else:
            bcs_model = YOLO(BCS_MODEL_PATH)
            logger.info("BCS model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load BCS model: {e}")
    
    # Load Validator model
    try:
        logger.info(f"Loading Validator model from {VALIDATOR_MODEL_PATH}...")
        if not os.path.exists(VALIDATOR_MODEL_PATH):
            logger.warning(f"Validator model file not found at {VALIDATOR_MODEL_PATH}. Image validation will be skipped.")
        else:
            validator_model = YOLO(VALIDATOR_MODEL_PATH)
            logger.info("Validator model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Validator model: {e}")

@app.get("/")
async def root():
    return {"message": "BCS Cow Classification API is running. Use /predict to classify images."}

@app.get("/health")
async def health_check():
    status = {
        "bcs_model": "loaded" if bcs_model is not None else "not loaded",
        "validator_model": "loaded" if validator_model is not None else "not loaded"
    }
    
    if bcs_model is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "reason": "BCS model not loaded", "details": status})
    
    return {"status": "healthy", "details": status}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if bcs_model is None:
        raise HTTPException(status_code=503, detail="BCS model is not loaded.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # ============================================================
        # STEP 1: Validate if image is cow rear
        # ============================================================
        if validator_model is not None:
            validation_results = validator_model(image, verbose=False)
            val_probs = validation_results[0].probs
            val_class_idx = val_probs.top1
            val_conf = val_probs.top1conf.item()
            val_class_name = validation_results[0].names[val_class_idx]
            
            logger.info(f"Validation result: {val_class_name} ({val_conf:.2%})")
            
            # Check if image is NOT a cow rear
            if val_class_name == "not_cow_rear" or (val_class_name == "cow_rear" and val_conf < VALIDATOR_CONFIDENCE_THRESHOLD):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Invalid image",
                        "message": "Gambar yang diberikan bukan bagian belakang sapi. Pastikan gambar menunjukkan tampilan belakang sapi dengan jelas untuk mendapatkan prediksi BCS yang akurat.",
                        "validation": {
                            "detected_as": val_class_name,
                            "confidence": float(val_conf)
                        }
                    }
                )
        else:
            logger.warning("Validator model not loaded, skipping image validation.")
        
        # ============================================================
        # STEP 2: Predict BCS
        # ============================================================
        results = bcs_model(image, verbose=False)
        
        # Process results
        result = results[0]
        probs = result.probs
        top1_index = probs.top1
        top1_conf = probs.top1conf.item()
        
        predicted_class = CLASS_NAMES.get(top1_index, str(top1_index))
        
        # Check for low confidence warning
        BCS_CONFIDENCE_THRESHOLD = 0.6  # 60%
        warning = None
        if top1_conf < BCS_CONFIDENCE_THRESHOLD:
            warning = f"Prediksi BCS mungkin tidak akurat (confidence: {top1_conf:.1%}). Sapi mungkin berada di luar range normal (3.25-4.25) atau gambar kurang jelas."
        
        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": float(top1_conf),
            "warning": warning,
            "all_probabilities": {CLASS_NAMES.get(i, str(i)): float(p) for i, p in enumerate(probs.data.tolist())},
            "validation": {
                "is_valid_cow_rear": True,
                "validator_active": validator_model is not None
            }
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)