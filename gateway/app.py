from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import requests
import numpy as np
from PIL import Image
import io
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Inference Gateway",
    description="Gateway for TensorFlow Serving - MobileNetV2 Image Classification",
    version="1.0.0"
)

TF_SERVING_URL = "http://tf-serving:8501/v1/models/mobilenet:predict"

# Metrics tracking (simple in-memory for demo)
metrics = {
    "requests_total": 0,
    "errors_total": 0,
    "start_time": datetime.now()
}

@app.get("/")
def root():
    """Root endpoint with API documentation"""
    return {
        "service": "ML Inference Gateway",
        "model": "MobileNetV2 (ImageNet)",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health - Service health check",
            "predict": "POST /predict - Image classification",
            "metrics": "GET /metrics - Service metrics",
            "model_info": "GET /model - Model information"
        },
        "usage": {
            "predict": "curl -X POST -F 'file=@image.jpg' http://localhost:8000/predict"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint - verifies TF Serving connectivity"""
    try:
        response = requests.get(
            "http://tf-serving:8501/v1/models/mobilenet",
            timeout=5
        )
        if response.status_code == 200:
            model_status = response.json()
            return {
                "status": "healthy",
                "gateway": "operational",
                "tf_serving": "connected",
                "model": model_status
            }
        else:
            raise Exception(f"TF Serving returned status {response.status_code}")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "gateway": "operational",
                "tf_serving": "disconnected",
                "error": str(e)
            }
        )

@app.get("/model")
def model_info():
    """Get model metadata from TF Serving"""
    try:
        response = requests.get(
            "http://tf-serving:8501/v1/models/mobilenet/metadata",
            timeout=5
        )
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify an image using MobileNetV2

    Args:
        file: Image file (JPEG, PNG)

    Returns:
        Top 5 predictions with class IDs and probabilities
    """
    metrics["requests_total"] += 1

    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Must be an image (JPEG, PNG)"
            )

        logger.info(f"Processing image: {file.filename} ({file.content_type})")

        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Store original size for response
        original_size = image.size

        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')

        # Resize to 224x224 (MobileNetV2 input requirement)
        image = image.resize((224, 224), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        image_array = np.expand_dims(image_array, axis=0)

        logger.info(f"Image preprocessed - shape: {image_array.shape}, dtype: {image_array.dtype}")

        # Prepare payload for TensorFlow Serving
        payload = {
            "instances": image_array.tolist()
        }

        # Call TensorFlow Serving
        logger.info(f"Sending request to TF Serving: {TF_SERVING_URL}")
        start_time = datetime.now()

        response = requests.post(
            TF_SERVING_URL,
            json=payload,
            timeout=30
        )

        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Inference completed in {inference_time:.2f}ms")

        if response.status_code != 200:
            logger.error(f"TF Serving error: {response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"TF Serving error: {response.text}"
            )

        predictions_response = response.json()

        # Process predictions - get top 5 classes
        pred_array = np.array(predictions_response['predictions'][0])
        top_5_indices = np.argsort(pred_array)[-5:][::-1]

        results = []
        for idx in top_5_indices:
            results.append({
                "class_id": int(idx),
                "probability": float(pred_array[idx]),
                "confidence_percent": round(float(pred_array[idx]) * 100, 2)
            })

        return {
            "success": True,
            "filename": file.filename,
            "original_size": f"{original_size[0]}x{original_size[1]}",
            "inference_time_ms": round(inference_time, 2),
            "predictions": results,
            "model": {
                "name": "mobilenet",
                "version": "1"
            },
            "note": "Class IDs correspond to ImageNet labels. Top prediction is first."
        }

    except HTTPException:
        metrics["errors_total"] += 1
        raise
    except Exception as e:
        metrics["errors_total"] += 1
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """
    Service metrics endpoint
    In production, integrate with Prometheus
    """
    uptime = (datetime.now() - metrics["start_time"]).total_seconds()

    return {
        "service": "ml-inference-gateway",
        "model": "mobilenet",
        "model_version": "1",
        "requests_total": metrics["requests_total"],
        "errors_total": metrics["errors_total"],
        "success_rate": round(
            (metrics["requests_total"] - metrics["errors_total"]) / max(metrics["requests_total"], 1) * 100,
            2
        ),
        "uptime_seconds": round(uptime, 2),
        "uptime_hours": round(uptime / 3600, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")