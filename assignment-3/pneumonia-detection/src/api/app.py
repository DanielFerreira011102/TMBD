from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tenacity import retry, stop_after_attempt, wait_exponential
import torch
import io
from PIL import Image
import logging
from pathlib import Path
import sys
from typing import Dict, Any
import asyncio
import os
from datetime import datetime

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.classifier import PneumoniaClassifier
from src.data.dataset import XRayDataset
from .config import settings
from .cache import RedisCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pneumonia Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
transform = None
cache = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = None
last_modified_time = None
MODEL_CHECK_INTERVAL = 10  # Check for model updates every 10 seconds

async def load_model():
    """Load the model and return its last modified time"""
    global model, transform, model_path
    
    model_path = Path(__file__).parent.parent.parent / "models" / "final_federated_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model, _ = PneumoniaClassifier.load_checkpoint(str(model_path), map_location=device)
    model.to(device)
    model.eval()
    
    # Get transforms if not already set
    if transform is None:
        transform = XRayDataset.get_default_transforms()
    
    return os.path.getmtime(model_path)

@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=30, min=30, max=3600),
    reraise=True
)
async def load_model_with_retry():
    """Try to load the model with retries"""
    return await load_model()

async def check_model_updates():
    """Periodically check for model updates"""
    global model, last_modified_time, cache
    
    while True:
        try:
            if model_path and os.path.exists(model_path):
                current_modified_time = os.path.getmtime(model_path)
                
                if last_modified_time is None or current_modified_time > last_modified_time:
                    logger.info("Detected model update, reloading model...")
                    last_modified_time = await load_model()
                    
                    # Clear prediction cache when model updates
                    if cache:
                        cache.redis_client.flushdb()
                        logger.info("Cleared prediction cache due to model update")
            
        except Exception as e:
            logger.error(f"Error checking for model updates: {str(e)}")
        
        await asyncio.sleep(MODEL_CHECK_INTERVAL)

@app.on_event("startup")
async def startup():
    """Initialize model and cache on startup"""
    global model, transform, cache, last_modified_time
    try:
        # Initialize cache first
        cache = RedisCache(
            host=settings.redis_host,
            port=settings.redis_port,
            ttl=settings.cache_ttl
        )
        
        # Wait for initial model load
        last_modified_time = await load_model_with_retry()
        
        # Start background task to check for model updates
        asyncio.create_task(check_model_updates())
        
        logger.info("Model, transforms, and cache initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise e

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check Redis connection
    try:
        cache.redis_client.ping()
    except:
        raise HTTPException(status_code=503, detail="Cache connection error")
    
    return {
        "status": "healthy",
        "model_path": str(model_path),
        "last_model_update": datetime.fromtimestamp(last_modified_time).isoformat() if last_modified_time else None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Make a prediction for an uploaded X-ray image"""
    if model is None or cache is None:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Check cache first
        cache_key = cache.generate_key(image_data)
        cached_result = cache.get_prediction(cache_key)
        
        if cached_result:
            logger.info("Returning cached prediction")
            return cached_result
        
        # Process image if not in cache
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            
        # Format results
        prediction_result = {
            "prediction": "PNEUMONIA" if predicted_class == 1 else "NORMAL",
            "confidence": {
                "normal": f"{probabilities[0].item() * 100:.2f}%",
                "pneumonia": f"{probabilities[1].item() * 100:.2f}%"
            },
            "cached": False,
            "model_version": datetime.fromtimestamp(last_modified_time).isoformat()
        }
        
        # Cache the result
        cache.set_prediction(cache_key, prediction_result)
        
        return prediction_result
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info() -> Dict[str, Any]:
    """Get information about the currently loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    return {
        "model_path": str(model_path),
        "last_update": datetime.fromtimestamp(last_modified_time).isoformat() if last_modified_time else None,
        "model_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
        "device": str(device)
    }

@app.get("/cache/stats")
async def cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    try:
        info = cache.redis_client.info()
        return {
            "used_memory": info.get("used_memory_human", "N/A"),
            "connected_clients": info.get("connected_clients", 0),
            "uptime_days": info.get("uptime_in_days", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))