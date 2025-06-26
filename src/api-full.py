"""
src/api.py

FastAPI Backend for AV Vision Pipeline

This creates a web API that serves our object detection model.
Perfect for connecting to a React frontend later.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import List, Dict
import logging

from .detector import AVObjectDetector, ObjectDetectionError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AV Vision Pipeline API",
    description="Object detection API for autonomous vehicles",
    version="1.0.0"
)

# Add CORS middleware (allows React frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector (happens once when API starts)
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize the detector when API starts"""
    global detector
    try:
        detector = AVObjectDetector()
        logger.info("✅ Object detector initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {e}")
        raise

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "AV Vision Pipeline API", 
        "status": "running",
        "model": detector.get_model_info() if detector else "not loaded"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    return {
        "status": "healthy",
        "detector": detector.get_model_info()
    }

@app.post("/detect", response_model=List[Dict])
async def detect_objects(file: UploadFile = File(...)):
    """
    Upload an image and get object detections
    
    This endpoint will be used by our React frontend
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not available")
    
    try:
        # Read uploaded image
        image_data = await file.read()
        
        # Convert to OpenCV format - this validates it's an image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image - please upload a valid image file")
        
        # Run detection
        detections = detector.detect_from_array(image)
        
        logger.info(f"Detected {len(detections)} objects in uploaded image")
        return detections
        
    except ObjectDetectionError as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")