from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import logging
from typing import Dict, Any
import asyncio
import time

# Import kelas DrowsinessDetector yang sudah ada
from drowsiness_detector import DrowsinessDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Drowsiness Detection API",
    description="API untuk deteksi kantuk menggunakan computer vision",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup"""
    global detector
    try:
        # Pastikan path model sesuai dengan lokasi file Anda
        detector = DrowsinessDetector(model_path="models/shape_predictor_68_face_landmarks.dat")
        if detector.is_ready():
            logger.info("Drowsiness detector initialized successfully")
        else:
            logger.error("Failed to initialize drowsiness detector")
    except Exception as e:
        logger.error(f"Error initializing detector: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Drowsiness Detection API",
        "version": "1.0.0",
        "status": "running",
        "detector_ready": detector.is_ready() if detector else False
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_ready": detector.is_ready() if detector else False,
        "timestamp": time.time()
    }

@app.post("/detect/image")
async def detect_drowsiness_image(file: UploadFile = File(...)):
    """
    Deteksi kantuk dari single image
    """
    if not detector or not detector.is_ready():
        raise HTTPException(status_code=503, detail="Detector not ready")
    
    try:
        # Baca file image
        contents = await file.read()
        
        # Convert ke OpenCV format
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Proses deteksi
        result = detector.detect_from_image(image)
        
        # Convert processed image to base64 for response
        processed_image = result['processed_image']
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response = {
            "success": True,
            "drowsiness_detected": result['drowsiness_detected'],
            "drowsiness_score": result['drowsiness_score'],
            "confidence": result['confidence'],
            "face_detected": result['face_detected'],
            "metrics": {
                "ear": round(result['ear'], 3),
                "mar": round(result['mar'], 3),
                "head_movement": round(result['head_movement'], 3)
            },
            "alerts": result['alerts'],
            "processed_image": processed_image_b64,
            "timestamp": time.time()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect/video-frame")
async def detect_drowsiness_video_frame(file: UploadFile = File(...)):
    """
    Deteksi kantuk dari video frame (dengan context dari frame sebelumnya)
    """
    if not detector or not detector.is_ready():
        raise HTTPException(status_code=503, detail="Detector not ready")
    
    try:
        # Baca file image
        contents = await file.read()
        
        # Convert ke OpenCV format
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Proses video frame
        result = detector.process_video_frame(frame)
        
        # Convert processed frame to base64
        processed_frame = result['processed_frame']
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response = {
            "success": True,
            "drowsiness_detected": result['drowsiness_detected'],
            "drowsiness_score": result['drowsiness_score'],
            "frame_count": result['frame_count'],
            "metrics": {
                "ear": round(result['metrics']['ear'], 3),
                "mar": round(result['metrics']['mar'], 3),
                "head_movement": round(result['metrics']['head_movement'], 3)
            },
            "alerts": result['alerts'],
            "processed_frame": processed_frame_b64,
            "timestamp": time.time()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error processing video frame: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current detector configuration"""
    if not detector or not detector.is_ready():
        raise HTTPException(status_code=503, detail="Detector not ready")
    
    try:
        config = detector.get_config()
        return {
            "success": True,
            "config": config,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}")

@app.post("/config")
async def update_config(config_data: Dict[str, Any]):
    """Update detector configuration"""
    if not detector or not detector.is_ready():
        raise HTTPException(status_code=503, detail="Detector not ready")
    
    try:
        detector.update_config(config_data)
        updated_config = detector.get_config()
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "config": updated_config,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

@app.post("/reset")
async def reset_detector():
    """Reset detector state"""
    if not detector or not detector.is_ready():
        raise HTTPException(status_code=503, detail="Detector not ready")
    
    try:
        detector.reset_state()
        return {
            "success": True,
            "message": "Detector state reset successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error resetting detector: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting detector: {str(e)}")

@app.get("/statistics")
async def get_statistics():
    """Get detector statistics"""
    if not detector or not detector.is_ready():
        raise HTTPException(status_code=503, detail="Detector not ready")
    
    try:
        stats = detector.get_statistics()
        return {
            "success": True,
            "statistics": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)