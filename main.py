from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
from typing import Optional, Dict, Any
import asyncio
import logging
import time
import os

# Import class DrowsinessDetector yang sudah ada
from Drowsiness_Detection import DrowsinessDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi FastAPI
app = FastAPI(
    title="Drowsiness Detection API",
    description="API untuk deteksi kantuk menggunakan analisis wajah dan dlib",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = None

def find_model_path():
    """Mencari path model yang tersedia"""
    possible_paths = [
        "models/shape_predictor_68_face_landmarks.dat",
        "shape_predictor_68_face_landmarks.dat",
        "./models/shape_predictor_68_face_landmarks.dat",
        "./shape_predictor_68_face_landmarks.dat"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

@app.on_event("startup")
async def startup_event():
    """Inisialisasi detector saat aplikasi dimulai"""
    global detector
    try:
        logger.info("Initializing Drowsiness Detector...")
        
        # Cari path model yang tersedia
        model_path = find_model_path()
        if model_path is None:
            logger.error("Model file 'shape_predictor_68_face_landmarks.dat' tidak ditemukan!")
            logger.error("Download dari: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            detector = None
            return
        
        logger.info(f"Using model path: {model_path}")
        detector = DrowsinessDetector(model_path=model_path)
        logger.info("Drowsiness Detection API started successfully")
        
    except Exception as e:
        logger.error(f"Error initializing detector: {e}")
        detector = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup saat aplikasi ditutup"""
    logger.info("Shutting down API...")
    if detector:
        detector.cleanup()

def is_detector_ready():
    """Check if detector is ready"""
    return detector is not None

def generate_alerts(result):
    """Generate alerts based on detection result"""
    alerts = []
    
    if result['metrics']['ear'] < detector.eye_ar_thresh:
        alerts.append("Eyes appear closed or droopy")
    
    if result['metrics']['mar'] > detector.mouth_ar_thresh:
        alerts.append("Yawning detected")
    
    if result['metrics']['head_movement'] > detector.head_movement_thresh:
        alerts.append("Excessive head movement detected")
    
    if result['is_drowsy']:
        alerts.append("DROWSINESS ALERT - Please take a break!")
    
    return alerts

def calculate_confidence(result):
    """Calculate confidence score based on face detection and metrics"""
    base_confidence = 0.7 if result['status'] != "No Face Detected" else 0.3
    
    # Adjust confidence based on metric stability
    ear_confidence = min(1.0, result['metrics']['ear'] * 5)  # Normalize EAR to confidence
    
    return round(min(1.0, base_confidence + (ear_confidence * 0.3)), 2)

@app.get("/")
async def root():
    """Endpoint root untuk health check"""
    return {
        "message": "Drowsiness Detection API",
        "status": "running",
        "version": "1.0.0",
        "detector_ready": is_detector_ready()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if is_detector_ready() else "detector_not_ready",
        "detector_ready": is_detector_ready(),
        "timestamp": time.time()
    }

@app.post("/detect/image")
async def detect_drowsiness_image(file: UploadFile = File(...)):
    """
    Deteksi kantuk dari gambar yang diupload
    """
    if not is_detector_ready():
        raise HTTPException(status_code=503, detail="Detector not ready. Please check if the model file exists.")
    
    try:
        # Validasi file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File harus berupa gambar")
        
        # Baca file gambar
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Tidak dapat membaca gambar")
        
        # Proses deteksi menggunakan method yang benar
        result = detector.detect_drowsiness(image)
        
        # Convert processed image ke base64 untuk response
        _, buffer = cv2.imencode('.jpg', result['frame'])
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Deteksi wajah untuk response
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detector(gray, 0)
        
        # Format response sesuai dengan struktur yang diharapkan
        response = {
            "success": True,
            "drowsiness_detected": result['is_drowsy'],
            "drowsiness_score": result['drowsiness_score'],
            "face_detected": len(faces) > 0,
            "metrics": {
                "ear": round(result['metrics']['ear'], 3),
                "mar": round(result['metrics']['mar'], 3),
                "head_movement": round(result['metrics']['head_movement'], 3)
            },
            "alerts": generate_alerts(result),
            "processed_image": processed_image_b64,
            "confidence": calculate_confidence(result),
            "status": result['status'],
            "timestamp": time.time()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error memproses gambar: {str(e)}")

@app.post("/detect/video-frame")
async def detect_drowsiness_video_frame(file: UploadFile = File(...)):
    """
    Deteksi kantuk dari frame video (dengan context dari frame sebelumnya)
    """
    if not is_detector_ready():
        raise HTTPException(status_code=503, detail="Detector not ready")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Tidak dapat membaca frame")
        
        # Proses deteksi frame video
        result = detector.detect_drowsiness(frame)
        
        # Convert processed frame ke base64
        _, buffer = cv2.imencode('.jpg', result['frame'])
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get frame count (simulasi)
        frame_count = getattr(detector, 'frame_count', 0) + 1
        setattr(detector, 'frame_count', frame_count)
        
        response = {
            "success": True,
            "drowsiness_detected": result['is_drowsy'],
            "drowsiness_score": result['drowsiness_score'],
            "frame_count": frame_count,
            "metrics": {
                "ear": round(result['metrics']['ear'], 3),
                "mar": round(result['metrics']['mar'], 3),
                "head_movement": round(result['metrics']['head_movement'], 3)
            },
            "alerts": generate_alerts(result),
            "processed_frame": processed_frame_b64,
            "status": result['status'],
            "timestamp": time.time()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error processing video frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error memproses frame: {str(e)}")

@app.get("/config")
async def get_detector_config():
    """
    Mendapatkan konfigurasi detector saat ini
    """
    if not is_detector_ready():
        raise HTTPException(status_code=503, detail="Detector not ready")
    
    try:
        config = {
            "eye_ar_thresh": detector.eye_ar_thresh,
            "eye_ar_consec_frames": detector.eye_ar_consec_frames,
            "mouth_ar_thresh": detector.mouth_ar_thresh,
            "mouth_ar_consec_frames": detector.mouth_ar_consec_frames,
            "head_movement_thresh": detector.head_movement_thresh,
            "head_movement_frames": detector.head_movement_frames,
            "drowsiness_threshold": detector.drowsiness_threshold,
            "max_drowsiness_score": detector.max_drowsiness_score
        }
        
        return {
            "success": True,
            "config": config,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}")

@app.post("/config")
async def update_detector_config(config_data: Dict[str, Any]):
    """
    Update konfigurasi detector
    """
    if not is_detector_ready():
        raise HTTPException(status_code=503, detail="Detector not ready")
    
    try:
        # Update configuration using set_thresholds method
        detector.set_thresholds(**config_data)
        
        # Get updated config
        updated_config = {
            "eye_ar_thresh": detector.eye_ar_thresh,
            "eye_ar_consec_frames": detector.eye_ar_consec_frames,
            "mouth_ar_thresh": detector.mouth_ar_thresh,
            "mouth_ar_consec_frames": detector.mouth_ar_consec_frames,
            "head_movement_thresh": detector.head_movement_thresh,
            "head_movement_frames": detector.head_movement_frames,
            "drowsiness_threshold": detector.drowsiness_threshold,
            "max_drowsiness_score": detector.max_drowsiness_score
        }
        
        return {
            "success": True,
            "message": "Konfigurasi berhasil diupdate",
            "config": updated_config,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=400, detail=f"Error updating config: {str(e)}")

@app.post("/reset")
async def reset_detector():
    """
    Reset detector state (counters, history, dll)
    """
    if not is_detector_ready():
        raise HTTPException(status_code=503, detail="Detector not ready")
    
    try:
        detector.reset_counters()
        
        # Reset frame count if exists
        if hasattr(detector, 'frame_count'):
            detector.frame_count = 0
        
        return {
            "success": True,
            "message": "Detector state berhasil direset",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error resetting detector: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting detector: {str(e)}")

@app.get("/statistics")
async def get_detector_stats():
    """
    Mendapatkan statistik detector
    """
    if not is_detector_ready():
        raise HTTPException(status_code=503, detail="Detector not ready")
    
    try:
        # Get current status
        status = detector.get_status()
        
        # Additional statistics
        stats = {
            "current_status": status,
            "configuration": {
                "eye_ar_thresh": detector.eye_ar_thresh,
                "mouth_ar_thresh": detector.mouth_ar_thresh,
                "head_movement_thresh": detector.head_movement_thresh,
                "drowsiness_threshold": detector.drowsiness_threshold
            },
            "history_lengths": {
                "ear_history": len(detector.ear_history),
                "mar_history": len(detector.mar_history),
                "head_movement_history": len(detector.head_movement_history)
            },
            "frame_count": getattr(detector, 'frame_count', 0)
        }
        
        return {
            "success": True,
            "statistics": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

# Endpoint untuk streaming info
@app.get("/stream/info")
async def stream_info():
    """
    Informasi tentang streaming
    """
    return {
        "message": "Untuk real-time streaming, gunakan endpoint /detect/video-frame secara berulang",
        "recommended_fps": "5-10 FPS untuk performa optimal",
        "note": "Detector otomatis maintain state antar frame"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")