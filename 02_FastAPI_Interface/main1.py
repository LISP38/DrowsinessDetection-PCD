from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
from typing import Optional
import asyncio
import logging

# Import modul deteksi kantuk
from drowsiness_detector import DrowsinessDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi FastAPI
app = FastAPI(
    title="Drowsiness Detection API",
    description="API untuk deteksi kantuk menggunakan analisis wajah dan MediaPipe",
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

# Inisialisasi detector
detector = DrowsinessDetector()

@app.on_event("startup")
async def startup_event():
    """Inisialisasi saat aplikasi dimulai"""
    logger.info("Starting Drowsiness Detection API...")
    
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup saat aplikasi ditutup"""
    logger.info("Shutting down API...")
    detector.cleanup()

@app.get("/")
async def root():
    """Endpoint root untuk health check"""
    return {
        "message": "Drowsiness Detection API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_ready": detector.is_ready()
    }

@app.post("/detect/image")
async def detect_drowsiness_image(file: UploadFile = File(...)):
    """
    Deteksi kantuk dari gambar yang diupload
    """
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
        
        # Proses deteksi
        result = detector.detect_from_image(image)
        
        # Convert processed image ke base64 untuk response
        _, buffer = cv2.imencode('.jpg', result['processed_image'])
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "drowsiness_detected": result['drowsiness_detected'],
            "drowsiness_score": result['drowsiness_score'],
            "metrics": {
                "ear": result['ear'],
                "mar": result['mar'],
                "head_movement": result['head_movement']
            },
            "alerts": result['alerts'],
            "processed_image": processed_image_b64,
            "confidence": result['confidence']
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error memproses gambar: {str(e)}")

@app.post("/detect/video-frame")
async def detect_drowsiness_video_frame(file: UploadFile = File(...)):
    """
    Deteksi kantuk dari frame video
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Tidak dapat membaca frame")
        
        # Proses deteksi dengan context frame sebelumnya
        result = detector.process_video_frame(frame)
        
        # Convert processed frame ke base64
        _, buffer = cv2.imencode('.jpg', result['processed_frame'])
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "drowsiness_detected": result['drowsiness_detected'],
            "drowsiness_score": result['drowsiness_score'],
            "metrics": result['metrics'],
            "alerts": result['alerts'],
            "processed_frame": processed_frame_b64,
            "frame_count": result['frame_count']
        }
        
    except Exception as e:
        logger.error(f"Error processing video frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error memproses frame: {str(e)}")

@app.get("/detector/config")
async def get_detector_config():
    """
    Mendapatkan konfigurasi detector saat ini
    """
    return detector.get_config()

@app.post("/detector/config")
async def update_detector_config(config: dict):
    """
    Update konfigurasi detector
    """
    try:
        detector.update_config(config)
        return {
            "status": "success",
            "message": "Konfigurasi berhasil diupdate",
            "new_config": detector.get_config()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error updating config: {str(e)}")

@app.post("/detector/reset")
async def reset_detector():
    """
    Reset detector state (counters, history, dll)
    """
    try:
        detector.reset_state()
        return {
            "status": "success",
            "message": "Detector state berhasil direset"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting detector: {str(e)}")

@app.get("/detector/stats")
async def get_detector_stats():
    """
    Mendapatkan statistik detector
    """
    return detector.get_statistics()

# Endpoint untuk streaming (opsional, untuk implementasi real-time)
@app.get("/stream/start")
async def start_stream():
    """
    Mulai streaming deteksi real-time
    (Implementasi ini memerlukan WebSocket atau Server-Sent Events)
    """
    return {
        "message": "Streaming endpoint - implementasi memerlukan WebSocket",
        "note": "Gunakan endpoint /detect/video-frame untuk processing frame-by-frame"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")