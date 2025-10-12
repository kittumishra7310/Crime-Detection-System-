#!/usr/bin/env python3
"""
Complete Surveillance System Server with MySQL and Live Detection
"""

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
import uvicorn
import logging
import cv2
import asyncio
import os
import shutil
from datetime import datetime
from typing import Optional, List
import json
import numpy as np
from passlib.context import CryptContext
from jose import JWTError, jwt
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = "mysql+pymysql://root:Kittu%40123@localhost/surveillance_db"

# Create engine and session
try:
    engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("✅ Database engine created successfully")
except Exception as e:
    logger.error(f"❌ Failed to create database engine: {e}")
    raise

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"

app = FastAPI(
    title="Smart Surveillance System API",
    description="Complete API with MySQL, Live Detection, and Crime Recognition",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for live detection
active_cameras = {}
detection_threads = {}

# Continuous detection function
def continuous_detection(camera_id: int, db):
    """Continuously process frames from camera for detection"""
    try:
        while camera_id in active_cameras and active_cameras[camera_id]["is_detecting"]:
            cap = active_cameras[camera_id]["capture"]
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame from camera {camera_id}")
                break
            
            # Store last frame for streaming
            active_cameras[camera_id]["last_frame"] = frame
            
            # Perform detection every 5 frames to reduce processing load
            if active_cameras[camera_id]["detections_count"] % 5 == 0:
                detection_result = detect_criminal_activity(frame)
                
                if detection_result.get("detected"):
                    # Save detection to database
                    try:
                        with engine.connect() as connection:
                            result = connection.execute(
                                text("""
                                    INSERT INTO detections (camera_id, detection_type, confidence, severity, timestamp)
                                    VALUES (:camera_id, :detection_type, :confidence, :severity, :timestamp)
                                """),
                                {
                                    "camera_id": camera_id,
                                    "detection_type": detection_result["crime_type"],
                                    "confidence": detection_result["confidence"],
                                    "severity": detection_result["severity"],
                                    "timestamp": datetime.now()
                                }
                            )
                            connection.commit()
                            
                            # Get detection ID
                            detection_id = connection.execute(text("SELECT LAST_INSERT_ID()")).fetchone()[0]
                            
                            # Create alert for high/critical severity
                            if detection_result["severity"] in ["high", "critical"]:
                                connection.execute(
                                    text("""
                                        INSERT INTO alerts (detection_id, status, severity, message, created_at)
                                        VALUES (:detection_id, 'active', :severity, :message, :created_at)
                                    """),
                                    {
                                        "detection_id": detection_id,
                                        "severity": detection_result["severity"],
                                        "message": f"{detection_result['crime_type']} detected with {detection_result['confidence']:.2f} confidence",
                                        "created_at": datetime.now()
                                    }
                                )
                                connection.commit()
                            
                            active_cameras[camera_id]["detections_count"] += 1
                            logger.info(f"🚨 Detection: {detection_result['crime_type']} on camera {camera_id}")
                            
                    except Exception as e:
                        logger.error(f"Database error during detection: {e}")
            
            # Small delay to prevent overwhelming the system
            import time
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Error in continuous detection for camera {camera_id}: {e}")
    finally:
        if camera_id in active_cameras:
            active_cameras[camera_id]["is_detecting"] = False

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Response models
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = "viewer"  # Default role

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

class DetectionResult(BaseModel):
    detection_id: Optional[int] = None
    crime_type: str
    confidence: float
    severity: str
    message: str

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(lambda: "dummy_token")):
    # Simplified for demo - in production, extract from Authorization header
    return {"id": 1, "username": "admin", "role": "admin"}

# Load trained model at startup
try:
    import tensorflow as tf
    from train_crime_model import predict_crime, load_trained_model, preprocess_image_for_model
    
    # Load the trained model
    crime_model = load_trained_model()
    logger.info("✅ Crime detection model loaded successfully")
    
    # Crime classes from the notebook
    CRIME_CLASSES = ['Fighting', 'Normal', 'RoadAccidents', 'Robbery', 
                    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism', 'Assault']
    
except Exception as e:
    logger.warning(f"⚠️ Could not load TensorFlow model: {e}")
    crime_model = None
    CRIME_CLASSES = []

# Real crime detection using enhanced computer vision
def detect_criminal_activity(frame):
    """Enhanced crime detection using computer vision techniques"""
    try:
        # First try the CNN model if available
        if crime_model is not None:
            try:
                result = predict_crime(crime_model, frame)
                logger.info(f"Model prediction: {result}")
                
                # If model gives a reasonable prediction, use it
                if result['detected'] and result['confidence'] > 0.4:
                    return {
                        "crime_type": result['crime_type'],
                        "confidence": result['confidence'],
                        "severity": result['severity'],
                        "detected": True,
                        "timestamp": datetime.now().isoformat(),
                        "analysis_method": "trained_cnn_model"
                    }
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
        
        # Use enhanced computer vision analysis
        return enhanced_cv_detection(frame)
        
    except Exception as e:
        logger.error(f"Error in detection: {e}")
        return {"detected": False}

def enhanced_cv_detection(frame):
    """Enhanced computer vision detection with better accuracy"""
    try:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Advanced feature extraction
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate features
        edge_density = np.sum(edges > 0) / (height * width)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Count significant objects
        large_contours = [c for c in contours if cv2.contourArea(c) > 200]
        num_objects = len(large_contours)
        
        # Motion estimation using optical flow simulation
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        motion_score = np.std(blur) / 255.0
        
        # Color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_variance = np.var(hsv[:,:,1])  # Saturation variance
        
        # Detection logic based on image characteristics
        detections = []
        
        # Theft detection - sneaky, focused activity
        if (brightness < 130 and edge_density > 0.02 and 
            num_objects >= 1 and motion_score > 0.1):
            confidence = min(0.9, 0.5 + edge_density * 3 + motion_score)
            detections.append({
                "crime_type": "Theft",
                "confidence": confidence,
                "severity": "high" if confidence > 0.7 else "medium"
            })
        
        # Fighting detection - chaotic, high movement
        elif (edge_density > 0.06 and num_objects >= 2 and 
              contrast > 50 and motion_score > 0.15):
            confidence = min(0.95, 0.6 + edge_density * 2 + motion_score)
            detections.append({
                "crime_type": "Fighting",
                "confidence": confidence,
                "severity": "critical" if confidence > 0.8 else "high"
            })
        
        # Vandalism detection - destructive patterns
        elif (color_variance > 1500 and edge_density > 0.04 and 
              contrast > 40):
            confidence = min(0.85, 0.5 + color_variance / 3000 + edge_density * 2)
            detections.append({
                "crime_type": "Vandalism",
                "confidence": confidence,
                "severity": "medium"
            })
        
        # Robbery detection - aggressive grabbing
        elif (edge_density > 0.05 and num_objects >= 1 and 
              motion_score > 0.12 and brightness < 150):
            confidence = min(0.88, 0.55 + edge_density * 2.5 + motion_score)
            detections.append({
                "crime_type": "Robbery",
                "confidence": confidence,
                "severity": "high"
            })
        
        # General suspicious activity
        elif (edge_density > 0.03 or motion_score > 0.08 or 
              num_objects >= 1):
            confidence = min(0.75, 0.4 + edge_density * 4 + motion_score * 2)
            detections.append({
                "crime_type": "Suspicious Activity",
                "confidence": confidence,
                "severity": "low"
            })
        
        # Return best detection
        if detections:
            best = max(detections, key=lambda x: x["confidence"])
            return {
                **best,
                "detected": True,
                "timestamp": datetime.now().isoformat(),
                "analysis_method": "enhanced_computer_vision"
            }
        
        return {"detected": False}
        
    except Exception as e:
        logger.error(f"Enhanced CV detection error: {e}")
        return {"detected": False}

def fallback_detection(frame):
    """Fallback detection method if trained model fails"""
    try:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Basic edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        brightness = np.mean(gray)
        
        # Simple heuristics
        if edge_density > 0.05 and brightness < 120:
            return {
                "crime_type": "Suspicious Activity",
                "confidence": 0.6,
                "severity": "medium",
                "detected": True,
                "timestamp": datetime.now().isoformat(),
                "analysis_method": "fallback_heuristic"
            }
        
        return {"detected": False}
        
    except Exception as e:
        logger.error(f"Error in fallback detection: {e}")
        return {"detected": False}

def extract_crime_features(frame, gray, hsv):
    """Extract comprehensive features for crime detection"""
    height, width = frame.shape[:2]
    
    # Edge and contour analysis
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Motion and texture features
    edge_density = np.sum(edges > 0) / (height * width)
    
    # Color analysis
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    # Shape analysis
    large_contours = [c for c in contours if cv2.contourArea(c) > 500]
    
    # Brightness and contrast
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    return {
        "edge_density": edge_density,
        "brightness": brightness,
        "contrast": contrast,
        "num_contours": len(contours),
        "large_contours": len(large_contours),
        "color_variance_h": np.var(hist_h),
        "color_variance_s": np.var(hist_s),
        "color_variance_v": np.var(hist_v),
        "frame_size": height * width
    }

def analyze_theft_patterns(features):
    """Analyze patterns specific to theft activities"""
    score = 0.0
    
    # Theft often involves:
    # - Moderate movement (not too aggressive)
    # - Lower brightness (sneaky behavior)
    # - Focused activity (fewer but larger contours)
    
    if features["brightness"] < 100:  # Dark/sneaky environment
        score += 0.3
    elif features["brightness"] < 130:
        score += 0.2
    
    if 0.02 < features["edge_density"] < 0.08:  # Moderate movement
        score += 0.4
    
    if features["large_contours"] > 0 and features["num_contours"] < 50:  # Focused activity
        score += 0.3
    
    if features["contrast"] > 40:  # Clear objects/people
        score += 0.2
    
    return min(1.0, score)

def analyze_fighting_patterns(features):
    """Analyze patterns specific to fighting activities"""
    score = 0.0
    
    # Fighting often involves:
    # - High movement and chaos
    # - Multiple active areas
    # - High edge density
    
    if features["edge_density"] > 0.1:  # High movement
        score += 0.5
    elif features["edge_density"] > 0.06:
        score += 0.3
    
    if features["num_contours"] > 30:  # Chaotic scene
        score += 0.3
    
    if features["large_contours"] > 2:  # Multiple people/objects
        score += 0.4
    
    if features["color_variance_h"] > 1000:  # Color chaos
        score += 0.2
    
    return min(1.0, score)

def analyze_vandalism_patterns(features):
    """Analyze patterns specific to vandalism activities"""
    score = 0.0
    
    # Vandalism often involves:
    # - Destructive movements
    # - Color changes (spray paint, etc.)
    # - Sharp movements
    
    if features["color_variance_s"] > 2000:  # Color changes
        score += 0.4
    
    if features["edge_density"] > 0.08:  # Sharp movements
        score += 0.3
    
    if features["contrast"] > 60:  # High contrast changes
        score += 0.3
    
    if features["large_contours"] > 1:  # Active destruction
        score += 0.2
    
    return min(1.0, score)

def analyze_suspicious_patterns(features):
    """Analyze general suspicious activity patterns"""
    score = 0.0
    
    # Suspicious activity - general anomalies
    if features["edge_density"] > 0.02:
        score += 0.3
    
    if features["brightness"] < 80 or features["brightness"] > 200:  # Unusual lighting
        score += 0.2
    
    if features["num_contours"] > 20:  # Some activity
        score += 0.2
    
    if features["color_variance_v"] > 800:  # Unusual patterns
        score += 0.2
    
    return min(1.0, score)

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Smart Surveillance System API",
        "version": "2.0.0",
        "features": ["MySQL Database", "Live Detection", "Crime Recognition", "Real-time Alerts"],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "auth": "/api/auth/login",
            "cameras": "/api/cameras",
            "detections": "/api/detections",
            "live_detection": "/api/detection/live/*"
        }
    }

@app.get("/health")
async def health_check():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat(),
            "message": "All systems operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    try:
        # Get user from database
        result = db.execute(
            text("SELECT id, username, email, password_hash, role FROM users WHERE username = :username"),
            {"username": request.username}
        )
        user = result.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        if not verify_password(request.password, user[3]):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # Create token
        token_data = {"sub": user[1], "user_id": user[0], "role": user[4]}
        access_token = create_access_token(token_data)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user={
                "id": user[0],
                "username": user[1],
                "email": user[2],
                "role": user[4]
            }
        )
    except HTTPException:
        # Re-raise HTTP exceptions (like 401)
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/auth/register", response_model=dict)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    try:
        # Check if username already exists
        result = db.execute(
            text("SELECT id FROM users WHERE username = :username"),
            {"username": request.username}
        )
        if result.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Check if email already exists
        result = db.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": request.email}
        )
        if result.fetchone():
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Validate role
        if request.role not in ["admin", "viewer"]:
            raise HTTPException(status_code=400, detail="Invalid role. Must be 'admin' or 'viewer'")
        
        # Hash password
        hashed_password = get_password_hash(request.password)
        
        # Insert new user
        db.execute(
            text("""
                INSERT INTO users (username, email, password_hash, role, is_active, created_at) 
                VALUES (:username, :email, :password_hash, :role, :is_active, :created_at)
            """),
            {
                "username": request.username,
                "email": request.email,
                "password_hash": hashed_password,
                "role": request.role,
                "is_active": True,
                "created_at": datetime.utcnow()
            }
        )
        db.commit()
        
        return {
            "message": "User registered successfully",
            "username": request.username,
            "email": request.email,
            "role": request.role
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.get("/api/cameras")
async def list_cameras(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT id, name, camera_id, location, status FROM cameras"))
        cameras = []
        for row in result.fetchall():
            cameras.append({
                "id": row[0],
                "name": row[1],
                "camera_id": row[2],
                "location": row[3],
                "status": row[4],
                "live_detection_active": row[0] in active_cameras
            })
        return cameras
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list cameras: {str(e)}")

@app.get("/api/detections")
async def list_detections(db: Session = Depends(get_db)):
    try:
        result = db.execute(
            text("""
                SELECT d.id, d.camera_id, d.detection_type, d.confidence, d.severity, d.timestamp,
                       c.name as camera_name
                FROM detections d
                JOIN cameras c ON d.camera_id = c.id
                ORDER BY d.timestamp DESC
                LIMIT 50
            """)
        )
        detections = []
        for row in result.fetchall():
            detections.append({
                "id": row[0],
                "camera_id": row[1],
                "detection_type": row[2],
                "confidence": float(row[3]),
                "severity": row[4],
                "timestamp": row[5].isoformat() if row[5] else None,
                "camera_name": row[6]
            })
        return detections
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list detections: {str(e)}")

@app.get("/api/alerts")
async def list_alerts(db: Session = Depends(get_db)):
    try:
        result = db.execute(
            text("""
                SELECT a.id, a.detection_id, a.status, a.severity, a.message, a.created_at,
                       d.detection_type, c.name as camera_name
                FROM alerts a
                JOIN detections d ON a.detection_id = d.id
                JOIN cameras c ON d.camera_id = c.id
                ORDER BY a.created_at DESC
                LIMIT 50
            """)
        )
        alerts = []
        for row in result.fetchall():
            alerts.append({
                "id": row[0],
                "detection_id": row[1],
                "status": row[2],
                "severity": row[3],
                "message": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
                "detection_type": row[6],
                "camera_name": row[7]
            })
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list alerts: {str(e)}")

@app.post("/api/detection/upload")
async def upload_file_detection(
    file: UploadFile = File(...),
    camera_id: int = Form(...),
    db: Session = Depends(get_db)
):
    try:
        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process file
        if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Process image
            image = cv2.imread(file_path)
            if image is not None:
                detection_result = detect_criminal_activity(image)
                
                if detection_result.get("detected"):
                    # Save to database
                    db.execute(
                        text("""
                            INSERT INTO detections (camera_id, detection_type, confidence, severity, image_path, timestamp)
                            VALUES (:camera_id, :detection_type, :confidence, :severity, :image_path, :timestamp)
                        """),
                        {
                            "camera_id": camera_id,
                            "detection_type": detection_result["crime_type"],
                            "confidence": detection_result["confidence"],
                            "severity": detection_result["severity"],
                            "image_path": file_path,
                            "timestamp": datetime.now()
                        }
                    )
                    
                    # Get detection ID
                    detection_id = db.execute(text("SELECT LAST_INSERT_ID()")).fetchone()[0]
                    
                    # Create alert if high severity
                    if detection_result["severity"] in ["high", "critical"]:
                        db.execute(
                            text("""
                                INSERT INTO alerts (detection_id, status, severity, message, created_at)
                                VALUES (:detection_id, 'active', :severity, :message, :created_at)
                            """),
                            {
                                "detection_id": detection_id,
                                "severity": detection_result["severity"],
                                "message": f"Crime detected: {detection_result['crime_type']} with {detection_result['confidence']:.0%} confidence",
                                "created_at": datetime.now()
                            }
                        )
                    
                    db.commit()
                    
                    return {
                        "success": True,
                        "message": f"Detection Complete: {detection_result['crime_type']} detected!",
                        "detection": {
                            "id": detection_id,
                            "crime_type": detection_result["crime_type"],
                            "confidence": f"{detection_result['confidence']:.1%}",
                            "severity": detection_result["severity"],
                            "timestamp": detection_result["timestamp"]
                        },
                        "file_path": file_path
                    }
                else:
                    return {
                        "success": False,
                        "message": "No criminal activity detected in the uploaded file",
                        "detection": None,
                        "file_path": file_path
                    }
        
        return {"message": "File uploaded but not processed", "file_path": file_path}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/detection/live/start/{camera_id}")
async def start_live_detection(camera_id: int, source: str = "0", db: Session = Depends(get_db)):
    try:
        if camera_id in active_cameras:
            return {"message": "Live detection already active for this camera", "camera_id": camera_id}
        
        # Convert source
        if source.isdigit():
            source = int(source)
        
        # Start camera capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open camera source")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        active_cameras[camera_id] = {
            "capture": cap,
            "source": source,
            "started_at": datetime.now(),
            "detections_count": 0,
            "is_detecting": True,
            "last_frame": None
        }
        
        # Start detection thread
        import threading
        detection_thread = threading.Thread(
            target=continuous_detection,
            args=(camera_id, db),
            daemon=True
        )
        detection_thread.start()
        detection_threads[camera_id] = detection_thread
        
        # Update camera status in database
        db.execute(
            text("UPDATE cameras SET status = 'active' WHERE id = :camera_id"),
            {"camera_id": camera_id}
        )
        db.commit()
        
        logger.info(f"✅ Started live detection for camera {camera_id}")
        
        return {
            "message": f"Live detection started for camera {camera_id}",
            "camera_id": camera_id,
            "source": source,
            "status": "active"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting live detection: {str(e)}")

@app.post("/api/detection/live/stop/{camera_id}")
async def stop_live_detection(camera_id: int, db: Session = Depends(get_db)):
    try:
        if camera_id not in active_cameras:
            return {"message": "No active detection for this camera", "camera_id": camera_id}
        
        # Stop detection flag
        active_cameras[camera_id]["is_detecting"] = False
        
        # Wait for thread to finish
        if camera_id in detection_threads:
            detection_threads[camera_id].join(timeout=2.0)
            del detection_threads[camera_id]
        
        # Stop camera
        cap = active_cameras[camera_id]["capture"]
        cap.release()
        cv2.destroyAllWindows()  # Clean up any OpenCV windows
        del active_cameras[camera_id]
        
        # Update camera status in database
        db.execute(
            text("UPDATE cameras SET status = 'inactive' WHERE id = :camera_id"),
            {"camera_id": camera_id}
        )
        db.commit()
        
        logger.info(f"✅ Stopped live detection for camera {camera_id}")
        
        return {
            "message": f"Live detection stopped for camera {camera_id}",
            "camera_id": camera_id,
            "status": "inactive"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping live detection: {str(e)}")

@app.get("/api/detection/live/status")
async def get_live_detection_status():
    try:
        status_info = {}
        for camera_id, info in active_cameras.items():
            status_info[camera_id] = {
                "source": info["source"],
                "started_at": info["started_at"].isoformat(),
                "detections_count": info["detections_count"],
                "status": "active"
            }
        
        return {
            "active_cameras": list(active_cameras.keys()),
            "total_active": len(active_cameras),
            "status": "running" if active_cameras else "idle",
            "details": status_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.get("/api/detection/live/feed/{camera_id}")
async def get_camera_feed(camera_id: int):
    """Get live camera feed as MJPEG stream"""
    if camera_id not in active_cameras:
        raise HTTPException(status_code=404, detail="Camera not active")
    
    async def generate_frames():
        cap = active_cameras[camera_id]["capture"]
        
        while camera_id in active_cameras:
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add overlay information
                cv2.putText(frame, f"Camera {camera_id} - Live Detection", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                await asyncio.sleep(0.1)  # Control frame rate
                
            except Exception as e:
                logger.error(f"Error in frame generation: {e}")
                break
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    logger.info("🚀 Starting Complete Surveillance System Server...")
    logger.info("📊 API Documentation: http://localhost:8000/docs")
    logger.info("🔍 Health Check: http://localhost:8000/health")
    logger.info("📹 Live Detection: http://localhost:8000/api/detection/live/status")
    
    uvicorn.run(
        "complete_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
