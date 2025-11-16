from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
import os
import shutil
import uuid
import cv2
import numpy as np
import asyncio
import json
import logging

# Local imports
from config import settings
from database import get_db, create_tables, User, Camera, Detection, Alert, SystemLog
from auth import (
    authenticate_user, create_access_token, get_current_user, 
    get_current_admin_user, get_password_hash
)
from schemas import (
    LoginRequest, TokenResponse, UserCreate, UserResponse, UserUpdate,
    CameraCreate, CameraResponse, CameraUpdate, DetectionResponse,
    AlertResponse, AlertUpdate, SystemStatus, DetectionResult,
    LiveDetectionResponse, FileDetectionResponse, AnalyticsResponse,
    PaginationParams
)
from ml_model import get_crime_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Smart Surveillance System API",
    description="Crime Detection Surveillance System with AI-powered detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Static files
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    create_tables()
    
    # Load ML model
    # model = get_crime_model()
    # if not model.load_model():
    #     logger.warning("Failed to load ML model")
    
    logger.info("System started successfully - Ready for user registration")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Authentication endpoints
@app.post("/api/auth/login", response_model=TokenResponse)
async def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return access token."""
    user = authenticate_user(db, login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.from_orm(user)
    )

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse.from_orm(current_user)

@app.post("/api/auth/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password,
        role=user_data.role or 'viewer'  # Default to 'viewer' if role is not provided
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse.from_orm(db_user)

@app.post("/api/auth/clerk-sync", response_model=UserResponse)
async def clerk_sync(user_data: UserCreate, db: Session = Depends(get_db)):
    """Sync Clerk user - create or update existing user."""
    # Check if user exists by username
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    
    if existing_user:
        # Update existing user's password
        existing_user.password_hash = get_password_hash(user_data.password)
        existing_user.email = user_data.email
        db.commit()
        db.refresh(existing_user)
        return UserResponse.from_orm(existing_user)
    
    # Check by email
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        # Update existing user's password and username
        existing_email.password_hash = get_password_hash(user_data.password)
        existing_email.username = user_data.username
        db.commit()
        db.refresh(existing_email)
        return UserResponse.from_orm(existing_email)
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password,
        role=user_data.role or 'viewer'
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse.from_orm(db_user)

# User management endpoints
@app.post("/api/admin/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Create a new user (admin only)."""
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password,
        role=user_data.role
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse.from_orm(db_user)

@app.get("/api/admin/users", response_model=List[UserResponse])
async def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """List all users (admin only)."""
    users = db.query(User).all()
    return [UserResponse.from_orm(user) for user in users]

@app.delete("/api/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Delete a user (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    db.delete(user)
    db.commit()
    
    return {"message": "User deleted successfully"}

# Camera management endpoints
@app.post("/api/admin/cameras", response_model=CameraResponse)
async def create_camera(
    camera_data: CameraCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Register a new camera (admin only)."""
    # Check if camera_id already exists
    existing_camera = db.query(Camera).filter(Camera.camera_id == camera_data.camera_id).first()
    if existing_camera:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera ID already exists"
        )
    
    db_camera = Camera(**camera_data.dict())
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    
    return CameraResponse.from_orm(db_camera)

@app.get("/api/cameras", response_model=List[CameraResponse])
async def list_cameras(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all cameras."""
    cameras = db.query(Camera).all()
    return [CameraResponse.from_orm(camera) for camera in cameras]

@app.put("/api/admin/cameras/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: int,
    camera_data: CameraUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Update camera information (admin only)."""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found"
        )
    
    update_data = camera_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(camera, field, value)
    
    db.commit()
    db.refresh(camera)
    
    return CameraResponse.from_orm(camera)

@app.delete("/api/admin/cameras/{camera_id}")
async def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Delete a camera (admin only)."""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found"
        )
    
    db.delete(camera)
    db.commit()
    
    return {"message": "Camera deleted successfully"}

@app.post("/api/detection/upload", response_model=FileDetectionResponse)
async def upload_for_detection(
    file: UploadFile = File(...),
    camera_id: int = Form(1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload a file for crime detection."""
    if not file.content_type.startswith(("image/", "video/")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only images and videos are supported."
        )

    # Generate a unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

    # Save the file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save file."
        )

    # Process the file with the detection model
    try:
        from live_detection import live_detection_manager
        
        logger.info(f"Processing uploaded file: {file.filename}")
        result = await live_detection_manager.process_uploaded_file(file_path, camera_id)
        
        if "error" in result:
            logger.error(f"Detection error: {result['error']}")
            return FileDetectionResponse(
                success=False,
                message=f"Detection failed: {result['error']}",
                filename=unique_filename
            )
        
        # For images or single detection
        if "crime_type" in result:
            detection_data = {
                "crime_type": result.get("crime_type", "Unknown"),
                "confidence": result.get("confidence", 0.0),
                "severity": result.get("severity", "low"),
                "id": result.get("detection_id", 0)
            }
            message = f"Detection complete: {result.get('crime_type', 'Unknown')} ({result.get('confidence', 0)*100:.1f}% confidence)"
        # For videos with multiple detections
        elif "detections" in result:
            num_detections = len(result["detections"])
            if num_detections > 0:
                latest = result["detections"][0]
                detection_data = {
                    "crime_type": latest.get("crime_type", "Unknown"),
                    "confidence": latest.get("confidence", 0.0),
                    "severity": latest.get("severity", "low"),
                    "id": latest.get("detection_id", 0)
                }
                message = f"Video processed: {num_detections} detection(s) found"
            else:
                detection_data = None
                message = "Video processed: No crimes detected"
        else:
            detection_data = None
            message = result.get("message", "File processed successfully")
        
        return FileDetectionResponse(
            success=True,
            message=message,
            filename=unique_filename,
            detection=detection_data
        )
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return FileDetectionResponse(
            success=False,
            message=f"Processing error: {str(e)}",
            filename=unique_filename
        )

# Include routers
from detection_routes import router as detection_router
from alert_routes import router as alert_router
from analytics_routes import router as analytics_router
from system_routes import router as system_router
from websocket_routes import router as websocket_router

async def video_streamer(camera_source: any = 0):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        logger.error(f"Cannot open camera source: {camera_source}")
        return

    logger.info(f"Camera {camera_source} opened for streaming.")
    try:
        while True:
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                logger.warning(f"Camera {camera_source} feed ended.")
                break

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
            await asyncio.sleep(0.03)  # Limit to ~30 FPS
    except asyncio.CancelledError:
        logger.info(f"Client disconnected, closing camera {camera_source}.")
    finally:
        cap.release()
        logger.info(f"Camera {camera_source} released.")

@app.get("/api/live/feed/{camera_id}")
async def camera_feed(camera_id: int):
    # In a real app, you'd get the camera source from the database
    # For now, we'll just use the default camera (0)
    return StreamingResponse(
        video_streamer(0),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

app.include_router(detection_router)
app.include_router(alert_router)
app.include_router(analytics_router)
app.include_router(system_router)
app.include_router(websocket_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)