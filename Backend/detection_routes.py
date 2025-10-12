from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.orm import Session
from typing import List, Optional
import shutil
import os
from datetime import datetime, timedelta
import asyncio
import json
import cv2

from database import get_db, Detection, Alert, Camera
from auth import get_current_user
from schemas import DetectionResponse, DetectionCreate, UserResponse, FileDetectionResponse
from ml_model import CrimeDetectionModel
from websocket_manager import websocket_manager
from live_detection import live_detection_manager
from config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
model = CrimeDetectionModel()

def save_detection_to_db(
    db: Session,
    camera_id: int,
    detection_result: dict,
    image_path: str = None,
    video_path: str = None
) -> Detection:
    """Save detection result to database."""
    detection = Detection(
        camera_id=camera_id,
        detection_type=detection_result["crime_type"],
        confidence=detection_result["confidence"],
        severity=detection_result["severity"],
        image_path=image_path,
        video_path=video_path,
        bounding_box=json.dumps(detection_result.get("bounding_box", {}))
    )
    
    db.add(detection)
    db.commit()
    db.refresh(detection)
    
    # Create alert if crime detected and confidence is high
    if detection_result["is_crime"] and detection_result["confidence"] >= settings.ALERT_THRESHOLD:
        alert = Alert(
            detection_id=detection.id,
            severity=detection_result["severity"],
            message=f"Crime detected: {detection_result['crime_type']} with {detection_result['confidence']:.2%} confidence"
        )
        db.add(alert)
        db.commit()
    
    return detection

@router.post("/api/detection/upload", response_model=FileDetectionResponse)
async def upload_file_detection(
    file: UploadFile = File(...),
    camera_id: int = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: UserResponse = Depends(get_current_user)
):
    """Upload and analyze image or video file for crime detection in the background."""
    try:
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.mkv'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        upload_dir = settings.UPLOAD_DIR
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the file in the background
        background_tasks.add_task(live_detection_manager.process_uploaded_file, file_path, camera_id)

        return {
            "message": "File upload successful. Processing in the background.",
            "success": True,
            "file_path": file_path
        }

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/live/start/{camera_id}")
async def start_live_detection(
    camera_id: int,
    source: str = "0",  # Default to webcam, can be RTSP URL
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start live detection on a camera feed."""
    try:
        # Check if camera exists
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        # Convert source to appropriate format
        if source.isdigit():
            source = int(source)  # Webcam index
        # Otherwise assume it's an RTSP URL or file path
        
        success = await live_detection_manager.start_camera_detection(camera_id, source)
        
        if success:
            # Update camera status
            camera.status = "active"
            db.commit()
            
            return {
                "message": f"Live detection started for camera {camera_id}",
                "camera_id": camera_id,
                "source": source,
                "status": "active"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start live detection")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting live detection: {str(e)}")

@router.post("/live/stop/{camera_id}")
async def stop_live_detection(
    camera_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Stop live detection on a camera."""
    try:
        success = await live_detection_manager.stop_camera_detection(camera_id)
        
        if success:
            # Update camera status
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            if camera:
                camera.status = "inactive"
                db.commit()
            
            return {
                "message": f"Live detection stopped for camera {camera_id}",
                "camera_id": camera_id,
                "status": "inactive"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to stop live detection")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping live detection: {str(e)}")

@router.get("/live/feed/{camera_id}")
async def get_camera_feed(
    camera_id: int,
    token: str = None
):
    """Get live camera feed as MJPEG stream with token authentication."""
    # Validate token if provided
    if token:
        try:
            from auth import verify_token
            payload = verify_token(token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication required")
    else:
        raise HTTPException(status_code=401, detail="Token required")
    async def generate_frames():
        # Check if camera is active - DO NOT auto-start
        if camera_id not in live_detection_manager.active_streams:
            # Send error frame instead of auto-starting
            error_frame = create_error_frame("Camera detection not active")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            return
        
        frame_count = 0
        consecutive_no_frames = 0
        max_no_frames = 50  # Stop after 50 consecutive failed frame reads (5 seconds)
        
        while True:
            try:
                # Check if camera is still active (detection not stopped)
                if camera_id not in live_detection_manager.active_streams:
                    logger.info(f"Camera {camera_id} detection stopped, ending feed")
                    break
                
                frame_data = live_detection_manager.get_camera_frame(camera_id)
                if frame_data:
                    consecutive_no_frames = 0
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                else:
                    consecutive_no_frames += 1
                    # No frame available, send placeholder only initially
                    if frame_count == 0:
                        placeholder_frame = create_placeholder_frame()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + placeholder_frame + b'\r\n')
                    
                    # If too many consecutive failed reads, stop the feed
                    if consecutive_no_frames >= max_no_frames:
                        logger.info(f"Camera {camera_id} feed stopped due to no frames")
                        break
                    
                    await asyncio.sleep(0.1)
                
                frame_count += 1
                    
            except Exception as e:
                logger.error(f"Error in camera feed: {e}")
                break
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

def create_placeholder_frame():
    """Create a placeholder frame when camera is not available."""
    import numpy as np
    # Create a simple placeholder image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Camera Feed Loading...", (150, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def create_error_frame(message: str):
    """Create an error frame with message."""
    import numpy as np
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, message, (150, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

@router.get("/live/status")
async def get_live_detection_status(
    current_user: UserResponse = Depends(get_current_user)
):
    """Get status of all active live detections."""
    try:
        active_cameras = await live_detection_manager.get_active_cameras()
        
        return {
            "active_cameras": active_cameras,
            "total_active": len(active_cameras),
            "status": "running" if active_cameras else "idle"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@router.get("/api/detections", response_model=List[DetectionResponse])
async def list_detections(
    camera_id: Optional[int] = None,
    detection_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List detections with optional filters."""
    
    query = db.query(Detection)
    
    if camera_id:
        query = query.filter(Detection.camera_id == camera_id)
    
    if detection_type:
        query = query.filter(Detection.detection_type == detection_type)
    
    if start_date:
        query = query.filter(Detection.timestamp >= start_date)
    
    if end_date:
        query = query.filter(Detection.timestamp <= end_date)
    
    detections = query.order_by(Detection.timestamp.desc()).limit(limit).all()
    
    return [DetectionResponse.from_orm(detection) for detection in detections]

@router.get("/api/detections/{detection_id}", response_model=DetectionResponse)
async def get_detection(
    detection_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get specific detection by ID."""
    
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection not found"
        )
    
    return DetectionResponse.from_orm(detection)
