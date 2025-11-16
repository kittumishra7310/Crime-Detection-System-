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
from ml_model import get_crime_model
from base_model import BaseDetectionModel
from websocket_manager import websocket_manager
from live_detection import LiveDetectionManager
from dependencies import get_live_detection_manager
from config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
# Use the global model instance from ml_model
model = get_crime_model()

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
    current_user: UserResponse = Depends(get_current_user),
    live_detection_manager: LiveDetectionManager = Depends(get_live_detection_manager)
):
    """Upload and analyze image or video file for crime detection."""
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

        # Process the file immediately and return results
        result = await live_detection_manager.process_uploaded_file(file_path, camera_id)
        
        # Format the response
        if isinstance(result, dict) and 'error' in result:
            return {
                "message": result['error'],
                "success": False,
                "filename": file.filename,
                "file_path": file_path
            }
        
        # For images, result is the prediction
        if file_ext in ['.jpg', '.jpeg', '.png']:
            is_crime = result.get('is_crime', False) if result else False
            crime_type = result.get('crime_type', 'Unknown') if result else 'Unknown'
            confidence = result.get('confidence', 0.0) if result else 0.0
            
            message = f"Detected: {crime_type} (confidence: {confidence:.2%})"
            if is_crime:
                message += " - ⚠️ SUSPICIOUS ACTIVITY DETECTED!"
            
            return {
                "message": message,
                "success": True,
                "filename": file.filename,
                "file_path": file_path,
                "detections": [result] if result else []
            }
        
        # For videos, result contains list of detections
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            detections = result.get('detections', []) if isinstance(result, dict) else []
            crime_count = len(detections)
            
            if crime_count > 0:
                message = f"⚠️ Found {crime_count} suspicious activities in video!"
            else:
                message = "Video processed - No suspicious activity detected"
            
            return {
                "message": message,
                "success": True,
                "filename": file.filename,
                "file_path": file_path,
                "detections": detections
            }

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/live/start/{camera_id}")
async def start_live_detection(
    camera_id: int,
    source: str = "0",  # Default to webcam, can be RTSP URL
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
    live_detection_manager: LiveDetectionManager = Depends(get_live_detection_manager)
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
    db: Session = Depends(get_db),
    live_detection_manager: LiveDetectionManager = Depends(get_live_detection_manager)
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

@router.post("/live/stop-all")
async def stop_all_live_detection(
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
    live_detection_manager: LiveDetectionManager = Depends(get_live_detection_manager)
):
    """Force stop all active live detections."""
    try:
        await live_detection_manager.force_stop_all()
        
        # Update all camera statuses to inactive
        cameras = db.query(Camera).all()
        for camera in cameras:
            camera.status = "inactive"
        db.commit()
        
        return {
            "message": "All live detections stopped",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping all detections: {str(e)}")

@router.get("/live/feed/{camera_id}")
async def get_camera_feed(
    camera_id: int,
    token: str = None,
    live_detection_manager: LiveDetectionManager = Depends(get_live_detection_manager)
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
    current_user: UserResponse = Depends(get_current_user),
    live_detection_manager: LiveDetectionManager = Depends(get_live_detection_manager)
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

@router.get("/live/status/{camera_id}")
async def get_camera_status(
    camera_id: int,
    current_user: UserResponse = Depends(get_current_user),
    live_detection_manager: LiveDetectionManager = Depends(get_live_detection_manager)
):
    """Get detailed status for a specific camera."""
    try:
        status_info = await live_detection_manager.get_camera_status(camera_id)
        
        if not status_info:
            raise HTTPException(status_code=404, detail="Camera not found or not active")
        
        return status_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting camera status: {str(e)}")

@router.get("/live/performance")
async def get_performance_metrics(
    current_user: UserResponse = Depends(get_current_user),
    live_detection_manager: LiveDetectionManager = Depends(get_live_detection_manager)
):
    """Get real-time performance metrics for all active detections."""
    try:
        metrics = await live_detection_manager.get_performance_metrics()
        
        return {
            "system_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")

@router.post("/live/configure/{camera_id}")
async def configure_camera_detection(
    camera_id: int,
    confidence_threshold: float = Form(0.7),
    frame_skip: int = Form(1),
    max_fps: int = Form(30),
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
    live_detection_manager: LiveDetectionManager = Depends(get_live_detection_manager)
):
    """Configure detection parameters for a camera."""
    try:
        # Check if camera exists
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        # Update camera configuration
        success = await live_detection_manager.update_camera_config(
            camera_id, 
            confidence_threshold=confidence_threshold,
            frame_skip=frame_skip,
            max_fps=max_fps
        )
        
        if success:
            return {
                "message": f"Camera {camera_id} configuration updated",
                "camera_id": camera_id,
                "config": {
                    "confidence_threshold": confidence_threshold,
                    "frame_skip": frame_skip,
                    "max_fps": max_fps
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update camera configuration")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error configuring camera: {str(e)}")

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

@router.get("/api/detections/history")
async def get_detection_history(
    camera_id: Optional[int] = None,
    hours: int = 24,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
    live_detection_manager: LiveDetectionManager = Depends(get_live_detection_manager)
):
    """Get detection history with statistics and analytics."""
    try:
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get detection history from live detection manager
        history_data = await live_detection_manager.get_detection_history(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Get database detections for the same period
        query = db.query(Detection).filter(
            Detection.timestamp >= start_time,
            Detection.timestamp <= end_time
        )
        
        if camera_id:
            query = query.filter(Detection.camera_id == camera_id)
        
        db_detections = query.all()
        
        # Calculate statistics
        total_detections = len(db_detections)
        crime_detections = len([d for d in db_detections if d.detection_type != 'normal'])
        
        # Group by crime type
        crime_types = {}
        for detection in db_detections:
            crime_type = detection.detection_type
            if crime_type not in crime_types:
                crime_types[crime_type] = 0
            crime_types[crime_type] += 1
        
        # Calculate hourly distribution
        hourly_distribution = {}
        for detection in db_detections:
            hour = detection.timestamp.hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        return {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            },
            "statistics": {
                "total_detections": total_detections,
                "crime_detections": crime_detections,
                "normal_detections": total_detections - crime_detections,
                "crime_rate": crime_detections / total_detections if total_detections > 0 else 0
            },
            "crime_types": crime_types,
            "hourly_distribution": hourly_distribution,
            "recent_detections": [DetectionResponse.from_orm(d) for d in db_detections[:10]],
            "live_data": history_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting detection history: {str(e)}")

@router.get("/api/detections/analytics")
async def get_detection_analytics(
    camera_id: Optional[int] = None,
    days: int = 7,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get advanced analytics for detections."""
    try:
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Get database detections
        query = db.query(Detection).filter(
            Detection.timestamp >= start_time,
            Detection.timestamp <= end_time
        )
        
        if camera_id:
            query = query.filter(Detection.camera_id == camera_id)
        
        detections = query.all()
        
        # Calculate analytics
        total_detections = len(detections)
        
        # Daily breakdown
        daily_stats = {}
        for detection in detections:
            date = detection.timestamp.date().isoformat()
            if date not in daily_stats:
                daily_stats[date] = {"total": 0, "crimes": 0}
            daily_stats[date]["total"] += 1
            if detection.detection_type != 'normal':
                daily_stats[date]["crimes"] += 1
        
        # Crime type breakdown
        crime_type_stats = {}
        for detection in detections:
            crime_type = detection.detection_type
            if crime_type not in crime_type_stats:
                crime_type_stats[crime_type] = {"count": 0, "avg_confidence": 0, "max_confidence": 0}
            crime_type_stats[crime_type]["count"] += 1
            crime_type_stats[crime_type]["avg_confidence"] += detection.confidence
            crime_type_stats[crime_type]["max_confidence"] = max(
                crime_type_stats[crime_type]["max_confidence"], 
                detection.confidence
            )
        
        # Calculate average confidence for each crime type
        for crime_type in crime_type_stats:
            count = crime_type_stats[crime_type]["count"]
            crime_type_stats[crime_type]["avg_confidence"] /= count
        
        # Peak hours analysis
        hourly_crimes = [0] * 24
        for detection in detections:
            if detection.detection_type != 'normal':
                hourly_crimes[detection.timestamp.hour] += 1
        
        peak_hours = []
        for hour, count in enumerate(hourly_crimes):
            if count > 0:
                peak_hours.append({"hour": hour, "crimes": count})
        
        peak_hours.sort(key=lambda x: x["crimes"], reverse=True)
        
        return {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": days
            },
            "summary": {
                "total_detections": total_detections,
                "total_crimes": len([d for d in detections if d.detection_type != 'normal']),
                "crime_rate": len([d for d in detections if d.detection_type != 'normal']) / total_detections if total_detections > 0 else 0
            },
            "daily_breakdown": daily_stats,
            "crime_types": crime_type_stats,
            "peak_hours": peak_hours[:5],  # Top 5 peak hours
            "confidence_analysis": {
                "avg_confidence": sum(d.confidence for d in detections) / total_detections if total_detections > 0 else 0,
                "max_confidence": max((d.confidence for d in detections), default=0),
                "min_confidence": min((d.confidence for d in detections), default=0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")

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

@router.get("/live/alerts")
async def get_live_alerts(
    camera_id: Optional[int] = None,
    limit: int = 50,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
    live_detection_manager: LiveDetectionManager = Depends(get_live_detection_manager)
):
    """Get recent alerts from live detections."""
    try:
        # Get alerts from database
        query = db.query(Alert).join(Detection).filter(
            Alert.created_at >= datetime.now() - timedelta(hours=1)
        )
        
        if camera_id:
            query = query.filter(Detection.camera_id == camera_id)
        
        alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()
        
        # Get live alerts from detection manager
        live_alerts = await live_detection_manager.get_live_alerts(camera_id=camera_id)
        
        return {
            "alerts": [
                {
                    "id": alert.id,
                    "detection_id": alert.detection_id,
                    "camera_id": alert.detection.camera_id,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.created_at.isoformat(),
                    "detection_type": alert.detection.detection_type,
                    "confidence": alert.detection.confidence
                }
                for alert in alerts
            ],
            "live_alerts": live_alerts,
            "total_count": len(alerts) + len(live_alerts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")

@router.get("/live/health")
async def get_system_health(
    current_user = Depends(get_current_user),
    live_detection_manager: LiveDetectionManager = Depends(get_live_detection_manager)
):
    """Get system health status including model and camera status."""
    try:
        # Get system metrics from live detection manager
        system_health = await live_detection_manager.get_system_health()
        
        # Get active cameras count
        active_cameras = await live_detection_manager.get_active_cameras()
        
        # Calculate system load
        total_cameras = len(active_cameras)
        avg_inference_time = system_health.get("avg_inference_time_ms", 0)
        avg_fps = system_health.get("avg_fps", 0)
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        if avg_inference_time > 200:  # > 200ms is concerning
            health_status = "degraded"
            issues.append(f"High inference time: {avg_inference_time:.1f}ms")
        
        if avg_fps < 10:  # < 10 FPS is poor performance
            health_status = "degraded"
            issues.append(f"Low FPS: {avg_fps:.1f}")
        
        if total_cameras > 10:  # High load
            health_status = "busy"
            issues.append(f"High camera load: {total_cameras} cameras")
        
        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "system_metrics": system_health,
            "active_cameras": total_cameras,
            "performance": {
                "avg_inference_time_ms": avg_inference_time,
                "avg_fps": avg_fps,
                "total_processed_frames": system_health.get("total_processed_frames", 0)
            },
            "issues": issues,
            "recommendations": generate_health_recommendations(health_status, issues)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system health: {str(e)}")

def generate_health_recommendations(health_status: str, issues: List[str]) -> List[str]:
    """Generate health recommendations based on current status."""
    recommendations = []
    
    if health_status == "degraded":
        if any("inference time" in issue for issue in issues):
            recommendations.append("Consider reducing frame skip or lowering confidence threshold")
            recommendations.append("Check GPU utilization and consider hardware upgrade")
        
        if any("FPS" in issue for issue in issues):
            recommendations.append("Optimize video processing pipeline")
            recommendations.append("Consider reducing camera resolution")
    
    elif health_status == "busy":
        recommendations.append("Consider load balancing across multiple instances")
        recommendations.append("Implement camera prioritization for critical feeds")
    
    elif health_status == "healthy":
        recommendations.append("System operating optimally")
        recommendations.append("Monitor performance trends for proactive maintenance")
    
    return recommendations
