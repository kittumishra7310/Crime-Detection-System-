from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List
import psutil
import os

from database import get_db, SystemLog, Detection, Alert, Camera, User
from auth import get_current_user, get_current_admin_user
from schemas import SystemStatus, SystemLog as SystemLogSchema
from ml_model import get_crime_model
from config import settings

router = APIRouter()

from live_detection import live_detection_manager

@router.get("/api/system/status", response_model=SystemStatus)
async def get_system_status(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get system status and health information."""
    
    uptime = psutil.boot_time()
    uptime_seconds = datetime.utcnow().timestamp() - uptime
    
    active_cameras = await live_detection_manager.get_active_cameras()
    
    yesterday = datetime.utcnow() - timedelta(days=1)
    total_detections = db.query(Detection).filter(Detection.timestamp >= yesterday).count()
    
    active_alerts = db.query(Alert).filter(Alert.status == "active").count()
    
    try:
        db.execute("SELECT 1")
        database_status = "healthy"
    except Exception:
        database_status = "error"
    
    model = get_crime_model()
    model_status = "loaded" if model.model is not None else "not_loaded"
    
    return SystemStatus(
        status="healthy" if database_status == "healthy" else "degraded",
        uptime=uptime_seconds,
        active_cameras=len(active_cameras),
        total_detections=total_detections,
        active_alerts=active_alerts,
        database_status=database_status,
        model_status=model_status
    )

@router.get("/api/system/logs", response_model=List[SystemLogSchema])
async def get_system_logs(
    level: str = None,
    component: str = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_admin_user)
):
    """Get system logs (admin only)."""
    
    query = db.query(SystemLog)
    
    if level:
        query = query.filter(SystemLog.level == level.upper())
    
    if component:
        query = query.filter(SystemLog.component == component)
    
    logs = query.order_by(SystemLog.timestamp.desc()).limit(limit).all()
    
    return [SystemLogSchema.from_orm(log) for log in logs]

@router.post("/api/system/logs")
async def create_system_log(
    level: str,
    message: str,
    component: str = "api",
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a system log entry."""
    
    log_entry = SystemLog(
        level=level.upper(),
        message=message,
        component=component,
        user_id=current_user.id
    )
    
    db.add(log_entry)
    db.commit()
    
    return {"message": "Log entry created successfully"}

@router.get("/api/system/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }

@router.post("/api/system/reload-model")
async def reload_model(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_admin_user)
):
    """Reload the ML model (admin only)."""
    
    try:
        model = get_crime_model()
        success = model.load_model()
        
        # Log the action
        log_entry = SystemLog(
            level="INFO",
            message=f"Model reload {'successful' if success else 'failed'}",
            component="model",
            user_id=current_user.id
        )
        db.add(log_entry)
        db.commit()
        
        if success:
            return {"message": "Model reloaded successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reload model"
            )
    
    except Exception as e:
        # Log the error
        log_entry = SystemLog(
            level="ERROR",
            message=f"Model reload error: {str(e)}",
            component="model",
            user_id=current_user.id
        )
        db.add(log_entry)
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reloading model: {str(e)}"
        )
