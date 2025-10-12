from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional

from database import get_db, Alert, Detection, User
from auth import get_current_user, get_current_admin_user
from schemas import AlertResponse, AlertUpdate, AlertStatus

router = APIRouter()

@router.get("/api/alerts", response_model=List[AlertResponse])
async def list_alerts(
    status_filter: Optional[AlertStatus] = None,
    severity: Optional[str] = None,
    camera_id: Optional[int] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List alerts with optional filters."""
    
    query = db.query(Alert).join(Detection)
    
    if status_filter:
        query = query.filter(Alert.status == status_filter)
    
    if severity:
        query = query.filter(Alert.severity == severity)
    
    if camera_id:
        query = query.filter(Detection.camera_id == camera_id)
    
    alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()
    
    return [AlertResponse.from_orm(alert) for alert in alerts]

@router.get("/api/alerts/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get specific alert by ID."""
    
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    return AlertResponse.from_orm(alert)

@router.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Acknowledge an alert."""
    
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    if alert.status == AlertStatus.acknowledged:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Alert already acknowledged"
        )
    
    alert.status = AlertStatus.acknowledged
    alert.acknowledged_by = current_user.id
    alert.acknowledged_at = datetime.utcnow()
    
    db.commit()
    db.refresh(alert)
    
    return {"message": "Alert acknowledged successfully"}

@router.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Resolve an alert."""
    
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    if alert.status == AlertStatus.resolved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Alert already resolved"
        )
    
    alert.status = AlertStatus.resolved
    alert.resolved_at = datetime.utcnow()
    
    if alert.status != AlertStatus.acknowledged:
        alert.acknowledged_by = current_user.id
        alert.acknowledged_at = datetime.utcnow()
    
    db.commit()
    db.refresh(alert)
    
    return {"message": "Alert resolved successfully"}

@router.put("/api/alerts/{alert_id}", response_model=AlertResponse)
async def update_alert(
    alert_id: int,
    alert_data: AlertUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update alert information."""
    
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    update_data = alert_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(alert, field, value)
    
    # Update acknowledgment info if status changed to acknowledged
    if alert_data.status == AlertStatus.acknowledged and alert.acknowledged_by is None:
        alert.acknowledged_by = current_user.id
        alert.acknowledged_at = datetime.utcnow()
    
    # Update resolution info if status changed to resolved
    if alert_data.status == AlertStatus.resolved and alert.resolved_at is None:
        alert.resolved_at = datetime.utcnow()
        if alert.acknowledged_by is None:
            alert.acknowledged_by = current_user.id
            alert.acknowledged_at = datetime.utcnow()
    
    db.commit()
    db.refresh(alert)
    
    return AlertResponse.from_orm(alert)

@router.get("/api/alerts/stats")
async def get_alert_stats(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get alert statistics."""
    
    total_alerts = db.query(Alert).count()
    active_alerts = db.query(Alert).filter(Alert.status == AlertStatus.active).count()
    acknowledged_alerts = db.query(Alert).filter(Alert.status == AlertStatus.acknowledged).count()
    resolved_alerts = db.query(Alert).filter(Alert.status == AlertStatus.resolved).count()
    
    # Get alerts by severity
    critical_alerts = db.query(Alert).filter(Alert.severity == "critical").count()
    high_alerts = db.query(Alert).filter(Alert.severity == "high").count()
    medium_alerts = db.query(Alert).filter(Alert.severity == "medium").count()
    low_alerts = db.query(Alert).filter(Alert.severity == "low").count()
    
    return {
        "total_alerts": total_alerts,
        "active_alerts": active_alerts,
        "acknowledged_alerts": acknowledged_alerts,
        "resolved_alerts": resolved_alerts,
        "by_severity": {
            "critical": critical_alerts,
            "high": high_alerts,
            "medium": medium_alerts,
            "low": low_alerts
        }
    }
