from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta
from typing import List, Optional

from database import get_db, Detection, Alert, Camera, User
from auth import get_current_user
from schemas import AnalyticsResponse, DetectionStats, AlertStats

router = APIRouter()

@router.get("/api/analytics/detections")
async def get_detection_analytics(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    camera_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get detection analytics for specified period."""
    
    # Default to last 30 days if no dates provided
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    # Base query
    query = db.query(Detection).filter(
        and_(
            Detection.timestamp >= start_date,
            Detection.timestamp <= end_date
        )
    )
    
    if camera_id:
        query = query.filter(Detection.camera_id == camera_id)
    
    # Total detections
    total_detections = query.count()
    
    # Crime vs normal detections
    crime_detections = query.filter(Detection.detection_type != 'NormalVideos').count()
    normal_detections = total_detections - crime_detections
    
    # Detections by type
    detection_types = db.query(
        Detection.detection_type,
        func.count(Detection.id).label('count')
    ).filter(
        and_(
            Detection.timestamp >= start_date,
            Detection.timestamp <= end_date
        )
    )
    
    if camera_id:
        detection_types = detection_types.filter(Detection.camera_id == camera_id)
    
    detection_types = detection_types.group_by(Detection.detection_type).all()
    by_type = {dt.detection_type: dt.count for dt in detection_types}
    
    # Detections by severity
    severity_stats = db.query(
        Detection.severity,
        func.count(Detection.id).label('count')
    ).filter(
        and_(
            Detection.timestamp >= start_date,
            Detection.timestamp <= end_date
        )
    )
    
    if camera_id:
        severity_stats = severity_stats.filter(Detection.camera_id == camera_id)
    
    severity_stats = severity_stats.group_by(Detection.severity).all()
    by_severity = {ss.severity: ss.count for ss in severity_stats}
    
    # Detections by camera
    camera_stats = db.query(
        Camera.name,
        func.count(Detection.id).label('count')
    ).join(Detection).filter(
        and_(
            Detection.timestamp >= start_date,
            Detection.timestamp <= end_date
        )
    ).group_by(Camera.name).all()
    
    by_camera = {cs.name: cs.count for cs in camera_stats}
    
    detection_stats = DetectionStats(
        total_detections=total_detections,
        crime_detections=crime_detections,
        normal_detections=normal_detections,
        by_type=by_type,
        by_severity=by_severity,
        by_camera=by_camera
    )
    
    # Alert analytics
    alert_query = db.query(Alert).join(Detection).filter(
        and_(
            Detection.timestamp >= start_date,
            Detection.timestamp <= end_date
        )
    )
    
    if camera_id:
        alert_query = alert_query.filter(Detection.camera_id == camera_id)
    
    total_alerts = alert_query.count()
    active_alerts = alert_query.filter(Alert.status == 'active').count()
    acknowledged_alerts = alert_query.filter(Alert.status == 'acknowledged').count()
    resolved_alerts = alert_query.filter(Alert.status == 'resolved').count()
    
    # Alerts by severity
    alert_severity_stats = db.query(
        Alert.severity,
        func.count(Alert.id).label('count')
    ).join(Detection).filter(
        and_(
            Detection.timestamp >= start_date,
            Detection.timestamp <= end_date
        )
    )
    
    if camera_id:
        alert_severity_stats = alert_severity_stats.filter(Detection.camera_id == camera_id)
    
    alert_severity_stats = alert_severity_stats.group_by(Alert.severity).all()
    alert_by_severity = {ass.severity: ass.count for ass in alert_severity_stats}
    
    alert_stats = AlertStats(
        total_alerts=total_alerts,
        active_alerts=active_alerts,
        acknowledged_alerts=acknowledged_alerts,
        resolved_alerts=resolved_alerts,
        by_severity=alert_by_severity
    )
    
    return AnalyticsResponse(
        detection_stats=detection_stats,
        alert_stats=alert_stats,
        period=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        start_date=start_date,
        end_date=end_date
    )

@router.get("/api/analytics/timeline")
async def get_detection_timeline(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    interval: str = Query("hour", regex="^(hour|day|week)$"),
    camera_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get detection timeline data for charts."""
    
    # Default to last 7 days if no dates provided
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=7)
    
    # Determine date truncation based on interval
    if interval == "hour":
        date_trunc = func.date_trunc('hour', Detection.timestamp)
    elif interval == "day":
        date_trunc = func.date_trunc('day', Detection.timestamp)
    else:  # week
        date_trunc = func.date_trunc('week', Detection.timestamp)
    
    # Query for timeline data
    query = db.query(
        date_trunc.label('period'),
        func.count(Detection.id).label('total_detections'),
        func.sum(func.case([(Detection.detection_type != 'NormalVideos', 1)], else_=0)).label('crime_detections')
    ).filter(
        and_(
            Detection.timestamp >= start_date,
            Detection.timestamp <= end_date
        )
    )
    
    if camera_id:
        query = query.filter(Detection.camera_id == camera_id)
    
    timeline_data = query.group_by(date_trunc).order_by(date_trunc).all()
    
    return {
        "timeline": [
            {
                "period": td.period.isoformat(),
                "total_detections": td.total_detections,
                "crime_detections": td.crime_detections or 0,
                "normal_detections": td.total_detections - (td.crime_detections or 0)
            }
            for td in timeline_data
        ],
        "interval": interval,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }

@router.get("/api/analytics/heatmap")
async def get_detection_heatmap(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get detection heatmap data by hour and day of week."""
    
    # Default to last 30 days if no dates provided
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    # Query for heatmap data
    heatmap_data = db.query(
        func.extract('dow', Detection.timestamp).label('day_of_week'),  # 0=Sunday, 6=Saturday
        func.extract('hour', Detection.timestamp).label('hour'),
        func.count(Detection.id).label('count')
    ).filter(
        and_(
            Detection.timestamp >= start_date,
            Detection.timestamp <= end_date,
            Detection.detection_type != 'NormalVideos'  # Only crime detections
        )
    ).group_by(
        func.extract('dow', Detection.timestamp),
        func.extract('hour', Detection.timestamp)
    ).all()
    
    # Convert to matrix format
    heatmap_matrix = {}
    for hd in heatmap_data:
        day = int(hd.day_of_week)
        hour = int(hd.hour)
        count = hd.count
        
        if day not in heatmap_matrix:
            heatmap_matrix[day] = {}
        heatmap_matrix[day][hour] = count
    
    # Fill missing values with 0
    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    result = []
    
    for day_num in range(7):
        day_data = {
            'day': days[day_num],
            'day_number': day_num,
            'hours': []
        }
        
        for hour in range(24):
            count = heatmap_matrix.get(day_num, {}).get(hour, 0)
            day_data['hours'].append({
                'hour': hour,
                'count': count
            })
        
        result.append(day_data)
    
    return {
        "heatmap": result,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }
