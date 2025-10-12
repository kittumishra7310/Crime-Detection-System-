from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# Enums
class UserRole(str, Enum):
    admin = "admin"
    viewer = "viewer"

class CameraStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    maintenance = "maintenance"

class AlertStatus(str, Enum):
    active = "active"
    acknowledged = "acknowledged"
    resolved = "resolved"

class SeverityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

# User Schemas
class UserBase(BaseModel):
    username: str
    email: str
    role: UserRole = UserRole.viewer

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# Authentication Schemas
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# Camera Schemas
class CameraBase(BaseModel):
    name: str
    camera_id: str
    rtsp_url: Optional[str] = None
    location: Optional[str] = None

class CameraCreate(CameraBase):
    pass

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    location: Optional[str] = None
    status: Optional[CameraStatus] = None

class CameraResponse(CameraBase):
    id: int
    status: CameraStatus
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# Detection Schemas
class DetectionBase(BaseModel):
    detection_type: str
    confidence: float
    severity: SeverityLevel
    bounding_box: Optional[str] = None

class DetectionCreate(DetectionBase):
    camera_id: int
    image_path: Optional[str] = None
    video_path: Optional[str] = None

class DetectionResponse(DetectionBase):
    id: int
    camera_id: int
    image_path: Optional[str]
    video_path: Optional[str]
    timestamp: datetime
    processed: bool
    camera: CameraResponse

    class Config:
        from_attributes = True

# Alert Schemas
class AlertBase(BaseModel):
    message: Optional[str] = None

class AlertCreate(AlertBase):
    detection_id: int
    severity: SeverityLevel

class AlertUpdate(BaseModel):
    status: Optional[AlertStatus] = None
    message: Optional[str] = None

class AlertResponse(AlertBase):
    id: int
    detection_id: int
    status: AlertStatus
    severity: SeverityLevel
    acknowledged_by: Optional[int]
    acknowledged_at: Optional[datetime]
    created_at: datetime
    resolved_at: Optional[datetime]
    detection: DetectionResponse

    class Config:
        from_attributes = True

# Detection Result Schemas
class DetectionResult(BaseModel):
    crime_type: str
    confidence: float
    severity: str
    is_crime: bool
    all_predictions: Dict[str, float]
    bounding_box: Optional[Dict[str, int]] = None

class LiveDetectionResponse(BaseModel):
    camera_id: str
    detection: DetectionResult
    timestamp: datetime
    image_url: Optional[str] = None

class FileDetectionResponse(BaseModel):
    filename: str
    detections: List[DetectionResult]
    total_frames: Optional[int] = None
    processing_time: float

# System Schemas
class SystemStatus(BaseModel):
    status: str
    uptime: float
    active_cameras: int
    total_detections: int
    active_alerts: int
    database_status: str
    model_status: str

class SystemLog(BaseModel):
    level: str
    message: str
    component: str
    timestamp: datetime

# Analytics Schemas
class DetectionStats(BaseModel):
    total_detections: int
    crime_detections: int
    normal_detections: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    by_camera: Dict[str, int]

class AlertStats(BaseModel):
    total_alerts: int
    active_alerts: int
    acknowledged_alerts: int
    resolved_alerts: int
    by_severity: Dict[str, int]

class AnalyticsResponse(BaseModel):
    detection_stats: DetectionStats
    alert_stats: AlertStats
    period: str
    start_date: datetime
    end_date: datetime

# File Upload Schemas
class FileUploadResponse(BaseModel):
    filename: str
    file_path: str
    file_size: int
    upload_time: datetime

# Pagination Schemas
class PaginationParams(BaseModel):
    page: int = 1
    size: int = 10
    
class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
