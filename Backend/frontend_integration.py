"""
Frontend Integration Helper
Provides utilities to integrate the FastAPI backend with the Next.js frontend.
"""

from fastapi import HTTPException, status
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

class FrontendIntegration:
    """Helper class for frontend-backend integration."""
    
    @staticmethod
    def format_api_response(data: Any, message: str = None, success: bool = True) -> Dict:
        """Format API response for consistent frontend consumption."""
        return {
            "success": success,
            "message": message,
            "data": data,
            "timestamp": "2025-09-15T23:07:16+05:30"
        }
    
    @staticmethod
    def format_error_response(error: str, code: int = 400) -> Dict:
        """Format error response for frontend."""
        return {
            "success": False,
            "error": error,
            "code": code,
            "timestamp": "2025-09-15T23:07:16+05:30"
        }
    
    @staticmethod
    def validate_frontend_request(data: Dict, required_fields: list) -> bool:
        """Validate request data from frontend."""
        for field in required_fields:
            if field not in data or data[field] is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )
        return True
    
    @staticmethod
    def convert_detection_for_frontend(detection) -> Dict:
        """Convert detection object for frontend consumption."""
        return {
            "id": detection.id,
            "camera_id": detection.camera_id,
            "camera_name": detection.camera.name if detection.camera else "Unknown",
            "detection_type": detection.detection_type,
            "confidence": round(detection.confidence * 100, 2),  # Convert to percentage
            "severity": detection.severity,
            "timestamp": detection.timestamp.isoformat(),
            "image_url": f"/uploads/{detection.image_path.split('/')[-1]}" if detection.image_path else None,
            "is_crime": detection.detection_type != "NormalVideos",
            "bounding_box": json.loads(detection.bounding_box) if detection.bounding_box else None
        }
    
    @staticmethod
    def convert_alert_for_frontend(alert) -> Dict:
        """Convert alert object for frontend consumption."""
        return {
            "id": alert.id,
            "detection_id": alert.detection_id,
            "status": alert.status,
            "severity": alert.severity,
            "message": alert.message,
            "created_at": alert.created_at.isoformat(),
            "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
            "detection": FrontendIntegration.convert_detection_for_frontend(alert.detection) if alert.detection else None
        }
    
    @staticmethod
    def convert_camera_for_frontend(camera) -> Dict:
        """Convert camera object for frontend consumption."""
        return {
            "id": camera.id,
            "name": camera.name,
            "camera_id": camera.camera_id,
            "location": camera.location,
            "status": camera.status,
            "rtsp_url": camera.rtsp_url,
            "created_at": camera.created_at.isoformat(),
            "stream_url": f"/api/detection/stream/{camera.camera_id}",
            "is_online": camera.status == "active"
        }

# API Response formatters for consistent frontend integration
def success_response(data: Any = None, message: str = "Success") -> Dict:
    """Create a successful API response."""
    return FrontendIntegration.format_api_response(data, message, True)

def error_response(error: str, code: int = 400) -> Dict:
    """Create an error API response."""
    return FrontendIntegration.format_error_response(error, code)

def paginated_response(items: list, total: int, page: int, size: int) -> Dict:
    """Create a paginated API response."""
    return {
        "success": True,
        "data": {
            "items": items,
            "pagination": {
                "total": total,
                "page": page,
                "size": size,
                "pages": (total + size - 1) // size
            }
        }
    }
