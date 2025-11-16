from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.orm import Session
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any

from database import get_db
from websocket_manager import websocket_manager
from auth import verify_token
from live_detection import live_detection_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class WebSocketMessageHandler:
    """Handle different types of WebSocket messages."""
    
    def __init__(self):
        self.message_handlers = {
            "ping": self.handle_ping,
            "start_detection": self.handle_start_detection,
            "stop_detection": self.handle_stop_detection,
            "get_status": self.handle_get_status,
            "get_detection_history": self.handle_get_detection_history,
            "get_performance_metrics": self.handle_get_performance_metrics,
            "set_confidence_threshold": self.handle_set_confidence_threshold,
            "get_camera_feed": self.handle_get_camera_feed,
            "force_stop_all": self.handle_force_stop_all
        }
    
    async def handle_message(self, message: Dict[str, Any], websocket: WebSocket, user_id: int, db: Session):
        """Route message to appropriate handler."""
        message_type = message.get("type")
        
        if message_type in self.message_handlers:
            try:
                return await self.message_handlers[message_type](message, websocket, user_id, db)
            except Exception as e:
                logger.error(f"Error handling message type {message_type}: {e}")
                return {
                    "type": "error",
                    "error": f"Error processing {message_type}: {str(e)}"
                }
        else:
            return {
                "type": "error",
                "error": f"Unknown message type: {message_type}"
            }
    
    async def handle_ping(self, message: Dict, websocket: WebSocket, user_id: int, db: Session):
        """Handle ping message."""
        return {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
    
    async def handle_start_detection(self, message: Dict, websocket: WebSocket, user_id: int, db: Session):
        """Handle start detection request."""
        camera_id = message.get("camera_id")
        source = message.get("source", "0")
        confidence_threshold = message.get("confidence_threshold")
        
        if not camera_id:
            return {"type": "error", "error": "camera_id is required"}
        
        try:
            success = await live_detection_manager.start_camera_detection(
                camera_id=camera_id,
                source=source,
                confidence_threshold=confidence_threshold
            )
            
            if success:
                return {
                    "type": "detection_started",
                    "camera_id": camera_id,
                    "source": source,
                    "message": f"Detection started for camera {camera_id}"
                }
            else:
                return {
                    "type": "error",
                    "error": f"Failed to start detection for camera {camera_id}"
                }
        except Exception as e:
            logger.error(f"Error starting detection: {e}")
            return {"type": "error", "error": f"Failed to start detection: {str(e)}"}
    
    async def handle_stop_detection(self, message: Dict, websocket: WebSocket, user_id: int, db: Session):
        """Handle stop detection request."""
        camera_id = message.get("camera_id")
        
        if not camera_id:
            return {"type": "error", "error": "camera_id is required"}
        
        try:
            success = await live_detection_manager.stop_camera_detection(camera_id)
            
            if success:
                return {
                    "type": "detection_stopped",
                    "camera_id": camera_id,
                    "message": f"Detection stopped for camera {camera_id}"
                }
            else:
                return {
                    "type": "error",
                    "error": f"Failed to stop detection for camera {camera_id}"
                }
        except Exception as e:
            logger.error(f"Error stopping detection: {e}")
            return {"type": "error", "error": f"Failed to stop detection: {str(e)}"}
    
    async def handle_get_status(self, message: Dict, websocket: WebSocket, user_id: int, db: Session):
        """Handle get status request."""
        camera_id = message.get("camera_id")
        
        try:
            if camera_id:
                # Get specific camera status
                status = await live_detection_manager.get_camera_status(camera_id)
                return {
                    "type": "camera_status",
                    "camera_id": camera_id,
                    "status": status
                }
            else:
                # Get all active cameras
                active_cameras = await live_detection_manager.get_active_cameras()
                return {
                    "type": "system_status",
                    "active_cameras": active_cameras,
                    "total_active": len(active_cameras)
                }
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"type": "error", "error": f"Failed to get status: {str(e)}"}
    
    async def handle_get_detection_history(self, message: Dict, websocket: WebSocket, user_id: int, db: Session):
        """Handle get detection history request."""
        camera_id = message.get("camera_id")
        limit = message.get("limit", 100)
        
        if not camera_id:
            return {"type": "error", "error": "camera_id is required"}
        
        try:
            history = await live_detection_manager.get_detection_history(camera_id, limit)
            return {
                "type": "detection_history",
                "camera_id": camera_id,
                "history": history,
                "count": len(history)
            }
        except Exception as e:
            logger.error(f"Error getting detection history: {e}")
            return {"type": "error", "error": f"Failed to get detection history: {str(e)}"}
    
    async def handle_get_performance_metrics(self, message: Dict, websocket: WebSocket, user_id: int, db: Session):
        """Handle get performance metrics request."""
        camera_id = message.get("camera_id")
        
        if not camera_id:
            return {"type": "error", "error": "camera_id is required"}
        
        try:
            metrics = await live_detection_manager.get_performance_metrics(camera_id)
            return {
                "type": "performance_metrics",
                "camera_id": camera_id,
                "metrics": metrics
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"type": "error", "error": f"Failed to get performance metrics: {str(e)}"}
    
    async def handle_set_confidence_threshold(self, message: Dict, websocket: WebSocket, user_id: int, db: Session):
        """Handle set confidence threshold request."""
        threshold = message.get("threshold")
        
        if threshold is None or not isinstance(threshold, (int, float)):
            return {"type": "error", "error": "threshold must be a number between 0.0 and 1.0"}
        
        if not 0.0 <= threshold <= 1.0:
            return {"type": "error", "error": "threshold must be between 0.0 and 1.0"}
        
        try:
            # This would need to be implemented in the model
            if hasattr(live_detection_manager.model, 'set_confidence_threshold'):
                live_detection_manager.model.set_confidence_threshold(threshold)
                return {
                    "type": "confidence_threshold_updated",
                    "threshold": threshold,
                    "message": f"Confidence threshold updated to {threshold}"
                }
            else:
                return {
                    "type": "error",
                    "error": "Model does not support threshold adjustment"
                }
        except Exception as e:
            logger.error(f"Error setting confidence threshold: {e}")
            return {"type": "error", "error": f"Failed to set confidence threshold: {str(e)}"}
    
    async def handle_get_camera_feed(self, message: Dict, websocket: WebSocket, user_id: int, db: Session):
        """Handle get camera feed request."""
        camera_id = message.get("camera_id")
        
        if not camera_id:
            return {"type": "error", "error": "camera_id is required"}
        
        try:
            # Get camera feed URL (this would be implemented in your API)
            feed_url = f"/api/live/feed/{camera_id}"
            return {
                "type": "camera_feed_url",
                "camera_id": camera_id,
                "feed_url": feed_url
            }
        except Exception as e:
            logger.error(f"Error getting camera feed: {e}")
            return {"type": "error", "error": f"Failed to get camera feed: {str(e)}"}
    
    async def handle_force_stop_all(self, message: Dict, websocket: WebSocket, user_id: int, db: Session):
        """Handle force stop all detections request."""
        try:
            await live_detection_manager.force_stop_all()
            return {
                "type": "all_detections_stopped",
                "message": "All active detections have been stopped"
            }
        except Exception as e:
            logger.error(f"Error force stopping all detections: {e}")
            return {"type": "error", "error": f"Failed to stop all detections: {str(e)}"}

# Create message handler instance
message_handler = WebSocketMessageHandler()

@router.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str, db: Session = Depends(get_db)):
    """Enhanced WebSocket endpoint for real-time crime detection updates."""
    
    try:
        # Verify token
        payload = verify_token(token)
        username = payload.get("sub")
        
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get user ID (simplified - in production, you'd query the database)
        user_id = hash(username) % 10000  # Simple user ID generation
        
        # Connect to WebSocket manager
        await websocket_manager.connect(websocket, user_id, metadata={"username": username})
        
        logger.info(f"WebSocket connection established for user {username}")
        
        try:
            while True:
                # Receive and process messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                logger.debug(f"Received message from user {username}: {message.get('type')}")
                
                # Handle the message
                response = await message_handler.handle_message(message, websocket, user_id, db)
                
                # Send response back to client
                if response:
                    await websocket_manager.send_personal_message(response, websocket)
                
        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket, user_id)
            logger.info(f"User {username} disconnected normally")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from user {username}: {e}")
            await websocket_manager.send_personal_message({
                "type": "error",
                "error": "Invalid message format"
            }, websocket)
            
        except Exception as e:
            logger.error(f"WebSocket error for user {username}: {e}")
            await websocket_manager.send_personal_message({
                "type": "error",
                "error": "Internal server error"
            }, websocket)
            
    except HTTPException as e:
        logger.error(f"Authentication failed: {e.detail}")
        await websocket.close(code=1008)  # Policy violation
        
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        try:
            await websocket.close(code=1011)  # Internal server error
        except:
            pass
