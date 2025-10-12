from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[int, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: int = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            self.user_connections[user_id] = websocket
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, user_id: int = None):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id and user_id in self.user_connections:
            del self.user_connections[user_id]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, user_id: int):
        """Send a message to a specific user."""
        if user_id in self.user_connections:
            websocket = self.user_connections[user_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {e}")
                self.disconnect(websocket, user_id)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
    
    async def send_alert(self, alert_data: dict):
        """Send alert notification to all connected clients."""
        message = {
            "type": "alert",
            "data": alert_data,
            "timestamp": alert_data.get("timestamp")
        }
        await self.broadcast(message)
    
    async def send_detection(self, detection_data: dict):
        """Send detection notification to all connected clients."""
        message = {
            "type": "detection",
            "data": detection_data,
            "timestamp": detection_data.get("timestamp")
        }
        await self.broadcast(message)
    
    async def send_system_update(self, update_data: dict):
        """Send system update to all connected clients."""
        message = {
            "type": "system_update",
            "data": update_data,
            "timestamp": update_data.get("timestamp")
        }
        await self.broadcast(message)

# Global WebSocket manager instance
websocket_manager = WebSocketManager()
