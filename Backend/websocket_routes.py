from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
import json
import logging

from database import get_db
from websocket_manager import websocket_manager
from auth import verify_token

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str, db: Session = Depends(get_db)):
    """WebSocket endpoint for real-time updates."""
    
    try:
        # Verify token
        payload = verify_token(token)
        username = payload.get("sub")
        
        # Get user ID (simplified - in production, you'd query the database)
        user_id = hash(username) % 10000  # Simple user ID generation
        
        await websocket_manager.connect(websocket, user_id)
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": "Connected to surveillance system",
            "user": username
        }))
        
        try:
            while True:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                
        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket, user_id)
            logger.info(f"User {username} disconnected")
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1008)  # Policy violation
