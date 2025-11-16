from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Optional, Set
import json
import asyncio
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Enhanced WebSocket manager for real-time crime detection system."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[int, WebSocket] = {}
        self.connection_metadata: Dict[WebSocket, Dict] = {}  # Store connection metadata
        self.message_queue: Optional[asyncio.Queue] = None
        self.broadcast_task: Optional[asyncio.Task] = None
        self.connection_semaphores: Dict[WebSocket, asyncio.Semaphore] = {}
        self.rate_limiter = RateLimiter(max_messages_per_second=10)
        
        # Initialize queue - will be created when first connection is made
        self._initialized = False
    
    async def initialize(self):
        """Initialize the WebSocket manager with async components."""
        if not self._initialized:
            self.message_queue = asyncio.Queue(maxsize=1000)
            self.start_broadcast_task()
            self._initialized = True
            logger.info("WebSocket manager initialized")
    
    def start_broadcast_task(self):
        """Start the background broadcast task."""
        try:
            if self.broadcast_task is None or self.broadcast_task.done():
                self.broadcast_task = asyncio.create_task(self._broadcast_worker())
                logger.info("WebSocket broadcast worker started")
        except RuntimeError as e:
            logger.warning(f"Could not start broadcast task immediately: {e}")
            # Task will be started when first connection is made
    
    async def stop_broadcast_task(self):
        """Stop the background broadcast task."""
        if self.broadcast_task and not self.broadcast_task.done():
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
            logger.info("WebSocket broadcast worker stopped")
    
    async def connect(self, websocket: WebSocket, user_id: int = None, metadata: Dict = None):
        """Accept a new WebSocket connection with enhanced metadata."""
        # Initialize the manager if not already done
        if not self._initialized:
            await self.initialize()
        
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Store connection metadata
        connection_info = {
            "user_id": user_id,
            "connected_at": datetime.utcnow().isoformat(),
            "last_ping": time.time(),
            "message_count": 0,
            "metadata": metadata or {}
        }
        self.connection_metadata[websocket] = connection_info
        
        if user_id:
            self.user_connections[user_id] = websocket
        
        # Create semaphore for this connection to prevent concurrent sends
        self.connection_semaphores[websocket] = asyncio.Semaphore(1)
        
        logger.info(f"WebSocket connected. User: {user_id}, Total connections: {len(self.active_connections)}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection",
            "message": "Connected to crime detection system",
            "user_id": user_id,
            "server_time": datetime.utcnow().isoformat(),
            "connection_id": id(websocket)
        }, websocket)
    
    def disconnect(self, websocket: WebSocket, user_id: int = None):
        """Remove a WebSocket connection with cleanup."""
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            
            if user_id and user_id in self.user_connections:
                del self.user_connections[user_id]
            
            # Clean up metadata and semaphores
            self.connection_metadata.pop(websocket, None)
            self.connection_semaphores.pop(websocket, None)
            
            logger.info(f"WebSocket disconnected. User: {user_id}, Total connections: {len(self.active_connections)}")
            
        except Exception as e:
            logger.error(f"Error during WebSocket disconnect: {e}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        if websocket not in self.active_connections:
            return
        
        try:
            # Use semaphore to prevent concurrent sends to the same connection
            semaphore = self.connection_semaphores.get(websocket)
            if semaphore:
                async with semaphore:
                    await websocket.send_text(json.dumps(message))
                    
                    # Update connection metadata
                    if websocket in self.connection_metadata:
                        self.connection_metadata[websocket]['message_count'] += 1
                        self.connection_metadata[websocket]['last_ping'] = time.time()
            else:
                # Fallback if semaphore not available
                await websocket.send_text(json.dumps(message))
                
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def send_personal_message_by_user_id(self, message: dict, user_id: int):
        """Send a message to a specific user by user ID."""
        if user_id in self.user_connections:
            websocket = self.user_connections[user_id]
            await self.send_personal_message(message, websocket)
    
    async def broadcast(self, message: dict, exclude_user_ids: Set[int] = None):
        """Broadcast a message to all connected clients with rate limiting."""
        if not self.active_connections:
            return
        
        # Initialize if needed
        if not self._initialized:
            await self.initialize()
        
        # Apply rate limiting
        if not self.rate_limiter.allow():
            logger.warning("Broadcast rate limit exceeded, message queued")
            await self.message_queue.put(message)
            return
        
        # Add timestamp to message
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
        
        # Queue message for background processing
        await self.message_queue.put(message)
    
    async def _broadcast_worker(self):
        """Background worker to handle message broadcasting."""
        while True:
            try:
                if self.message_queue is None:
                    await asyncio.sleep(0.1)
                    continue
                    
                message = await self.message_queue.get()
                await self._process_broadcast(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast worker: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_broadcast(self, message: dict):
        """Process and send broadcast message to all connections."""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message)
        disconnected = []
        
        # Send to all connections concurrently
        tasks = []
        for connection in self.active_connections:
            task = asyncio.create_task(self._send_to_connection(connection, message_str))
            tasks.append((connection, task))
        
        # Wait for all sends to complete
        for connection, task in tasks:
            try:
                await task
            except Exception as e:
                logger.error(f"Error sending to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def _send_to_connection(self, websocket: WebSocket, message_str: str):
        """Send message to a single connection with error handling."""
        try:
            semaphore = self.connection_semaphores.get(websocket)
            if semaphore:
                async with semaphore:
                    await websocket.send_text(message_str)
            else:
                await websocket.send_text(message_str)
        except Exception as e:
            logger.error(f"Error sending to connection: {e}")
            raise
    
    async def send_alert(self, alert_data: dict, severity: str = "low"):
        """Send alert notification to all connected clients."""
        message = {
            "type": "alert",
            "severity": severity,
            "data": alert_data,
            "timestamp": alert_data.get("timestamp", datetime.utcnow().isoformat())
        }
        await self.broadcast(message)
    
    async def send_detection(self, detection_data: dict):
        """Send detection notification to all connected clients."""
        message = {
            "type": "detection",
            "data": detection_data,
            "timestamp": detection_data.get("timestamp", datetime.utcnow().isoformat())
        }
        await self.broadcast(message)
    
    async def send_system_update(self, update_data: dict):
        """Send system update to all connected clients."""
        message = {
            "type": "system_update",
            "data": update_data,
            "timestamp": update_data.get("timestamp", datetime.utcnow().isoformat())
        }
        await self.broadcast(message)
    
    async def send_detection_result(self, detection_result: dict):
        """Send real-time detection result to all connected clients."""
        message = {
            "type": "detection_result",
            "data": detection_result,
            "timestamp": detection_result.get("timestamp", datetime.utcnow().isoformat())
        }
        await self.broadcast(message)
    
    async def send_buffering_status(self, camera_id: int, frames_collected: int, frames_needed: int):
        """Send buffering status update."""
        message = {
            "type": "buffering_status",
            "camera_id": camera_id,
            "frames_collected": frames_collected,
            "frames_needed": frames_needed,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(message)
    
    async def send_performance_metrics(self, metrics: dict):
        """Send performance metrics update."""
        message = {
            "type": "performance_metrics",
            "data": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(message)
    
    def get_connection_stats(self) -> dict:
        """Get WebSocket connection statistics."""
        total_connections = len(self.active_connections)
        active_users = len(self.user_connections)
        
        connection_info = []
        for websocket, metadata in self.connection_metadata.items():
            connection_info.append({
                "connection_id": id(websocket),
                "user_id": metadata.get("user_id"),
                "connected_at": metadata.get("connected_at"),
                "message_count": metadata.get("message_count", 0),
                "last_ping": metadata.get("last_ping", 0)
            })
        
        return {
            "total_connections": total_connections,
            "active_users": active_users,
            "message_queue_size": self.message_queue.qsize(),
            "connections": connection_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cleanup(self):
        """Clean up WebSocket manager resources."""
        await self.stop_broadcast_task()
        
        # Close all connections
        for websocket in self.active_connections[:]:
            try:
                await websocket.close()
            except Exception:
                pass
        
        self.active_connections.clear()
        self.user_connections.clear()
        self.connection_metadata.clear()
        self.connection_semaphores.clear()
        
        logger.info("WebSocket manager cleanup completed")


class RateLimiter:
    """Simple rate limiter for WebSocket messages."""
    
    def __init__(self, max_messages_per_second: int = 10):
        self.max_messages_per_second = max_messages_per_second
        self.message_times = []
        self.lock = asyncio.Lock()
    
    async def allow(self) -> bool:
        """Check if message is allowed under rate limit."""
        async with self.lock:
            current_time = time.time()
            
            # Remove old message times (older than 1 second)
            self.message_times = [t for t in self.message_times if current_time - t < 1.0]
            
            # Check if we're under the limit
            if len(self.message_times) < self.max_messages_per_second:
                self.message_times.append(current_time)
                return True
            
            return False

# Global WebSocket manager instance
websocket_manager = WebSocketManager()
