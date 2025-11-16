import cv2
import numpy as np
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import threading
import time
from collections import deque
import json
from ml_model import get_crime_model
from base_model import BaseDetectionModel
from database import SessionLocal, Detection, Alert
from websocket_manager import websocket_manager
import base64

logger = logging.getLogger(__name__)

class LiveDetectionManager:
    def __init__(self):
        self.model: BaseDetectionModel = get_crime_model()
        self.active_tasks: Dict[int, asyncio.Task] = {}
        self.active_feeds: Dict[int, Any] = {}
        self.active_streams: Dict[int, bool] = {}
        
        # Real-time processing optimizations
        self.detection_history: Dict[int, deque] = {}  # Camera ID -> detection history
        self.frame_queues: Dict[int, asyncio.Queue] = {}  # Camera ID -> frame queue
        self.detection_stats: Dict[int, Dict] = {}  # Camera ID -> statistics
        self.max_history_size = 1000  # Keep last 1000 detections per camera
        self.max_queue_size = 30  # Maximum frames in queue
        
        # Performance monitoring
        self.start_times: Dict[int, float] = {}  # Camera start times
        self.frame_counts: Dict[int, int] = {}  # Frames processed per camera
        self.detection_counts: Dict[int, int] = {}  # Detections per camera
        
        # Performance optimization
        self.frame_skip_count = 2  # Process every 3rd frame for performance
        self.detection_cooldown = 2.0  # Minimum seconds between detections
        self.last_detection_time: Dict[int, float] = {}  # Last detection time per camera
        self.confidence_threshold = 0.3  # Minimum confidence for detection (lowered for testing)
        
        # Memory management
        self.max_memory_usage = 1024 * 1024 * 1024  # 1GB max memory usage
        self.cleanup_interval = 60  # Cleanup every 60 seconds
        self.last_cleanup = time.time()
        
        # Log which model type is active
        from ml_model import get_model_info
        model_info = get_model_info()
        logger.info(f"LiveDetectionManager initialized with {model_info['type']} model")
        
        # Initialize camera-specific data structures
        self._init_camera_data()
        
        # Background cleanup task will be started on first use
        self._cleanup_started = False
    
    async def initialize(self):
        """Initialize async components when first needed."""
        if not self._cleanup_started:
            await self._start_cleanup_task()

    def _init_camera_data(self):
        """Initialize camera-specific data structures."""
        # This method can be extended for camera-specific initialization
        pass
    
    async def _start_cleanup_task(self):
        """Start background cleanup task for memory management."""
        if self._cleanup_started:
            return
            
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self._perform_cleanup()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        # Start cleanup task in background
        asyncio.create_task(cleanup_loop())
        self._cleanup_started = True
    
    async def _perform_cleanup(self):
        """Perform memory cleanup and optimization."""
        try:
            current_time = time.time()
            
            # Clean up old detection history
            for camera_id in list(self.detection_history.keys()):
                history = self.detection_history[camera_id]
                # Remove detections older than 1 hour
                cutoff_time = current_time - 3600
                while history and history[0].get('timestamp', 0) < cutoff_time:
                    history.popleft()
            
            # Clean up old frame queues
            for camera_id in list(self.frame_queues.keys()):
                queue = self.frame_queues[camera_id]
                # Clear old frames if queue is too large
                while queue.qsize() > self.max_queue_size:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            
            # Reset detection counts if no activity for 5 minutes
            for camera_id in list(self.last_detection_time.keys()):
                if current_time - self.last_detection_time[camera_id] > 300:
                    self.detection_counts[camera_id] = 0
                    self.frame_counts[camera_id] = 0
            
            logger.info("Memory cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _should_process_frame(self, camera_id: int) -> bool:
        """Check if frame should be processed based on performance settings."""
        # Skip frames for performance
        frame_count = self.frame_counts.get(camera_id, 0)
        if frame_count % (self.frame_skip_count + 1) != 0:
            return False
        
        # Check detection cooldown
        current_time = time.time()
        last_detection = self.last_detection_time.get(camera_id, 0)
        if current_time - last_detection < self.detection_cooldown:
            return False
        
        return True
    
    def _update_performance_stats(self, camera_id: int, processing_time: float, 
                                detection_made: bool = False):
        """Update performance statistics for a camera."""
        if camera_id not in self.detection_stats:
            self.detection_stats[camera_id] = {
                'total_frames': 0,
                'total_detections': 0,
                'avg_processing_time': 0,
                'last_detection_time': 0,
                'detection_rate': 0,
                'fps': 0
            }
        
        stats = self.detection_stats[camera_id]
        stats['total_frames'] += 1
        if detection_made:
            stats['total_detections'] += 1
            stats['last_detection_time'] = time.time()
        
        # Update average processing time
        if stats['total_frames'] == 1:
            stats['avg_processing_time'] = processing_time
        else:
            stats['avg_processing_time'] = (stats['avg_processing_time'] * (stats['total_frames'] - 1) + 
                                          processing_time) / stats['total_frames']
        
        # Calculate detection rate (detections per minute)
        if camera_id in self.start_times:
            runtime = time.time() - self.start_times[camera_id]
            if runtime > 0:
                stats['detection_rate'] = (stats['total_detections'] * 60) / runtime
                stats['fps'] = stats['total_frames'] / runtime
    
    def _init_camera_session(self, camera_id: int):
        """Initialize data structures for a specific camera session."""
        self.detection_history[camera_id] = deque(maxlen=self.max_history_size)
        self.frame_queues[camera_id] = asyncio.Queue(maxsize=self.max_queue_size)
        self.detection_stats[camera_id] = {
            'start_time': time.time(),
            'frames_processed': 0,
            'detections_made': 0,
            'crimes_detected': 0,
            'last_detection_time': None,
            'detection_types': {},
            'avg_confidence': 0.0,
            'total_confidence': 0.0,
            # Add fields for _update_performance_stats
            'total_frames': 0,
            'total_detections': 0,
            'avg_processing_time': 0,
            'detection_rate': 0,
            'fps': 0
        }
        self.start_times[camera_id] = time.time()
        self.frame_counts[camera_id] = 0
        self.detection_counts[camera_id] = 0
        logger.info(f"Initialized camera session data for camera {camera_id}")

    async def start_camera_detection(self, camera_id: int, source: any = 0, confidence_threshold: float = None) -> bool:
        """
        Start camera detection with real-time processing capabilities.
        
        Args:
            camera_id: Unique identifier for the camera
            source: Camera source (0 for webcam, RTSP URL, or file path)
            confidence_threshold: Optional confidence threshold override
            
        Returns:
            bool: Success status
        """
        # Initialize async components on first use
        await self.initialize()
        
        logger.info(f"Starting camera detection for camera {camera_id} with source {source}")
        
        # Ensure any existing detection is fully stopped
        await self.stop_camera_detection(camera_id)
        
        # Initialize camera session data
        self._init_camera_session(camera_id)
        
        # Set confidence threshold if provided
        if confidence_threshold is not None:
            self.model.set_confidence_threshold(confidence_threshold)
        
        # Give extra time for camera to be fully released
        await asyncio.sleep(0.5)

        try:
            # Handle different source types
            if isinstance(source, str):
                if source.isdigit():
                    source = int(source)
                elif source.startswith('rtsp://') or source.startswith('http://'):
                    logger.info(f"Using RTSP/HTTP stream: {source}")
                elif source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    logger.info(f"Using video file: {source}")
                else:
                    logger.warning(f"Unknown source type: {source}")

            # Try to open the camera with retries and different backends
            cap = None
            for attempt in range(3):
                try:
                    # Try different backends for better compatibility
                    if isinstance(source, int) and source == 0:
                        # For webcam, try different backends
                        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                        for backend in backends:
                            cap = cv2.VideoCapture(source, backend)
                            if cap.isOpened():
                                logger.info(f"Camera opened with backend {backend}")
                                break
                            if cap:
                                cap.release()
                    else:
                        # For other sources, use default backend
                        cap = cv2.VideoCapture(source)
                    
                    if cap and cap.isOpened():
                        break
                        
                except Exception as cap_error:
                    logger.warning(f"Camera open attempt {attempt + 1} failed: {cap_error}")
                
                if cap:
                    try:
                        cap.release()
                    except:
                        pass
                
                if attempt < 2:  # Don't sleep on last attempt
                    await asyncio.sleep(0.5)
            
            if not cap or not cap.isOpened():
                logger.error(f"Failed to open camera source: {source} after 3 attempts")
                return False

            # Set camera properties for optimal performance
            if isinstance(source, int):
                # Webcam specific settings
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
                cap.set(cv2.CAP_PROP_FPS, 30)  # Target 30 FPS
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for faster processing
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                # Stream/file settings
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Log camera properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(f"Camera properties: {width}x{height} @ {fps} FPS")
            
            self.active_feeds[camera_id] = cap
            self.active_streams[camera_id] = True
            
            # Create detection loop task
            task = asyncio.create_task(self._detection_loop(camera_id, cap))
            self.active_tasks[camera_id] = task
            
            logger.info(f"Started live detection for camera {camera_id}")
            return True

        except Exception as e:
            logger.error(f"Error starting camera detection: {e}")
            # Clean up on error
            if cap:
                try:
                    cap.release()
                except:
                    pass
            return False

    async def stop_camera_detection(self, camera_id: int) -> bool:
        """Stop camera detection and release all resources."""
        logger.info(f"Stopping camera detection for camera {camera_id}")
        
        # Remove from active streams first to stop new frame processing
        if camera_id in self.active_streams:
            self.active_streams.pop(camera_id)
            logger.info(f"Removed camera {camera_id} from active streams")
        
        # Save detection statistics before cleanup
        if camera_id in self.detection_stats:
            stats = self.detection_stats[camera_id]
            session_duration = time.time() - stats['start_time']
            logger.info(f"Camera {camera_id} session stats: {stats['frames_processed']} frames, "
                       f"{stats['detections_made']} detections, {stats['crimes_detected']} crimes, "
                       f"duration: {session_duration:.1f}s")
        
        # Cancel the detection task
        if camera_id in self.active_tasks:
            task = self.active_tasks.pop(camera_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Detection task for camera {camera_id} cancelled successfully")
            except Exception as e:
                logger.error(f"Error cancelling task for camera {camera_id}: {e}")

        # Release the camera feed with proper cleanup
        if camera_id in self.active_feeds:
            cap = self.active_feeds.pop(camera_id)
            try:
                # Release in a thread to avoid blocking
                await asyncio.to_thread(self._release_camera, cap)
                logger.info(f"Camera {camera_id} released successfully")
            except Exception as e:
                logger.error(f"Error releasing camera {camera_id}: {e}")
        
        # Clean up camera-specific data
        self._cleanup_camera_data(camera_id)
        
        # Give the camera a moment to fully release
        await asyncio.sleep(0.5)
        
        logger.info(f"Camera {camera_id} stopped completely")
        return True
    
    def _cleanup_camera_data(self, camera_id: int):
        """Clean up camera-specific data structures."""
        try:
            # Remove camera data structures
            self.detection_history.pop(camera_id, None)
            self.frame_queues.pop(camera_id, None)
            self.detection_stats.pop(camera_id, None)
            self.start_times.pop(camera_id, None)
            self.frame_counts.pop(camera_id, None)
            self.detection_counts.pop(camera_id, None)
            logger.info(f"Cleaned up data for camera {camera_id}")
        except Exception as e:
            logger.error(f"Error cleaning up camera {camera_id} data: {e}")
    
    def _release_camera(self, cap: cv2.VideoCapture):
        """Helper method to properly release camera resources."""
        try:
            if cap and cap.isOpened():
                # Read and discard any buffered frames
                try:
                    cap.read()
                except:
                    pass
                # Release the camera
                cap.release()
                logger.info("Camera released successfully")
            # Destroy any OpenCV windows
            cv2.destroyAllWindows()
            # Force garbage collection of the capture object
            if cap:
                del cap
        except Exception as e:
            logger.error(f"Error in _release_camera: {e}")

    async def _detection_loop(self, camera_id: int, cap: cv2.VideoCapture):
        """
        Main detection loop with real-time processing and WebSocket integration.
        """
        logger.info(f"Starting detection loop for camera {camera_id}")
        frame_skip_counter = 0
        
        # Initialize camera session
        self._init_camera_session(camera_id)
        self.start_times[camera_id] = time.time()
        
        try:
            while camera_id in self.active_streams:
                try:
                    start_time = time.time()
                    
                    ret, frame = await asyncio.to_thread(cap.read)
                    if not ret:
                        logger.warning(f"Camera {camera_id} disconnected.")
                        break

                    # Update frame count
                    self.frame_counts[camera_id] = self.frame_counts.get(camera_id, 0) + 1
                    frame_skip_counter += 1

                    # Performance optimization: skip frames based on settings
                    if not self._should_process_frame(camera_id):
                        continue

                    # Process frame with performance monitoring
                    processing_start = time.time()
                    await self._process_frame(camera_id, frame)
                    processing_time = time.time() - processing_start
                    
                    # Update performance statistics
                    self._update_performance_stats(camera_id, processing_time)
                    
                    # Dynamic sleep based on processing time to maintain target FPS
                    target_frame_time = 0.033  # Target ~30 FPS
                    actual_frame_time = time.time() - start_time
                    sleep_time = max(0, target_frame_time - actual_frame_time)
                    
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

                except asyncio.CancelledError:
                    logger.info(f"Detection loop for camera {camera_id} cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in detection loop for camera {camera_id}: {e}")
                    # Continue processing despite errors
                    await asyncio.sleep(0.1)
                    continue
                    
        finally:
            # Always clean up the camera when loop exits
            logger.info(f"Detection loop exiting for camera {camera_id}, cleaning up...")
            if camera_id in self.active_feeds:
                cap_to_release = self.active_feeds.pop(camera_id, None)
                if cap_to_release:
                    try:
                        await asyncio.to_thread(self._release_camera, cap_to_release)
                        logger.info(f"Camera {camera_id} released in finally block")
                    except Exception as e:
                        logger.error(f"Error releasing camera in finally: {e}")
            
            # Remove from active streams
            self.active_streams.pop(camera_id, None)
            # Remove task reference
            self.active_tasks.pop(camera_id, None)

    async def _process_frame(self, camera_id: int, frame: np.ndarray):
        """
        Process a single frame with real-time detection and WebSocket updates.
        """
        try:
            # Run model prediction
            prediction = await asyncio.to_thread(self.model.predict_frame, frame)
            
            # Handle buffering status
            if prediction and prediction.get('status') == 'Buffering frames':
                # Send buffering status update
                await self._send_buffering_update(camera_id, prediction)
                return
            
            # Handle prediction errors
            if prediction and prediction.get('status') == 'Error':
                logger.error(f"Prediction error for camera {camera_id}: {prediction.get('error', 'Unknown error')}")
                await self._send_error_update(camera_id, prediction.get('error', 'Unknown error'))
                return
            
            # Apply confidence threshold and cooldown
            if prediction and prediction.get('is_crime', False):
                confidence = prediction.get('confidence', 0.0)
                current_time = time.time()
                last_detection = self.last_detection_time.get(camera_id, 0)
                
                # Check confidence threshold
                if confidence < self.confidence_threshold:
                    logger.debug(f"Detection below confidence threshold: {confidence} < {self.confidence_threshold}")
                    prediction['is_crime'] = False
                # Check detection cooldown
                elif current_time - last_detection < self.detection_cooldown:
                    logger.debug(f"Detection in cooldown period: {current_time - last_detection:.1f}s < {self.detection_cooldown}s")
                    prediction['is_crime'] = False
                else:
                    # Valid detection - update last detection time
                    self.last_detection_time[camera_id] = current_time
                    self.detection_counts[camera_id] = self.detection_counts.get(camera_id, 0) + 1
            
            # Update detection statistics
            if camera_id in self.detection_stats and prediction:
                stats = self.detection_stats[camera_id]
                stats['detections_made'] += 1
                
                if prediction.get('is_crime', False):
                    stats['crimes_detected'] += 1
                    stats['last_detection_time'] = datetime.utcnow()
                    
                    # Update detection types
                    crime_type = prediction.get('crime_type', 'Unknown')
                    stats['detection_types'][crime_type] = stats['detection_types'].get(crime_type, 0) + 1
                    
                    # Update confidence statistics
                    confidence = prediction.get('confidence', 0.0)
                    stats['total_confidence'] += confidence
                    stats['avg_confidence'] = stats['total_confidence'] / stats['detections_made']
            
            # Log prediction results
            if prediction:
                logger.info(f"Camera {camera_id} - Detected: {prediction.get('crime_type', 'N/A')} "
                           f"(confidence: {prediction.get('confidence', 0.0):.3f}, "
                           f"is_crime: {prediction.get('is_crime', False)}, "
                           f"inference_time: {prediction.get('inference_time_ms', 0):.1f}ms)")
            
            # Handle crime detection
            if prediction and prediction.get('is_crime', False):
                detection_id = await self._save_detection(camera_id, prediction, frame)
                
                # Create alert for high/critical severity
                if prediction.get('severity') in ['high', 'critical']:
                    await self._create_alert(detection_id, prediction)
                
                # Send real-time detection update
                await self._send_detection_update(camera_id, prediction, detection_id, frame)
                
                # Add to detection history
                if camera_id in self.detection_history:
                    self.detection_history[camera_id].append({
                        'timestamp': datetime.utcnow(),
                        'detection_id': detection_id,
                        'prediction': prediction
                    })
            else:
                # Send normal status update (no crime detected)
                await self._send_status_update(camera_id, prediction)

        except Exception as e:
            logger.error(f"Error processing frame for camera {camera_id}: {e}")
            await self._send_error_update(camera_id, str(e))
    
    async def _send_buffering_update(self, camera_id: int, prediction: Dict):
        """Send buffering status update via WebSocket."""
        message = {
            "type": "buffering_status",
            "camera_id": camera_id,
            "frames_collected": prediction.get('frames_collected', 0),
            "frames_needed": prediction.get('frames_needed', 16),
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket_manager.broadcast(message)
    
    async def _send_status_update(self, camera_id: int, prediction: Dict):
        """Send normal status update via WebSocket."""
        if not prediction:
            return
            
        message = {
            "type": "detection_status",
            "camera_id": camera_id,
            "status": "normal",
            "crime_type": prediction.get('crime_type', 'Normal_Videos_event'),
            "confidence": prediction.get('confidence', 0.0),
            "inference_time_ms": prediction.get('inference_time_ms', 0),
            "fps": prediction.get('fps', 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket_manager.broadcast(message)
    
    async def _send_error_update(self, camera_id: int, error_message: str):
        """Send error update via WebSocket."""
        message = {
            "type": "detection_error",
            "camera_id": camera_id,
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket_manager.broadcast(message)
    
    async def _send_detection_update(self, camera_id: int, prediction: Dict, detection_id: int, frame: np.ndarray):
        """Send real-time detection update via WebSocket with frame data."""
        try:
            # Encode frame as base64 for transmission
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            message = {
                "type": "detection_result",
                "camera_id": camera_id,
                "detection_id": detection_id,
                "crime_type": prediction.get('crime_type', 'Unknown'),
                "confidence": prediction.get('confidence', 0.0),
                "severity": prediction.get('severity', 'low'),
                "is_crime": prediction.get('is_crime', False),
                "inference_time_ms": prediction.get('inference_time_ms', 0),
                "fps": prediction.get('fps', 0),
                "all_predictions": prediction.get('all_predictions', {}),
                "frame_data": frame_base64,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await websocket_manager.broadcast(message)
            
        except Exception as e:
            logger.error(f"Error sending detection update: {e}")
            # Send update without frame data if encoding fails
            message = {
                "type": "detection_result",
                "camera_id": camera_id,
                "detection_id": detection_id,
                "crime_type": prediction.get('crime_type', 'Unknown'),
                "confidence": prediction.get('confidence', 0.0),
                "severity": prediction.get('severity', 'low'),
                "is_crime": prediction.get('is_crime', False),
                "inference_time_ms": prediction.get('inference_time_ms', 0),
                "fps": prediction.get('fps', 0),
                "all_predictions": prediction.get('all_predictions', {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            await websocket_manager.broadcast(message)

    async def _save_detection(self, camera_id: int, prediction: Dict, frame: np.ndarray) -> int:
        """Save detection to database with enhanced metadata."""
        try:
            with SessionLocal() as db:
                # Save frame image
                _, buffer = cv2.imencode('.jpg', frame)
                timestamp = datetime.utcnow()
                image_path = f"uploads/detection_{camera_id}_{timestamp.timestamp()}.jpg"
                
                # Ensure uploads directory exists
                import os
                os.makedirs('uploads', exist_ok=True)
                
                with open(image_path, "wb") as f:
                    f.write(buffer)

                # Create detection record with enhanced metadata
                detection = Detection(
                    camera_id=camera_id,
                    detection_type=prediction['crime_type'],
                    confidence=prediction['confidence'],
                    severity=prediction['severity'],
                    timestamp=timestamp,
                    image_path=image_path,
                    bounding_box=json.dumps({
                        'inference_time_ms': prediction.get('inference_time_ms', 0),
                        'fps': prediction.get('fps', 0),
                        'model_performance': self.model.get_performance_metrics() if hasattr(self.model, 'get_performance_metrics') else {}
                    })
                )
                
                db.add(detection)
                db.commit()
                db.refresh(detection)
                
                logger.info(f"Saved detection {detection.id} for camera {camera_id}: "
                           f"{prediction['crime_type']} (confidence: {prediction['confidence']:.3f})")
                
                return detection.id
                
        except Exception as e:
            logger.error(f"Error saving detection for camera {camera_id}: {e}")
            return 0

    async def _create_alert(self, detection_id: int, prediction: Dict):
        """Create alert with enhanced messaging and severity handling."""
        try:
            with SessionLocal() as db:
                # Create detailed alert message
                crime_type = prediction.get('crime_type', 'Unknown Crime')
                confidence = prediction.get('confidence', 0.0)
                severity = prediction.get('severity', 'low')
                
                # Generate appropriate message based on severity
                if severity == 'critical':
                    message = f"🚨 CRITICAL ALERT: {crime_type} detected with {confidence:.1%} confidence"
                elif severity == 'high':
                    message = f"⚠️ HIGH PRIORITY: {crime_type} detected with {confidence:.1%} confidence"
                elif severity == 'medium':
                    message = f"📋 MEDIUM PRIORITY: {crime_type} detected with {confidence:.1%} confidence"
                else:
                    message = f"ℹ️ LOW PRIORITY: {crime_type} detected with {confidence:.1%} confidence"
                
                alert = Alert(
                    detection_id=detection_id,
                    severity=severity,
                    message=message
                )
                
                db.add(alert)
                db.commit()
                
                logger.warning(f"Alert created for detection {detection_id}: {message}")
                
                # Send real-time alert via WebSocket
                await self._send_alert_notification(alert, prediction)
                
        except Exception as e:
            logger.error(f"Error creating alert for detection {detection_id}: {e}")
    
    async def _send_alert_notification(self, alert: Alert, prediction: Dict):
        """Send real-time alert notification via WebSocket."""
        try:
            message = {
                "type": "alert_notification",
                "alert_id": alert.id,
                "detection_id": alert.detection_id,
                "severity": alert.severity,
                "message": alert.message,
                "crime_type": prediction.get('crime_type', 'Unknown'),
                "confidence": prediction.get('confidence', 0.0),
                "timestamp": alert.timestamp.isoformat() if alert.timestamp else datetime.utcnow().isoformat()
            }
            
            await websocket_manager.broadcast(message)
            logger.info(f"Alert notification sent via WebSocket for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")

    async def _send_realtime_update(self, camera_id: int, prediction: Dict, detection_id: int):
        """Legacy method for backward compatibility - now handled by specific update methods."""
        # This method is kept for backward compatibility but the new methods handle specific cases
        pass

    def get_camera_frame(self, camera_id: int) -> Optional[bytes]:
        """Get current frame from camera for streaming."""
        if camera_id in self.active_feeds:
            cap = self.active_feeds[camera_id]
            try:
                ret, frame = cap.read()
                if ret:
                    # Encode as JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    return buffer.tobytes()
            except Exception as e:
                logger.error(f"Error getting frame from camera {camera_id}: {e}")
        return None

    async def get_active_cameras(self) -> List[int]:
        """Get list of active camera IDs."""
        return list(self.active_tasks.keys())

    async def get_camera_status(self, camera_id: int) -> Dict[str, Any]:
        """Get detailed status for a specific camera."""
        is_active = camera_id in self.active_streams
        
        status = {
            "camera_id": camera_id,
            "is_active": is_active,
            "status": "active" if is_active else "inactive",
            "start_time": self.start_times.get(camera_id),
            "frames_processed": self.frame_counts.get(camera_id, 0),
            "detections_made": self.detection_counts.get(camera_id, 0)
        }
        
        if camera_id in self.detection_stats:
            stats = self.detection_stats[camera_id]
            status.update({
                "session_duration": time.time() - stats['start_time'] if is_active else 0,
                "crimes_detected": stats['crimes_detected'],
                "detection_types": stats['detection_types'],
                "avg_confidence": stats['avg_confidence'],
                "last_detection_time": stats['last_detection_time'].isoformat() if stats['last_detection_time'] else None
            })
        
        return status

    async def get_detection_history(self, camera_id: int, limit: int = 100) -> List[Dict]:
        """Get detection history for a camera."""
        if camera_id in self.detection_history:
            history = list(self.detection_history[camera_id])
            return history[-limit:] if limit > 0 else history
        return []

    async def get_performance_metrics(self, camera_id: int) -> Dict[str, Any]:
        """Get performance metrics for a camera."""
        if hasattr(self.model, 'get_performance_metrics'):
            model_metrics = self.model.get_performance_metrics()
        else:
            model_metrics = {}
        
        camera_status = await self.get_camera_status(camera_id)
        
        return {
            "camera_id": camera_id,
            "model_performance": model_metrics,
            "camera_status": camera_status,
            "active_streams": len(self.active_streams),
            "total_detections": sum(self.detection_counts.values()),
            "total_frames": sum(self.frame_counts.values())
        }

    async def force_stop_all(self):
        """Force stop all active camera detections."""
        logger.info("Force stopping all camera detections")
        camera_ids = list(self.active_tasks.keys())
        
        # Stop all cameras concurrently
        stop_tasks = [self.stop_camera_detection(cam_id) for cam_id in camera_ids]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Clear all dictionaries as a safety measure
        self.active_tasks.clear()
        self.active_feeds.clear()
        self.active_streams.clear()
        
        # Clean up all camera data
        for camera_id in list(self.detection_history.keys()):
            self._cleanup_camera_data(camera_id)
        
        logger.info("All cameras force stopped")

    async def cleanup(self):
        """Clean up all resources."""
        await self.force_stop_all()
        
        # Reset model buffer if available
        if hasattr(self.model, 'reset_buffer'):
            self.model.reset_buffer()
        
        logger.info("LiveDetectionManager cleanup completed")

    async def process_uploaded_file(self, file_path: str, camera_id: int) -> Dict[str, Any]:
        # Initialize async components on first use
        await self.initialize()
        
        file_ext = file_path.lower().split('.')[-1]
        if file_ext in ['jpg', 'jpeg', 'png']:
            return await self._process_image(file_path, camera_id)
        elif file_ext in ['mp4', 'avi', 'mov']:
            return await self._process_video(file_path, camera_id)
        else:
            return {"error": "Unsupported file format"}

    async def _process_image(self, image_path: str, camera_id: int) -> Dict[str, Any]:
        image = await asyncio.to_thread(cv2.imread, image_path)
        if image is None:
            return {"error": "Failed to load image"}
        
        prediction = await asyncio.to_thread(self.model.predict_frame, image)
        
        # Log the prediction
        if prediction:
            logger.info(f"Image analysis - Detected: {prediction['crime_type']} "
                       f"(confidence: {prediction['confidence']:.2f}, is_crime: {prediction['is_crime']})")
        
        # Save to database if it's a crime
        if prediction and prediction['is_crime']:
            await self._save_detection(camera_id, prediction, image)
            logger.info(f"Saved crime detection to database")
        
        return prediction

    async def _process_video(self, video_path: str, camera_id: int) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Failed to open video"}

        detections = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                break
            
            frame_count += 1
            prediction = await asyncio.to_thread(self.model.predict_frame, frame)
            
            if prediction is None:
                continue

            # Log what was detected
            logger.info(f"Video frame {frame_count} - Detected: {prediction.get('crime_type', 'N/A')} "
                       f"(confidence: {prediction.get('confidence', 0.0):.2f}, is_crime: {prediction.get('is_crime', False)})")
            
            if prediction and prediction.get('is_crime'):
                detection_id = await self._save_detection(camera_id, prediction, frame)
                detections.append({
                    "detection_id": detection_id,
                    "timestamp": frame_count / fps,
                    **prediction
                })
        
        cap.release()
        return {"detections": detections, "message": f"Video processed. Found {len(detections)} detections."}

    def get_camera_frame(self, camera_id: int) -> Optional[bytes]:
        if camera_id in self.active_feeds:
            cap = self.active_feeds[camera_id]
            ret, frame = cap.read() # This is blocking, but for MJPEG stream it's often acceptable
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                return buffer.tobytes()
        return None

    async def get_active_cameras(self) -> List[int]:
        return list(self.active_tasks.keys())

    async def force_stop_all(self):
        """Force stop all active camera detections."""
        logger.info("Force stopping all camera detections")
        camera_ids = list(self.active_tasks.keys())
        for cam_id in camera_ids:
            await self.stop_camera_detection(cam_id)
        
        # Clear all dictionaries as a safety measure
        self.active_tasks.clear()
        self.active_feeds.clear()
        self.active_streams.clear()
        logger.info("All cameras force stopped")

    async def cleanup(self):
        await self.force_stop_all()

# Global instance
live_detection_manager = LiveDetectionManager()

