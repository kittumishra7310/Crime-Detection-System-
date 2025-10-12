import cv2
import numpy as np
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
from ml_model import get_crime_model, CrimeDetectionModel
from database import SessionLocal, Detection, Alert
from websocket_manager import websocket_manager
import base64

logger = logging.getLogger(__name__)

class LiveDetectionManager:
    def __init__(self):
        self.model: CrimeDetectionModel = get_crime_model()
        self.active_tasks: Dict[int, asyncio.Task] = {}
        self.active_feeds: Dict[int, Any] = {}

    async def start_camera_detection(self, camera_id: int, source: any = 0) -> bool:
        await self.stop_camera_detection(camera_id)

        try:
            if isinstance(source, str) and source.isdigit():
                source = int(source)

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error(f"Failed to open camera source: {source}")
                return False

            self.active_feeds[camera_id] = cap
            task = asyncio.create_task(self._detection_loop(camera_id, cap))
            self.active_tasks[camera_id] = task
            logger.info(f"Started live detection for camera {camera_id}")
            return True

        except Exception as e:
            logger.error(f"Error starting camera detection: {e}")
            return False

    async def stop_camera_detection(self, camera_id: int) -> bool:
        if camera_id in self.active_tasks:
            task = self.active_tasks.pop(camera_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Detection task for camera {camera_id} cancelled.")

        if camera_id in self.active_feeds:
            cap = self.active_feeds.pop(camera_id)
            await asyncio.to_thread(cap.release)
            logger.info(f"Camera {camera_id} released.")
        
        return True

    async def _detection_loop(self, camera_id: int, cap: cv2.VideoCapture):
        detection_interval = 1.0  # seconds
        while True:
            try:
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret:
                    logger.warning(f"Camera {camera_id} disconnected.")
                    break

                await self._process_frame(camera_id, frame)
                await asyncio.sleep(detection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in detection loop for camera {camera_id}: {e}")
                break
        
        await self.stop_camera_detection(camera_id)

    async def _process_frame(self, camera_id: int, frame: np.ndarray):
        try:
            prediction = await asyncio.to_thread(self.model.predict_frame, frame)
            
            if prediction and prediction['is_crime']:
                detection_id = await self._save_detection(camera_id, prediction, frame)
                if prediction['severity'] in ['high', 'critical']:
                    await self._create_alert(detection_id, prediction)
                await self._send_realtime_update(camera_id, prediction, detection_id)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    async def _save_detection(self, camera_id: int, prediction: Dict, frame: np.ndarray) -> int:
        try:
            with SessionLocal() as db:
                _, buffer = cv2.imencode('.jpg', frame)
                image_path = f"uploads/detection_{datetime.utcnow().timestamp()}.jpg"
                with open(image_path, "wb") as f:
                    f.write(buffer)

                detection = Detection(
                    camera_id=camera_id,
                    detection_type=prediction['crime_type'],
                    confidence=prediction['confidence'],
                    severity=prediction['severity'],
                    timestamp=datetime.utcnow(),
                    image_path=image_path
                )
                db.add(detection)
                db.commit()
                return detection.id
        except Exception as e:
            logger.error(f"Error saving detection: {e}")
            return 0

    async def _create_alert(self, detection_id: int, prediction: Dict):
        try:
            with SessionLocal() as db:
                alert = Alert(
                    detection_id=detection_id,
                    severity=prediction['severity'],
                    message=f"Crime detected: {prediction['crime_type']}"
                )
                db.add(alert)
                db.commit()
        except Exception as e:
            logger.error(f"Error creating alert: {e}")

    async def _send_realtime_update(self, camera_id: int, prediction: Dict, detection_id: int):
        message = {
            "type": "live_detection",
            "camera_id": camera_id,
            "detection_id": detection_id,
            **prediction
        }
        await websocket_manager.broadcast(message)

    async def process_uploaded_file(self, file_path: str, camera_id: int) -> Dict[str, Any]:
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
        if prediction and prediction['is_crime']:
            await self._save_detection(camera_id, prediction, image)
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
            if frame_count % int(fps) == 0:  # Process one frame per second
                prediction = await asyncio.to_thread(self.model.predict_frame, frame)
                if prediction and prediction['is_crime']:
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

    async def cleanup(self):
        await asyncio.gather(*[self.stop_camera_detection(cam_id) for cam_id in list(self.active_tasks.keys())])

# Global instance
live_detection_manager = LiveDetectionManager()

