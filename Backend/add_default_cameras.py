"""
Script to add default cameras to the database.
Run this script to populate the database with sample cameras.
"""

import sys
from database import SessionLocal, Camera
from sqlalchemy.exc import IntegrityError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_default_cameras():
    """Add default cameras to the database."""
    db = SessionLocal()
    
    default_cameras = [
        {
            "name": "Webcam",
            "camera_id": "webcam_0",
            "rtsp_url": "0",  # Webcam index
            "location": "Local Computer",
            "status": "inactive"
        },
        {
            "name": "Front Entrance",
            "camera_id": "cam_001",
            "rtsp_url": "rtsp://example.com/stream1",
            "location": "Building A - Main Entrance",
            "status": "inactive"
        },
        {
            "name": "Parking Lot",
            "camera_id": "cam_002",
            "rtsp_url": "rtsp://example.com/stream2",
            "location": "Building A - Parking Area",
            "status": "inactive"
        },
        {
            "name": "Back Exit",
            "camera_id": "cam_003",
            "rtsp_url": "rtsp://example.com/stream3",
            "location": "Building A - Back Exit",
            "status": "inactive"
        },
        {
            "name": "Hallway Camera",
            "camera_id": "cam_004",
            "rtsp_url": "rtsp://example.com/stream4",
            "location": "Building A - 1st Floor Hallway",
            "status": "inactive"
        }
    ]
    
    added_count = 0
    updated_count = 0
    
    try:
        for cam_data in default_cameras:
            # Check if camera already exists
            existing_camera = db.query(Camera).filter(
                Camera.camera_id == cam_data["camera_id"]
            ).first()
            
            if existing_camera:
                # Update existing camera
                for key, value in cam_data.items():
                    if key != "camera_id":  # Don't update the unique identifier
                        setattr(existing_camera, key, value)
                updated_count += 1
                logger.info(f"Updated camera: {cam_data['name']} (ID: {cam_data['camera_id']})")
            else:
                # Add new camera
                camera = Camera(**cam_data)
                db.add(camera)
                added_count += 1
                logger.info(f"Added camera: {cam_data['name']} (ID: {cam_data['camera_id']})")
        
        db.commit()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Camera setup complete!")
        logger.info(f"Added: {added_count} new cameras")
        logger.info(f"Updated: {updated_count} existing cameras")
        logger.info(f"Total cameras in database: {db.query(Camera).count()}")
        logger.info(f"{'='*60}\n")
        
        # List all cameras
        all_cameras = db.query(Camera).all()
        logger.info("Current cameras in database:")
        for cam in all_cameras:
            logger.info(f"  - ID: {cam.id}, Name: {cam.name}, Camera ID: {cam.camera_id}, Status: {cam.status}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error adding cameras: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    logger.info("Adding default cameras to database...")
    success = add_default_cameras()
    sys.exit(0 if success else 1)
