#!/usr/bin/env python3
"""
Database initialization script for the surveillance system.
This script creates the database tables and sets up initial data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from database import Base, User, Camera
from auth import get_password_hash
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize the database with tables and default data."""
    
    try:
        # Create engine - use SQLite for easy setup
        database_url = "sqlite:///./surveillance.db"
        engine = create_engine(database_url, connect_args={"check_same_thread": False})
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Create session
        from sqlalchemy.orm import sessionmaker
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        # Create default admin user if not exists
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(
                username="admin",
                email="admin@surveillance.com",
                password_hash=get_password_hash("admin123"),
                role="admin"
            )
            db.add(admin_user)
            logger.info("Default admin user created (username: admin, password: admin123)")
        
        # Create demo viewer user
        viewer_user = db.query(User).filter(User.username == "viewer").first()
        if not viewer_user:
            viewer_user = User(
                username="viewer",
                email="viewer@surveillance.com",
                password_hash=get_password_hash("viewer123"),
                role="viewer"
            )
            db.add(viewer_user)
            logger.info("Demo viewer user created (username: viewer, password: viewer123)")
        
        # Create demo cameras
        demo_cameras = [
            {
                "name": "Main Entrance",
                "camera_id": "CAM-001",
                "location": "Building Main Entrance",
                "rtsp_url": "rtsp://demo:demo@192.168.1.100:554/stream1"
            },
            {
                "name": "Parking Lot",
                "camera_id": "CAM-002", 
                "location": "Employee Parking Area",
                "rtsp_url": "rtsp://demo:demo@192.168.1.101:554/stream1"
            },
            {
                "name": "Lobby Camera",
                "camera_id": "CAM-003",
                "location": "Reception Lobby",
                "rtsp_url": "rtsp://demo:demo@192.168.1.102:554/stream1"
            }
        ]
        
        for cam_data in demo_cameras:
            existing_cam = db.query(Camera).filter(Camera.camera_id == cam_data["camera_id"]).first()
            if not existing_cam:
                camera = Camera(**cam_data)
                db.add(camera)
                logger.info(f"Demo camera created: {cam_data['name']}")
        
        # Commit all changes
        db.commit()
        db.close()
        
        logger.info("Database initialization completed successfully!")
        
        print("\n" + "="*50)
        print("DATABASE INITIALIZATION COMPLETE")
        print("="*50)
        print("Default Users Created:")
        print("  Admin: username=admin, password=admin123")
        print("  Viewer: username=viewer, password=viewer123")
        print("\nDemo Cameras Created:")
        for cam in demo_cameras:
            print(f"  {cam['name']} ({cam['camera_id']})")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    init_database()
