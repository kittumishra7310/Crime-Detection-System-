#!/usr/bin/env python3
"""
MySQL Database Initialization Script for Surveillance System
Creates database, tables, and default data
"""

import os
import urllib.parse
import logging
from datetime import datetime

import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base, User, Camera, SystemLog
from auth import get_password_hash
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mysql_config():
    """Get MySQL config from environment variables."""
    return {
        'host': settings.DATABASE_HOST,
        'user': settings.DATABASE_USER,
        'password': settings.DATABASE_PASSWORD,
        'database': settings.DATABASE_NAME
    }

def create_database(config):
    """Create the database if it doesn't exist."""
    try:
        conn = mysql.connector.connect(
            host=config['host'],
            user=config['user'],
            password=config['password']
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config['database']}")
        logger.info(f"Database '{config['database']}' created or already exists.")
        return True
    except Error as e:
        logger.error(f"Error creating database: {e}")
        return False
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def create_tables(config):
    """Create all tables using SQLAlchemy."""
    try:
        encoded_password = urllib.parse.quote_plus(config['password'])
        db_url = f"mysql+pymysql://{config['user']}:{encoded_password}@{config['host']}/{config['database']}"
        engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(bind=engine)
        logger.info("All tables created successfully.")
        return engine
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return None

def create_default_data(engine):
    """Create default users and cameras."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with SessionLocal() as db:
        try:
            # Create admin user
            if not db.query(User).filter(User.username == "admin").first():
                admin = User(
                    username="admin", email="admin@example.com",
                    password_hash=get_password_hash("admin123"), role="admin"
                )
                db.add(admin)
                logger.info("Created admin user.")

            # Create viewer user
            if not db.query(User).filter(User.username == "viewer").first():
                viewer = User(
                    username="viewer", email="viewer@example.com",
                    password_hash=get_password_hash("viewer123"), role="viewer"
                )
                db.add(viewer)
                logger.info("Created viewer user.")

            # Create demo cameras
            demo_cameras = [
                {"name": "Main Entrance", "location": "Building A", "status": "inactive"},
                {"name": "Parking Lot", "location": "West Side", "status": "inactive"},
                {"name": "Lobby", "location": "Building A", "status": "inactive"}
            ]
            for cam_data in demo_cameras:
                if not db.query(Camera).filter(Camera.name == cam_data["name"]).first():
                    db.add(Camera(**cam_data))
            logger.info("Created demo cameras.")

            # Create initial system log
            log = SystemLog(message="Database initialized.", level="INFO")
            db.add(log)
            
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating default data: {e}")
            return False

def main():
    """Main initialization function."""
    logger.info("Starting database initialization...")
    
    config = get_mysql_config()
    
    if not create_database(config):
        return
        
    engine = create_tables(config)
    if not engine:
        return

    if not create_default_data(engine):
        return

    logger.info("✅ Database initialization completed successfully!")

if __name__ == "__main__":
    main()
