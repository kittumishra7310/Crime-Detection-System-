#!/usr/bin/env python3
"""
Simplified server for quick testing and demo.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import sqlite3
import hashlib
import jwt
from datetime import datetime, timedelta
import os

# Simple FastAPI app
app = FastAPI(title="Crime Detection API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple database setup
DB_PATH = "simple_surveillance.db"
SECRET_KEY = "demo-secret-key"

def init_simple_db():
    """Initialize simple SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT,
            password_hash TEXT,
            role TEXT DEFAULT 'viewer'
        )
    ''')
    
    # Create cameras table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY,
            name TEXT,
            camera_id TEXT UNIQUE,
            location TEXT,
            status TEXT DEFAULT 'active'
        )
    ''')
    
    # Create detections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY,
            camera_id INTEGER,
            detection_type TEXT,
            confidence REAL,
            severity TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY,
            detection_id INTEGER,
            status TEXT DEFAULT 'active',
            severity TEXT,
            message TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert default users
    admin_hash = hashlib.sha256("admin123".encode()).hexdigest()
    viewer_hash = hashlib.sha256("viewer123".encode()).hexdigest()
    
    cursor.execute('INSERT OR IGNORE INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)',
                   ('admin', 'admin@surveillance.com', admin_hash, 'admin'))
    cursor.execute('INSERT OR IGNORE INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)',
                   ('viewer', 'viewer@surveillance.com', viewer_hash, 'viewer'))
    
    # Insert demo cameras
    cameras = [
        ('Main Entrance', 'CAM-001', 'Building Main Entrance'),
        ('Parking Lot', 'CAM-002', 'Employee Parking Area'),
        ('Lobby Camera', 'CAM-003', 'Reception Lobby')
    ]
    
    for name, cam_id, location in cameras:
        cursor.execute('INSERT OR IGNORE INTO cameras (name, camera_id, location) VALUES (?, ?, ?)',
                       (name, cam_id, location))
    
    conn.commit()
    conn.close()

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

class CameraResponse(BaseModel):
    id: int
    name: str
    camera_id: str
    location: str
    status: str

class DetectionResponse(BaseModel):
    id: int
    camera_id: int
    detection_type: str
    confidence: float
    severity: str
    timestamp: str

class AlertResponse(BaseModel):
    id: int
    detection_id: int
    status: str
    severity: str
    message: str
    created_at: str

# Helper functions
def get_db():
    return sqlite3.connect(DB_PATH)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Initialize database on startup
@app.on_event("startup")
async def startup():
    init_simple_db()
    print("✅ Simple database initialized")
    print("✅ Demo users: admin/admin123, viewer/viewer123")

# Routes
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(login_data: LoginRequest):
    conn = get_db()
    cursor = conn.cursor()
    
    # Hash the provided password
    password_hash = hashlib.sha256(login_data.password.encode()).hexdigest()
    
    # Check credentials
    cursor.execute('SELECT id, username, email, role FROM users WHERE username = ? AND password_hash = ?',
                   (login_data.username, password_hash))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    token_data = {"sub": user[1], "role": user[3]}
    token = jwt.encode(token_data, SECRET_KEY, algorithm="HS256")
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user={
            "id": user[0],
            "username": user[1],
            "email": user[2],
            "role": user[3]
        }
    )

@app.get("/api/cameras", response_model=List[CameraResponse])
async def list_cameras():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, camera_id, location, status FROM cameras')
    cameras = cursor.fetchall()
    conn.close()
    
    return [
        CameraResponse(
            id=cam[0],
            name=cam[1],
            camera_id=cam[2],
            location=cam[3],
            status=cam[4]
        )
        for cam in cameras
    ]

@app.get("/api/detections", response_model=List[DetectionResponse])
async def list_detections():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, camera_id, detection_type, confidence, severity, timestamp FROM detections ORDER BY timestamp DESC LIMIT 50')
    detections = cursor.fetchall()
    conn.close()
    
    return [
        DetectionResponse(
            id=det[0],
            camera_id=det[1],
            detection_type=det[2],
            confidence=det[3],
            severity=det[4],
            timestamp=det[5]
        )
        for det in detections
    ]

@app.get("/api/alerts", response_model=List[AlertResponse])
async def list_alerts():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, detection_id, status, severity, message, created_at FROM alerts ORDER BY created_at DESC LIMIT 50')
    alerts = cursor.fetchall()
    conn.close()
    
    return [
        AlertResponse(
            id=alert[0],
            detection_id=alert[1],
            status=alert[2],
            severity=alert[3],
            message=alert[4] or "",
            created_at=alert[5]
        )
        for alert in alerts
    ]

@app.get("/api/system/status")
async def system_status():
    return {
        "status": "healthy",
        "active_cameras": 3,
        "total_detections": 0,
        "active_alerts": 0,
        "database_status": "healthy",
        "model_status": "loaded"
    }

@app.post("/api/detection/demo")
async def create_demo_detection():
    """Create a demo detection for testing."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Insert demo detection
    cursor.execute('''
        INSERT INTO detections (camera_id, detection_type, confidence, severity)
        VALUES (1, 'Fighting', 0.85, 'high')
    ''')
    detection_id = cursor.lastrowid
    
    # Create alert if confidence is high
    cursor.execute('''
        INSERT INTO alerts (detection_id, severity, message)
        VALUES (?, 'high', 'Crime detected: Fighting with 85% confidence')
    ''', (detection_id,))
    
    conn.commit()
    conn.close()
    
    return {"message": "Demo detection created", "detection_id": detection_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
