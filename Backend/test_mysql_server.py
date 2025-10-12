#!/usr/bin/env python3
"""
Simple test server to verify MySQL connection and basic functionality
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = "mysql+pymysql://root:Kittu%40123@localhost/surveillance_db"

# Create engine and session
try:
    engine = create_engine(DATABASE_URL, echo=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("✅ Database engine created successfully")
except Exception as e:
    logger.error(f"❌ Failed to create database engine: {e}")
    raise

app = FastAPI(
    title="Surveillance System Test API",
    description="Test API for MySQL connection and basic functionality",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Response models
class HealthResponse(BaseModel):
    status: str
    database: str
    message: str

class DatabaseTestResponse(BaseModel):
    tables: list
    users_count: int
    cameras_count: int

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="online",
        database="mysql",
        message="Surveillance System Test API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()
        
        return HealthResponse(
            status="healthy",
            database="connected",
            message="All systems operational"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@app.get("/test-db", response_model=DatabaseTestResponse)
async def test_database(db: Session = Depends(get_db)):
    """Test database operations"""
    try:
        # Get table names
        tables_result = db.execute(text("SHOW TABLES"))
        tables = [row[0] for row in tables_result.fetchall()]
        
        # Count users
        users_result = db.execute(text("SELECT COUNT(*) FROM users"))
        users_count = users_result.fetchone()[0]
        
        # Count cameras
        cameras_result = db.execute(text("SELECT COUNT(*) FROM cameras"))
        cameras_count = cameras_result.fetchone()[0]
        
        return DatabaseTestResponse(
            tables=tables,
            users_count=users_count,
            cameras_count=cameras_count
        )
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database test failed: {str(e)}")

@app.get("/users")
async def list_users(db: Session = Depends(get_db)):
    """List all users"""
    try:
        result = db.execute(text("SELECT id, username, email, role, is_active FROM users"))
        users = []
        for row in result.fetchall():
            users.append({
                "id": row[0],
                "username": row[1],
                "email": row[2],
                "role": row[3],
                "is_active": bool(row[4])
            })
        return {"users": users}
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list users: {str(e)}")

@app.get("/cameras")
async def list_cameras(db: Session = Depends(get_db)):
    """List all cameras"""
    try:
        result = db.execute(text("SELECT id, name, camera_id, location, status FROM cameras"))
        cameras = []
        for row in result.fetchall():
            cameras.append({
                "id": row[0],
                "name": row[1],
                "camera_id": row[2],
                "location": row[3],
                "status": row[4]
            })
        return {"cameras": cameras}
    except Exception as e:
        logger.error(f"Failed to list cameras: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list cameras: {str(e)}")

if __name__ == "__main__":
    logger.info("🚀 Starting Surveillance System Test Server...")
    logger.info("📊 API Documentation: http://localhost:8000/docs")
    logger.info("🔍 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "test_mysql_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
