#!/usr/bin/env python3
"""
Production server startup script for the surveillance system.
"""

import uvicorn
import sys
import os
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI server."""
    
    # Ensure required directories exist
    Path(settings.UPLOAD_DIR).mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("Starting Smart Surveillance System Backend...")
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"Model path: {settings.MODEL_PATH}")
    
    # Start server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
