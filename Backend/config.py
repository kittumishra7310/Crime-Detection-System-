from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "mysql+pymysql://root:Kittu%40123@localhost/surveillance_db"
    
    # JWT
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Clerk Authentication
    NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: Optional[str] = None
    CLERK_SECRET_KEY: Optional[str] = None
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # File Storage
    UPLOAD_DIR: str = "uploads"
    MODEL_PATH: str = "models/crime_detection_model.h5"
    
    # Model Configuration
    MODEL_TYPE: str = "videomae"  # Options: "pytorch", "tensorflow", "videomae"
    PYTORCH_MODEL_PATH: str = "Backend/models/optimizer.pt"
    TENSORFLOW_MODEL_PATH: str = "Backend/models/crime_detection_model.h5"
    VIDEOMAE_MODEL_PATH: str = "Backend/models/videomae-large-finetuned-UCF-Crime"
    
    # PyTorch Specific Configuration
    PYTORCH_DEVICE: str = "auto"  # Options: "auto", "cuda", "cpu"
    PYTORCH_INPUT_SIZE: tuple = (224, 224)  # Input dimensions for PyTorch model
    PYTORCH_NORMALIZE_MEAN: list = [0.485, 0.456, 0.406]  # ImageNet normalization mean
    PYTORCH_NORMALIZE_STD: list = [0.229, 0.224, 0.225]   # ImageNet normalization std
    
    # Detection Settings
    CONFIDENCE_THRESHOLD: float = 0.7
    ALERT_THRESHOLD: float = 0.8
    
    # CORS
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    @field_validator('MODEL_TYPE')
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate that MODEL_TYPE is 'pytorch', 'tensorflow', or 'videomae'."""
        v = v.lower().strip()
        if v not in ['pytorch', 'tensorflow', 'videomae']:
            logger.warning(f"Invalid MODEL_TYPE '{v}', defaulting to 'tensorflow'")
            return 'tensorflow'
        return v
    
    @field_validator('PYTORCH_DEVICE')
    @classmethod
    def validate_pytorch_device(cls, v: str) -> str:
        """Validate that PYTORCH_DEVICE is valid."""
        v = v.lower().strip()
        if v not in ['auto', 'cuda', 'cpu']:
            logger.warning(f"Invalid PYTORCH_DEVICE '{v}', defaulting to 'auto'")
            return 'auto'
        return v
    
    @field_validator('CONFIDENCE_THRESHOLD', 'ALERT_THRESHOLD')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate that thresholds are between 0 and 1."""
        if not 0 <= v <= 1:
            logger.warning(f"Threshold {v} out of range [0, 1], clamping")
            return max(0.0, min(1.0, v))
        return v
    
    class Config:
        env_file = ".env"

settings = Settings()

# Create upload directory if it doesn't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
