"""
Crime Detection Model Interface

This module provides the main interface for crime detection models.
It uses the ModelFactory to create and manage model instances based on
configuration, supporting both PyTorch and TensorFlow implementations.
"""

import logging
from typing import Dict
from model_factory import ModelFactory
from base_model import BaseDetectionModel
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance - created using factory pattern with fallback
logger.info("Initializing crime detection model...")
try:
    crime_model = ModelFactory.create_with_fallback(settings)
except Exception as e:
    logger.critical(f"Failed to initialize crime detection model: {e}", exc_info=True)
    # Optionally, re-raise the exception or set crime_model to a dummy/None
    # For now, we'll let the application potentially crash if the model is critical
    # or handle it gracefully if possible.
    crime_model = None # Set to None to indicate failure

# Log which model type is active
model_info = ModelFactory.get_model_info(crime_model)
logger.info(f"Active model type: {model_info['type']}")
logger.info(f"Model path: {model_info['model_path']}")
logger.info(f"Model loaded: {model_info['is_loaded']}")
logger.info(f"Crime classes: {len(model_info['crime_classes'])} classes")

def get_crime_model() -> BaseDetectionModel:
    """
    Get the global crime detection model instance.
    
    This function provides backward compatibility with existing code
    while using the new model abstraction layer.
    
    Returns:
        BaseDetectionModel: The active crime detection model
    """
    return crime_model

def reload_model() -> BaseDetectionModel:
    """
    Reload the crime detection model based on current configuration.
    
    This function allows dynamic model reloading without restarting
    the application. Useful for switching between model types or
    updating model files.
    
    Returns:
        BaseDetectionModel: The newly loaded model instance
    """
    global crime_model
    logger.info("Reloading crime detection model...")
    crime_model = ModelFactory.create_with_fallback(settings)
    model_info = ModelFactory.get_model_info(crime_model)
    logger.info(f"Model reloaded: {model_info['type']}")
    return crime_model

def get_model_info() -> Dict:
    """
    Get information about the currently active model.
    
    Returns:
        Dict: Model information including type, path, and status
    """
    return ModelFactory.get_model_info(crime_model)
