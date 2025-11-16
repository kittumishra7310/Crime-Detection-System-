"""
Model Factory Pattern Implementation

This module implements the factory pattern for creating crime detection models.
It provides a unified interface for instantiating different model types (PyTorch
or TensorFlow) based on configuration, with automatic fallback support.
"""

import logging
from typing import Optional
from base_model import BaseDetectionModel
from pytorch_model import PyTorchCrimeModel
from videomae_model import VideoMAEModel
from tensorflow_model import TensorFlowCrimeModel
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating crime detection model instances.
    
    This class implements the factory pattern to abstract model creation
    and provide automatic fallback mechanisms when models fail to load.
    """
    
    @staticmethod
    def create_model(model_type: str, config=None) -> Optional[BaseDetectionModel]:
        """
        Create and load the specified model type.
        
        Args:
            model_type: Type of model to create ('pytorch' or 'tensorflow')
            config: Optional configuration object (defaults to global settings)
        
        Returns:
            BaseDetectionModel: Loaded model instance, or None if creation fails
        
        Raises:
            ValueError: If model_type is invalid
        """
        if config is None:
            config = settings
        
        # Normalize model type to lowercase
        model_type = model_type.lower().strip()
        
        # Validate model type
        if model_type not in ['pytorch', 'tensorflow', 'videomae']:
            logger.error(f"Invalid model type: {model_type}")
            raise ValueError(f"Invalid model type '{model_type}'. Must be 'pytorch', 'tensorflow', or 'videomae'")
        
        logger.info(f"Creating {model_type} model...")
        
        try:
            # Create the appropriate model instance
            if model_type == 'pytorch':
                model = PyTorchCrimeModel(model_path=config.PYTORCH_MODEL_PATH)
            elif model_type == 'videomae':
                model = VideoMAEModel(model_path=config.VIDEOMAE_MODEL_PATH)
            else:  # tensorflow
                model = TensorFlowCrimeModel(model_path=config.TENSORFLOW_MODEL_PATH)
            
            # Attempt to load the model
            if model.load_model():
                logger.info(f"{model_type.capitalize()} model created and loaded successfully")
                return model
            else:
                logger.error(f"Failed to load {model_type} model")
                return None

        except Exception as e:
            logger.error(f"Error creating {model_type} model: {str(e)}")
            return None
    
    @staticmethod
    def create_with_fallback(config=None) -> BaseDetectionModel:
        """
        Create model with automatic fallback mechanism.
        
        This method attempts to create the model specified in configuration.
        If that fails, it automatically falls back to the alternative model type.
        If both fail, it returns a TensorFlow model with default architecture.
        
        Fallback order:
        1. Try configured model type (from config.MODEL_TYPE)
        2. If PyTorch was primary and failed, try TensorFlow
        3. If TensorFlow was primary and failed, try PyTorch
        4. If both fail, return TensorFlow with default model
        
        Args:
            config: Optional configuration object (defaults to global settings)
        
        Returns:
            BaseDetectionModel: A loaded model instance (never None)
        """
        if config is None:
            config = settings
        
        primary_model_type = config.MODEL_TYPE.lower().strip()
        logger.info(f"Attempting to create primary model: {primary_model_type}")
        
        # Try to create the primary model
        try:
            primary_model = ModelFactory.create_model(primary_model_type, config)
            if primary_model is not None:
                logger.info(f"Successfully using {primary_model_type} model")
                return primary_model
            else:
                logger.warning(f"Primary model ({primary_model_type}) failed to load")
        except ValueError as e:
            logger.error(f"Invalid primary model type: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with primary model: {str(e)}")
        
        # Determine fallback model type
        fallback_model_type = 'tensorflow' if primary_model_type == 'pytorch' else 'pytorch'
        logger.info(f"Attempting fallback to {fallback_model_type} model")

        # Try fallback model
        try:
            fallback_model = ModelFactory.create_model(fallback_model_type, config)
            if fallback_model is not None:
                logger.warning(f"Using fallback model: {fallback_model_type}")
                return fallback_model
            else:
                logger.error(f"Fallback model ({fallback_model_type}) also failed to load")
        except Exception as e:
            logger.error(f"Error with fallback model: {str(e)}")
        
        # Last resort: create TensorFlow model with default architecture
        logger.critical("Both models failed. Creating TensorFlow model with default architecture")
        try:
            default_model = TensorFlowCrimeModel(model_path=config.TENSORFLOW_MODEL_PATH)
            default_model.load_model()  # This will create a default model if file doesn't exist
            logger.info("Default TensorFlow model created as last resort")
            return default_model
        except Exception as e:
            logger.critical(f"Failed to create even default model: {str(e)}")
            # This should never happen, but if it does, create a minimal model
            default_model = TensorFlowCrimeModel(model_path=config.TENSORFLOW_MODEL_PATH)
            default_model._create_default_model()
            return default_model
    
    @staticmethod
    def get_model_info(model: BaseDetectionModel) -> dict:
        """
        Get information about the loaded model.
        
        Args:
            model: The model instance to inspect
        
        Returns:
            dict: Model information including type, path, and status
        """
        model_type = "unknown"
        if isinstance(model, PyTorchCrimeModel):
            model_type = "pytorch"
        elif isinstance(model, TensorFlowCrimeModel):
            model_type = "tensorflow"
        elif isinstance(model, VideoMAEModel):
            model_type = "videomae"
        
        return {
            "type": model_type,
            "model_path": model.model_path,
            "is_loaded": model.model is not None,
            "crime_classes": model.crime_classes,
            "confidence_threshold": model.confidence_threshold
        }
