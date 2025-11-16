"""
Base Detection Model Abstract Class

This module defines the abstract base class for all crime detection models.
It provides a common interface that both PyTorch and TensorFlow implementations
must follow, enabling seamless model switching and backward compatibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseDetectionModel(ABC):
    """
    Abstract base class for crime detection models.
    
    This class defines the interface that all crime detection model implementations
    must follow. It ensures consistency across different model types (PyTorch, TensorFlow)
    and enables the strategy pattern for runtime model selection.
    
    Attributes:
        model: The underlying ML model instance (type varies by implementation)
        crime_classes: List of crime class labels supported by the model
    """
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the model from disk and prepare it for inference.
        
        This method should:
        - Load the model file from the configured path
        - Initialize the model architecture
        - Move the model to the appropriate device (GPU/CPU)
        - Set the model to evaluation mode
        - Handle any loading errors gracefully
        
        Returns:
            bool: True if model loaded successfully, False otherwise
            
        Raises:
            FileNotFoundError: If model file does not exist
            RuntimeError: If model loading fails due to incompatibility
        """
        pass
    
    @abstractmethod
    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Predict crime type from a video frame or image.
        
        This method performs the complete prediction pipeline:
        - Preprocess the input frame
        - Run model inference
        - Post-process the output
        - Determine severity level
        - Format the result
        
        Args:
            frame: Input image as numpy array in BGR format (OpenCV default)
                   Shape should be (height, width, 3)
        
        Returns:
            Dict containing:
                - crime_type (str): Predicted crime class label
                - confidence (float): Prediction confidence score (0.0 to 1.0)
                - severity (str): Severity level ("low", "medium", "high", "critical")
                - is_crime (bool): True if detection is a valid crime event
                - all_predictions (Dict[str, float]): All class probabilities
                
        Example:
            {
                "crime_type": "Assault",
                "confidence": 0.85,
                "severity": "high",
                "is_crime": True,
                "all_predictions": {
                    "Assault": 0.85,
                    "Theft": 0.10,
                    "NormalVideos": 0.05
                }
            }
            
        Raises:
            ValueError: If frame format is invalid
            RuntimeError: If inference fails
        """
        pass
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> Any:
        """
        Preprocess image for model input.
        
        This method transforms the raw input image into the format expected
        by the specific model implementation. Preprocessing steps typically include:
        - Color space conversion (BGR to RGB)
        - Resizing to model input dimensions
        - Pixel normalization
        - Data type conversion
        - Adding batch dimension
        
        Args:
            image: Input image as numpy array in BGR format
                   Shape: (height, width, 3)
        
        Returns:
            Preprocessed input in model-specific format:
            - PyTorch: torch.Tensor
            - TensorFlow: numpy array or tf.Tensor
            
        Raises:
            ValueError: If image dimensions or format are invalid
        """
        pass
    
    @abstractmethod
    def _determine_severity(self, crime_type: str, confidence: float) -> str:
        """
        Determine severity level based on crime type and confidence.
        
        This method classifies the detection into severity levels to help
        prioritize security responses. The severity is determined by both
        the type of crime detected and the confidence of the prediction.
        
        Args:
            crime_type: The predicted crime class label
            confidence: Prediction confidence score (0.0 to 1.0)
        
        Returns:
            str: Severity level, one of:
                - "low": Minor incidents or low confidence detections
                - "medium": Moderate threats requiring attention
                - "high": Serious crimes requiring immediate response
                - "critical": Life-threatening situations requiring urgent action
                
        Example:
            - "Shooting" with 0.9 confidence -> "critical"
            - "Shoplifting" with 0.7 confidence -> "medium"
            - "NormalVideos" with any confidence -> "low"
        """
        pass
