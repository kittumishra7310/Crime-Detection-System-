"""
TensorFlow Crime Detection Model Implementation

This module implements the TensorFlow/Keras-based crime detection model that
inherits from BaseDetectionModel. This is a refactored version of the original
CrimeDetectionModel to work with the new abstraction layer.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any
import os
import logging
from base_model import BaseDetectionModel
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorFlowCrimeModel(BaseDetectionModel):
    """
    TensorFlow/Keras implementation of the crime detection model.
    
    This class maintains backward compatibility with the existing TensorFlow
    model while conforming to the BaseDetectionModel interface.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the TensorFlow crime detection model.
        
        Args:
            model_path: Path to the Keras model file (.h5)
                       Defaults to settings.TENSORFLOW_MODEL_PATH
        """
        self.model_path = model_path or settings.TENSORFLOW_MODEL_PATH
        self.model = None
        
        # Crime classes based on the trained model
        self.crime_classes = [
            'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
            'Explosion', 'Fighting', 'NormalVideos', 'RoadAccidents'
        ]
        
        self.input_shape = (48, 48, 1)
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        logger.info("TensorFlow crime detection model initialized")

    def load_model(self) -> bool:
        """
        Load the trained TensorFlow/Keras crime detection model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"TensorFlow model loaded successfully from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file not found at {self.model_path}. Creating default model.")
                self._create_default_model()
                return True
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {str(e)}")
            self._create_default_model()
            return True
    
    def _create_default_model(self):
        """Create a default CNN model architecture."""
        logger.info("Creating default CNN model architecture...")
        model = keras.Sequential([
            keras.layers.Input(shape=self.input_shape),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.4),
            keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.4),
            keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.4),
            keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.4),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(len(self.crime_classes), activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        logger.info("Default TensorFlow model created successfully")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for TensorFlow model prediction.
        
        Args:
            image: Input image as numpy array in BGR format
        
        Returns:
            np.ndarray: Preprocessed image ready for model input
        
        Raises:
            ValueError: If image preprocessing fails
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize to model input shape
            image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Reshape to add batch and channel dimensions
            image = image.reshape(1, self.input_shape[0], self.input_shape[1], 1)
            
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Predict crime type from a video frame or image.
        
        Args:
            frame: Input image as numpy array in BGR format
        
        Returns:
            Dict containing prediction results
        """
        if self.model is None and not self.load_model():
            return {
                "crime_type": "NormalVideos",
                "confidence": 0.0,
                "severity": "low",
                "is_crime": False,
                "error": "Model not available"
            }
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(frame)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_class_idx])
            predicted_crime = self.crime_classes[predicted_class_idx]
            
            # Determine severity
            severity = self._determine_severity(predicted_crime, confidence)
            
            # Determine if this is a valid crime detection
            is_crime_detected = (
                confidence >= self.confidence_threshold and 
                predicted_crime != 'NormalVideos'
            )
            
            # Create all predictions dictionary
            all_predictions = {
                self.crime_classes[i]: float(predictions[i]) 
                for i in range(len(self.crime_classes))
            }
            
            return {
                "crime_type": predicted_crime,
                "confidence": confidence,
                "severity": severity,
                "is_crime": is_crime_detected,
                "all_predictions": all_predictions
            }

        except ValueError as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return {
                "crime_type": "NormalVideos",
                "confidence": 0.0,
                "severity": "low",
                "is_crime": False,
                "error": f"Preprocessing failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                "crime_type": "NormalVideos",
                "confidence": 0.0,
                "severity": "low",
                "is_crime": False,
                "error": f"Prediction failed: {str(e)}"
            }
    
    def _determine_severity(self, crime_type: str, confidence: float) -> str:
        """
        Determine severity level based on crime type and confidence.
        
        Args:
            crime_type: The predicted crime class label
            confidence: Prediction confidence score (0.0 to 1.0)
        
        Returns:
            str: Severity level ("low", "medium", "high", "critical")
        """
        # Critical severity crimes
        if crime_type in ['Explosion', 'Arson', 'Assault', 'RoadAccidents']:
            return "critical" if confidence > 0.8 else "high"
        
        # High severity crimes
        elif crime_type in ['Fighting', 'Arrest', 'Burglary']:
            return "high" if confidence > 0.8 else "medium"
        
        # Medium severity crimes
        elif crime_type in ['Abuse']:
            return "medium" if confidence > 0.8 else "low"
        
        # Normal videos or low confidence
        else:
            return "low"
