import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Dict, Optional
import os
import logging
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrimeDetectionModel:
    """Crime Detection ML Model based on UCF Crime Dataset."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        # Correct crime classes based on the trained model
        self.crime_classes = [
            'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
            'Explosion', 'Fighting', 'NormalVideos', 'RoadAccidents'
        ]
        self.input_shape = (48, 48, 1)
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
    def load_model(self) -> bool:
        """Load the trained crime detection model."""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file not found at {self.model_path}. A new untrained model will be created.")
                self._create_default_model()
                return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
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
        logger.info("Default model created successfully")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model prediction."""
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
            image = image.astype(np.float32) / 255.0
            image = image.reshape(1, self.input_shape[0], self.input_shape[1], 1)
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def predict_crime(self, image: np.ndarray) -> Dict:
        """Predict crime type from image."""
        if self.model is None and not self.load_model():
            return {
                "crime_type": "NormalVideos", "confidence": 0.0, "severity": "low",
                "is_crime": False, "message": "Model not available."
            }
        
        try:
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                raise ValueError("Image preprocessing failed")

            predictions = self.model.predict(processed_image, verbose=0)[0]
            predicted_class_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_class_idx])
            predicted_crime = self.crime_classes[predicted_class_idx]
            
            severity = self._determine_severity(predicted_crime, confidence)
            is_crime_detected = confidence >= self.confidence_threshold and predicted_crime != 'NormalVideos'
            
            return {
                "crime_type": predicted_crime,
                "confidence": confidence,
                "severity": severity,
                "is_crime": is_crime_detected,
                "all_predictions": {self.crime_classes[i]: float(predictions[i]) for i in range(len(self.crime_classes))}
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                "crime_type": "NormalVideos", "confidence": 0.0, "severity": "low",
                "is_crime": False, "message": f"Prediction failed: {str(e)}"
            }

    def _determine_severity(self, crime_type: str, confidence: float) -> str:
        """Determine severity level based on the corrected crime classes."""
        if crime_type in ['Explosion', 'Arson', 'Assault', 'RoadAccidents']:
            return "critical" if confidence > 0.8 else "high"
        elif crime_type in ['Fighting', 'Arrest', 'Burglary']:
            return "high" if confidence > 0.8 else "medium"
        elif crime_type in ['Abuse']:
            return "medium" if confidence > 0.8 else "low"
        else: # NormalVideos
            return "low"

    def predict_frame(self, frame: np.ndarray) -> Dict:
        """Alias for predict_crime for compatibility."""
        return self.predict_crime(frame)

# Global model instance
crime_model = CrimeDetectionModel()

def get_crime_model() -> CrimeDetectionModel:
    """Get the global crime detection model instance."""
    return crime_model
