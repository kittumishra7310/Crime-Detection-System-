"""
PyTorch Crime Detection Model Implementation

This module implements the PyTorch-based crime detection model that inherits
from BaseDetectionModel. It handles model loading, preprocessing, inference,
and post-processing for PyTorch models.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, Any
import logging
import os
from base_model import BaseDetectionModel
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyTorchCrimeModel(BaseDetectionModel):
    """
    PyTorch implementation of the crime detection model.
    
    This class provides PyTorch-specific implementation of the crime detection
    interface, including GPU acceleration support and optimized inference.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the PyTorch crime detection model.
        
        Args:
            model_path: Path to the PyTorch model file (.pt or .pth)
                       Defaults to settings.PYTORCH_MODEL_PATH
        """
        self.model_path = model_path or settings.PYTORCH_MODEL_PATH
        self.model = None
        
        # Device selection logic
        if settings.PYTORCH_DEVICE == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif settings.PYTORCH_DEVICE == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        
        logger.info(f"PyTorch device set to: {self.device}")
        
        # Enable cuDNN benchmark for performance optimization
        # This optimizes convolution algorithms for consistent input sizes
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled for performance optimization")
        
        # Crime classes - will be updated based on model inspection
        self.crime_classes = [
            'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
            'Explosion', 'Fighting', 'NormalVideos', 'RoadAccidents',
            'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
        ]
        
        # Input configuration
        self.input_size = settings.PYTORCH_INPUT_SIZE
        self.normalize_mean = settings.PYTORCH_NORMALIZE_MEAN
        self.normalize_std = settings.PYTORCH_NORMALIZE_STD
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD

    def load_model(self) -> bool:
        """
        Load the PyTorch model from disk and prepare it for inference.
        
        This method loads the model file, moves it to the appropriate device
        (GPU or CPU), and sets it to evaluation mode for inference.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found at {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading PyTorch model from {self.model_path}")
            
            # Load the model
            # Using weights_only=False to support older PyTorch models
            # In production, consider using weights_only=True for security
            self.model = torch.load(self.model_path, map_location=self.device)
            
            # If the loaded object is a state dict, we need the model architecture
            if isinstance(self.model, dict):
                logger.error("Loaded a state dict instead of a complete model. Model architecture required.")
                return False
            
            # Move model to the appropriate device
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode (disables dropout, batch norm training behavior)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Model type: {type(self.model)}")
            
            # Perform a warm-up inference to initialize CUDA kernels
            if self.device.type == 'cuda':
                try:
                    dummy_input = torch.randn(1, 3, *self.input_size).to(self.device)
                    with torch.no_grad():
                        _ = self.model(dummy_input)
                    logger.info("Model warm-up completed")
                except Exception as e:
                    logger.warning(f"Model warm-up failed: {str(e)}")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {str(e)}")
            return False
        except RuntimeError as e:
            logger.error(f"Runtime error loading model: {str(e)}")
            # Try to fall back to CPU if CUDA error
            if self.device.type == 'cuda':
                logger.info("Attempting to load model on CPU instead")
                try:
                    self.device = torch.device('cpu')
                    self.model = torch.load(self.model_path, map_location=self.device)
                    if not isinstance(self.model, dict):
                        self.model = self.model.to(self.device)
                        self.model.eval()
                        logger.info("Model loaded successfully on CPU after CUDA failure")
                        return True
                except Exception as cpu_error:
                    logger.error(f"Failed to load on CPU as well: {str(cpu_error)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading model: {str(e)}")
            return False

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for PyTorch model input.
        
        This method performs the following transformations:
        1. Convert BGR to RGB (OpenCV uses BGR by default)
        2. Resize to model input dimensions
        3. Normalize pixel values
        4. Convert to PyTorch tensor
        5. Add batch dimension
        6. Move to appropriate device
        
        Args:
            image: Input image as numpy array in BGR format
                   Shape: (height, width, 3)
        
        Returns:
            torch.Tensor: Preprocessed tensor ready for model input
                         Shape: (1, 3, height, width)
        
        Raises:
            ValueError: If image format is invalid
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid image: empty or None")
            
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected 3-channel image, got shape {image.shape}")
            
            # Step 1: Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Step 2: Resize to model input dimensions
            image_resized = cv2.resize(image_rgb, self.input_size)
            
            # Step 3: Convert to float and normalize to [0, 1]
            image_float = image_resized.astype(np.float32) / 255.0
            
            # Step 4: Apply ImageNet normalization (or custom normalization)
            # Normalize each channel: (pixel - mean) / std
            for i in range(3):
                image_float[:, :, i] = (image_float[:, :, i] - self.normalize_mean[i]) / self.normalize_std[i]
            
            # Step 5: Convert to PyTorch tensor and change from HWC to CHW format
            # PyTorch expects (channels, height, width)
            image_tensor = torch.from_numpy(image_float).permute(2, 0, 1)
            
            # Step 6: Add batch dimension (1, C, H, W)
            image_tensor = image_tensor.unsqueeze(0)
            
            # Step 7: Move to appropriate device (GPU/CPU)
            image_tensor = image_tensor.to(self.device)
            
            return image_tensor
            
        except cv2.error as e:
            logger.error(f"OpenCV error during preprocessing: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Predict crime type from a video frame or image.
        
        Args:
            frame: Input image as numpy array in BGR format
        
        Returns:
            Dict containing prediction results with keys:
                - crime_type: Predicted crime class label
                - confidence: Prediction confidence score (0.0 to 1.0)
                - severity: Severity level
                - is_crime: True if valid crime detection
                - all_predictions: All class probabilities
        """
        # Check if model is loaded
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return {
                "crime_type": "NormalVideos",
                "confidence": 0.0,
                "severity": "low",
                "is_crime": False,
                "error": "Model not loaded"
            }
        
        try:
            # Preprocess the frame
            input_tensor = self.preprocess_image(frame)
            
            # Run inference with no gradient computation
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Apply softmax if not already applied
            if output.dim() == 2 and output.shape[1] == len(self.crime_classes):
                probabilities = torch.softmax(output, dim=1)
            else:
                probabilities = output
            
            # Move to CPU and convert to numpy
            probabilities = probabilities.cpu().numpy()[0]

            # Get predicted class and confidence
            predicted_class_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class_idx])
            
            # Map to crime class label
            if predicted_class_idx < len(self.crime_classes):
                predicted_crime = self.crime_classes[predicted_class_idx]
            else:
                logger.warning(f"Predicted index {predicted_class_idx} out of range")
                predicted_crime = "NormalVideos"
                confidence = 0.0
            
            # Determine severity
            severity = self._determine_severity(predicted_crime, confidence)
            
            # Determine if this is a valid crime detection
            is_crime_detected = (
                confidence >= self.confidence_threshold and 
                predicted_crime != 'NormalVideos'
            )
            
            # Create all predictions dictionary
            all_predictions = {
                self.crime_classes[i]: float(probabilities[i]) 
                for i in range(min(len(self.crime_classes), len(probabilities)))
            }
            
            return {
                "crime_type": predicted_crime,
                "confidence": confidence,
                "severity": severity,
                "is_crime": is_crime_detected,
                "all_predictions": all_predictions
            }

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory: {str(e)}")
            logger.info("Attempting to clear CUDA cache and retry on CPU")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Try to reload model on CPU
            self.device = torch.device('cpu')
            if self.load_model():
                logger.info("Model reloaded on CPU, retrying prediction")
                return self.predict_frame(frame)
            else:
                return {
                    "crime_type": "NormalVideos",
                    "confidence": 0.0,
                    "severity": "low",
                    "is_crime": False,
                    "error": "CUDA OOM and CPU fallback failed"
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
            logger.error(f"Prediction error: {str(e)}")
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
        # Critical severity crimes (life-threatening)
        if crime_type in ['Shooting', 'Explosion', 'Arson', 'Assault']:
            return "critical" if confidence > 0.8 else "high"
        
        # High severity crimes (serious threats)
        elif crime_type in ['Robbery', 'Fighting', 'Arrest', 'Burglary', 'RoadAccidents']:
            return "high" if confidence > 0.8 else "medium"
        
        # Medium severity crimes (property crimes, minor incidents)
        elif crime_type in ['Stealing', 'Shoplifting', 'Vandalism', 'Abuse']:
            return "medium" if confidence > 0.7 else "low"
        
        # Normal videos or low confidence
        else:
            return "low"
