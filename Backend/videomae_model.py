"""
VideoMAE Crime Detection Model Implementation

This module implements the VideoMAE-based crime detection model that inherits
from BaseDetectionModel. It handles model loading, video frame buffering,
preprocessing, and inference for the VideoMAE model.
"""

import torch
import numpy as np
import cv2
from typing import Dict, Any, List, Optional
import logging
from collections import deque
import threading
import time
from base_model import BaseDetectionModel
from config import settings
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoMAEModel(BaseDetectionModel):
    """
    VideoMAE implementation of the crime detection model with real-time processing.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the VideoMAE crime detection model.
        
        Args:
            model_path: Path to the VideoMAE model directory.
                       Defaults to settings.VIDEOMAE_MODEL_PATH
        """
        self.model_path = model_path or settings.VIDEOMAE_MODEL_PATH
        self.model = None
        self.image_processor = None
        
        if settings.PYTORCH_DEVICE == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(settings.PYTORCH_DEVICE)
            
        logger.info(f"VideoMAE device set to: {self.device}")

        # Crime classes from the UCF-Crime dataset
        self.crime_classes = [
            'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
            'Explosion', 'Fighting', 'Normal_Videos_event', 'RoadAccidents',
            'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
        ]
        self.reverse_mapping = {i: label for i, label in enumerate(self.crime_classes)}
        
        # Model parameters
        self.input_size = (224, 224)
        self.num_frames = 16
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Real-time processing optimizations
        self.frame_buffer = deque(maxlen=self.num_frames)
        self.processing_lock = threading.Lock()
        self.last_prediction_time = 0
        self.min_prediction_interval = 0.1  # Minimum 100ms between predictions
        
        # Performance metrics
        self.inference_times = deque(maxlen=100)  # Keep last 100 inference times
        self.frame_count = 0
        self.prediction_count = 0
        
        # Accuracy improvements
        self.prediction_history = deque(maxlen=5)  # Keep last 5 predictions for smoothing
        self.confidence_boost_factor = 1.2  # Boost confidence for repeated detections
        self.min_consecutive_detections = 2  # Require 2 consecutive detections to confirm

    def load_model(self) -> bool:
        """
        Load the VideoMAE model from the specified path.
        """
        try:
            logger.info(f"Loading VideoMAE model from {self.model_path}")
            
            # Load model with optimizations
            self.model = VideoMAEForVideoClassification.from_pretrained(
                self.model_path,
                id2label=self.reverse_mapping,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            ).to(self.device)
            
            # Load image processor for proper preprocessing
            self.image_processor = VideoMAEImageProcessor.from_pretrained(self.model_path)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Enable optimizations for inference
            if hasattr(torch, 'compile') and self.device.type == 'cuda':
                try:
                    self.model = torch.compile(self.model, mode='max-autotune')
                    logger.info("Model compiled with torch.compile for CUDA optimization")
                except Exception as compile_err:
                    logger.warning(f"Failed to compile model: {compile_err}")
            
            logger.info(f"VideoMAE model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Error loading VideoMAE model: {str(e)}")
            return False

    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a list of frames for VideoMAE model input using the proper format.
        
        Args:
            frames: List of numpy arrays (BGR format) representing video frames
            
        Returns:
            torch.Tensor: Preprocessed tensor in format [batch_size, num_channels, num_frames, height, width]
        """
        try:
            # Use the image processor if available for proper preprocessing
            if self.image_processor:
                # Convert frames to RGB PIL Images (required by VideoMAEImageProcessor)
                from PIL import Image
                pil_frames = []
                for frame in frames:
                    # Enhance frame quality before processing
                    enhanced_frame = self._enhance_frame_quality(frame)
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                    # Resize to target size with high-quality interpolation
                    rgb_frame = cv2.resize(rgb_frame, self.input_size, interpolation=cv2.INTER_LANCZOS4)
                    # Convert to PIL Image
                    pil_image = Image.fromarray(rgb_frame)
                    pil_frames.append(pil_image)
                
                # Use the image processor for proper normalization
                # Pass list of PIL Images
                processed = self.image_processor(
                    pil_frames,
                    return_tensors="pt"
                )
                
                # Extract the pixel values tensor
                video_tensor = processed.pixel_values
                
                # Move to device
                return video_tensor.to(self.device)
            
            # Fallback manual preprocessing
            processed_frames = []
            for frame in frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.input_size)
                processed_frames.append(frame)
            
            # Stack frames: (num_frames, height, width, channels)
            frames_array = np.stack(processed_frames, axis=0)
            
            # Convert to tensor and normalize
            video_tensor = torch.tensor(frames_array, dtype=torch.float32)
            
            # Normalize to [0, 1] and then apply ImageNet normalization
            video_tensor = video_tensor / 255.0
            
            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
            
            # Rearrange dimensions: (num_frames, height, width, channels) -> (num_channels, num_frames, height, width)
            video_tensor = video_tensor.permute(3, 0, 1, 2)
            
            # Add batch dimension and normalize
            video_tensor = video_tensor.unsqueeze(0)  # [1, channels, frames, height, width]
            video_tensor = (video_tensor - mean) / std
            
            return video_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing frames: {e}")
            raise

    def predict_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a single frame and predict when buffer is full.
        
        Args:
            frame: Single video frame as numpy array (BGR format)
            
        Returns:
            Dict containing prediction results, or None if still buffering
        """
        self.frame_count += 1
        
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        # Check if we have enough frames for prediction
        if len(self.frame_buffer) < self.num_frames:
            return {
                "crime_type": "Normal_Videos_event",
                "confidence": 0.0,
                "severity": "low",
                "is_crime": False,
                "status": "Buffering frames",
                "frames_collected": len(self.frame_buffer),
                "frames_needed": self.num_frames
            }
        
        # Rate limiting: check if enough time has passed since last prediction
        current_time = time.time()
        if current_time - self.last_prediction_time < self.min_prediction_interval:
            return {
                "crime_type": "Normal_Videos_event",
                "confidence": 0.0,
                "severity": "low",
                "is_crime": False,
                "status": "Rate limited",
                "frames_collected": len(self.frame_buffer)
            }
        
        # Ensure thread-safe processing
        if not self.processing_lock.acquire(blocking=False):
            return {
                "crime_type": "Normal_Videos_event",
                "confidence": 0.0,
                "severity": "low",
                "is_crime": False,
                "status": "Processing busy",
                "frames_collected": len(self.frame_buffer)
            }
        
        try:
            # Make a copy of the buffer for processing
            frames_to_predict = list(self.frame_buffer)
            
            # Start timing
            start_time = time.time()
            
            # Preprocess frames
            input_tensor = self.preprocess_frames(frames_to_predict)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            # Process outputs
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label_idx = torch.argmax(probs, dim=-1).item()
            confidence = float(probs[0, predicted_label_idx])
            predicted_crime = self.reverse_mapping[predicted_label_idx]
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.inference_times.append(inference_time)
            self.prediction_count += 1
            self.last_prediction_time = current_time
            
            # Get all prediction probabilities
            all_predictions = {
                self.crime_classes[i]: float(probs[0, i]) 
                for i in range(len(self.crime_classes))
            }
            
            # Apply temporal smoothing and accuracy improvements
            smoothed_result = self._apply_temporal_smoothing(
                predicted_crime, 
                confidence, 
                all_predictions
            )
            predicted_crime = smoothed_result['crime_type']
            confidence = smoothed_result['confidence']
            
            # Determine if crime was detected with improved logic
            is_crime_detected = (
                confidence >= self.confidence_threshold and 
                predicted_crime != 'Normal_Videos_event' and
                smoothed_result['is_confident']
            )
            
            # Calculate average inference time
            avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
            
            result = {
                "crime_type": predicted_crime,
                "confidence": confidence,
                "severity": self._determine_severity(predicted_crime, confidence),
                "is_crime": is_crime_detected,
                "all_predictions": all_predictions,
                "inference_time_ms": inference_time,
                "avg_inference_time_ms": avg_inference_time,
                "fps": 1000.0 / avg_inference_time if avg_inference_time > 0 else 0,
                "frames_processed": self.frame_count,
                "predictions_made": self.prediction_count,
                "status": "Prediction completed"
            }
            
            # Log prediction results
            logger.info(f"VideoMAE Prediction: {predicted_crime} (confidence: {confidence:.3f}, "
                       f"inference_time: {inference_time:.1f}ms, fps: {result['fps']:.1f})")
            
            return result

        except Exception as e:
            logger.error(f"VideoMAE prediction error: {str(e)}")
            return {
                "crime_type": "Normal_Videos_event",
                "confidence": 0.0,
                "severity": "low",
                "is_crime": False,
                "error": f"Prediction failed: {str(e)}",
                "status": "Error"
            }
        finally:
            self.processing_lock.release()

    def _enhance_frame_quality(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame quality for better model accuracy.
        
        Applies:
        1. Denoising
        2. Contrast enhancement (CLAHE)
        3. Sharpening
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Enhanced frame
        """
        try:
            # Apply denoising
            denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Apply subtle sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) / 9.0
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Blend original and sharpened (70% sharpened, 30% original)
            result = cv2.addWeighted(sharpened, 0.7, enhanced, 0.3, 0)
            
            return result
            
        except Exception as e:
            logger.warning(f"Frame enhancement failed: {e}, using original frame")
            return frame
    
    def _apply_temporal_smoothing(self, crime_type: str, confidence: float, 
                                   all_predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Apply temporal smoothing to reduce false positives and improve accuracy.
        
        This method:
        1. Tracks prediction history
        2. Boosts confidence for repeated detections
        3. Requires consecutive detections for confirmation
        4. Filters out noise and inconsistent predictions
        
        Args:
            crime_type: Predicted crime type
            confidence: Raw confidence score
            all_predictions: All class probabilities
            
        Returns:
            Dict with smoothed crime_type, confidence, and is_confident flag
        """
        # Add current prediction to history
        self.prediction_history.append({
            'crime_type': crime_type,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # If we don't have enough history, return as-is but mark as not confident
        if len(self.prediction_history) < self.min_consecutive_detections:
            return {
                'crime_type': crime_type,
                'confidence': confidence,
                'is_confident': False
            }
        
        # Count consecutive detections of the same crime type
        recent_predictions = list(self.prediction_history)[-self.min_consecutive_detections:]
        same_crime_count = sum(1 for p in recent_predictions if p['crime_type'] == crime_type)
        
        # Calculate average confidence for this crime type in recent history
        same_crime_confidences = [p['confidence'] for p in recent_predictions 
                                  if p['crime_type'] == crime_type]
        avg_confidence = sum(same_crime_confidences) / len(same_crime_confidences) if same_crime_confidences else confidence
        
        # Boost confidence if we have consecutive detections
        if same_crime_count >= self.min_consecutive_detections:
            # Boost confidence but cap at 0.99
            boosted_confidence = min(0.99, confidence * self.confidence_boost_factor)
            
            # Use average confidence if it's higher (more stable)
            final_confidence = max(boosted_confidence, avg_confidence)
            
            return {
                'crime_type': crime_type,
                'confidence': final_confidence,
                'is_confident': True
            }
        
        # Check if there's a dominant crime type in recent history
        crime_counts = {}
        for p in recent_predictions:
            crime_counts[p['crime_type']] = crime_counts.get(p['crime_type'], 0) + 1
        
        # Find the most common crime type
        most_common_crime = max(crime_counts, key=crime_counts.get)
        most_common_count = crime_counts[most_common_crime]
        
        # If the most common crime appears in majority of recent predictions
        if most_common_count >= len(recent_predictions) * 0.6:  # 60% threshold
            # Use the most common crime with averaged confidence
            common_crime_confidences = [p['confidence'] for p in recent_predictions 
                                       if p['crime_type'] == most_common_crime]
            avg_common_confidence = sum(common_crime_confidences) / len(common_crime_confidences)
            
            return {
                'crime_type': most_common_crime,
                'confidence': avg_common_confidence,
                'is_confident': True
            }
        
        # Not enough consistent detections - return current but mark as not confident
        return {
            'crime_type': crime_type,
            'confidence': confidence * 0.8,  # Reduce confidence for inconsistent predictions
            'is_confident': False
        }
    
    def _determine_severity(self, crime_type: str, confidence: float) -> str:
        """
        Determine severity level based on crime type and confidence.
        """
        if crime_type in ['Shooting', 'Explosion', 'Arson', 'Assault']:
            return "critical" if confidence > 0.8 else "high"
        elif crime_type in ['Robbery', 'Fighting', 'Arrest', 'Burglary', 'RoadAccidents']:
            return "high" if confidence > 0.8 else "medium"
        elif crime_type in ['Stealing', 'Shoplifting', 'Vandalism', 'Abuse']:
            return "medium" if confidence > 0.7 else "low"
        elif crime_type == 'Normal_Videos_event':
            return "low"
        else:
            return "low"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        """
        avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
        return {
            "frames_processed": self.frame_count,
            "predictions_made": self.prediction_count,
            "avg_inference_time_ms": avg_inference_time,
            "fps": 1000.0 / avg_inference_time if avg_inference_time > 0 else 0,
            "buffer_size": len(self.frame_buffer),
            "device": str(self.device)
        }
    
    def reset_buffer(self):
        """
        Reset the frame buffer.
        """
        self.frame_buffer.clear()
        logger.info("Frame buffer reset")
    
    def set_confidence_threshold(self, threshold: float):
        """
        Update the confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold updated to {self.confidence_threshold}")

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        raise NotImplementedError("VideoMAEModel works on frame sequences, not single images.")

