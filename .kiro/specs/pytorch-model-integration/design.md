# Design Document: VideoMAE Model Integration

## Overview

This design document describes the integration of a VideoMAE-based crime detection model into the existing crime detection system. VideoMAE is a video transformer model that processes sequences of frames to understand temporal patterns in videos. The solution maintains backward compatibility with TensorFlow models while enabling the use of state-of-the-art video understanding models for improved detection capabilities. The design follows a strategy pattern to abstract model implementations and allow runtime model selection.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Live Detection Manager                    │
│                  (Existing - No Changes)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ predict_frame()
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Crime Model Interface (Updated)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  - model_type: str                                    │  │
│  │  - active_model: BaseDetectionModel                   │  │
│  │  + predict_frame(frame) -> Dict                       │  │
│  │  + load_model() -> bool                               │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ delegates to
                         ▼
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│ VideoMAEModel       │       │ TensorFlowCrimeModel│
│                     │       │                     │
│ - model: VideoMAE   │       │ - model: keras.Model│
│ - frame_buffer      │       │                     │
│ - device: str       │       │ + predict()         │
│ + predict()         │       │ + preprocess()      │
│ + preprocess()      │       │                     │
│ + buffer_frames()   │       │                     │
└─────────────────────┘       └─────────────────────┘
```

### Design Principles

1. **Strategy Pattern**: Use abstract base class to define common interface for all model implementations
2. **Backward Compatibility**: Existing code continues to work without modifications
3. **Configuration-Driven**: Model selection controlled via configuration file
4. **Graceful Degradation**: Automatic fallback to TensorFlow model if PyTorch model fails
5. **Performance Optimization**: GPU acceleration when available, efficient tensor operations

## Components and Interfaces

### 1. Base Detection Model (New Abstract Class)

**File**: `Backend/base_model.py`

```python
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class BaseDetectionModel(ABC):
    """Abstract base class for crime detection models."""
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model from disk."""
        pass
    
    @abstractmethod
    def predict_frame(self, frame: np.ndarray) -> Dict:
        """Predict crime type from a frame."""
        pass
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> any:
        """Preprocess image for model input."""
        pass
    
    @abstractmethod
    def _determine_severity(self, crime_type: str, confidence: float) -> str:
        """Determine severity level."""
        pass
```

### 2. VideoMAE Crime Model (New Implementation)

**File**: `Backend/videomae_model.py`

**Key Responsibilities**:
- Load VideoMAE model from HuggingFace format directory
- Handle GPU/CPU device placement
- Buffer frames for sequence-based inference
- Preprocess frame sequences for VideoMAE model input
- Execute inference and post-process results
- Map predictions to UCF-Crime classes

**Key Methods**:

```python
class VideoMAEModel(BaseDetectionModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.image_processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_buffer = deque(maxlen=16)  # Buffer for 16 frames
        self.num_frames = 16
        self.crime_classes = [...]  # UCF-Crime classes
        
    def load_model(self) -> bool:
        """Load VideoMAE model from HuggingFace directory."""
        
    def predict_frame(self, frame: np.ndarray) -> Dict:
        """
        Buffer frame and predict when 16 frames are collected.
        Returns:
        {
            "crime_type": str,
            "confidence": float,
            "severity": str,
            "is_crime": bool,
            "all_predictions": Dict[str, float],
            "status": str,  # buffering/prediction/rate_limited
            "frames_collected": int
        }
        """
        
    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Convert frame sequence to VideoMAE tensor format."""
```

**Preprocessing Pipeline**:
1. Buffer frames until 16 frames are collected
2. Convert BGR to RGB for each frame
3. Resize frames to 224x224 dimensions
4. Use VideoMAE image processor for normalization
5. Stack frames into tensor (batch, channels, frames, height, width)
6. Move to appropriate device (GPU/CPU)

### 3. TensorFlow Crime Model (Refactored)

**File**: `Backend/tensorflow_model.py`

**Changes**:
- Refactor existing `CrimeDetectionModel` to inherit from `BaseDetectionModel`
- Maintain all existing functionality
- Ensure consistent output format with PyTorch model

```python
class TensorFlowCrimeModel(BaseDetectionModel):
    # Existing implementation from ml_model.py
    # Refactored to inherit from BaseDetectionModel
```

### 4. Model Factory (New Component)

**File**: `Backend/model_factory.py`

**Responsibilities**:
- Create appropriate model instance based on configuration
- Handle model loading errors and fallback logic
- Provide singleton access to active model

```python
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, config: Settings) -> BaseDetectionModel:
        """
        Create and load the specified model type.
        
        Args:
            model_type: 'pytorch' or 'tensorflow'
            config: Application settings
            
        Returns:
            Loaded model instance
            
        Raises:
            ValueError: If model_type is invalid
        """
        
    @staticmethod
    def create_with_fallback(config: Settings) -> BaseDetectionModel:
        """
        Create model with automatic fallback.
        Try PyTorch first, fall back to TensorFlow if it fails.
        """
```

### 5. Updated ML Model Interface (Modified)

**File**: `Backend/ml_model.py`

**Changes**:
- Import and use `ModelFactory`
- Maintain backward compatibility with existing `get_crime_model()` function
- Delegate to appropriate model implementation

```python
from model_factory import ModelFactory
from config import settings

# Global model instance
crime_model = ModelFactory.create_with_fallback(settings)

def get_crime_model() -> BaseDetectionModel:
    """Get the global crime detection model instance."""
    return crime_model
```

### 6. Configuration Updates

**File**: `Backend/config.py`

**New Settings**:
```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Model Configuration
    MODEL_TYPE: str = "videomae"  # Options: "videomae", "pytorch", "tensorflow"
    VIDEOMAE_MODEL_PATH: str = "models/videomae-large-finetuned-UCF-Crime"
    PYTORCH_MODEL_PATH: str = "models/optimizer.pt"  # Legacy, not used
    TENSORFLOW_MODEL_PATH: str = "models/crime_detection_model.h5"
    
    # PyTorch/VideoMAE specific
    PYTORCH_DEVICE: str = "auto"  # Options: "auto", "cuda", "cpu"
    PYTORCH_INPUT_SIZE: tuple = (224, 224)
    PYTORCH_NORMALIZE_MEAN: list = [0.485, 0.456, 0.406]  # ImageNet defaults
    PYTORCH_NORMALIZE_STD: list = [0.229, 0.224, 0.225]   # ImageNet defaults
```

## Data Models

### Model Output Format (Standardized)

Both PyTorch and TensorFlow models must return predictions in this format:

```python
{
    "crime_type": str,           # e.g., "Assault", "Theft", "NormalVideos"
    "confidence": float,         # 0.0 to 1.0
    "severity": str,             # "low", "medium", "high", "critical"
    "is_crime": bool,            # True if confidence >= threshold and not "NormalVideos"
    "all_predictions": {         # All class probabilities
        "Assault": 0.85,
        "Theft": 0.10,
        ...
    }
}
```

### Crime Classes Mapping

The VideoMAE model uses the UCF-Crime dataset classes:

```python
CRIME_CLASSES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
    'Explosion', 'Fighting', 'Normal_Videos_event', 'RoadAccidents',
    'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]
```

Note: The model uses `Normal_Videos_event` instead of `NormalVideos` for non-crime events.

## Error Handling

### Model Loading Errors

1. **PyTorch model file not found**:
   - Log warning with file path
   - Attempt to load TensorFlow model
   - Continue operation with fallback model

2. **PyTorch model incompatible**:
   - Log error with model architecture details
   - Fall back to TensorFlow model
   - Alert administrator via logs

3. **Both models fail to load**:
   - Log critical error
   - Create dummy model that returns "NormalVideos" for all predictions
   - System continues but with no detection capability

### Runtime Errors

1. **CUDA out of memory**:
   - Catch CUDA OOM exception
   - Reload model on CPU
   - Log device change
   - Continue processing

2. **Preprocessing errors**:
   - Validate frame dimensions and type
   - Return error prediction with confidence 0.0
   - Log frame details for debugging

3. **Inference errors**:
   - Catch all exceptions during model.forward()
   - Return safe default prediction
   - Log full stack trace

### Error Response Format

```python
{
    "crime_type": "NormalVideos",
    "confidence": 0.0,
    "severity": "low",
    "is_crime": False,
    "error": "Error message here"
}
```

## Testing Strategy

### Unit Tests

**File**: `Backend/tests/test_pytorch_model.py`

1. **Model Loading Tests**:
   - Test successful model loading
   - Test handling of missing model file
   - Test GPU/CPU device placement
   - Test model architecture validation

2. **Preprocessing Tests**:
   - Test frame resizing
   - Test color space conversion
   - Test normalization
   - Test tensor shape and dtype

3. **Prediction Tests**:
   - Test prediction output format
   - Test confidence score calculation
   - Test crime class mapping
   - Test severity determination

4. **Error Handling Tests**:
   - Test invalid input handling
   - Test CUDA OOM handling
   - Test model inference errors

### Integration Tests

**File**: `Backend/tests/test_model_integration.py`

1. **Model Factory Tests**:
   - Test PyTorch model creation
   - Test TensorFlow model creation
   - Test fallback mechanism
   - Test configuration-based selection

2. **Live Detection Integration**:
   - Test frame processing with PyTorch model
   - Test detection saving to database
   - Test alert creation
   - Test WebSocket notifications

3. **Performance Tests**:
   - Measure inference latency
   - Test concurrent predictions
   - Test memory usage
   - Compare PyTorch vs TensorFlow performance

### Manual Testing

1. **Upload Detection**:
   - Upload test images with known crimes
   - Verify correct classification
   - Check confidence scores
   - Validate severity levels

2. **Live Detection**:
   - Start camera feed
   - Verify real-time processing
   - Check frame rate and latency
   - Test multiple concurrent cameras

3. **Model Switching**:
   - Change configuration to use TensorFlow
   - Restart system
   - Verify model switch
   - Compare detection results

## Performance Considerations

### Optimization Strategies

1. **Model Optimization**:
   - Use `torch.jit.script()` or `torch.jit.trace()` for model optimization
   - Enable `torch.backends.cudnn.benchmark` for consistent input sizes
   - Use half-precision (FP16) on compatible GPUs

2. **Batch Processing**:
   - Process multiple frames in batches when possible
   - Implement frame queue for batch accumulation
   - Balance latency vs throughput

3. **Memory Management**:
   - Use `torch.no_grad()` during inference
   - Clear CUDA cache periodically
   - Implement frame buffer size limits

4. **Preprocessing Optimization**:
   - Use OpenCV's GPU module if available
   - Cache preprocessing parameters
   - Minimize data type conversions

### Expected Performance Metrics

- **Single Frame Inference**: < 100ms on GPU, < 500ms on CPU
- **Preprocessing Overhead**: < 50ms
- **Memory Usage**: < 2GB GPU memory, < 1GB system memory
- **Concurrent Cameras**: Support 4+ cameras on GPU, 2+ on CPU

## Migration Path

### Phase 1: Implementation (No Breaking Changes)

1. Create new files: `base_model.py`, `pytorch_model.py`, `model_factory.py`
2. Refactor `ml_model.py` to use factory pattern
3. Add configuration options
4. Maintain backward compatibility

### Phase 2: Testing and Validation

1. Run unit tests
2. Perform integration testing
3. Compare PyTorch vs TensorFlow results
4. Benchmark performance

### Phase 3: Deployment

1. Update configuration to use PyTorch model
2. Monitor system performance
3. Validate detection accuracy
4. Document any issues

### Rollback Plan

If PyTorch integration causes issues:
1. Change `MODEL_TYPE` to `"tensorflow"` in config
2. Restart application
3. System reverts to TensorFlow model
4. No code changes required

## Dependencies

### New Python Packages

Add to `Backend/requirements.txt`:
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
```

### Version Compatibility

- Python: 3.8+
- PyTorch: 2.0+ (for better performance and API stability)
- Transformers: 4.30+ (for VideoMAE support)
- CUDA: 11.8+ (if using GPU)
- Existing packages: No changes required

## Security Considerations

1. **Model File Validation**:
   - Verify model file integrity before loading
   - Check file size limits
   - Validate model architecture

2. **Input Validation**:
   - Validate frame dimensions
   - Check for malformed images
   - Limit processing time per frame

3. **Resource Limits**:
   - Set maximum GPU memory usage
   - Implement timeout for inference
   - Limit concurrent model instances

## Monitoring and Logging

### Key Metrics to Log

1. **Model Performance**:
   - Inference time per frame
   - Preprocessing time
   - Model type in use
   - Device (GPU/CPU)

2. **Detection Statistics**:
   - Predictions per minute
   - Crime detection rate
   - Confidence score distribution
   - Severity level distribution

3. **System Health**:
   - GPU memory usage
   - CPU usage
   - Model loading success/failure
   - Fallback events

### Log Levels

- **INFO**: Model loaded, predictions made, device selection
- **WARNING**: Fallback to TensorFlow, low confidence detections
- **ERROR**: Model loading failures, inference errors
- **CRITICAL**: Both models failed, system degraded

## Future Enhancements

1. **Model Ensemble**: Combine PyTorch and TensorFlow predictions for improved accuracy
2. **Dynamic Model Loading**: Hot-swap models without restart
3. **Model Versioning**: Support multiple model versions simultaneously
4. **A/B Testing**: Compare model performance in production
5. **Model Quantization**: Reduce model size for edge deployment
