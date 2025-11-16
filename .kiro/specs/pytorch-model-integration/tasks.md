# Implementation Plan

- [x] 1. Set up PyTorch and VideoMAE dependencies and configuration
  - Install PyTorch, torchvision, and transformers packages in the Backend environment
  - Add VideoMAE-related configuration parameters to `Backend/config.py` (MODEL_TYPE, VIDEOMAE_MODEL_PATH, PYTORCH_DEVICE, input size, normalization parameters)
  - Update `Backend/requirements.txt` with torch, torchvision, and transformers dependencies
  - _Requirements: 1.1, 6.1, 6.2_

- [x] 2. Create base model abstraction layer
  - [x] 2.1 Implement BaseDetectionModel abstract class
    - Create `Backend/base_model.py` with abstract base class defining the common interface
    - Define abstract methods: `load_model()`, `predict_frame()`, `preprocess_image()`, `_determine_severity()`
    - Add type hints and docstrings for all methods
    - _Requirements: 1.4, 5.1, 5.3_

- [x] 3. Implement VideoMAE model wrapper
  - [x] 3.1 Create VideoMAEModel class
    - Create `Backend/videomae_model.py` with VideoMAEModel class inheriting from BaseDetectionModel
    - Implement `__init__()` method to initialize model path, device, frame buffer, and crime classes
    - Implement device selection logic (auto-detect CUDA, fallback to CPU)
    - Initialize frame buffer with deque for 16 frames
    - _Requirements: 1.1, 4.1, 4.2_
  
  - [x] 3.2 Implement model loading functionality
    - Implement `load_model()` method to load VideoMAE model from HuggingFace directory
    - Load VideoMAEImageProcessor for proper preprocessing
    - Add error handling for missing model directory with logging
    - Verify model loads successfully and move to appropriate device
    - Set model to evaluation mode
    - _Requirements: 1.1, 1.2, 4.3_
  
  - [x] 3.3 Implement frame buffering and preprocessing pipeline
    - Implement frame buffer to accumulate 16 frames
    - Implement `preprocess_frames()` method to convert frame sequences to VideoMAE tensors
    - Add BGR to RGB color space conversion for each frame
    - Implement frame resizing to 224x224 dimensions
    - Use VideoMAE image processor for proper normalization
    - Stack frames into tensor with shape (batch, channels, frames, height, width)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [x] 3.4 Implement prediction and post-processing
    - Implement `predict_frame()` method with frame buffering logic
    - Return buffering status when less than 16 frames collected
    - Execute model inference when buffer is full
    - Extract predicted class index and confidence score from model logits
    - Map output indices to UCF-Crime class labels
    - Implement `_determine_severity()` method for severity classification
    - Format output to match standardized prediction format
    - Add is_crime detection logic based on confidence threshold
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [x] 3.5 Add error handling, logging, and performance tracking
    - Add try-catch blocks for CUDA out of memory errors with CPU fallback
    - Add error handling for preprocessing failures
    - Add error handling for inference errors
    - Implement thread-safe processing with locks
    - Add rate limiting to prevent excessive predictions
    - Track inference times and calculate FPS
    - Implement logging for model loading, device selection, and predictions
    - Return safe default predictions on errors
    - _Requirements: 1.2, 4.4, 5.5_

- [x] 4. Refactor TensorFlow model to use base class
  - [x] 4.1 Create TensorFlowCrimeModel class
    - Create `Backend/tensorflow_model.py` by refactoring existing `Backend/ml_model.py`
    - Make TensorFlowCrimeModel inherit from BaseDetectionModel
    - Move existing CrimeDetectionModel implementation to TensorFlowCrimeModel
    - Ensure all methods match BaseDetectionModel interface
    - Maintain all existing functionality without breaking changes
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 5. Implement model factory pattern
  - [x] 5.1 Create ModelFactory class
    - Create `Backend/model_factory.py` with ModelFactory class
    - Implement `create_model()` static method for model instantiation based on type
    - Add model type validation (videomae/pytorch/tensorflow)
    - Implement model loading with error handling
    - _Requirements: 5.1, 5.4, 6.1, 6.2, 6.4_
  
  - [x] 5.2 Implement fallback mechanism
    - Implement `create_with_fallback()` static method
    - Add logic to try configured model first, then fall back to alternative
    - Add comprehensive logging for fallback events
    - Ensure graceful degradation if both models fail
    - _Requirements: 1.2, 5.2, 5.5_

- [x] 6. Update ml_model.py interface
  - Modify `Backend/ml_model.py` to use ModelFactory
  - Replace direct CrimeDetectionModel instantiation with factory call
  - Maintain backward compatibility with existing `get_crime_model()` function
  - Update global crime_model instance to use factory pattern
  - Add logging to indicate which model type is active
  - _Requirements: 1.4, 1.5, 5.3, 5.4, 5.5_

- [x] 7. Inspect and configure VideoMAE model
  - [x] 7.1 Analyze VideoMAE model structure
    - Inspect the VideoMAE model from HuggingFace directory
    - Verify input shape requirements (16 frames, 224x224)
    - Identify output classes from UCF-Crime dataset
    - Document model architecture details
    - _Requirements: 1.3, 2.1, 3.2_
  
  - [x] 7.2 Update configuration with model specifications
    - Update `Backend/config.py` with correct input dimensions (224x224)
    - Set appropriate normalization parameters (VideoMAE uses ImageNet normalization)
    - Configure crime class labels in correct order from UCF-Crime
    - Update VIDEOMAE_MODEL_PATH to point to model directory
    - _Requirements: 1.1, 2.3, 3.2, 6.1_

- [x] 8. Add model performance optimizations
  - Implement `torch.no_grad()` context for inference
  - Enable `torch.backends.cudnn.benchmark` for consistent input sizes
  - Add model warm-up call during initialization
  - Implement frame batching capability for future use
  - _Requirements: 4.3, 4.4, 4.5_

- [x] 9. Update detection routes and live detection manager
  - Verify `Backend/detection_routes.py` works with new model interface
  - Verify `Backend/live_detection.py` works with new model interface
  - Test that no code changes are needed due to abstraction layer
  - Add logging to track model performance metrics
  - _Requirements: 1.5, 5.3, 5.4_

- [x] 10. Add configuration management
  - [x] 10.1 Implement model type configuration
    - Add MODEL_TYPE environment variable support (videomae/pytorch/tensorflow)
    - Implement configuration validation
    - Add configuration reload capability
    - Document configuration options in README or config comments
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [x] 10.2 Create environment variable template
    - Update `Backend/.env.example` with VideoMAE configuration variables
    - Add comments explaining each configuration option
    - Provide example values for different deployment scenarios
    - _Requirements: 6.1, 6.5_

- [ ] 11. Integration testing and validation
  - [x] 11.1 Test PyTorch model loading and inference
    - Test model loads successfully from VideoMAE directory
    - Test GPU device placement when CUDA is available
    - Test CPU fallback when CUDA is not available
    - Verify prediction output format matches specification
    - Test with sample frame sequences from different crime categories
    - Verify frame buffering works correctly (16 frames)
    - _Requirements: 1.1, 1.2, 1.3, 4.1, 4.2_
  
  - [x] 11.2 Test model fallback mechanism
    - Test fallback to TensorFlow when VideoMAE model directory is missing
    - Test fallback when VideoMAE model fails to load
    - Verify system continues operating with fallback model
    - Check that appropriate warnings are logged
    - _Requirements: 1.2, 5.2, 5.5_
  
  - [ ] 11.3 Test live detection integration
    - Start live detection with VideoMAE model
    - Verify frames are buffered and processed correctly
    - Check detection results are saved to database
    - Verify alerts are created for high-severity detections
    - Test WebSocket notifications are sent
    - _Requirements: 1.5, 3.5, 4.4_
  
  - [ ] 11.4 Test file upload detection
    - Upload test videos and verify VideoMAE model processes them
    - Verify frame-by-frame buffering and processing
    - Verify detection results match expected crime types
    - Check confidence scores are reasonable
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.3_
  
  - [ ] 11.5 Test configuration switching
    - Change MODEL_TYPE to "tensorflow" and restart
    - Verify system uses TensorFlow model
    - Change MODEL_TYPE to "videomae" and restart
    - Verify system switches back to VideoMAE model
    - _Requirements: 6.1, 6.2, 6.5_

- [ ] 12. Performance benchmarking
  - [ ] 12.1 Measure inference performance
    - Measure frame sequence inference time on GPU (16 frames)
    - Measure frame sequence inference time on CPU (16 frames)
    - Compare VideoMAE vs TensorFlow inference times
    - Verify latency meets requirements (< 500ms per prediction)
    - Track FPS and average inference times
    - _Requirements: 4.4_
  
  - [ ] 12.2 Test concurrent processing
    - Test multiple camera feeds simultaneously
    - Measure system performance with 2-4 concurrent cameras
    - Monitor GPU/CPU memory usage
    - Verify no blocking occurs between concurrent requests
    - _Requirements: 4.5_

- [ ] 13. Documentation and deployment preparation
  - [ ] 13.1 Update documentation
    - Document VideoMAE model integration in README or integration guide
    - Add configuration guide for model selection
    - Document frame buffering behavior and requirements
    - Document troubleshooting steps for common issues
    - Add performance tuning recommendations
    - _Requirements: 6.1, 6.4_
  
  - [ ] 13.2 Create deployment checklist
    - Document PyTorch and Transformers installation requirements
    - List CUDA version requirements for GPU support
    - Provide VideoMAE model directory deployment instructions
    - Document the 16-frame buffering requirement
    - Create rollback procedure documentation
    - _Requirements: 1.1, 4.1_
