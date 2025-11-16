# Requirements Document

## Introduction

This document outlines the requirements for integrating a VideoMAE-based crime detection model into the existing real-time crime detection system. The VideoMAE model is a video transformer model that processes sequences of frames for temporal crime detection. This feature enables the use of a state-of-the-art video understanding model for improved detection accuracy in real-time surveillance scenarios.

## Glossary

- **Detection System**: The existing crime detection application that processes video feeds and images to identify criminal activities
- **VideoMAE Model**: The pre-trained video transformer model stored in `models/videomae-large-finetuned-UCF-Crime` used for crime classification from video sequences
- **TensorFlow Model**: The legacy Keras-based model used by the Detection System for fallback
- **Live Detection Manager**: The component responsible for managing real-time camera feeds and processing frames
- **Crime Model Interface**: The abstraction layer that provides prediction capabilities to the Detection System
- **Frame Buffer**: A sliding window of frames used to accumulate the required number of frames for VideoMAE inference
- **Frame Processing Pipeline**: The sequence of operations that transform raw camera frames into model predictions
- **Model Loader**: The component responsible for loading and initializing the VideoMAE Model
- **Confidence Threshold**: The minimum prediction confidence value required to classify a detection as valid

## Requirements

### Requirement 1

**User Story:** As a system administrator, I want to use the VideoMAE model for crime detection, so that I can leverage improved temporal detection accuracy in real-time surveillance.

#### Acceptance Criteria

1. WHEN the Detection System initializes, THE Model Loader SHALL load the VideoMAE Model from the directory path `Backend/models/videomae-large-finetuned-UCF-Crime`
2. IF the VideoMAE Model directory does not exist at the specified path, THEN THE Model Loader SHALL log an error message and fall back to the TensorFlow Model
3. WHEN the VideoMAE Model loads successfully, THE Model Loader SHALL verify the model is compatible with the expected input format (16 frames)
4. THE Crime Model Interface SHALL provide a unified prediction method that works with VideoMAE Model and TensorFlow Model
5. WHEN the Live Detection Manager processes a frame, THE Crime Model Interface SHALL use the VideoMAE Model if it is loaded successfully

### Requirement 2

**User Story:** As a developer, I want the VideoMAE model to buffer frames and process them as sequences, so that I can leverage temporal information for better crime detection.

#### Acceptance Criteria

1. WHEN a frame is passed to the Crime Model Interface, THE Frame Buffer SHALL accumulate frames until 16 frames are collected
2. THE Frame Processing Pipeline SHALL convert BGR color format frames to RGB format expected by the VideoMAE Model
3. THE Frame Processing Pipeline SHALL resize frames to 224x224 dimensions required by the VideoMAE Model
4. THE Frame Processing Pipeline SHALL normalize pixel values using the VideoMAE image processor specifications
5. WHEN 16 frames are collected, THE Frame Processing Pipeline SHALL convert the frame sequence to a PyTorch tensor with shape (batch, channels, frames, height, width)

### Requirement 3

**User Story:** As a security operator, I want the system to classify detected crimes into specific categories, so that I can respond appropriately to different threat levels.

#### Acceptance Criteria

1. WHEN the VideoMAE Model generates predictions, THE Crime Model Interface SHALL extract the predicted crime class from the model output logits
2. THE Crime Model Interface SHALL map the VideoMAE Model output indices to human-readable crime type labels from the UCF-Crime dataset
3. WHEN a prediction is made, THE Crime Model Interface SHALL calculate the confidence score as a value between 0.0 and 1.0 using softmax
4. THE Crime Model Interface SHALL determine the severity level based on the crime type and confidence score
5. WHEN the confidence score exceeds the Confidence Threshold, THE Crime Model Interface SHALL mark the detection as a valid crime event

### Requirement 4

**User Story:** As a system operator, I want the VideoMAE model to process frame sequences efficiently, so that real-time detection does not introduce significant latency.

#### Acceptance Criteria

1. THE Model Loader SHALL load the VideoMAE Model onto GPU device if CUDA is available
2. IF CUDA is not available, THEN THE Model Loader SHALL load the VideoMAE Model onto CPU device
3. WHEN processing frames, THE Crime Model Interface SHALL execute VideoMAE Model inference in evaluation mode to disable dropout and batch normalization training behavior
4. THE Crime Model Interface SHALL process frame sequences with a target latency of less than 500 milliseconds per prediction
5. WHEN multiple cameras are active, THE Crime Model Interface SHALL use thread-safe frame buffering to handle concurrent prediction requests without blocking

### Requirement 5

**User Story:** As a developer, I want to maintain backward compatibility with the existing TensorFlow model, so that the system can operate if the VideoMAE model is unavailable.

#### Acceptance Criteria

1. THE Crime Model Interface SHALL support both VideoMAE Model and TensorFlow Model implementations
2. WHEN the VideoMAE Model fails to load, THE Crime Model Interface SHALL automatically use the TensorFlow Model
3. THE Crime Model Interface SHALL provide the same output format regardless of which model is active
4. WHEN switching between models, THE Detection System SHALL not require code changes in the Live Detection Manager
5. THE Crime Model Interface SHALL log which model type is currently active for debugging purposes

### Requirement 6

**User Story:** As a system administrator, I want to configure which model to use, so that I can test and compare different models without code changes.

#### Acceptance Criteria

1. THE Detection System SHALL read a configuration parameter that specifies the preferred model type (videomae, pytorch, or tensorflow)
2. WHERE the configuration specifies VideoMAE Model, THE Model Loader SHALL attempt to load the VideoMAE Model first
3. WHERE the configuration specifies TensorFlow Model, THE Model Loader SHALL use the TensorFlow Model exclusively
4. THE Detection System SHALL validate the configuration parameter and reject invalid model type values
5. WHEN the configuration changes, THE Detection System SHALL allow model reloading without requiring a full system restart
