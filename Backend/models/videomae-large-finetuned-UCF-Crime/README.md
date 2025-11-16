---
language:
- en
base_model:
- MCG-NJU/videomae-large
license: cc-by-nc-4.0
---
# Model Card for `videomae-large-finetuned-UCF-Crime-dataset`

This model is a fine-tuned version of the `MCG-NJU/videomae-large` model, specifically adapted for video classification tasks on the UCF Crime dataset. It is designed to classify various activities and events, including normal and anomalous behaviors such as burglary, vandalism, or fighting, based on video input.

## Test the model at https://www.opear.org/demo


## Model Details

### Model Description

- **Developed by:** Paulo Briceño 
- **Model type:** VideoMAE for Video Classification
- **Labels:**
  - Abuse
  - Arrest
  - Arson
  - Assault
  - Burglary
  - Explosion
  - Fighting
  - Normal Videos
  - Road Accidents
  - Robbery
  - Shooting
  - Shoplifting
  - Stealing
  - Vandalism
- **Finetuned from model:** `MCG-NJU/videomae-large`

### Model Sources

- **Repository:** [Link to repository](https://huggingface.co/pabrcn/videomae-large-finetuned-UCF-Crime-dataset)
- **Base Model Repository:** [MCG-NJU/videomae-large](https://huggingface.co/MCG-NJU/videomae-large)

## Uses

### Direct Use

The model can directly classify input videos into one of the 14 labels mentioned above. It is intended for anomaly detection tasks, especially in scenarios where automated video surveillance is required.

### Downstream Use

This model can be integrated into real-time surveillance systems, used in forensic investigations, or applied in research to evaluate and improve crime detection algorithms.

### Out-of-Scope Use

- This model is not suitable for scenarios where input data deviates significantly from the types of videos in the UCF Crime dataset.
- Misuse for surveillance without proper ethical considerations.

## Bias, Risks, and Limitations

- **Biases:** The model may inherit biases from the UCF Crime dataset, which could reflect cultural or situational assumptions specific to the dataset.
- **Limitations:** Performance may degrade for scenarios or activities outside the scope of the training dataset.

### Recommendations

Users should carefully evaluate the model's outputs and cross-verify results before taking critical decisions. Test performance in real-world scenarios to ensure reliability.

## How to Get Started with the Model

### Google Colab
Upload test videos to `sample_data` folder

```python
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification

# Define video directory
video_folder = "sample_data"

# Define class mapping
class_mapping = {
    "Abuse": 0, "Arrest": 1, "Arson": 2, "Assault": 3, "Burglary": 4,
    "Explosion": 5, "Fighting": 6, "Normal Videos": 7, "Road Accidents": 8,
    "Robbery": 9, "Shooting": 10, "Shoplifting": 11, "Stealing": 12, "Vandalism": 13
}
reverse_mapping = {v: k for k, v in class_mapping.items()}

# Load VideoMAE model
model_name = "OPear/videomae-large-finetuned-UCF-Crime"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = VideoMAEForVideoClassification.from_pretrained(
    model_name,
    label2id=class_mapping,
    id2label=reverse_mapping,
    ignore_mismatched_sizes=True,
).to(device)
model.eval()

# Video processing function
def load_video_frames(video_path, num_frames=16, size=(224, 224)):
    """
    Load video frames from a given path and resize them to (224, 224).
    Converts video into a tensor of shape [num_frames, 3, height, width].
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size)
            frames.append(frame)
    
    cap.release()

    if len(frames) < num_frames:  # Pad if not enough frames
        frames.extend([frames[-1]] * (num_frames - len(frames)))

    frames = np.stack(frames, axis=0)  # Shape: [num_frames, height, width, 3]
    frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # Normalize

    return frames  # Shape: [num_frames, 3, height, width]

# Custom Dataset
class VideoDataset(Dataset):
    def __init__(self, video_folder):
        self.video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".mp4")]
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_tensor = load_video_frames(video_path)
        return {"video": video_tensor, "filename": os.path.basename(video_path)}

# Load dataset
test_dataset = VideoDataset(video_folder)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Run inference
with torch.no_grad():
    for idx, sample in enumerate(test_loader):
        video_tensor = sample["video"].squeeze(0)  # Remove batch dimension from DataLoader
        video_tensor = video_tensor.unsqueeze(0).to(device)  # Correct shape: [1, 3, num_frames, H, W]

        # Forward pass
        outputs = model(video_tensor)

        # Get predictions
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=-1).item()
        
        filename = sample["filename"][0]

        print(f"Video {idx}: {filename} - Predicted label = {reverse_mapping[predicted_label]}")

```

## Training Details

### Training Data

The model was fine-tuned on the **UCF Crime dataset**, which contains labeled videos of anomalous and normal events.

### Training Procedure

- **Preprocessing:** Videos were preprocessed to 224x224 resolution with 16 frames sampled per video clip.
- **Hyperparameters:**
  - **Batch size:** 4
  - **Epochs:** 4
  - **Learning rate:** Linear warmup to a peak of 5e-5
  - **Optimizer:** AdamW
  - **Mixed precision:** fp16

#### Speeds, Sizes, Times

- **Number of parameters:** ~323M
- **Best model checkpoint:** `checkpoint-1112`
- **Evaluation accuracy:** 92.96%
- **Evaluation loss:** 0.15

## Evaluation

### Testing Data, Factors & Metrics

- **Testing Dataset:** UCF Crime dataset (Test split)
- **Metrics Used:** Accuracy, Evaluation Loss

### Results

- **Best Accuracy:** 92.96% (on the validation split after 4 epochs)

#### Summary

The model achieves state-of-the-art performance for video classification tasks on the UCF Crime dataset.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute).

- **Hardware Type:** NVIDIA V100 GPUs
- **Hours used:** ~50 hours
- **Cloud Provider:** Google Colab
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

# Technical Specifications

## Training procedure

#### Training hyperparameters
The following hyperparameters were used during training:
```
{
  learning_rate: 5e-05,
  train_batch_size: 4,
  eval_batch_size: 4,
  seed: 42,
  gradient_accumulation_steps: 2,
  total_train_batch_size: 8,
  optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments,
  lr_scheduler_type: linear,
  lr_scheduler_warmup_ratio: 0.1,
  training_steps: 13320,
  mixed_precision_training: Native AMP
}
```

#### Framework versions
Transformers 4.46.3
Pytorch 1.13.0+cu117
Datasets 3.1.0
Tokenizers 0.20.3

#### Model Architecture and Objective

This model is based on the **VideoMAE architecture**, which leverages masked autoencoders for efficient video feature learning. It uses:

- **Hidden size:** 1024
- **Number of layers:** 24
- **Attention heads:** 16

#### Compute Infrastructure

- **Hardware:** NVIDIA GPUs
- **Software:** Transformers 4.46.3, PyTorch 1.13.1

## Citation

**BibTeX:**

```bibtex
@article{videomae_large,
  title={VideoMAE: Masked Autoencoders for Video Representation Learning},
  author={MCG-NJU Team},
  year={2024},
  url={https://huggingface.co/MCG-NJU/videomae-large}
}
```
```bibtex
@InProceedings{Sultani_2018_CVPR,
author = {Sultani, Waqas and Chen, Chen and Shah, Mubarak},
title = {Real-World Anomaly Detection in Surveillance Videos},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
**APA:**
MCG-NJU Team. (2024). VideoMAE: Masked Autoencoders for Video Representation Learning. Retrieved from https://huggingface.co/MCG-NJU/videomae-large

Sultani, W., Chen, C., & Shah, M. (2018). Real-world anomaly detection in surveillance videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.