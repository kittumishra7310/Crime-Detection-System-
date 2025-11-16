# Frontend Issues and Fixes

## Issues Identified

### 1. Stop Detection Button Not Working
**Symptoms**: Button doesn't stop the detection
**Possible Causes**:
- `isDetecting` state not being set correctly
- Backend not stopping properly
- WebSocket not disconnecting

### 2. Camera View Not Showing
**Symptoms**: No video feed displayed
**Possible Causes**:
- `feedUrl` not being set
- Backend feed endpoint not working
- Authentication token issues

## Quick Fixes

### Fix 1: Add Debug Logging

Add this to your browser console to debug:
```javascript
// Check if detection is running
console.log('Is Detecting:', document.querySelector('[data-detecting]'))

// Check feed URL
console.log('Feed URL:', document.querySelector('img[src*="feed"]')?.src)
```

### Fix 2: Force Stop Detection

If stop button doesn't work, run this in browser console:
```javascript
// Force stop via API
fetch('http://localhost:8000/live/stop-all', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${localStorage.getItem('token')}`,
    'Content-Type': 'application/json'
  }
}).then(r => r.json()).then(console.log)
```

### Fix 3: Check Backend Status

```bash
# Check if backend is running
curl http://localhost:8000/api/system/status

# Check active cameras
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/live/status
```

## Recommended Actions

### 1. Restart Both Servers

```bash
# Terminal 1 - Backend
cd Backend
python main.py

# Terminal 2 - Frontend  
npm run dev
```

### 2. Clear Browser Cache

- Open DevTools (F12)
- Go to Application tab
- Clear Storage
- Refresh page

### 3. Check Camera Status in Database

```bash
python Backend/add_default_cameras.py
```

### 4. Test API Endpoints

```bash
# Get cameras
curl http://localhost:8000/api/cameras

# Start detection (replace TOKEN and CAMERA_ID)
curl -X POST http://localhost:8000/live/start/1 \
  -H "Authorization: Bearer TOKEN" \
  -d "source=0"

# Stop detection
curl -X POST http://localhost:8000/live/stop/1 \
  -H "Authorization: Bearer TOKEN"
```

## Model Fine-Tuning

Fine-tuning the VideoMAE model requires:

### Requirements
1. **Training Data**: Labeled video clips of crimes
2. **GPU**: CUDA-capable GPU (8GB+ VRAM recommended)
3. **Time**: Several hours to days depending on dataset size
4. **Expertise**: Understanding of PyTorch and transformers

### Steps for Fine-Tuning

1. **Prepare Dataset**
```python
# Create dataset structure
dataset/
  ├── train/
  │   ├── Abuse/
  │   ├── Robbery/
  │   └── ...
  └── val/
      ├── Abuse/
      ├── Robbery/
      └── ...
```

2. **Install Training Dependencies**
```bash
pip install transformers[torch] datasets accelerate
```

3. **Create Training Script**
```python
from transformers import VideoMAEForVideoClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("your_dataset")

# Load model
model = VideoMAEForVideoClassification.from_pretrained(
    "Backend/models/videomae-large-finetuned-UCF-Crime"
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    warmup_steps=500,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"]
)

trainer.train()
```

### Alternative: Use Pre-trained Model

Instead of fine-tuning, you can:
1. Use the accuracy improvements I already implemented
2. Adjust confidence thresholds
3. Collect data and fine-tune later

The improvements I made (temporal smoothing, frame enhancement, etc.) will significantly improve accuracy without retraining.

## Next Steps

1. **Immediate**: Fix frontend issues (restart servers, check API)
2. **Short-term**: Test with the accuracy improvements
3. **Long-term**: Collect data for fine-tuning if needed

## Testing Checklist

- [ ] Backend server running on port 8000
- [ ] Frontend server running on port 3000
- [ ] Cameras added to database
- [ ] User logged in with valid token
- [ ] WebSocket connection established
- [ ] Camera feed URL accessible
- [ ] Detection starts successfully
- [ ] Detection stops successfully
- [ ] Alerts showing in frontend

## Common Errors and Solutions

### Error: "Camera not found"
**Solution**: Run `python Backend/add_default_cameras.py`

### Error: "Authentication required"
**Solution**: Log out and log in again

### Error: "Failed to open camera"
**Solution**: Check if another app is using the webcam

### Error: "WebSocket connection failed"
**Solution**: Restart backend server

### Error: "Feed not loading"
**Solution**: Check backend logs for camera feed errors
