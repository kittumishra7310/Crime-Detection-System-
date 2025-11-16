# Camera Detection Troubleshooting

## Issue: "Camera detection not active" message

This error appears when the frontend tries to access the camera feed, but the backend detection has either:
1. Not started yet
2. Started but immediately failed
3. Started but the camera is not accessible

## Quick Fixes Applied:

1. ✅ Increased delay to 2 seconds before accessing feed
2. ✅ Added error checking for start detection result
3. ✅ Added authentication token to feed URL

## How to Diagnose:

### Check Backend Logs:
Look for these messages in the backend terminal:

**Success indicators:**
```
Starting camera detection for camera 1 with source 0
Camera opened with backend...
Started live detection for camera 1
```

**Failure indicators:**
```
Failed to open camera source: 0
Camera 1 disconnected
Error starting camera detection
```

### Common Issues:

#### 1. Camera Access Permission
**macOS**: System Preferences > Security & Privacy > Camera
- Make sure Terminal/Python has camera access

**Linux**: Check if camera device exists
```bash
ls /dev/video*
```

#### 2. Camera Already in Use
Another application might be using the webcam:
- Close Zoom, Skype, Photo Booth, etc.
- Check with: `lsof | grep video` (Linux/Mac)

#### 3. Wrong Camera Source
The default source is "0" (first webcam). Try:
- Source "1" for second camera
- Or use a video file path for testing

### Testing with Video File:

Instead of webcam, test with a video file:

1. Place a video file in `Backend/test_video.mp4`
2. In the frontend, when starting detection, use source: `"test_video.mp4"`

### Manual Backend Test:

Test if the backend can access the camera:

```python
import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Camera accessible")
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame captured: {frame.shape}")
    else:
        print("❌ Cannot read frames")
else:
    print("❌ Cannot open camera")
cap.release()
```

### Check Active Streams:

Add this to your backend to see active cameras:
```python
# In Backend/main.py, add a debug endpoint:
@app.get("/api/debug/active-cameras")
async def debug_active_cameras():
    from detection_routes import live_detection_manager
    return {
        "active_streams": list(live_detection_manager.active_streams.keys()),
        "active_tasks": list(live_detection_manager.active_tasks.keys()),
        "active_feeds": list(live_detection_manager.active_feeds.keys())
    }
```

Then check: `http://localhost:8000/api/debug/active-cameras`

## Next Steps:

1. **Check backend terminal** for error messages
2. **Verify camera access** permissions
3. **Try with video file** instead of webcam
4. **Check if camera is in use** by another app
5. **Look at backend logs** when clicking "Start Detection"

If the backend shows the camera starting successfully but the frontend still shows "Camera detection not active", the issue is likely a timing problem or the camera is crashing immediately after starting.
