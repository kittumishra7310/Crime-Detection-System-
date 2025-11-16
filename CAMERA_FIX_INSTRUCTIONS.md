# Camera View and Stop Button Fix Instructions

## Issues Fixed
1. ✅ Stop detection button not working
2. ✅ Camera view not showing
3. ✅ Better error handling

## Changes Made

### 1. Fixed Start Detection Logic
- Better success detection from backend response
- Proper state management order
- Added camera reload after start

### 2. Fixed Stop Detection Logic
- Clear feed URL first
- Force stop image loading
- Reset all related states
- Added delay before reloading cameras

## Testing Steps

### 1. Restart Servers

```bash
# Terminal 1 - Backend
cd Backend
python main.py

# Terminal 2 - Frontend
npm run dev
```

### 2. Test Camera Detection

1. **Login** to the application
2. **Select a camera** from the dropdown (use "Webcam" for testing)
3. **Click "Start Detection"**
   - Should see "Starting..." message
   - After 2 seconds, camera feed should appear
   - Status should show "Live detection started successfully"
4. **Click "Stop Detection"**
   - Should see "Stopping..." message
   - Camera feed should disappear immediately
   - Status should show "Live detection stopped successfully"

### 3. If Issues Persist

#### Issue: Stop button doesn't work
**Solution**: Use browser console:
```javascript
// Force stop all detections
fetch('http://localhost:8000/live/stop-all', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${localStorage.getItem('token')}`,
    'Content-Type': 'application/json'
  }
}).then(r => r.json()).then(console.log)

// Then refresh the page
location.reload()
```

#### Issue: Camera view not showing
**Checklist**:
- [ ] Backend server running on port 8000
- [ ] Camera added to database (run `python Backend/add_default_cameras.py`)
- [ ] User logged in (check localStorage.getItem('token'))
- [ ] Detection started successfully (check browser console for errors)
- [ ] Feed URL correct (should be `http://localhost:8000/live/feed/CAMERA_ID?token=...`)

**Debug in browser console**:
```javascript
// Check if detection is active
console.log('Is Detecting:', document.querySelector('[class*="isDetecting"]'))

// Check feed URL
const img = document.querySelector('img[src*="feed"]')
console.log('Feed URL:', img?.src)
console.log('Image loaded:', img?.complete)

// Check for errors
img?.addEventListener('error', (e) => console.error('Image load error:', e))
```

#### Issue: "Failed to start detection"
**Possible causes**:
1. **Camera already in use**: Close other apps using the webcam
2. **Permission denied**: Allow camera access in browser
3. **Backend error**: Check backend logs

**Check backend logs**:
```bash
# Look for errors in backend terminal
# Should see:
# INFO:live_detection:Starting camera detection for camera X with source 0
# INFO:live_detection:Camera opened with backend 0
# INFO:live_detection:Started live detection for camera X
```

## Manual Testing

### Test 1: Start Detection
```bash
# Get token from browser localStorage
TOKEN="your_token_here"

# Start detection
curl -X POST "http://localhost:8000/live/start/4?source=0" \
  -H "Authorization: Bearer $TOKEN"

# Should return:
# {"message":"Live detection started for camera 4","camera_id":4,"source":"0","status":"active"}
```

### Test 2: Check Feed
```bash
# Open in browser (replace TOKEN and CAMERA_ID)
http://localhost:8000/live/feed/4?token=YOUR_TOKEN

# Should show MJPEG stream
```

### Test 3: Stop Detection
```bash
# Stop detection
curl -X POST "http://localhost:8000/live/stop/4" \
  -H "Authorization: Bearer $TOKEN"

# Should return:
# {"message":"Live detection stopped for camera 4","camera_id":4,"status":"inactive"}
```

## Common Errors and Solutions

### Error: "Camera not found"
```bash
# Add cameras to database
python Backend/add_default_cameras.py
```

### Error: "Authentication required"
```bash
# Log out and log in again
# Or clear localStorage and re-authenticate
localStorage.clear()
location.reload()
```

### Error: "Failed to open camera source"
**Solutions**:
1. Close other apps using webcam (Zoom, Skype, etc.)
2. Check camera permissions in browser settings
3. Try a different camera source
4. Restart computer if camera is stuck

### Error: "WebSocket connection failed"
**Solutions**:
1. Check if backend is running
2. Check CORS settings
3. Restart backend server

## Verification

After fixes, you should be able to:
- ✅ Select a camera from dropdown
- ✅ Start detection and see camera feed
- ✅ See real-time detection results
- ✅ Stop detection and feed disappears
- ✅ Start detection again without issues

## Additional Features

### Force Stop Button
If stop button doesn't work, a "Force Stop" button will appear when detection is active. This will:
- Stop all active detections
- Clear all states
- Reload camera list

### Debug Mode
Enable debug logging in browser console:
```javascript
localStorage.setItem('debug', 'true')
location.reload()
```

## Support

If issues persist after following these steps:
1. Check browser console for errors (F12)
2. Check backend logs for errors
3. Verify database has cameras
4. Verify authentication token is valid
5. Try with a different browser
6. Restart both servers

## Success Indicators

When everything is working:
- Backend logs show camera opened successfully
- Frontend shows camera feed
- Detection results appear in real-time
- Stop button works immediately
- No errors in console
