# Camera Feed and Detection Display Fix

## Issues Identified:

### 1. Camera Feed Not Visible
- The feed URL is set to `http://localhost:8000/live/feed/1`
- This should be an MJPEG stream but might not be working properly
- The `<img>` tag is trying to load it as a static image

### 2. Detection Results Not Showing in Frontend
- Backend is sending `detection_status` messages
- Frontend WebSocket handler now supports these messages (just fixed)
- Detection results should now appear in the UI

## What Was Fixed:

### ✅ WebSocket Message Handling
Added support for these message types in `services/websocket.ts`:
- `detection_status` - Status updates (buffering, normal, etc.)
- `buffering_status` - Frame buffering progress
- `detection_result` - Crime detection results

## What Still Needs Attention:

### Camera Feed Display
The camera feed uses MJPEG streaming which should work with an `<img>` tag, but there might be issues with:
1. **CORS** - The backend might not be allowing the frontend to access the stream
2. **Stream Format** - The MJPEG stream might not be properly formatted
3. **Camera Source** - The camera (source "0") might not be accessible

## Testing Steps:

1. **Test the feed URL directly**:
   - Open `http://localhost:8000/live/feed/1` in a browser
   - You should see the camera feed as an MJPEG stream
   - If it doesn't work, the backend stream has issues

2. **Check backend logs**:
   - Look for errors related to camera access
   - Check if frames are being captured
   - Verify MJPEG encoding is working

3. **Test detection results**:
   - Start detection on a camera
   - Watch the browser console for WebSocket messages
   - Detection results should now appear in:
     - Recent Detections panel (right side)
     - Toast notifications (top-right)
     - Detection stats (top cards)

## Quick Fixes Applied:

1. ✅ Added `detection_status`, `buffering_status`, and `detection_result` message handlers
2. ✅ WebSocket now properly forwards all detection-related messages to the UI
3. ✅ Detection overlay component is ready to display results

## Next Steps:

If camera feed still doesn't show:
1. Check if webcam is accessible: `ls /dev/video*` (Linux) or check System Preferences (Mac)
2. Verify backend can access camera
3. Test with a video file instead of webcam
4. Check CORS settings in backend

The detection results should now be visible in the frontend!
