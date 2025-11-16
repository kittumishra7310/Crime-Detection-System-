#!/usr/bin/env python3
"""Force release camera by opening and immediately closing it."""
import cv2
import time

def force_release_camera(camera_id=0):
    """Force release a camera by opening and closing it multiple times."""
    print(f"Attempting to force release camera {camera_id}...")
    
    for attempt in range(3):
        try:
            print(f"  Attempt {attempt + 1}/3...")
            cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                print(f"    Camera opened successfully")
                # Read one frame to ensure it's working
                ret, frame = cap.read()
                if ret:
                    print(f"    Frame read successfully")
                else:
                    print(f"    Failed to read frame")
            else:
                print(f"    Failed to open camera")
            
            # Release the camera
            cap.release()
            print(f"    Camera released")
            
            # Wait a bit
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    Error: {e}")
    
    # Final cleanup
    cv2.destroyAllWindows()
    print("Camera force release complete!")

if __name__ == "__main__":
    force_release_camera(0)
