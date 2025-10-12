#!/usr/bin/env python3
"""
Camera Permission Test for macOS
Tests webcam access and provides solutions for permission issues
"""

import cv2
import sys
import subprocess
import os

def test_camera_access():
    """Test if camera can be accessed"""
    print("🔍 Testing camera access...")
    
    try:
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not accessible")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print("✅ Camera access successful")
            print(f"📹 Frame size: {frame.shape}")
            return True
        else:
            print("❌ Could not read from camera")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def check_system_info():
    """Check system information"""
    print("\n🖥️  System Information:")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")
    
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except:
        print("OpenCV: Not installed")

def provide_solutions():
    """Provide solutions for camera permission issues"""
    print("\n🔧 Solutions for Camera Permission Issues:")
    print("=" * 50)
    
    print("1. Grant Camera Permission:")
    print("   • Go to System Preferences > Security & Privacy > Camera")
    print("   • Add Terminal or your IDE to allowed apps")
    print("   • Restart Terminal/IDE after granting permission")
    
    print("\n2. Alternative: Use Image/Video Upload Instead:")
    print("   • Upload test images/videos for crime detection")
    print("   • Test the system without live camera")
    
    print("\n3. Test with Different Camera Sources:")
    print("   • Camera 0: Built-in camera")
    print("   • Camera 1: External USB camera")
    print("   • RTSP URL: Network camera stream")
    
    print("\n4. Run as Administrator:")
    print("   • Try running with sudo (not recommended for production)")

def create_test_image():
    """Create a test image for upload testing"""
    try:
        import numpy as np
        
        # Create a test image with some patterns
        height, width = 480, 640
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some patterns to simulate activity
        cv2.rectangle(image, (100, 100), (300, 200), (0, 255, 0), 2)
        cv2.circle(image, (400, 300), 50, (255, 0, 0), -1)
        cv2.putText(image, "Test Crime Scene", (150, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        test_image_path = "test_crime_scene.jpg"
        cv2.imwrite(test_image_path, image)
        
        print(f"\n📸 Created test image: {test_image_path}")
        return test_image_path
        
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return None

def main():
    print("🎯 Camera Permission Test for Surveillance System")
    print("=" * 55)
    
    check_system_info()
    
    # Test camera access
    camera_works = test_camera_access()
    
    if not camera_works:
        provide_solutions()
        
        # Create test image as alternative
        test_image = create_test_image()
        
        print("\n🚀 Alternative Testing Options:")
        print("=" * 35)
        
        if test_image:
            print(f"1. Test with uploaded image:")
            print(f"   curl -X POST http://localhost:8000/api/detection/upload \\")
            print(f"        -F 'file=@{test_image}' \\")
            print(f"        -F 'camera_id=1'")
        
        print("\n2. Check system status:")
        print("   curl http://localhost:8000/api/detection/live/status")
        
        print("\n3. View API documentation:")
        print("   Open http://localhost:8000/docs")
        
    else:
        print("\n✅ Camera is working! You can run the live detection demo.")
        print("   python live_detection_demo.py")

if __name__ == "__main__":
    main()
