#!/usr/bin/env python3
"""
Live Detection Demo Script
Demonstrates webcam integration with crime detection and MySQL database
"""

import cv2
import asyncio
import requests
import json
import time
from datetime import datetime
import logging
import threading
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveDetectionDemo:
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.token = None
        self.camera_id = 1  # Use camera ID 1 from database
        self.running = False
        
    async def login(self, username="admin", password="admin123"):
        """Login to get authentication token"""
        try:
            response = requests.post(
                f"{self.api_base_url}/api/auth/login",
                json={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                logger.info(f"✅ Logged in as {username}")
                return True
            else:
                logger.error(f"❌ Login failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Login error: {e}")
            return False
    
    def get_headers(self):
        """Get authentication headers"""
        if self.token:
            return {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
        else:
            return {"Content-Type": "application/json"}
    
    async def start_live_detection(self, source=0):
        """Start live detection on webcam"""
        try:
            response = requests.post(
                f"{self.api_base_url}/api/detection/live/start/{self.camera_id}",
                params={"source": str(source)},
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                logger.info("✅ Live detection started successfully")
                return True
            else:
                logger.error(f"❌ Failed to start live detection: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting live detection: {e}")
            return False
    
    async def stop_live_detection(self):
        """Stop live detection"""
        try:
            response = requests.post(
                f"{self.api_base_url}/api/detection/live/stop/{self.camera_id}",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                logger.info("✅ Live detection stopped successfully")
                return True
            else:
                logger.error(f"❌ Failed to stop live detection: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error stopping live detection: {e}")
            return False
    
    def get_detection_status(self):
        """Get live detection status"""
        try:
            response = requests.get(
                f"{self.api_base_url}/api/detection/live/status",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Failed to get status: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            return None
    
    def upload_test_image(self, image_path):
        """Upload and test image detection"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {'camera_id': self.camera_id}
                
                response = requests.post(
                    f"{self.api_base_url}/api/detection/upload",
                    files=files,
                    data=data,
                    headers={"Authorization": f"Bearer {self.token}"}
                )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Image processed: {result}")
                return result
            else:
                logger.error(f"❌ Upload failed: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            return None
    
    def get_recent_detections(self):
        """Get recent detections from database"""
        try:
            response = requests.get(
                f"{self.api_base_url}/api/detections",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Failed to get detections: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error getting detections: {e}")
            return []
    
    def get_alerts(self):
        """Get active alerts"""
        try:
            response = requests.get(
                f"{self.api_base_url}/api/alerts",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Failed to get alerts: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error getting alerts: {e}")
            return []
    
    def display_webcam_with_detection(self):
        """Display webcam feed with detection overlay"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("❌ Cannot open webcam")
            return
        
        logger.info("📹 Webcam feed started. Press 'q' to quit, 's' to capture screenshot")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add detection overlay
            cv2.putText(frame, "Live Crime Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Camera ID: {self.camera_id}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Status: {'ACTIVE' if self.running else 'INACTIVE'}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.running else (0, 0, 255), 2)
            
            # Show timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Live Crime Detection - Surveillance System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot for testing
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, frame)
                logger.info(f"📸 Screenshot saved: {screenshot_path}")
                
                # Test upload
                result = self.upload_test_image(screenshot_path)
                if result:
                    logger.info(f"🔍 Detection result: {result}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    async def run_demo(self):
        """Run the complete live detection demo"""
        logger.info("🚀 Starting Live Detection Demo")
        logger.info("=" * 50)
        
        # Step 1: Login
        logger.info("Step 1: Authenticating...")
        if not await self.login():
            logger.error("❌ Authentication failed. Exiting.")
            return
        
        # Step 2: Check system status
        logger.info("Step 2: Checking system status...")
        status = self.get_detection_status()
        if status:
            logger.info(f"📊 System Status: {status}")
        
        # Step 3: Start live detection
        logger.info("Step 3: Starting live detection...")
        if not await self.start_live_detection():
            logger.error("❌ Failed to start live detection. Exiting.")
            return
        
        # Step 4: Display webcam feed
        logger.info("Step 4: Starting webcam display...")
        self.running = True
        
        # Run webcam in separate thread
        webcam_thread = threading.Thread(target=self.display_webcam_with_detection)
        webcam_thread.daemon = True
        webcam_thread.start()
        
        # Step 5: Monitor detections and alerts
        logger.info("Step 5: Monitoring detections and alerts...")
        logger.info("📹 Webcam window opened. Press 'q' in webcam window to quit.")
        
        try:
            while self.running:
                # Check for new detections every 5 seconds
                await asyncio.sleep(5)
                
                detections = self.get_recent_detections()
                alerts = self.get_alerts()
                
                if detections:
                    logger.info(f"🔍 Recent detections: {len(detections)}")
                    for detection in detections[-3:]:  # Show last 3
                        logger.info(f"   - {detection.get('detection_type', 'Unknown')} "
                                  f"({detection.get('confidence', 0):.2f} confidence)")
                
                if alerts:
                    logger.info(f"🚨 Active alerts: {len(alerts)}")
                    for alert in alerts[-3:]:  # Show last 3
                        logger.info(f"   - {alert.get('severity', 'Unknown')} severity: "
                                  f"{alert.get('message', 'No message')}")
                
                # Check if webcam thread is still alive
                if not webcam_thread.is_alive():
                    break
                    
        except KeyboardInterrupt:
            logger.info("🛑 Demo interrupted by user")
        
        # Step 6: Cleanup
        logger.info("Step 6: Cleaning up...")
        self.running = False
        await self.stop_live_detection()
        
        logger.info("✅ Demo completed successfully!")
        logger.info("=" * 50)

async def main():
    """Main demo function"""
    demo = LiveDetectionDemo()
    
    print("\n" + "="*60)
    print("🎯 SMART SURVEILLANCE SYSTEM - LIVE DETECTION DEMO")
    print("="*60)
    print("Features:")
    print("• MySQL Database Integration")
    print("• Live Webcam Detection")
    print("• Real-time Crime Detection")
    print("• Image/Video Upload Processing")
    print("• Alert System")
    print("• Authentication & Security")
    print("="*60)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Backend server is running")
        else:
            print("❌ Backend server not responding properly")
            return
    except:
        print("❌ Backend server not running. Please start it first:")
        print("   cd Backend && python test_mysql_server.py")
        return
    
    print("\n🚀 Starting demo in 3 seconds...")
    await asyncio.sleep(3)
    
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())
