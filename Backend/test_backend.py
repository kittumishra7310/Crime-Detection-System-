#!/usr/bin/env python3
"""
Backend testing script to verify all components are working.
"""

import requests
import json
import sys
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test basic health check."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print("❌ Health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend server")
        return False

def test_authentication():
    """Test authentication system."""
    try:
        # Test login with default admin credentials
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
        
        if response.status_code == 200:
            data = response.json()
            token = data.get("access_token")
            print("✅ Authentication test passed")
            return token
        else:
            print("❌ Authentication test failed")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Authentication test error: {e}")
        return None

def test_api_endpoints(token):
    """Test various API endpoints."""
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test cameras endpoint
    try:
        response = requests.get(f"{BASE_URL}/api/cameras", headers=headers)
        if response.status_code == 200:
            print("✅ Cameras API test passed")
        else:
            print("❌ Cameras API test failed")
    except Exception as e:
        print(f"❌ Cameras API test error: {e}")
    
    # Test alerts endpoint
    try:
        response = requests.get(f"{BASE_URL}/api/alerts", headers=headers)
        if response.status_code == 200:
            print("✅ Alerts API test passed")
        else:
            print("❌ Alerts API test failed")
    except Exception as e:
        print(f"❌ Alerts API test error: {e}")
    
    # Test system status
    try:
        response = requests.get(f"{BASE_URL}/api/system/status", headers=headers)
        if response.status_code == 200:
            print("✅ System status API test passed")
        else:
            print("❌ System status API test failed")
    except Exception as e:
        print(f"❌ System status API test error: {e}")

def main():
    """Run all tests."""
    print("🚀 Starting Backend Integration Tests")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Backend server is not running!")
        print("Please start the server with: python start_server.py")
        sys.exit(1)
    
    # Test 2: Authentication
    token = test_authentication()
    if not token:
        print("\n❌ Authentication failed!")
        print("Make sure the database is initialized with: python init_db.py")
        sys.exit(1)
    
    # Test 3: API endpoints
    test_api_endpoints(token)
    
    print("\n" + "=" * 50)
    print("🎉 Backend Integration Tests Complete!")
    print("✅ Backend is ready for frontend integration")
    print("\nNext steps:")
    print("1. Start the Next.js frontend: npm run dev")
    print("2. Access the application at: http://localhost:3000")
    print("3. Login with admin/admin123 or viewer/viewer123")

if __name__ == "__main__":
    main()
