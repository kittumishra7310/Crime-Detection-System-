// API service for backend communication
const API_BASE_URL = "" // All requests are now relative to the frontend server

class ApiService {
  private getAuthHeaders() {
    const token = localStorage.getItem("token")
    return {
      "Content-Type": "application/json",
      ...(token && { Authorization: `Bearer ${token}` }),
    }
  }

  private getFileUploadHeaders() {
    const token = localStorage.getItem("token")
    return {
      ...(token && { Authorization: `Bearer ${token}` }),
    }
  }

  // Authentication
  async login(username: string, password: string) {
    const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    })
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `Login failed: ${response.statusText}`);
    }
    
    const data = await response.json()
    if (data.access_token) {
      localStorage.setItem("token", data.access_token)
      localStorage.setItem("user", JSON.stringify(data.user))
    }
    return data
  }

  async register(username: string, email: string, password: string, role: string = "viewer") {
    const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password, role }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Registration failed");
    }

    return response.json();
  }

  async logout() {
    localStorage.removeItem("token")
    localStorage.removeItem("user")
  }

  // System Health
  async getHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
  }

  // Cameras
  async getCameras() {
    try {
      const response = await fetch(`/api/proxy/cameras`, {
        headers: this.getAuthHeaders(),
      });

      if (!response.ok) {
        if (response.status === 401 || response.status === 403) {
          // Handle unauthorized/forbidden (e.g., redirect to login)
          this.logout();
          window.location.href = '/login';
          throw new Error('Session expired. Please log in again.');
        }
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to fetch cameras: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error in getCameras:', error);
      throw error;
    }
  }

  // Detections
  async getDetections() {
    const response = await fetch(`${API_BASE_URL}/api/detections`, {
      headers: this.getAuthHeaders(),
    })
    return response.json()
  }

  // Alerts
  async getAlerts() {
    const response = await fetch(`${API_BASE_URL}/api/alerts`, {
      headers: this.getAuthHeaders(),
    })
    return response.json()
  }

  // File Upload
  async uploadFile(file: File, cameraId: number = 1) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('camera_id', cameraId.toString())

    const response = await fetch(`${API_BASE_URL}/api/detection/upload`, {
      method: "POST",
      headers: this.getFileUploadHeaders(),
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`)
    }

    return response.json()
  }

  // Live Detection
  async startLiveDetection(cameraId: number, source: string = "0") {
    try {
      const response = await fetch(`/api/proxy/live/start/${cameraId}?source=${source}`, {
        method: "POST",
        headers: this.getAuthHeaders(),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to start live detection: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error in startLiveDetection:', error);
      throw error;
    }
  }

  async stopLiveDetection(cameraId: number) {
    try {
      const response = await fetch(`/api/proxy/live/stop/${cameraId}`, {
        method: "POST",
        headers: this.getAuthHeaders(),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to stop live detection: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error in stopLiveDetection:', error);
      throw error;
    }
  }

  async getLiveDetectionStatus() {
    const response = await fetch(`${API_BASE_URL}/live/status`, {
      headers: this.getAuthHeaders(),
    })
    return response.json()
  }

  // Get live camera feed URL with authentication token
  getLiveFeedUrl(cameraId: number) {
    // This points to the Next.js proxy route for the live feed
    return `/api/proxy/live/feed/${cameraId}`;
  }


  // Legacy methods for backward compatibility
  async getLiveDetection() {
    return this.getLiveDetectionStatus()
  }

  async getHistory() {
    return this.getDetections()
  }
}

export const apiService = new ApiService()
