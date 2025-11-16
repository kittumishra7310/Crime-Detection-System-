// API service for backend communication
const API_BASE_URL = "http://localhost:8000" // Direct backend connection

class ApiService {
  private getToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem("token");
  }

  private async getAuthHeaders() {
    // Try to sync with backend first to ensure we have a valid token
    // But don't fail if sync doesn't work - use existing token if available
    try {
      await this.syncClerkWithBackend()
    } catch (syncError) {
      console.warn("Auth sync failed, using existing token if available:", syncError)
    }
    
    const token = this.getToken()
    return {
      "Content-Type": "application/json",
      ...(token && { Authorization: `Bearer ${token}` }),
    }
  }

  private async getFileUploadHeaders() {
    const token = this.getToken()
    return {
      ...(token && { Authorization: `Bearer ${token}` }),
    }
  }

  // Method to sync Clerk authentication with backend JWT
  public async syncClerkWithBackend() {
    if (typeof window === 'undefined') return;
    
    try {
      // Check if we have a token already
      let token = localStorage.getItem("token")
      if (token) {
        console.log("Token already exists, skipping sync")
        return
      }
      
      // Simple approach: check if we're in a browser with Clerk
      const clerk = (window as any).Clerk;
      if (!clerk?.user) {
        console.log("No Clerk user found, skipping sync")
        return
      }
      
      console.log("Syncing Clerk user with backend:", {
        id: clerk.user.id,
        username: clerk.user.username,
        email: clerk.user.emailAddresses?.[0]?.emailAddress
      });
      
      // Use Clerk user ID as the username to ensure uniqueness
      const clerkUserId = clerk.user.id
      const email = clerk.user.emailAddresses?.[0]?.emailAddress || `clerk_${clerkUserId}@clerk.local`
      const username = clerk.user.username || `clerk_user_${clerkUserId.substring(0, 8)}`
      const password = 'clerk_auth_' + clerkUserId
      
      // Try to register/login the user with backend
      const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: username,
          email: email,
          password: password,
          role: 'viewer'
        }),
      });
      
      if (response.ok) {
        const data = await response.json()
        if (data.access_token) {
          localStorage.setItem("token", data.access_token)
          localStorage.setItem("user", JSON.stringify(data))
          console.log("✓ User registered and token obtained")
          return
        }
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.warn("Registration failed:", errorData);
      }
      
      // If registration fails, try login
      const loginResponse = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: username,
          password: password
        }),
      });
      
      if (loginResponse.ok) {
        const data = await loginResponse.json()
        if (data.access_token) {
          localStorage.setItem("token", data.access_token)
          localStorage.setItem("user", JSON.stringify(data))
          console.log("✓ User logged in and token obtained")
        }
      } else {
        const errorData = await loginResponse.json().catch(() => ({}));
        console.warn("Login failed:", errorData);
      }
      
    } catch (error) {
      console.warn("Failed to sync Clerk with backend:", error)
    }
  }

  // Authentication - Now handled by Clerk
  // These methods are kept for backward compatibility but won't be used
  async login(username: string, password: string) {
    throw new Error("Authentication is handled by Clerk. Please use Clerk's login flow.")
  }

  async register(username: string, email: string, password: string, role: string = "viewer") {
    throw new Error("Registration is handled by Clerk. Please use Clerk's registration flow.")
  }

  async logout() {
    // Clerk handles logout - this is just a placeholder
    return Promise.resolve()
  }

  // System Health
  async getHealth() {
    const response = await fetch(`${API_BASE_URL}/api/system/status`, {
      headers: await this.getAuthHeaders()
    });
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
  }

  // Cameras
  async getCameras() {
    const headers = await this.getAuthHeaders()
    
    const response = await fetch(`${API_BASE_URL}/api/cameras`, {
      headers,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to fetch cameras: ${response.statusText}`);
    }

    return await response.json();
  }

  // Detections
  async getDetections() {
    const headers = await this.getAuthHeaders()
    
    const response = await fetch(`${API_BASE_URL}/api/detections`, {
      headers,
    })
    return response.json()
  }

  // Alerts
  async getAlerts() {
    // Sync Clerk authentication with backend
    await this.syncClerkWithBackend()
    
    const response = await fetch(`${API_BASE_URL}/api/alerts`, {
      headers: await this.getAuthHeaders(),
    })
    return response.json()
  }

  // File Upload
  async uploadFile(file: File, cameraId: number = 1) {
    // Sync Clerk authentication with backend
    await this.syncClerkWithBackend()
    
    const formData = new FormData()
    formData.append('file', file)
    formData.append('camera_id', cameraId.toString())

    const response = await fetch(`${API_BASE_URL}/api/detection/upload`, {
      method: "POST",
      headers: await this.getFileUploadHeaders(),
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
      // Sync Clerk authentication with backend
      await this.syncClerkWithBackend()
      
      const response = await fetch(`${API_BASE_URL}/live/start/${cameraId}?source=${source}`, {
        method: "POST",
        headers: await this.getAuthHeaders(),
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
      // Sync Clerk authentication with backend
      await this.syncClerkWithBackend()
      
      const response = await fetch(`${API_BASE_URL}/live/stop/${cameraId}`, {
        method: "POST",
        headers: await this.getAuthHeaders(),
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

  async stopAllLiveDetection() {
    try {
      // Sync Clerk authentication with backend
      await this.syncClerkWithBackend()
      
      const response = await fetch(`${API_BASE_URL}/live/stop-all`, {
        method: "POST",
        headers: await this.getAuthHeaders(),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to stop all detections: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error in stopAllLiveDetection:', error);
      throw error;
    }
  }

  async getLiveDetectionStatus() {
    // Sync Clerk authentication with backend
    await this.syncClerkWithBackend()
    
    const response = await fetch(`${API_BASE_URL}/live/status`, {
      headers: await this.getAuthHeaders(),
    })
    return response.json()
  }

  // Get live camera feed URL with authentication token
  getLiveFeedUrl(cameraId: number) {
    const token = this.getToken();
    if (!token) {
      console.warn('No authentication token available for camera feed');
      return `${API_BASE_URL}/live/feed/${cameraId}`;
    }
    return `${API_BASE_URL}/live/feed/${cameraId}?token=${encodeURIComponent(token)}`;
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
