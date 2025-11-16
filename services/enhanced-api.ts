// Enhanced authentication service for Clerk + JWT integration
import { apiService } from './api';

class AuthSyncService {
  private syncInProgress = false;
  private syncAttempts = 0;
  private maxSyncAttempts = 3;

  /**
   * Sync Clerk authentication with backend JWT system
   * This ensures the frontend has a valid JWT token for API calls
   */
  async syncClerkWithBackend(): Promise<boolean> {
    if (typeof window === 'undefined') {
      console.log('AuthSync: Server-side rendering detected, skipping sync');
      return false;
    }

    if (this.syncInProgress) {
      console.log('AuthSync: Sync already in progress');
      return false;
    }

    // Check if we already have a valid token
    const existingToken = localStorage.getItem('token');
    if (existingToken && this.isTokenValid(existingToken)) {
      console.log('AuthSync: Valid token already exists');
      return true;
    }

    this.syncInProgress = true;
    this.syncAttempts++;

    try {
      console.log('AuthSync: Starting Clerk sync...');

      // Wait for Clerk to be available
      await this.waitForClerk();
      
      const clerk = (window as any).Clerk;
      if (!clerk?.user) {
        console.log('AuthSync: No Clerk user found');
        return false;
      }

      console.log('AuthSync: Found Clerk user:', {
        id: clerk.user.id,
        username: clerk.user.username,
        email: clerk.user.emailAddresses?.[0]?.emailAddress
      });

      // Generate backend credentials from Clerk user
      const clerkUserId = clerk.user.id;
      const email = clerk.user.emailAddresses?.[0]?.emailAddress || `clerk_${clerkUserId}@clerk.local`;
      const username = clerk.user.username || `clerk_user_${clerkUserId.substring(0, 8)}`;
      const password = 'clerk_auth_' + clerkUserId;

      // Try to register/login with backend
      const success = await this.authenticateWithBackend(username, email, password);
      
      if (success) {
        console.log('✅ AuthSync: Successfully synced with backend');
        this.syncAttempts = 0; // Reset attempts on success
      } else {
        console.log('❌ AuthSync: Failed to sync with backend');
      }

      return success;
    } catch (error) {
      console.error('AuthSync: Error during sync:', error);
      return false;
    } finally {
      this.syncInProgress = false;
    }
  }

  /**
   * Wait for Clerk to be available
   */
  private async waitForClerk(): Promise<void> {
    return new Promise((resolve) => {
      const checkClerk = () => {
        if ((window as any).Clerk?.loaded) {
          resolve();
        } else {
          setTimeout(checkClerk, 100);
        }
      };
      checkClerk();
    });
  }

  /**
   * Authenticate with backend using Clerk user data
   */
  private async authenticateWithBackend(username: string, email: string, password: string): Promise<boolean> {
    try {
      // Try registration first
      const registerResponse = await fetch(`http://localhost:8000/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username,
          email,
          password,
          role: 'viewer'
        }),
      });

      if (registerResponse.ok) {
        const data = await registerResponse.json();
        if (data.access_token) {
          localStorage.setItem('token', data.access_token);
          localStorage.setItem('user', JSON.stringify(data));
          console.log('AuthSync: User registered and token obtained');
          return true;
        }
      }

      // If registration fails, try login
      console.log('AuthSync: Registration failed, trying login...');
      const loginResponse = await fetch(`http://localhost:8000/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username,
          password
        }),
      });

      if (loginResponse.ok) {
        const data = await loginResponse.json();
        if (data.access_token) {
          localStorage.setItem('token', data.access_token);
          localStorage.setItem('user', JSON.stringify(data));
          console.log('AuthSync: User logged in and token obtained');
          return true;
        }
      }

      console.log('AuthSync: Both registration and login failed');
      return false;
    } catch (error) {
      console.error('AuthSync: Authentication error:', error);
      return false;
    }
  }

  /**
   * Check if a JWT token is valid (not expired)
   */
  private isTokenValid(token: string): boolean {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      const currentTime = Math.floor(Date.now() / 1000);
      return payload.exp > currentTime;
    } catch (error) {
      console.log('AuthSync: Token validation failed:', error);
      return false;
    }
  }

  /**
   * Force a fresh authentication sync
   */
  async forceSync(): Promise<boolean> {
    console.log('AuthSync: Forcing authentication sync...');
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    return this.syncClerkWithBackend();
  }

  /**
   * Get current authentication status
   */
  getAuthStatus() {
    if (typeof window === 'undefined') {
      return { hasToken: false, hasClerk: false, isValid: false };
    }

    const token = localStorage.getItem('token');
    const clerk = (window as any).Clerk;
    
    return {
      hasToken: !!token,
      hasClerk: !!clerk?.user,
      isValid: token ? this.isTokenValid(token) : false,
      clerkUser: clerk?.user ? {
        id: clerk.user.id,
        username: clerk.user.username,
        email: clerk.user.emailAddresses?.[0]?.emailAddress
      } : null
    };
  }
}

// Create singleton instance
export const authSyncService = new AuthSyncService();

// Enhanced API service that ensures authentication before requests
export class EnhancedApiService {
  private apiService = apiService;

  /**
   * Ensure authentication is synced before making API calls
   */
  private async ensureAuthenticated(): Promise<boolean> {
    const authStatus = authSyncService.getAuthStatus();
    
    if (!authStatus.hasClerk) {
      console.log('EnhancedAPI: No Clerk user, cannot authenticate');
      return false;
    }

    if (!authStatus.hasToken || !authStatus.isValid) {
      console.log('EnhancedAPI: Token missing or invalid, syncing...');
      return await authSyncService.syncClerkWithBackend();
    }

    return true;
  }

  /**
   * Enhanced API methods that ensure authentication
   */
  async getCameras() {
    if (!(await this.ensureAuthenticated())) {
      throw new Error('Authentication required');
    }
    return this.apiService.getCameras();
  }

  async getDetections() {
    if (!(await this.ensureAuthenticated())) {
      throw new Error('Authentication required');
    }
    return this.apiService.getDetections();
  }

  async getAlerts() {
    if (!(await this.ensureAuthenticated())) {
      throw new Error('Authentication required');
    }
    return this.apiService.getAlerts();
  }

  async startLiveDetection(cameraId: number, source: string = "0") {
    if (!(await this.ensureAuthenticated())) {
      throw new Error('Authentication required');
    }
    return this.apiService.startLiveDetection(cameraId, source);
  }

  async stopLiveDetection(cameraId: number) {
    if (!(await this.ensureAuthenticated())) {
      throw new Error('Authentication required');
    }
    return this.apiService.stopLiveDetection(cameraId);
  }

  async stopAllLiveDetection() {
    if (!(await this.ensureAuthenticated())) {
      throw new Error('Authentication required');
    }
    return this.apiService.stopAllLiveDetection();
  }

  async getLiveDetectionStatus() {
    if (!(await this.ensureAuthenticated())) {
      throw new Error('Authentication required');
    }
    return this.apiService.getLiveDetectionStatus();
  }

  async uploadFile(file: File, cameraId: number = 1) {
    if (!(await this.ensureAuthenticated())) {
      throw new Error('Authentication required');
    }
    return this.apiService.uploadFile(file, cameraId);
  }

  /**
   * Get authentication status
   */
  getAuthStatus() {
    return authSyncService.getAuthStatus();
  }

  /**
   * Force authentication sync
   */
  async forceAuthSync() {
    return authSyncService.forceSync();
  }
}

// Create enhanced API service instance
export const enhancedApiService = new EnhancedApiService();

// Export for convenience
export { authSyncService };