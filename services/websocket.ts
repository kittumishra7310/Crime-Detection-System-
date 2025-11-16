// WebSocket service for real-time crime detection
import { apiService } from './api';

export interface WebSocketMessage {
  type: string;
  data?: any;
  camera_id?: number;
  timestamp?: string;
}

export interface DetectionUpdate {
  camera_id: number;
  detection_id: string;
  crime_type: string;
  confidence: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string;
  frame_data?: string;
  is_real_time: boolean;
}

export interface CameraStatus {
  camera_id: number;
  status: 'active' | 'inactive' | 'error';
  source: string;
  fps: number;
  total_frames: number;
  detections_count: number;
  last_detection?: string;
  error_message?: string;
}

export interface PerformanceMetrics {
  avg_inference_time_ms: number;
  avg_fps: number;
  total_processed_frames: number;
  total_detections: number;
  active_cameras: number;
  system_load: number;
  memory_usage_mb: number;
  gpu_usage_percent?: number;
}

export interface SystemAlert {
  id: string;
  type: 'detection' | 'system' | 'error';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  camera_id?: number;
  timestamp: string;
  data?: any;
}

class WebSocketService {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private messageHandlers: Map<string, Set<(data: any) => void>> = new Map();
  private connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error' = 'disconnected';
  private connectionListeners: Set<(status: string) => void> = new Set();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private lastPingTime = 0;
  private messageQueue: WebSocketMessage[] = [];
  private isProcessingQueue = false;

  constructor() {
    if (typeof window !== 'undefined') {
      this.setupEventListeners();
    }
  }

  private setupEventListeners() {
    if (typeof window === 'undefined') return;
    
    // Handle auth changes
    window.addEventListener('auth-changed', () => {
      if (this.connectionStatus === 'connected') {
        this.disconnect();
        this.connect();
      }
    });

    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && this.connectionStatus === 'connected') {
        // Reduce activity when page is hidden
        this.stopHeartbeat();
      } else if (!document.hidden && this.connectionStatus === 'connected') {
        // Resume activity when page is visible
        this.startHeartbeat();
      }
    });
  }

  private getWebSocketUrl(): string {
    if (typeof window === 'undefined') {
      return ''; // Return empty string for server-side
    }
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    const port = process.env.NEXT_PUBLIC_WS_PORT || '8000';
    const token = localStorage.getItem('token');
    
    return token ? `${protocol}//${host}:${port}/ws/${token}` : '';
  }

  public connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.connectionStatus === 'connected') {
        resolve();
        return;
      }

      if (this.connectionStatus === 'connecting') {
        // Wait for current connection attempt to complete
        const checkConnection = setInterval(() => {
          if (this.connectionStatus !== 'connecting') {
            clearInterval(checkConnection);
            if (this.connectionStatus === 'connected') {
              resolve();
            } else {
              reject(new Error('Connection failed'));
            }
          }
        }, 100);
        return;
      }

      this.connectionStatus = 'connecting';
      this.notifyConnectionStatus('connecting');

      try {
        // Wait a bit for authentication to be ready
        setTimeout(() => {
          const wsUrl = this.getWebSocketUrl();
          
          if (!wsUrl) {
            const error = new Error('Cannot create WebSocket URL - token may be missing or running server-side');
            this.connectionStatus = 'error';
            this.notifyConnectionStatus('error');
            reject(error);
            return;
          }
          
          this.socket = new WebSocket(wsUrl);

          this.socket.onopen = () => {
            console.log('WebSocket connected');
            this.connectionStatus = 'connected';
            this.reconnectAttempts = 0;
            this.notifyConnectionStatus('connected');
            this.startHeartbeat();
            this.processMessageQueue();
            resolve();
          };

          this.socket.onmessage = (event) => {
            try {
              const message = JSON.parse(event.data);
              this.handleMessage(message);
            } catch (error) {
              console.error('Error parsing WebSocket message:', error);
            }
          };

          this.socket.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.connectionStatus = 'disconnected';
            this.notifyConnectionStatus('disconnected');
            this.stopHeartbeat();
            
            // Attempt reconnection if not a normal close
            if (event.code !== 1000 && event.code !== 1001) {
              this.scheduleReconnect();
            }
          };

          this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.connectionStatus = 'error';
            this.notifyConnectionStatus('error');
            reject(new Error('WebSocket connection failed'));
          };
        }, 1000); // Wait 1 second for authentication to be ready
      } catch (error) {
        this.connectionStatus = 'error';
        this.notifyConnectionStatus('error');
        reject(error);
      }
    });
  }

  public disconnect(): void {
    this.stopHeartbeat();
    if (this.socket) {
      this.socket.close(1000, 'User disconnected');
      this.socket = null;
    }
    this.connectionStatus = 'disconnected';
    this.notifyConnectionStatus('disconnected');
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.notifyConnectionStatus('error');
      
      // Notify all listeners about connection failure
      this.notifyListeners('connection_failed', {
        attempts: this.reconnectAttempts,
        maxAttempts: this.maxReconnectAttempts,
        timestamp: new Date().toISOString()
      });
      return;
    }

    this.reconnectAttempts++;
    
    // Exponential backoff with jitter to prevent thundering herd
    const baseDelay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
    const jitter = Math.random() * 0.1 * baseDelay; // Add up to 10% jitter
    const delay = baseDelay + jitter;
    
    console.log(`Scheduling reconnection attempt ${this.reconnectAttempts} in ${Math.round(delay)}ms`);
    
    setTimeout(() => {
      this.connect().then(() => {
        // Reset reconnect attempts on successful connection
        this.reconnectAttempts = 0;
        console.log('Reconnection successful');
      }).catch(error => {
        console.error('Reconnection attempt failed:', error);
        // Continue with next reconnection attempt
        this.scheduleReconnect();
      });
    }, delay);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatInterval = setInterval(() => {
      if (this.connectionStatus === 'connected' && this.socket) {
        this.lastPingTime = Date.now();
        this.sendMessage({ type: 'ping' });
      }
    }, 30000); // 30 second heartbeat
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private handleMessage(message: WebSocketMessage): void {
    const startTime = Date.now();
    
    try {
      // Validate message structure
      if (!message || typeof message.type !== 'string') {
        console.warn('Invalid message structure:', message);
        return;
      }
      
      // Ensure message has required properties
      if (!message.hasOwnProperty('data') && message.type !== 'pong' && message.type !== 'ping') {
        console.warn('Message missing data property:', message.type);
      }
      
      // Handle different message types with validation
      switch (message.type) {
        case 'pong':
          const latency = Date.now() - this.lastPingTime;
          this.notifyListeners('latency', latency);
          break;
          
        case 'detection_update':
          if (this.validateDetectionData(message.data)) {
            this.notifyListeners('detection', message.data);
          } else {
            console.warn('Invalid detection data:', message.data);
          }
          break;
          
        case 'detection_status':
          // Handle detection status updates (buffering, normal, etc.)
          this.notifyListeners('detection_status', message.data);
          break;
          
        case 'buffering_status':
          // Handle frame buffering status
          this.notifyListeners('buffering_status', message.data);
          break;
          
        case 'detection_result':
          // Handle detection results (crimes detected)
          if (message.data) {
            this.notifyListeners('detection', message.data);
          }
          break;
          
        case 'camera_status':
          if (this.validateCameraStatusData(message.data)) {
            this.notifyListeners('camera_status', message.data);
          } else {
            console.warn('Invalid camera status data:', message.data);
          }
          break;
          
        case 'performance_metrics':
          if (this.validatePerformanceData(message.data)) {
            this.notifyListeners('performance', message.data);
          } else {
            console.warn('Invalid performance data:', message.data);
          }
          break;
          
        case 'system_alert':
          if (this.validateSystemAlertData(message.data)) {
            this.notifyListeners('alert', message.data);
          } else {
            console.warn('Invalid system alert data:', message.data);
          }
          break;
          
        case 'detection_history':
          if (Array.isArray(message.data)) {
            this.notifyListeners('history', message.data);
          } else {
            console.warn('Invalid history data:', message.data);
          }
          break;
          
        case 'error':
          const errorData = message.data || message.error || 'Unknown error';
          console.error('Server error:', errorData);
          this.notifyListeners('error', errorData);
          break;
          
        default:
          // For unknown message types, still notify but with warning
          console.warn('Unknown message type:', message.type);
          this.notifyListeners(message.type, message.data);
      }
      
      // Log slow message processing
      const processingTime = Date.now() - startTime;
      if (processingTime > 50) {
        console.warn(`Slow message processing: ${processingTime}ms for ${message.type}`);
      }
      
    } catch (error) {
      console.error('Error handling WebSocket message:', error, message);
    }
  }
  
  private validateDetectionData(data: any): data is DetectionUpdate {
    return (
      data &&
      typeof data.camera_id === 'number' &&
      typeof data.detection_id === 'string' &&
      typeof data.crime_type === 'string' &&
      typeof data.confidence === 'number' &&
      data.confidence >= 0 && data.confidence <= 1 &&
      ['low', 'medium', 'high', 'critical'].includes(data.severity) &&
      typeof data.timestamp === 'string'
    );
  }
  
  private validateCameraStatusData(data: any): data is CameraStatus {
    return (
      data &&
      typeof data.camera_id === 'number' &&
      typeof data.status === 'string' &&
      ['online', 'offline', 'error', 'buffering'].includes(data.status)
    );
  }
  
  private validateSystemAlertData(data: any): data is SystemAlert {
    return (
      data &&
      typeof data.message === 'string' &&
      typeof data.severity === 'string' &&
      ['low', 'medium', 'high', 'critical'].includes(data.severity) &&
      typeof data.timestamp === 'string'
    );
  }
  
  private validatePerformanceData(data: any): data is PerformanceMetrics {
    return (
      data &&
      typeof data.camera_id === 'number' &&
      typeof data.fps === 'number' &&
      typeof data.processing_time === 'number' &&
      data.fps >= 0 && data.processing_time >= 0
    );
  }

  public sendMessage(message: WebSocketMessage): void {
    if (this.connectionStatus === 'connected' && this.socket && this.socket.readyState === WebSocket.OPEN) {
      try {
        this.socket.send(JSON.stringify(message));
      } catch (error) {
        console.error('Error sending WebSocket message:', error);
        this.messageQueue.push(message);
      }
    } else {
      // Queue message if not connected
      this.messageQueue.push(message);
    }
  }

  private async processMessageQueue(): Promise<void> {
    if (this.isProcessingQueue || this.messageQueue.length === 0) {
      return;
    }

    this.isProcessingQueue = true;

    while (this.messageQueue.length > 0 && this.connectionStatus === 'connected') {
      const message = this.messageQueue.shift();
      if (message) {
        this.sendMessage(message);
      }
      
      // Small delay to prevent overwhelming the connection
      await new Promise(resolve => setTimeout(resolve, 10));
    }

    this.isProcessingQueue = false;
  }

  // Subscribe to specific message types
  public subscribe(type: string, handler: (data: any) => void): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, new Set());
    }
    
    this.messageHandlers.get(type)!.add(handler);
    
    // Return unsubscribe function
    return () => {
      const handlers = this.messageHandlers.get(type);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          this.messageHandlers.delete(type);
        }
      }
    };
  }

  private notifyListeners(type: string, data: any): void {
    const handlers = this.messageHandlers.get(type);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in ${type} handler:`, error);
        }
      });
    }
  }

  private notifyConnectionStatus(status: string): void {
    this.connectionListeners.forEach(listener => {
      try {
        listener(status);
      } catch (error) {
        console.error('Error in connection status listener:', error);
      }
    });
  }

  public onConnectionStatusChange(handler: (status: string) => void): () => void {
    this.connectionListeners.add(handler);
    
    // Return unsubscribe function
    return () => {
      this.connectionListeners.delete(handler);
    };
  }

  // Convenience methods for common operations
  public startDetection(cameraId: number, source: string = "0"): void {
    this.sendMessage({
      type: "start_detection",
      data: { camera_id: cameraId, source },
      camera_id: cameraId
    });
  }

  public stopDetection(cameraId: number): void {
    this.sendMessage({
      type: "stop_detection",
      data: { camera_id: cameraId },
      camera_id: cameraId
    });
  }

  public getStatus(cameraId?: number): void {
    this.sendMessage({
      type: "get_status",
      data: cameraId ? { camera_id: cameraId } : {},
      camera_id: cameraId
    });
  }

  public getDetectionHistory(cameraId?: number, hours: number = 24): void {
    this.sendMessage({
      type: "get_detection_history",
      data: { 
        camera_id: cameraId, 
        hours,
        timestamp: new Date().toISOString()
      },
      camera_id: cameraId
    });
  }

  public updateConfidenceThreshold(cameraId: number, threshold: number): void {
    this.sendMessage({
      type: "update_confidence_threshold",
      data: { camera_id: cameraId, threshold },
      camera_id: cameraId
    });
  }

  public subscribeToCamera(cameraId: number): void {
    this.sendMessage({
      type: "subscribe_camera",
      data: { camera_id: cameraId },
      camera_id: cameraId
    });
  }

  public unsubscribeFromCamera(cameraId: number): void {
    this.sendMessage({
      type: "unsubscribe_camera",
      data: { camera_id: cameraId },
      camera_id: cameraId
    });
  }

  // Getters for connection status
  public getConnectionStatus(): string {
    return this.connectionStatus;
  }

  public isConnected(): boolean {
    return this.connectionStatus === 'connected';
  }

  public isConnecting(): boolean {
    return this.connectionStatus === 'connecting';
  }

  // Cleanup method
  public destroy(): void {
    this.disconnect();
    this.messageHandlers.clear();
    this.connectionListeners.clear();
    this.messageQueue = [];
  }
}

// Create singleton instance
export const websocketService = new WebSocketService();

// React hook for WebSocket connection
export function useWebSocket() {
  const [connectionStatus, setConnectionStatus] = useState(websocketService.getConnectionStatus());
  const [isConnected, setIsConnected] = useState(websocketService.isConnected());

  useEffect(() => {
    const unsubscribe = websocketService.onConnectionStatusChange((status) => {
      setConnectionStatus(status);
      setIsConnected(status === 'connected');
    });

    // Auto-connect on mount if not connected
    if (!websocketService.isConnected() && !websocketService.isConnecting()) {
      websocketService.connect().catch(console.error);
    }

    return () => {
      unsubscribe();
    };
  }, []);

  return {
    websocketService,
    connectionStatus,
    isConnected,
    connect: () => websocketService.connect(),
    disconnect: () => websocketService.disconnect()
  };
}

// React hook for subscribing to WebSocket messages
export function useWebSocketSubscription<T>(
  type: string, 
  handler: (data: T) => void,
  dependencies: any[] = []
) {
  // Use useCallback to ensure handler identity is stable
  const stableHandler = useCallback(handler, dependencies);
  
  useEffect(() => {
    if (!type) return; // Don't subscribe if type is falsy
    
    const unsubscribe = websocketService.subscribe(type, stableHandler);
    return () => {
      unsubscribe();
    };
  }, [type, stableHandler]);
}

export default websocketService;