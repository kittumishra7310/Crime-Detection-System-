"use client"

// React hooks for WebSocket - Client-only imports
import { useState, useEffect, useCallback } from 'react';
import { websocketService } from './websocket';

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