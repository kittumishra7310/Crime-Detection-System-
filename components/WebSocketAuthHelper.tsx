"use client"

import { useEffect, useState } from 'react';
import { websocketService } from '@/services/websocket';
import { enhancedApiService } from '@/services/enhanced-api';

export function WebSocketAuthHelper({ children }: { children: React.ReactNode }) {
  const [wsConnected, setWsConnected] = useState(false);
  const [authReady, setAuthReady] = useState(false);

  useEffect(() => {
    // Check authentication status
    const checkAuthAndConnect = async () => {
      const authStatus = enhancedApiService.getAuthStatus();
      console.log('WebSocketAuthHelper: Auth status:', authStatus);

      if (authStatus.hasToken && authStatus.isValid) {
        setAuthReady(true);
        
        // Connect WebSocket
        try {
          await websocketService.connect();
          setWsConnected(true);
          console.log('✅ WebSocketAuthHelper: WebSocket connected successfully');
        } catch (error) {
          console.error('❌ WebSocketAuthHelper: WebSocket connection failed:', error);
          setWsConnected(false);
        }
      } else {
        console.log('WebSocketAuthHelper: Authentication not ready, waiting...');
        setAuthReady(false);
      }
    };

    // Initial check
    checkAuthAndConnect();

    // Set up interval to check authentication status
    const authCheckInterval = setInterval(() => {
      checkAuthAndConnect();
    }, 5000); // Check every 5 seconds

    // Listen for connection status changes
    const unsubscribe = websocketService.onConnectionStatusChange((status) => {
      console.log('WebSocketAuthHelper: Connection status changed:', status);
      setWsConnected(status === 'connected');
    });

    return () => {
      clearInterval(authCheckInterval);
      unsubscribe();
      websocketService.disconnect();
    };
  }, []);

  return (
    <>
      {children}
      {/* Connection status indicator */}
      <div className="fixed bottom-4 right-4 z-50">
        <div className="bg-background border rounded-lg p-3 shadow-lg">
          <div className="flex items-center gap-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${
              wsConnected ? 'bg-green-500' : authReady ? 'bg-yellow-500' : 'bg-red-500'
            }`} />
            <span className="text-muted-foreground">
              {wsConnected ? 'WebSocket Connected' : authReady ? 'Connecting...' : 'Auth Required'}
            </span>
          </div>
        </div>
      </div>
    </>
  );
}