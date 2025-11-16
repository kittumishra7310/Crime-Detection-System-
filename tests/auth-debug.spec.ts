import { test, expect } from '@playwright/test';

// Test configuration
const BASE_URL = 'http://localhost:3000';
const API_URL = 'http://localhost:8000';

test.describe('Authentication Debug Tests', () => {
  test('Debug authentication flow and API calls', async ({ page }) => {
    console.log('=== Starting Authentication Debug Test ===');
    
    // Capture console logs
    const logs: string[] = [];
    page.on('console', msg => {
      const text = `[${msg.type()}] ${msg.text()}`;
      logs.push(text);
      console.log(text);
    });
    
    // Capture network errors
    const errors: any[] = [];
    page.on('response', response => {
      if (response.status() >= 400) {
        errors.push({
          url: response.url(),
          status: response.status(),
          statusText: response.statusText()
        });
        console.error(`❌ Network error: ${response.status()} ${response.statusText()} - ${response.url()}`);
      }
    });
    
    // Navigate to dashboard
    console.log('Navigating to dashboard...');
    await page.goto(`${BASE_URL}/dashboard`);
    await page.waitForLoadState('networkidle', { timeout: 10000 });
    
    // Check localStorage for auth data
    const localStorageData = await page.evaluate(() => {
      return {
        token: localStorage.getItem('token'),
        user: localStorage.getItem('user'),
        clerkSession: localStorage.getItem('__clerk_session')
      };
    });
    
    console.log('LocalStorage auth data:', {
      hasToken: !!localStorageData.token,
      hasUser: !!localStorageData.user,
      hasClerkSession: !!localStorageData.clerkSession
    });
    
    // Test API endpoints directly
    console.log('Testing API endpoints...');
    
    const apiEndpoints = [
      '/api/cameras',
      '/api/detections',
      '/api/backend/system/status'
    ];
    
    for (const endpoint of apiEndpoints) {
      try {
        console.log(`Testing ${endpoint}...`);
        const response = await page.evaluate(async (url) => {
          try {
            const result = await fetch(url);
            return {
              status: result.status,
              statusText: result.statusText,
              body: await result.text().catch(() => '')
            };
          } catch (error) {
            return {
              error: error.message,
              status: 0
            };
          }
        }, `${API_URL}${endpoint}`);
        
        console.log(`${endpoint}:`, {
          status: response.status,
          statusText: response.statusText,
          error: response.error
        });
        
        if (response.status === 401) {
          console.error(`❌ ${endpoint}: 401 Unauthorized`);
        } else if (response.status >= 200 && response.status < 300) {
          console.log(`✅ ${endpoint}: Success`);
        } else {
          console.warn(`⚠️  ${endpoint}: ${response.status} ${response.statusText}`);
        }
      } catch (error) {
        console.error(`❌ ${endpoint}: Failed to test`, error);
      }
    }
    
    // Test WebSocket connection
    console.log('Testing WebSocket connection...');
    
    const wsResult = await page.evaluate(async () => {
      return new Promise((resolve) => {
        let ws: WebSocket | null = null;
        const timeout = setTimeout(() => {
          if (ws) ws.close();
          resolve({
            success: false,
            error: 'Connection timeout'
          });
        }, 5000);
        
        try {
          const token = localStorage.getItem('token');
          const wsUrl = `ws://localhost:8000/ws/${token || 'null'}`;
          
          ws = new WebSocket(wsUrl);
          
          ws.onopen = () => {
            clearTimeout(timeout);
            ws!.close();
            resolve({
              success: true,
              url: wsUrl,
              hasToken: !!token
            });
          };
          
          ws.onerror = (error) => {
            clearTimeout(timeout);
            ws!.close();
            resolve({
              success: false,
              error: 'WebSocket error',
              url: wsUrl,
              hasToken: !!localStorage.getItem('token')
            });
          };
          
          ws.onclose = (event) => {
            clearTimeout(timeout);
            resolve({
              success: false,
              error: `WebSocket closed: ${event.code} ${event.reason}`,
              url: wsUrl,
              hasToken: !!localStorage.getItem('token')
            });
          };
        } catch (error) {
          clearTimeout(timeout);
          resolve({
            success: false,
            error: error.message
          });
        }
      });
    });
    
    console.log('WebSocket test result:', wsResult);
    
    // Check for React hooks errors
    console.log('Checking for React hooks errors...');
    const reactErrors = logs.filter(log => 
      log.includes('Rendered more hooks than during the previous render') ||
      log.includes('Warning: React has detected a change in the order of Hooks') ||
      log.includes('Cannot update a component')
    );
    
    if (reactErrors.length > 0) {
      console.error('React hooks errors found:', reactErrors);
    } else {
      console.log('No React hooks errors found');
    }
    
    // Check for authentication-related errors
    console.log('Checking for authentication errors...');
    const authErrors = logs.filter(log =>
      log.includes('401') ||
      log.includes('Unauthorized') ||
      log.includes('authentication') ||
      log.includes('auth')
    );
    
    if (authErrors.length > 0) {
      console.error('Authentication errors found:', authErrors);
    } else {
      console.log('No authentication errors found');
    }
    
    // Summary
    console.log('\n=== DEBUG TEST SUMMARY ===');
    console.log('Console logs captured:', logs.length);
    console.log('Network errors captured:', errors.length);
    console.log('React hooks errors:', reactErrors.length);
    console.log('Authentication errors:', authErrors.length);
    
    // Save debug report
    const debugReport = {
      timestamp: new Date().toISOString(),
      localStorageData,
      apiEndpoints: ['/api/cameras', '/api/detections', '/api/backend/system/status'].map(endpoint => ({ endpoint, status: 'tested' })),
      websocketResult: wsResult,
      consoleLogs: logs.slice(-50),
      networkErrors: errors,
      reactErrors,
      authErrors
    };
    
    console.log('Debug report:', JSON.stringify(debugReport, null, 2));
    
    // Fail the test if critical errors are found
    const hasCriticalErrors = errors.length > 0 || reactErrors.length > 0;
    
    if (hasCriticalErrors) {
      console.error('❌ Critical errors found during debug test');
      console.log('Network errors:', errors);
      console.log('Console logs (last 20):', logs.slice(-20));
    } else {
      console.log('✅ No critical errors found');
    }
    
    expect(hasCriticalErrors).toBe(false);
  });
});