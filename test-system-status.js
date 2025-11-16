#!/usr/bin/env node

/**
 * Test the corrected system status endpoint
 */

const http = require('http');

// Helper function to make HTTP requests
function makeRequest(options, postData = null, token = null) {
  return new Promise((resolve, reject) => {
    if (token) {
      options.headers = options.headers || {};
      options.headers['Authorization'] = `Bearer ${token}`;
    }
    
    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        resolve({
          status: res.statusCode,
          statusText: res.statusMessage,
          headers: res.headers,
          body: data
        });
      });
    });
    
    req.on('error', (error) => {
      reject(error);
    });
    
    if (postData) {
      req.write(postData);
    }
    
    req.end();
  });
}

async function testSystemStatus() {
  console.log('=== Testing System Status Endpoint ===');
  
  // First, get a JWT token
  console.log('Getting authentication token...');
  const loginOptions = {
    hostname: 'localhost',
    port: 8000,
    path: '/api/auth/login',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    }
  };
  
  const loginData = {
    username: 'test_user_1763210728901', // Use the user from previous test
    password: 'test_password'
  };
  
  let token = null;
  try {
    const loginResponse = await makeRequest(loginOptions, JSON.stringify(loginData));
    console.log('Login response:', {
      status: loginResponse.status,
      statusText: loginResponse.statusText
    });
    
    if (loginResponse.status === 200) {
      const data = JSON.parse(loginResponse.body);
      token = data.access_token;
      console.log('✅ Token obtained successfully');
    }
  } catch (error) {
    console.error('❌ Login failed:', error.message);
    return;
  }
  
  if (!token) {
    console.error('❌ No token obtained');
    return;
  }
  
  // Test the corrected system status endpoint
  console.log('\nTesting system status endpoint...');
  
  const endpoints = [
    '/api/system/status',
    '/api/backend/system/status' // Old incorrect path
  ];
  
  for (const endpoint of endpoints) {
    try {
      console.log(`Testing ${endpoint}...`);
      const options = {
        hostname: 'localhost',
        port: 8000,
        path: endpoint,
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      };
      
      const response = await makeRequest(options, null, token);
      
      console.log(`${endpoint}:`, {
        status: response.status,
        statusText: response.statusText,
        body: response.body.substring(0, 300)
      });
      
      if (response.status === 200) {
        console.log(`✅ ${endpoint}: Success`);
      } else if (response.status === 404) {
        console.log(`❌ ${endpoint}: Not found`);
      } else if (response.status === 401) {
        console.log(`❌ ${endpoint}: Unauthorized`);
      } else if (response.status === 403) {
        console.log(`❌ ${endpoint}: Forbidden`);
      } else {
        console.warn(`⚠️  ${endpoint}: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      console.error(`❌ ${endpoint}: Failed to connect - ${error.message}`);
    }
  }
}

// Run the test
testSystemStatus().catch(error => {
  console.error('❌ System status test failed:', error);
  process.exit(1);
});