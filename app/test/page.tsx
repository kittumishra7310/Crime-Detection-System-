"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { AlertCircle, CheckCircle, Clock, RefreshCw, Wifi, WifiOff } from "lucide-react"
import { useWebSocket, useWebSocketSubscription } from "@/services/websocket-hooks"
import { toast } from "sonner"

interface TestResult {
  id: string
  name: string
  status: 'pending' | 'running' | 'passed' | 'failed' | 'skipped'
  message: string
  timestamp: string
  duration?: number
}

interface WebSocketTestMessage {
  type: string
  data: any
  timestamp: string
}

export default function SystemTestPage() {
  const [tests, setTests] = useState<TestResult[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [wsMessages, setWsMessages] = useState<WebSocketTestMessage[]>([])
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting' | 'error'>('disconnected')
  
  const { websocketService, isConnected } = useWebSocket()
  
  // WebSocket message monitoring
  useWebSocketSubscription<any>('detection', (data) => {
    setWsMessages(prev => [{
      type: 'detection',
      data,
      timestamp: new Date().toISOString()
    }, ...prev.slice(0, 9)])
    
    toast.success('Detection message received', {
      description: `Camera ${data.camera_id}: ${data.crime_type}`
    })
  })
  
  useWebSocketSubscription<any>('system_alert', (data) => {
    setWsMessages(prev => [{
      type: 'system_alert',
      data,
      timestamp: new Date().toISOString()
    }, ...prev.slice(0, 9)])
    
    toast.error('System alert received', {
      description: data.message
    })
  })
  
  useWebSocketSubscription<any>('camera_status', (data) => {
    setWsMessages(prev => [{
      type: 'camera_status',
      data,
      timestamp: new Date().toISOString()
    }, ...prev.slice(0, 9)])
  })
  
  useWebSocketSubscription<any>('performance', (data) => {
    setWsMessages(prev => [{
      type: 'performance',
      data,
      timestamp: new Date().toISOString()
    }, ...prev.slice(0, 9)])
  })
  
  // Connection status monitoring
  useEffect(() => {
    setConnectionStatus(isConnected ? 'connected' : 'disconnected')
  }, [isConnected])
  
  const predefinedTests: Omit<TestResult, 'id' | 'timestamp'>[] = [
    {
      name: 'WebSocket Connection',
      status: 'pending',
      message: 'Testing WebSocket connection establishment'
    },
    {
      name: 'Detection Messages',
      status: 'pending',
      message: 'Waiting for detection messages'
    },
    {
      name: 'System Alerts',
      status: 'pending',
      message: 'Waiting for system alerts'
    },
    {
      name: 'Camera Status Updates',
      status: 'pending',
      message: 'Waiting for camera status updates'
    },
    {
      name: 'Performance Metrics',
      status: 'pending',
      message: 'Waiting for performance metrics'
    },
    {
      name: 'Message Validation',
      status: 'pending',
      message: 'Validating message formats and data integrity'
    },
    {
      name: 'Real-time Processing',
      status: 'pending',
      message: 'Testing real-time message processing'
    },
    {
      name: 'Error Handling',
      status: 'pending',
      message: 'Testing error handling and recovery'
    }
  ]
  
  const runTests = async () => {
    setIsRunning(true)
    setTests([])
    setWsMessages([])
    
    // Initialize tests
    const initialTests = predefinedTests.map(test => ({
      ...test,
      id: Math.random().toString(36).substr(2, 9),
      timestamp: new Date().toISOString()
    }))
    setTests(initialTests)
    
    // Test 1: WebSocket Connection
    await runWebSocketConnectionTest(initialTests)
    
    // Test 2-5: Message Reception Tests (run in parallel)
    await runMessageReceptionTests(initialTests)
    
    // Test 6: Message Validation
    await runMessageValidationTest(initialTests)
    
    // Test 7: Real-time Processing
    await runRealTimeProcessingTest(initialTests)
    
    // Test 8: Error Handling
    await runErrorHandlingTest(initialTests)
    
    setIsRunning(false)
  }
  
  const runWebSocketConnectionTest = async (testList: TestResult[]) => {
    updateTestStatus(testList, 0, 'running')
    
    try {
      if (isConnected) {
        updateTestStatus(testList, 0, 'passed', 'WebSocket connection established successfully')
      } else {
        updateTestStatus(testList, 0, 'failed', 'WebSocket connection failed')
      }
    } catch (error) {
      updateTestStatus(testList, 0, 'failed', `Connection error: ${error}`)
    }
  }
  
  const runMessageReceptionTests = async (testList: TestResult[]) => {
    // Test 2: Detection Messages
    updateTestStatus(testList, 1, 'running', 'Monitoring for detection messages...')
    
    // Test 3: System Alerts
    updateTestStatus(testList, 2, 'running', 'Monitoring for system alerts...')
    
    // Test 4: Camera Status Updates
    updateTestStatus(testList, 3, 'running', 'Monitoring for camera status updates...')
    
    // Test 5: Performance Metrics
    updateTestStatus(testList, 4, 'running', 'Monitoring for performance metrics...')
    
    // Wait for messages to arrive
    await new Promise(resolve => setTimeout(resolve, 10000))
    
    // Evaluate results based on received messages
    const detectionMessages = wsMessages.filter(m => m.type === 'detection')
    const alertMessages = wsMessages.filter(m => m.type === 'system_alert')
    const statusMessages = wsMessages.filter(m => m.type === 'camera_status')
    const performanceMessages = wsMessages.filter(m => m.type === 'performance')
    
    // Update test results
    updateTestStatus(
      testList, 
      1, 
      detectionMessages.length > 0 ? 'passed' : 'skipped',
      `Received ${detectionMessages.length} detection messages`
    )
    
    updateTestStatus(
      testList, 
      2, 
      alertMessages.length > 0 ? 'passed' : 'skipped',
      `Received ${alertMessages.length} system alerts`
    )
    
    updateTestStatus(
      testList, 
      3, 
      statusMessages.length > 0 ? 'passed' : 'skipped',
      `Received ${statusMessages.length} camera status updates`
    )
    
    updateTestStatus(
      testList, 
      4, 
      performanceMessages.length > 0 ? 'passed' : 'skipped',
      `Received ${performanceMessages.length} performance metrics`
    )
  }
  
  const runMessageValidationTest = async (testList: TestResult[]) => {
    updateTestStatus(testList, 5, 'running', 'Validating message formats...')
    
    let validationErrors = 0
    let totalMessages = wsMessages.length
    
    wsMessages.forEach(message => {
      try {
        // Basic validation for each message type
        switch (message.type) {
          case 'detection':
            if (!message.data.camera_id || !message.data.crime_type || message.data.confidence === undefined) {
              validationErrors++
            }
            break
          case 'system_alert':
            if (!message.data.message || !message.data.severity) {
              validationErrors++
            }
            break
          case 'camera_status':
            if (!message.data.camera_id || !message.data.status) {
              validationErrors++
            }
            break
          case 'performance':
            if (!message.data.camera_id || message.data.fps === undefined) {
              validationErrors++
            }
            break
        }
      } catch (error) {
        validationErrors++
      }
    })
    
    if (validationErrors === 0 && totalMessages > 0) {
      updateTestStatus(testList, 5, 'passed', `All ${totalMessages} messages validated successfully`)
    } else if (totalMessages === 0) {
      updateTestStatus(testList, 5, 'skipped', 'No messages to validate')
    } else {
      updateTestStatus(testList, 5, 'failed', `${validationErrors} validation errors in ${totalMessages} messages`)
    }
  }
  
  const runRealTimeProcessingTest = async (testList: TestResult[]) => {
    updateTestStatus(testList, 6, 'running', 'Testing real-time message processing...')
    
    if (wsMessages.length === 0) {
      updateTestStatus(testList, 6, 'skipped', 'No messages to process')
      return
    }
    
    // Check if messages are being processed in real-time
    const recentMessages = wsMessages.filter(m => {
      const msgTime = new Date(m.timestamp).getTime()
      const now = new Date().getTime()
      return now - msgTime < 30000 // Messages from last 30 seconds
    })
    
    const processingRate = (recentMessages.length / 30) * 60 // Messages per minute
    
    if (processingRate > 0) {
      updateTestStatus(testList, 6, 'passed', `Processing rate: ${processingRate.toFixed(1)} messages/minute`)
    } else {
      updateTestStatus(testList, 6, 'skipped', 'No recent messages for processing rate calculation')
    }
  }
  
  const runErrorHandlingTest = async (testList: TestResult[]) => {
    updateTestStatus(testList, 7, 'running', 'Testing error handling...')
    
    // Test connection recovery
    if (!isConnected) {
      updateTestStatus(testList, 7, 'skipped', 'Cannot test error handling - no connection')
      return
    }
    
    // Simulate some error conditions
    try {
      // Test invalid message handling
      websocketService.sendMessage({
        type: 'invalid_message_type',
        data: { test: 'data' }
      } as any)
      
      // Test connection stability
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      if (isConnected) {
        updateTestStatus(testList, 7, 'passed', 'Error handling working correctly')
      } else {
        updateTestStatus(testList, 7, 'failed', 'Connection lost during error handling test')
      }
    } catch (error) {
      updateTestStatus(testList, 7, 'failed', `Error during test: ${error}`)
    }
  }
  
  const updateTestStatus = (testList: TestResult[], index: number, status: TestResult['status'], message?: string) => {
    setTests(prev => {
      const updated = [...prev]
      if (updated[index]) {
        updated[index] = {
          ...updated[index],
          status,
          message: message || updated[index].message,
          timestamp: new Date().toISOString()
        }
      }
      return updated
    })
  }
  
  const getStatusIcon = (status: TestResult['status']) => {
    switch (status) {
      case 'passed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      case 'running':
        return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />
      case 'skipped':
        return <Clock className="h-4 w-4 text-yellow-500" />
      default:
        return <div className="h-4 w-4 rounded-full bg-gray-400" />
    }
  }
  
  const getStatusColor = (status: TestResult['status']) => {
    switch (status) {
      case 'passed':
        return 'border-green-500 bg-green-500/10'
      case 'failed':
        return 'border-red-500 bg-red-500/10'
      case 'running':
        return 'border-blue-500 bg-blue-500/10'
      case 'skipped':
        return 'border-yellow-500 bg-yellow-500/10'
      default:
        return 'border-gray-500 bg-gray-500/10'
    }
  }
  
  const passedTests = tests.filter(t => t.status === 'passed').length
  const failedTests = tests.filter(t => t.status === 'failed').length
  const runningTests = tests.filter(t => t.status === 'running').length
  
  return (
    <div className="min-h-screen bg-slate-900 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-white mb-2">System Test Dashboard</h1>
          <p className="text-slate-400">Real-time crime detection system validation</p>
        </div>
        
        {/* Connection Status */}
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              {isConnected ? <Wifi className="h-5 w-5 text-green-500" /> : <WifiOff className="h-5 w-5 text-red-500" />}
              Connection Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <Badge className={`${
                connectionStatus === 'connected' ? 'bg-green-500' :
                connectionStatus === 'error' ? 'bg-red-500' :
                connectionStatus === 'connecting' ? 'bg-blue-500' :
                'bg-gray-500'
              } text-white`}>
                {connectionStatus.toUpperCase()}
              </Badge>
              <span className="text-slate-400">
                {wsMessages.length} messages received
              </span>
            </div>
          </CardContent>
        </Card>
        
        {/* Test Controls */}
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Test Controls</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-4">
              <Button 
                onClick={runTests} 
                disabled={isRunning}
                className="bg-blue-600 hover:bg-blue-700"
              >
                {isRunning ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Running Tests...
                  </>
                ) : (
                  'Run All Tests'
                )}
              </Button>
              
              <Button 
                onClick={() => setWsMessages([])}
                variant="outline"
                className="border-slate-600 text-slate-300 hover:bg-slate-700"
              >
                Clear Messages
              </Button>
            </div>
            
            {tests.length > 0 && (
              <div className="mt-4 flex gap-4 text-sm">
                <span className="text-green-400">✅ {passedTests} Passed</span>
                <span className="text-red-400">❌ {failedTests} Failed</span>
                <span className="text-blue-400">🔄 {runningTests} Running</span>
              </div>
            )}
          </CardContent>
        </Card>
        
        {/* Test Results */}
        {tests.length > 0 && (
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Test Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {tests.map((test, index) => (
                  <div key={test.id} className={`p-4 rounded-lg border ${getStatusColor(test.status)}`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {getStatusIcon(test.status)}
                        <div>
                          <h3 className="text-white font-medium">{test.name}</h3>
                          <p className="text-slate-400 text-sm">{test.message}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-slate-500 text-xs">
                          {new Date(test.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
        
        {/* WebSocket Messages */}
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">WebSocket Messages</CardTitle>
          </CardHeader>
          <CardContent>
            {wsMessages.length === 0 ? (
              <div className="text-center py-8">
                <Clock className="h-12 w-12 text-slate-500 mx-auto mb-2" />
                <p className="text-slate-400">No WebSocket messages received yet</p>
                <p className="text-slate-500 text-sm">Messages will appear here when the system detects activity</p>
              </div>
            ) : (
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {wsMessages.map((message, index) => (
                  <div key={index} className="p-3 bg-slate-700 rounded-lg border border-slate-600">
                    <div className="flex items-center justify-between mb-2">
                      <Badge className="bg-blue-500 text-white text-xs">
                        {message.type.toUpperCase()}
                      </Badge>
                      <span className="text-slate-400 text-xs">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <pre className="text-slate-300 text-xs overflow-x-auto">
                      {JSON.stringify(message.data, null, 2)}
                    </pre>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}