"use client"

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { 
  Play, 
  Square, 
  Camera, 
  AlertTriangle, 
  CheckCircle, 
  Eye, 
  Settings,
  Activity,
  Clock,
  BarChart3,
  Zap,
  Cpu,
  MemoryStick,
  HardDrive
} from "lucide-react"
import { apiService } from "@/services/api"
import { 
  websocketService, 
  DetectionUpdate,
  CameraStatus,
  PerformanceMetrics,
  SystemAlert 
} from "@/services/websocket"
import { useWebSocket, useWebSocketSubscription } from "@/services/websocket-hooks"
import { toast } from "sonner"
import { VideoStreamOverlay } from "@/components/VideoStreamOverlay"

interface Camera {
  id: number
  name: string
  location: string
  status: string
  stream_url?: string
}

interface LiveDetectionProps {
  onDetectionAlert?: (alert: any) => void
  showAdvancedControls?: boolean
}

interface DetectionStats {
  totalDetections: number;
  crimeDetections: number;
  normalDetections: number;
  crimeRate: number;
  lastDetection: string | null;
}

interface CameraConfig {
  confidenceThreshold: number;
  frameSkip: number;
  maxFps: number;
}

export function LiveDetection({ onDetectionAlert, showAdvancedControls = false }: LiveDetectionProps) {
  const [cameras, setCameras] = useState<Camera[]>(() => [])
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const [error, setError] = useState<string>("")
  const [status, setStatus] = useState<string>("")
  const [feedUrl, setFeedUrl] = useState<string>("")
  const [isLoading, setIsLoading] = useState(true)
  const [cameraStatus, setCameraStatus] = useState<CameraStatus | null>(null)
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null)
  const [recentDetections, setRecentDetections] = useState<DetectionUpdate[]>([])
  const [detectionStats, setDetectionStats] = useState<DetectionStats>({
    totalDetections: 0,
    crimeDetections: 0,
    normalDetections: 0,
    crimeRate: 0,
    lastDetection: null
  })
  const [cameraConfig, setCameraConfig] = useState<CameraConfig>({
    confidenceThreshold: 0.7,
    frameSkip: 1,
    maxFps: 30
  })
  const [showConfig, setShowConfig] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  
  const videoRef = useRef<HTMLImageElement>(null)
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  
  // WebSocket integration
  const { websocketService, connectionStatus, isConnected } = useWebSocket()

  // Load cameras on component mount
  useEffect(() => {
    loadCameras()
  }, [])

  // Memoized detection handler to prevent re-creation on every render
  const handleDetectionUpdate = useCallback((detection: DetectionUpdate) => {
    console.log('🔍 Detection received:', detection);
    if (detection.camera_id === selectedCamera) {
      console.log('✅ Detection matches selected camera, updating UI');
      // Add to recent detections
      setRecentDetections(prev => {
        const updated = [detection, ...prev.slice(0, 9)] // Keep last 10
        console.log('📊 Updated recent detections:', updated.length);
        return updated
      })

      // Update stats
      setDetectionStats(prev => {
        const newTotal = prev.totalDetections + 1
        const isCrime = detection.confidence > 0.5 // Simple threshold for crime detection
        const newCrimeDetections = isCrime ? prev.crimeDetections + 1 : prev.crimeDetections
        const newNormalDetections = newTotal - newCrimeDetections
        
        return {
          totalDetections: newTotal,
          crimeDetections: newCrimeDetections,
          normalDetections: newNormalDetections,
          crimeRate: newCrimeDetections / newTotal,
          lastDetection: detection.timestamp
        }
      })

      // Show toast notification for crimes
      if (detection.confidence > cameraConfig.confidenceThreshold) {
        const severity = detection.confidence > 0.8 ? 'critical' : 
                        detection.confidence > 0.7 ? 'high' : 
                        detection.confidence > 0.6 ? 'medium' : 'low'
        
        toast.error(`🚨 Crime Detected: ${detection.crime_type}`, {
          description: `Confidence: ${(detection.confidence * 100).toFixed(1)}%`,
          duration: 5000,
          position: 'top-right'
        })

        // Call parent alert handler
        if (onDetectionAlert) {
          onDetectionAlert({
            ...detection,
            severity,
            message: `Crime detected: ${detection.crime_type}`
          })
        }
      }
    }
  }, [selectedCamera, cameraConfig.confidenceThreshold, onDetectionAlert])

  // Memoized camera status handler
  const handleCameraStatus = useCallback((status: CameraStatus) => {
    if (status.camera_id === selectedCamera) {
      setCameraStatus(status)
    }
  }, [selectedCamera])

  // Memoized performance handler
  const handlePerformanceMetrics = useCallback((metrics: PerformanceMetrics) => {
    setPerformanceMetrics(metrics)
  }, [])

  // Memoized alert handler
  const handleSystemAlert = useCallback((alert: SystemAlert) => {
    if (alert.camera_id === selectedCamera || !alert.camera_id) {
      const toastFn = alert.severity === 'critical' ? toast.error : 
                     alert.severity === 'high' ? toast.warning : toast.info
      
      toastFn(alert.message, {
        description: alert.data?.details || 'System alert',
        duration: 4000,
        position: 'top-right'
      })
    }
  }, [selectedCamera])

  // WebSocket event handlers - now using memoized handlers
  useWebSocketSubscription<DetectionUpdate>('detection', handleDetectionUpdate)
  useWebSocketSubscription<CameraStatus>('camera_status', handleCameraStatus)
  useWebSocketSubscription<PerformanceMetrics>('performance', handlePerformanceMetrics)
  useWebSocketSubscription<SystemAlert>('alert', handleSystemAlert)

  // Subscribe to camera when selected
  useEffect(() => {
    if (selectedCamera && isConnected) {
      websocketService.subscribeToCamera(selectedCamera)
      websocketService.getStatus(selectedCamera)
      
      return () => {
        websocketService.unsubscribeFromCamera(selectedCamera)
      }
    }
  }, [selectedCamera, isConnected])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Clear polling interval on unmount
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
    };
  }, []);

  const loadCameras = async () => {
    setIsLoading(true);
    setError("");
    
    try {
      const response = await apiService.getCameras();
      console.log('API Response:', response); // Log the raw response
      
      // Ensure response is an array and has the expected structure
      const cameraList = Array.isArray(response) 
        ? response.map(cam => ({
            id: cam.id || 0,
            name: cam.name || 'Unknown Camera',
            location: cam.location || 'Unknown Location',
            status: cam.status || 'offline',
            stream_url: cam.stream_url || ''
          }))
        : [];
      
      console.log('Processed cameras:', cameraList);
      setCameras(cameraList);
      
      if (cameraList.length > 0 && !selectedCamera) {
        setSelectedCamera(cameraList[0].id);
      }
    } catch (err: any) {
      console.error('Error loading cameras:', err);
      setError(`Failed to load cameras: ${err.message}`);
      setCameras([]);
    } finally {
      setIsLoading(false);
    }
  }

  const startDetection = async () => {
    if (!selectedCamera) {
      setError("Please select a camera first")
      return
    }

    setError("")
    setStatus("Starting live detection...")
    setIsConnecting(true)
    console.log('=== START DETECTION CALLED ===')
    console.log('Current isDetecting:', isDetecting)

    try {
      // Ensure WebSocket is connected
      if (!isConnected) {
        await websocketService.connect()
      }

      // Start detection via API
      const result = await apiService.startLiveDetection(selectedCamera, "0")
      console.log('Start detection result:', result)
      
      // Check if the result indicates success (backend returns different formats)
      const isSuccess = result && (result.status === 'active' || result.message?.includes('started'))
      
      if (!isSuccess && result.success === false) {
        throw new Error(result.message || 'Failed to start detection')
      }
      
      // Wait for backend to initialize camera (increased delay)
      console.log('Waiting for backend to initialize camera...')
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Get the live feed URL
      const newFeedUrl = apiService.getLiveFeedUrl(selectedCamera);
      console.log('Feed URL:', newFeedUrl)
      
      // Update all states together
      console.log('Setting isDetecting to TRUE')
      setIsDetecting(true)
      setFeedUrl(newFeedUrl)
      setStatus("Live detection started successfully")
      setIsConnecting(false)
      
      // Subscribe to camera updates
      websocketService.subscribeToCamera(selectedCamera)
      websocketService.getStatus(selectedCamera)
      
      // Show success toast
      toast.success("Live detection started", {
        description: `Camera ${selectedCamera} is now actively monitoring for criminal activity`,
        duration: 3000
      })
      
      // Log after state update
      setTimeout(() => {
        console.log('After state update, isDetecting should be true')
      }, 100)
      
    } catch (err: any) {
      console.error('Start detection error:', err)
      setError(`Failed to start detection: ${err.message}`)
      setStatus("")
      setIsDetecting(false)
      setIsConnecting(false)
      
      toast.error("Failed to start detection", {
        description: err.message,
        duration: 5000
      })
    }
  }

  const stopDetection = async () => {
    if (!selectedCamera) return

    setStatus("Stopping live detection...")
    setError("")
    setIsConnecting(true)

    try {
      console.log('Stopping detection for camera:', selectedCamera)
      
      // Stop via WebSocket first
      websocketService.stopDetection(selectedCamera)
      
      // Clear polling if it exists
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
        pollingIntervalRef.current = null
      }
      
      // Stop the backend detection
      const result = await apiService.stopLiveDetection(selectedCamera)
      console.log('Stop detection result:', result)
      
      // IMPORTANT: Clear states in the right order
      // 1. Stop the feed first
      setFeedUrl("")
      
      // 2. Force the image to stop loading
      if (videoRef.current) {
        videoRef.current.src = ""
        videoRef.current.removeAttribute('src')
      }
      
      // 3. Update detection state
      setIsDetecting(false)
      
      // 4. Reset other states
      setCameraStatus(null)
      setRecentDetections([])
      setPerformanceMetrics(null)
      setIsConnecting(false)
      
      setStatus("Live detection stopped successfully")
      
      // Show success toast
      toast.success("Live detection stopped", {
        description: `Camera ${selectedCamera} monitoring has been stopped`,
        duration: 3000
      })
      
      // Reload cameras to update status
      setTimeout(() => loadCameras(), 500)
      
    } catch (err: any) {
      console.error('Stop detection error:', err)
      setError(`Failed to stop detection: ${err.message}`)
      setIsConnecting(false)
      
      // Force stop on frontend even if backend fails
      setIsDetecting(false)
      setFeedUrl("")
      if (videoRef.current) {
        videoRef.current.src = ""
        videoRef.current.removeAttribute('src')
      }
      
      toast.error("Failed to stop detection", {
        description: err.message,
        duration: 5000
      })
    }
  }

  const updateCameraConfig = async () => {
    if (!selectedCamera) return

    try {
      const response = await fetch(`/api/live/configure/${selectedCamera}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: new URLSearchParams({
          confidence_threshold: cameraConfig.confidenceThreshold.toString(),
          frame_skip: cameraConfig.frameSkip.toString(),
          max_fps: cameraConfig.maxFps.toString()
        })
      })

      if (!response.ok) {
        throw new Error('Failed to update configuration')
      }

      const result = await response.json()
      
      // Update via WebSocket as well
      websocketService.updateConfidenceThreshold(selectedCamera, cameraConfig.confidenceThreshold)
      
      toast.success("Configuration updated", {
        description: "Camera settings have been applied",
        duration: 3000
      })
      
    } catch (err: any) {
      toast.error("Failed to update configuration", {
        description: err.message,
        duration: 5000
      })
    }
  }

  const resetStats = () => {
    setRecentDetections([])
    setDetectionStats({
      totalDetections: 0,
      crimeDetections: 0,
      normalDetections: 0,
      crimeRate: 0,
      lastDetection: null
    })
    setCameraStatus(null)
  }




  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active':
      case 'online':
        return 'bg-green-500'
      case 'inactive':
      case 'offline':
        return 'bg-red-500'
      case 'maintenance':
        return 'bg-yellow-500'
      case 'error':
        return 'bg-red-600'
      default:
        return 'bg-gray-500'
    }
  }

  const getConnectionStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'bg-green-500'
      case 'connecting':
        return 'bg-yellow-500'
      case 'error':
        return 'bg-red-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-600'
      case 'high':
        return 'bg-red-500'
      case 'medium':
        return 'bg-yellow-500'
      case 'low':
        return 'bg-blue-500'
      default:
        return 'bg-gray-500'
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`
    } else {
      return `${secs}s`
    }
  }

  // Safely find the selected camera
  const selectedCameraData = cameras?.find(c => c.id === selectedCamera) || null;

  return (
    <div className="space-y-6">
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-slate-100 flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Live Crime Detection
          </CardTitle>
          <CardDescription className="text-slate-400">
            Monitor live camera feeds for criminal activity detection
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Camera Selection */}
          <div className="space-y-2">
            <label className="text-slate-200 text-sm font-medium">Select Camera:</label>
            <Select value={selectedCamera?.toString()} onValueChange={(value) => setSelectedCamera(parseInt(value))}>
              <SelectTrigger className="bg-slate-700 border-slate-600 text-slate-100">
                <SelectValue placeholder="Choose a camera" />
              </SelectTrigger>
              <SelectContent className="bg-slate-700 border-slate-600">
                {cameras.map((camera) => (
                  <SelectItem key={camera.id} value={camera.id.toString()} className="text-slate-100">
                    <div className="flex items-center gap-2">
                      <Badge className={`${getStatusColor(camera.status)} text-white text-xs`}>
                        {camera.status}
                      </Badge>
                      {camera.name} - {camera.location}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Camera Info */}
          {selectedCameraData && (
            <div className="bg-slate-700 p-3 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-200 font-medium">{selectedCameraData.name}</p>
                  <p className="text-slate-400 text-sm">{selectedCameraData.location}</p>
                </div>
                <Badge className={`${getStatusColor(selectedCameraData.status)} text-white`}>
                  {selectedCameraData.status}
                </Badge>
              </div>
            </div>
          )}

          {/* Connection Status */}
          <div className="flex items-center justify-between p-3 bg-slate-700 rounded-lg">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${getConnectionStatusColor(connectionStatus)}`}></div>
              <span className="text-slate-200 text-sm">WebSocket: {connectionStatus}</span>
            </div>
            <div className="flex items-center gap-2">
              {isConnected && (
                <Badge className="bg-green-600 text-white text-xs">
                  <Zap className="h-3 w-3 mr-1" />
                  Real-time
                </Badge>
              )}
              {showAdvancedControls && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowConfig(!showConfig)}
                  className="text-slate-300 hover:text-slate-100"
                >
                  <Settings className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>

          {/* Advanced Configuration */}
          {showConfig && showAdvancedControls && (
            <Card className="bg-slate-700 border-slate-600">
              <CardHeader className="pb-3">
                <CardTitle className="text-slate-100 text-sm flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Camera Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-slate-200 text-sm">Confidence Threshold: {cameraConfig.confidenceThreshold}</label>
                  <Slider
                    value={[cameraConfig.confidenceThreshold]}
                    onValueChange={(value) => setCameraConfig(prev => ({ ...prev, confidenceThreshold: value[0] }))}
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    className="w-full"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-slate-200 text-sm">Frame Skip</label>
                    <input
                      type="number"
                      value={cameraConfig.frameSkip}
                      onChange={(e) => setCameraConfig(prev => ({ ...prev, frameSkip: parseInt(e.target.value) || 1 }))}
                      min={1}
                      max={10}
                      className="w-full bg-slate-600 border-slate-500 text-slate-100 rounded px-3 py-2"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-slate-200 text-sm">Max FPS</label>
                    <input
                      type="number"
                      value={cameraConfig.maxFps}
                      onChange={(e) => setCameraConfig(prev => ({ ...prev, maxFps: parseInt(e.target.value) || 30 }))}
                      min={5}
                      max={60}
                      className="w-full bg-slate-600 border-slate-500 text-slate-100 rounded px-3 py-2"
                    />
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button
                    onClick={updateCameraConfig}
                    disabled={!isDetecting}
                    size="sm"
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    Apply Settings
                  </Button>
                  <Button
                    onClick={resetStats}
                    variant="outline"
                    size="sm"
                    className="border-slate-500 text-slate-300 hover:text-slate-100"
                  >
                    Reset Stats
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Control Buttons */}
          <div className="flex gap-3">
            <Button
              onClick={startDetection}
              disabled={isDetecting || !selectedCamera || isConnecting}
              className="flex-1 bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isConnecting ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Connecting...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  {isDetecting ? 'Detection Running' : 'Start Detection'}
                </>
              )}
            </Button>
            <Button
              onClick={stopDetection}
              disabled={!isDetecting || isConnecting}
              variant="destructive"
              className="flex-1 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isConnecting ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Stopping...
                </>
              ) : (
                <>
                  <Square className="h-4 w-4 mr-2" />
                  Stop Detection
                </>
              )}
            </Button>
          </div>

          {/* Status Display */}
          {status && (
            <Alert className={`${isDetecting ? 'bg-green-900/20 border-green-800' : 'bg-blue-900/20 border-blue-800'}`}>
              <CheckCircle className="h-4 w-4" />
              <AlertDescription className={isDetecting ? 'text-green-400' : 'text-blue-400'}>
                {status}
              </AlertDescription>
            </Alert>
          )}

          {/* Error Display */}
          {error && (
            <Alert className="bg-red-900/20 border-red-800">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription className="text-red-400">{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Live Feed Display */}
      {feedUrl && isDetecting && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Feed with Overlay */}
          <div className="lg:col-span-2">
            <VideoStreamOverlay
              feedUrl={feedUrl}
              cameraName={selectedCameraData?.name || "Unknown Camera"}
              cameraStatus={cameraStatus}
              recentDetections={recentDetections}
              showDetections={true}
              showPerformance={true}
              onError={(error) => setError(error)}
            />
          </div>

          {/* Real-time Statistics Panel */}
          <div className="space-y-4">
            {/* Detection Statistics */}
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-slate-100 text-sm flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Detection Statistics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-slate-700 p-3 rounded">
                    <div className="text-slate-400 text-xs">Total</div>
                    <div className="text-slate-100 text-lg font-bold">{detectionStats.totalDetections}</div>
                  </div>
                  <div className="bg-slate-700 p-3 rounded">
                    <div className="text-slate-400 text-xs">Crimes</div>
                    <div className="text-red-400 text-lg font-bold">{detectionStats.crimeDetections}</div>
                  </div>
                  <div className="bg-slate-700 p-3 rounded">
                    <div className="text-slate-400 text-xs">Normal</div>
                    <div className="text-green-400 text-lg font-bold">{detectionStats.normalDetections}</div>
                  </div>
                  <div className="bg-slate-700 p-3 rounded">
                    <div className="text-slate-400 text-xs">Rate</div>
                    <div className="text-slate-100 text-lg font-bold">{(detectionStats.crimeRate * 100).toFixed(1)}%</div>
                  </div>
                </div>
                {detectionStats.lastDetection && (
                  <div className="text-slate-400 text-xs">
                    Last detection: {formatTimestamp(detectionStats.lastDetection)}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Performance Metrics */}
            {performanceMetrics && (
              <Card className="bg-slate-800 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-slate-100 text-sm flex items-center gap-2">
                    <Activity className="h-4 w-4" />
                    Performance
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400 text-xs">Inference Time</span>
                    <span className="text-slate-100 font-mono">{performanceMetrics.avg_inference_time_ms.toFixed(1)}ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400 text-xs">Average FPS</span>
                    <span className="text-slate-100 font-mono">{performanceMetrics.avg_fps.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400 text-xs">System Load</span>
                    <span className="text-slate-100 font-mono">{(performanceMetrics.system_load * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400 text-xs">Memory Usage</span>
                    <span className="text-slate-100 font-mono">{performanceMetrics.memory_usage_mb.toFixed(0)}MB</span>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Recent Detections */}
            {recentDetections.length > 0 && (
              <Card className="bg-slate-800 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-slate-100 text-sm flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    Recent Detections
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 max-h-48 overflow-y-auto">
                  {recentDetections.map((detection, index) => (
                    <div key={index} className="bg-slate-700 p-2 rounded text-xs">
                      <div className="flex justify-between items-start">
                        <div>
                          <div className="text-slate-200 font-medium">{detection.crime_type}</div>
                          <div className="text-slate-400">{(detection.confidence * 100).toFixed(1)}% confidence</div>
                        </div>
                        <Badge className={`${getSeverityColor(detection.severity)} text-white text-xs`}>
                          {detection.severity}
                        </Badge>
                      </div>
                      <div className="text-slate-500 mt-1">
                        {formatTimestamp(detection.timestamp)}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
