"use client"

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Bell, AlertTriangle, CheckCircle, X, Clock, RefreshCw, Volume2, VolumeX } from "lucide-react"
import { apiService } from "@/services/api"
import { 
  websocketService, 
  SystemAlert,
  DetectionUpdate 
} from "@/services/websocket"
import { useWebSocket, useWebSocketSubscription } from "@/services/websocket-hooks"
import { toast } from "sonner"

interface AlertData {
  id: number
  detection_id: number
  alert_type: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  message: string
  camera_id: number
  camera_name?: string
  created_at: string
  acknowledged: boolean
  is_real_time?: boolean
  detection_type?: string
  confidence?: number
}

interface RealTimeAlertsProps {
  onNewAlert?: (alert: AlertData) => void
  maxAlerts?: number
  autoRefresh?: boolean
  soundEnabled?: boolean
  filterCameraId?: number
}

export function RealTimeAlerts({ 
  onNewAlert, 
  maxAlerts = 50,
  autoRefresh = true,
  soundEnabled = true,
  filterCameraId
}: RealTimeAlertsProps) {
  const [alerts, setAlerts] = useState<AlertData[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>("")
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [soundEnabledState, setSoundEnabledState] = useState(soundEnabled)
  const [unreadCount, setUnreadCount] = useState(0)
  
  // WebSocket integration
  const { websocketService, isConnected } = useWebSocket()

  // Sound notification function
  const playAlertSound = useCallback((severity: string) => {
    if (!soundEnabledState) return
    
    try {
      // Create audio context for different severity levels
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
      const oscillator = audioContext.createOscillator()
      const gainNode = audioContext.createGain()
      
      oscillator.connect(gainNode)
      gainNode.connect(audioContext.destination)
      
      // Different sounds for different severity levels
      switch (severity.toLowerCase()) {
        case 'critical':
          oscillator.frequency.setValueAtTime(800, audioContext.currentTime)
          oscillator.frequency.exponentialRampToValueAtTime(400, audioContext.currentTime + 0.5)
          break
        case 'high':
          oscillator.frequency.setValueAtTime(600, audioContext.currentTime)
          oscillator.frequency.exponentialRampToValueAtTime(300, audioContext.currentTime + 0.3)
          break
        case 'medium':
          oscillator.frequency.setValueAtTime(500, audioContext.currentTime)
          break
        default:
          oscillator.frequency.setValueAtTime(400, audioContext.currentTime)
      }
      
      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime)
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5)
      
      oscillator.start(audioContext.currentTime)
      oscillator.stop(audioContext.currentTime + 0.5)
    } catch (error) {
      console.warn('Failed to play alert sound:', error)
    }
  }, [soundEnabledState])

  // WebSocket subscription for real-time alerts
  useWebSocketSubscription<SystemAlert>('system_alert', (alert) => {
    // Filter by camera if specified
    if (filterCameraId && alert.camera_id !== filterCameraId) {
      return
    }
    
    // Create alert data from system alert
    const newAlert: AlertData = {
      id: Date.now() + Math.random(), // Temporary ID for frontend
      detection_id: alert.detection_id || 0,
      alert_type: alert.alert_type || 'system',
      severity: alert.severity || 'medium',
      message: alert.message,
      camera_id: alert.camera_id || 0,
      camera_name: alert.camera_name,
      created_at: new Date().toISOString(),
      acknowledged: false,
      is_real_time: true,
      detection_type: alert.detection_type,
      confidence: alert.confidence
    }
    
    // Add to alerts
    setAlerts(prev => [newAlert, ...prev.slice(0, maxAlerts - 1)])
    
    // Play sound notification
    playAlertSound(alert.severity || 'medium')
    
    // Show toast notification
    toast.error(`🚨 ${alert.alert_type || 'Alert'}: ${alert.message}`, {
      description: `Camera ${alert.camera_id} - ${alert.severity || 'medium'} severity`,
      duration: 5000,
      position: 'top-right'
    })
    
    // Update unread count
    setUnreadCount(prev => prev + 1)
    
    // Call callback
    onNewAlert?.(newAlert)
  }, [filterCameraId, maxAlerts, onNewAlert, playAlertSound])

  // WebSocket subscription for detection updates
  useWebSocketSubscription<DetectionUpdate>('detection', (detection) => {
    // Filter by camera if specified
    if (filterCameraId && detection.camera_id !== filterCameraId) {
      return
    }
    
    // Create alert from detection if confidence is high enough (lowered threshold)
    if (detection.confidence > 0.3) {
      const detectionAlert: AlertData = {
        id: Date.now() + Math.random(),
        detection_id: parseInt(detection.detection_id) || 0,
        alert_type: 'crime_detection',
        severity: detection.severity || 'medium',
        message: `Crime detected: ${detection.crime_type} (${(detection.confidence * 100).toFixed(1)}% confidence)`,
        camera_id: detection.camera_id,
        created_at: detection.timestamp,
        acknowledged: false,
        is_real_time: true,
        detection_type: detection.crime_type,
        confidence: detection.confidence
      }
      
      // Add to alerts
      setAlerts(prev => [detectionAlert, ...prev.slice(0, maxAlerts - 1)])
      
      // Play sound notification
      playAlertSound(detection.severity || 'medium')
      
      // Show toast notification
      toast.error(`🚨 Crime Detected: ${detection.crime_type}`, {
        description: `Camera ${detection.camera_id} - ${(detection.confidence * 100).toFixed(1)}% confidence`,
        duration: 5000,
        position: 'top-right'
      })
      
      // Update unread count
      setUnreadCount(prev => prev + 1)
      
      // Call callback
      onNewAlert?.(detectionAlert)
    }
  }, [filterCameraId, maxAlerts, onNewAlert, playAlertSound])
  
  // WebSocket subscription for detection_result messages (from live detection)
  useWebSocketSubscription<any>('detection_result', (result) => {
    // Filter by camera if specified
    if (filterCameraId && result.camera_id !== filterCameraId) {
      return
    }
    
    // Create alert from detection result if it's a crime
    if (result.is_crime && result.confidence > 0.3) {
      const detectionAlert: AlertData = {
        id: Date.now() + Math.random(),
        detection_id: result.detection_id || 0,
        alert_type: 'live_detection',
        severity: result.severity || 'medium',
        message: `Live detection: ${result.crime_type} (${(result.confidence * 100).toFixed(1)}% confidence)`,
        camera_id: result.camera_id,
        created_at: result.timestamp || new Date().toISOString(),
        acknowledged: false,
        is_real_time: true,
        detection_type: result.crime_type,
        confidence: result.confidence
      }
      
      // Add to alerts
      setAlerts(prev => [detectionAlert, ...prev.slice(0, maxAlerts - 1)])
      
      // Play sound notification
      playAlertSound(result.severity || 'medium')
      
      // Show toast notification
      toast.error(`🚨 Live Detection: ${result.crime_type}`, {
        description: `Camera ${result.camera_id} - ${(result.confidence * 100).toFixed(1)}% confidence`,
        duration: 5000,
        position: 'top-right'
      })
      
      // Update unread count
      setUnreadCount(prev => prev + 1)
      
      // Call callback
      onNewAlert?.(detectionAlert)
    }
  }, [filterCameraId, maxAlerts, onNewAlert, playAlertSound])

  useEffect(() => {
    loadAlerts()
    
    // Poll for new alerts every 30 seconds if autoRefresh is enabled
    let interval: NodeJS.Timeout | null = null
    if (autoRefresh && !isConnected) {
      interval = setInterval(loadAlerts, 30000)
    }
    
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [autoRefresh, filterCameraId, isConnected])

  const loadAlerts = async () => {
    try {
      setError("")
      const alertsData = await apiService.getAlerts()
      
      // Check for new alerts
      if (alerts.length > 0 && alertsData.length > alerts.length) {
        const newAlerts = alertsData.slice(0, alertsData.length - alerts.length)
        newAlerts.forEach((alert: AlertData) => onNewAlert?.(alert))
      }
      
      setAlerts(alertsData)
    } catch (err: any) {
      setError(`Failed to load alerts: ${err.message}`)
    }
  }

  const acknowledgeAlert = async (alertId: number) => {
    try {
      // Note: This would need to be implemented in the backend
      // For now, we'll just update the local state
      setAlerts(prev => prev.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      ))
    } catch (err: any) {
      setError(`Failed to acknowledge alert: ${err.message}`)
    }
  }

  const dismissAlert = (alertId: number) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId))
  }

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical': return 'bg-red-500'
      case 'high': return 'bg-orange-500'
      case 'medium': return 'bg-yellow-500'
      case 'low': return 'bg-blue-500'
      default: return 'bg-gray-500'
    }
  }

  const getSeverityIcon = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
      case 'high':
        return <AlertTriangle className="h-4 w-4" />
      case 'medium':
        return <Bell className="h-4 w-4" />
      case 'low':
        return <Clock className="h-4 w-4" />
      default:
        return <Bell className="h-4 w-4" />
    }
  }

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    
    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`
    return date.toLocaleDateString()
  }

  const unacknowledgedAlerts = alerts.filter(alert => !alert.acknowledged)
  const acknowledgedAlerts = alerts.filter(alert => alert.acknowledged)

  const refreshAlerts = async () => {
    setIsRefreshing(true)
    await loadAlerts()
    setIsRefreshing(false)
  }

  const clearUnreadCount = () => {
    setUnreadCount(0)
  }

  return (
    <Card className="bg-slate-800 border-slate-700">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-slate-100 flex items-center gap-2">
              <Bell className="h-5 w-5" />
              Real-Time Alerts
              {unacknowledgedAlerts.length > 0 && (
                <Badge className="bg-red-500 text-white animate-pulse">
                  {unacknowledgedAlerts.length} New
                </Badge>
              )}
              {unreadCount > 0 && (
                <Badge 
                  variant="outline" 
                  className="bg-yellow-500/20 border-yellow-500 text-yellow-400 cursor-pointer"
                  onClick={clearUnreadCount}
                >
                  {unreadCount} Unread
                </Badge>
              )}
            </CardTitle>
            <CardDescription className="text-slate-400">
              Live security alerts and notifications
            </CardDescription>
          </div>
          
          <div className="flex items-center gap-2">
            {/* Connection Status */}
            <div className="flex items-center gap-1">
              <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
              <span className="text-xs text-slate-400">
                {isConnected ? 'Live' : 'Offline'}
              </span>
            </div>
            
            {/* Sound Toggle */}
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setSoundEnabledState(!soundEnabledState)}
              className={`${soundEnabledState ? 'text-green-400' : 'text-slate-400'} hover:text-green-300`}
            >
              {soundEnabledState ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
            </Button>
            
            {/* Refresh Button */}
            <Button
              size="sm"
              variant="ghost"
              onClick={refreshAlerts}
              disabled={isRefreshing}
              className="text-slate-400 hover:text-slate-200"
            >
              <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert className="bg-red-900/20 border-red-800">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="text-red-400">{error}</AlertDescription>
          </Alert>
        )}

        {loading && (
          <div className="text-center py-8">
            <RefreshCw className="h-12 w-12 text-blue-500 mx-auto mb-2 animate-spin" />
            <p className="text-slate-400">Loading alerts...</p>
            <p className="text-slate-500 text-sm">Connecting to monitoring system</p>
          </div>
        )}

        {alerts.length === 0 && !loading && (
          <div className="text-center py-8">
            {isConnected ? (
              <>
                <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-2" />
                <p className="text-slate-400">No alerts at this time</p>
                <p className="text-slate-500 text-sm">
                  {filterCameraId ? `Monitoring camera ${filterCameraId}` : 'System monitoring all cameras'}
                </p>
              </>
            ) : (
              <>
                <Bell className="h-12 w-12 text-yellow-500 mx-auto mb-2" />
                <p className="text-slate-400">Connecting to alert system...</p>
                <p className="text-slate-500 text-sm">Real-time alerts will appear here</p>
              </>
            )}
          </div>
        )}

        {/* Unacknowledged Alerts */}
        {unacknowledgedAlerts.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-slate-200 font-medium text-sm flex items-center gap-2">
              New Alerts 
              {isConnected && (
                <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse" />
              )}
            </h4>
            {unacknowledgedAlerts.map((alert) => (
              <div key={alert.id} className={`bg-slate-700 border p-4 rounded-lg transition-all duration-300 ${
                alert.is_real_time ? 'border-red-500 shadow-lg shadow-red-500/20 animate-pulse' : 'border-slate-600'
              }`}>
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Badge className={`${getSeverityColor(alert.severity)} text-white`}>
                      {getSeverityIcon(alert.severity)}
                      {alert.severity.toUpperCase()}
                    </Badge>
                    <span className="text-slate-300 text-sm">{alert.alert_type}</span>
                    {alert.is_real_time && (
                      <Badge variant="outline" className="border-green-500 text-green-400 text-xs">
                        LIVE
                      </Badge>
                    )}
                  </div>
                  <div className="flex gap-1">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => acknowledgeAlert(alert.id)}
                      className="text-slate-400 hover:text-green-400 h-6 w-6 p-0"
                      title="Acknowledge alert"
                    >
                      <CheckCircle className="h-4 w-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => dismissAlert(alert.id)}
                      className="text-slate-400 hover:text-red-400 h-6 w-6 p-0"
                      title="Dismiss alert"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                
                <p className="text-slate-200 mb-2">{alert.message}</p>
                
                {alert.confidence && (
                  <div className="mb-2">
                    <div className="flex justify-between text-xs text-slate-400 mb-1">
                      <span>Confidence</span>
                      <span>{(alert.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-600 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          alert.confidence > 0.8 ? 'bg-red-500' :
                          alert.confidence > 0.6 ? 'bg-orange-500' :
                          alert.confidence > 0.4 ? 'bg-yellow-500' : 'bg-blue-500'
                        }`}
                        style={{ width: `${alert.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                )}
                
                <div className="flex justify-between items-center text-xs text-slate-400">
                  <span>Camera {alert.camera_id} {alert.camera_name && `- ${alert.camera_name}`}</span>
                  <span className="flex items-center gap-1">
                    {formatTimestamp(alert.created_at)}
                    {alert.is_real_time && (
                      <div className="h-1 w-1 bg-green-500 rounded-full animate-pulse" />
                    )}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Acknowledged Alerts */}
        {acknowledgedAlerts.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-slate-400 font-medium text-sm">Acknowledged</h4>
            {acknowledgedAlerts.slice(0, 5).map((alert) => (
              <div key={alert.id} className="bg-slate-700/50 p-3 rounded-lg opacity-75">
                <div className="flex items-start justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <Badge className={`${getSeverityColor(alert.severity)} text-white opacity-75`}>
                      {alert.severity.toUpperCase()}
                    </Badge>
                    <span className="text-slate-400 text-sm">{alert.alert_type}</span>
                  </div>
                  <CheckCircle className="h-4 w-4 text-green-500" />
                </div>
                
                <p className="text-slate-300 text-sm mb-1">{alert.message}</p>
                
                <div className="flex justify-between items-center text-xs text-slate-500">
                  <span>Camera {alert.camera_id}</span>
                  <span>{formatTimestamp(alert.created_at)}</span>
                </div>
              </div>
            ))}
            
            {acknowledgedAlerts.length > 5 && (
              <p className="text-slate-500 text-sm text-center">
                +{acknowledgedAlerts.length - 5} more acknowledged alerts
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
