"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Bell, AlertTriangle, CheckCircle, X, Clock } from "lucide-react"
import { apiService } from "@/services/api"

interface AlertData {
  id: number
  detection_id: number
  alert_type: string
  severity: string
  message: string
  camera_id: number
  camera_name?: string
  created_at: string
  acknowledged: boolean
}

interface RealTimeAlertsProps {
  onNewAlert?: (alert: AlertData) => void
}

export function RealTimeAlerts({ onNewAlert }: RealTimeAlertsProps) {
  const [alerts, setAlerts] = useState<AlertData[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>("")

  useEffect(() => {
    loadAlerts()
    
    // Poll for new alerts every 5 seconds
    const interval = setInterval(loadAlerts, 5000)
    
    return () => clearInterval(interval)
  }, [])

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

  return (
    <Card className="bg-slate-800 border-slate-700">
      <CardHeader>
        <CardTitle className="text-slate-100 flex items-center gap-2">
          <Bell className="h-5 w-5" />
          Real-Time Alerts
          {unacknowledgedAlerts.length > 0 && (
            <Badge className="bg-red-500 text-white animate-pulse">
              {unacknowledgedAlerts.length} New
            </Badge>
          )}
        </CardTitle>
        <CardDescription className="text-slate-400">
          Live security alerts and notifications
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert className="bg-red-900/20 border-red-800">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="text-red-400">{error}</AlertDescription>
          </Alert>
        )}

        {alerts.length === 0 && !loading && (
          <div className="text-center py-8">
            <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-2" />
            <p className="text-slate-400">No alerts at this time</p>
            <p className="text-slate-500 text-sm">System monitoring active</p>
          </div>
        )}

        {/* Unacknowledged Alerts */}
        {unacknowledgedAlerts.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-slate-200 font-medium text-sm">New Alerts</h4>
            {unacknowledgedAlerts.map((alert) => (
              <div key={alert.id} className="bg-slate-700 border border-slate-600 p-4 rounded-lg">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Badge className={`${getSeverityColor(alert.severity)} text-white`}>
                      {getSeverityIcon(alert.severity)}
                      {alert.severity.toUpperCase()}
                    </Badge>
                    <span className="text-slate-300 text-sm">{alert.alert_type}</span>
                  </div>
                  <div className="flex gap-1">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => acknowledgeAlert(alert.id)}
                      className="text-slate-400 hover:text-green-400 h-6 w-6 p-0"
                    >
                      <CheckCircle className="h-4 w-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => dismissAlert(alert.id)}
                      className="text-slate-400 hover:text-red-400 h-6 w-6 p-0"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                
                <p className="text-slate-200 mb-2">{alert.message}</p>
                
                <div className="flex justify-between items-center text-xs text-slate-400">
                  <span>Camera {alert.camera_id} {alert.camera_name && `- ${alert.camera_name}`}</span>
                  <span>{formatTimestamp(alert.created_at)}</span>
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
