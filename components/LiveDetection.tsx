"use client"

import React, { useState, useEffect, useRef } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Play, Square, Camera, AlertTriangle, CheckCircle, Eye } from "lucide-react"
import { apiService } from "@/services/api"

interface Camera {
  id: number
  name: string
  location: string
  status: string
  stream_url?: string
}

interface LiveDetectionProps {
  onDetectionAlert?: (alert: any) => void
}

export function LiveDetection({ onDetectionAlert }: LiveDetectionProps) {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const [error, setError] = useState<string>("")
  const [status, setStatus] = useState<string>("")
  const [feedUrl, setFeedUrl] = useState<string>("")
  const videoRef = useRef<HTMLImageElement>(null);

  // Load cameras on component mount
  useEffect(() => {
    loadCameras()
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    // Cleanup on unmount
    return () => {
      // No-op, cleanup is handled by the backend
    };
  }, []);

  const loadCameras = async () => {
    try {
      const cameraList = await apiService.getCameras()
      setCameras(cameraList)
      if (cameraList.length > 0 && !selectedCamera) {
        setSelectedCamera(cameraList[0].id)
      }
    } catch (err: any) {
      setError(`Failed to load cameras: ${err.message}`)
    }
  }

  const startDetection = async () => {
    if (!selectedCamera) {
      setError("Please select a camera first")
      return
    }

    setError("")
    setStatus("Starting live detection...")

    try {
      const result = await apiService.startLiveDetection(selectedCamera, "0")
      setIsDetecting(true)
      setStatus("Live detection started successfully")
      
      // Get the live feed URL
      const newFeedUrl = apiService.getLiveFeedUrl(selectedCamera);
      setFeedUrl(newFeedUrl);

      // Start polling for status updates
      startStatusPolling()
      
    } catch (err: any) {
      setError(`Failed to start detection: ${err.message}`)
      setStatus("")
    }
  }

  const stopDetection = async () => {
    if (!selectedCamera) return

    setStatus("Stopping live detection...")

    try {
      await apiService.stopLiveDetection(selectedCamera)
      setIsDetecting(false)
      setStatus("Live detection stopped")
      setFeedUrl("")
      
      setFeedUrl("");
      
    } catch (err: any) {
      setError(`Failed to stop detection: ${err.message}`)
    }
  }


  const startStatusPolling = () => {
    if (!selectedCamera) return

    const pollStatus = setInterval(async () => {
      try {
        const statusResponse = await apiService.getLiveDetectionStatus()
        if (!statusResponse.is_active) {
          setIsDetecting(false)
          setStatus("Detection stopped")
          clearInterval(pollStatus)
        }
      } catch (err) {
        // Ignore polling errors to avoid spam
      }
    }, 5000) // Poll every 5 seconds
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
      default:
        return 'bg-gray-500'
    }
  }

  const selectedCameraData = cameras.find(c => c.id === selectedCamera)

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

          {/* Control Buttons */}
          <div className="flex gap-3">
            <Button
              onClick={startDetection}
              disabled={isDetecting || !selectedCamera}
              className="flex-1 bg-green-600 hover:bg-green-700"
            >
              <Play className="h-4 w-4 mr-2" />
              Start Detection
            </Button>
            <Button
              onClick={stopDetection}
              disabled={!isDetecting}
              variant="destructive"
              className="flex-1"
            >
              <Square className="h-4 w-4 mr-2" />
              Stop Detection
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
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-slate-100 flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Live Feed - {selectedCameraData?.name}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative bg-slate-900 rounded-lg overflow-hidden">
              <img
                ref={videoRef}
                src={feedUrl}
                alt="Live camera feed"
                className="w-full h-auto max-h-96 object-contain"
                onError={() => setError("Failed to load camera feed. Is the backend running and camera accessible?")}
              />
              <div className="absolute top-2 right-2">
                <Badge className="bg-red-500 text-white animate-pulse">
                  ● LIVE
                </Badge>
              </div>
            </div>
            <p className="text-slate-400 text-sm mt-2 text-center">
              Live detection active - Criminal activity will be automatically detected and reported
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
