"use client"

import React, { useRef, useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AlertTriangle, Eye, Maximize2, Minimize2 } from "lucide-react"
import { DetectionUpdate, CameraStatus } from "@/services/websocket"

interface VideoStreamOverlayProps {
  feedUrl: string
  cameraName: string
  cameraStatus?: CameraStatus
  recentDetections?: DetectionUpdate[]
  showDetections?: boolean
  showPerformance?: boolean
  className?: string
  onError?: (error: string) => void
}

interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
  label: string
  confidence: number
  color: string
}

export function VideoStreamOverlay({
  feedUrl,
  cameraName,
  cameraStatus,
  recentDetections = [],
  showDetections = true,
  showPerformance = true,
  className = "",
  onError
}: VideoStreamOverlayProps) {
  const videoRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [currentDetection, setCurrentDetection] = useState<DetectionUpdate | null>(null)
  const [boundingBoxes, setBoundingBoxes] = useState<BoundingBox[]>([])

  // Update current detection from recent detections
  useEffect(() => {
    if (recentDetections.length > 0) {
      const latest = recentDetections[0]
      setCurrentDetection(latest)
      
      // Generate bounding boxes from detection data
      if (latest.frame_data && showDetections) {
        try {
          // Parse frame data to extract bounding box information
          const frameInfo = JSON.parse(latest.frame_data)
          const boxes: BoundingBox[] = []
          
          if (frameInfo.objects) {
            frameInfo.objects.forEach((obj: any) => {
              boxes.push({
                x: obj.x || 0,
                y: obj.y || 0,
                width: obj.width || 100,
                height: obj.height || 100,
                label: obj.label || latest.crime_type,
                confidence: latest.confidence,
                color: getSeverityColor(latest.severity)
              })
            })
          }
          
          setBoundingBoxes(boxes)
        } catch (error) {
          console.warn('Failed to parse frame data:', error)
          setBoundingBoxes([])
        }
      } else {
        setBoundingBoxes([])
      }
      
      // Clear detection after 3 seconds
      const timer = setTimeout(() => {
        setCurrentDetection(null)
        setBoundingBoxes([])
      }, 3000)
      
      return () => clearTimeout(timer)
    }
  }, [recentDetections, showDetections])

  // Draw overlays on canvas
  useEffect(() => {
    if (!canvasRef.current || !imageLoaded) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw bounding boxes
    if (boundingBoxes.length > 0) {
      boundingBoxes.forEach(box => {
        // Draw box
        ctx.strokeStyle = box.color
        ctx.lineWidth = 2
        ctx.strokeRect(box.x, box.y, box.width, box.height)
        
        // Draw label background
        const label = `${box.label} ${(box.confidence * 100).toFixed(1)}%`
        const textMetrics = ctx.measureText(label)
        const labelWidth = textMetrics.width + 8
        const labelHeight = 20
        
        ctx.fillStyle = box.color
        ctx.fillRect(box.x, box.y - labelHeight, labelWidth, labelHeight)
        
        // Draw label text
        ctx.fillStyle = 'white'
        ctx.font = '12px Arial'
        ctx.fillText(label, box.x + 4, box.y - 6)
      })
    }

    // Draw current detection alert
    if (currentDetection && showDetections) {
      const alertText = `🚨 ${currentDetection.crime_type} - ${(currentDetection.confidence * 100).toFixed(1)}%`
      ctx.fillStyle = 'rgba(239, 68, 68, 0.9)'
      ctx.fillRect(10, 10, ctx.measureText(alertText).width + 20, 30)
      
      ctx.fillStyle = 'white'
      ctx.font = 'bold 14px Arial'
      ctx.fillText(alertText, 20, 28)
    }
  }, [boundingBoxes, currentDetection, imageLoaded, showDetections])

  // Handle image load and canvas sizing
  useEffect(() => {
    if (!videoRef.current || !canvasRef.current || !containerRef.current) return

    const updateCanvasSize = () => {
      const img = videoRef.current!
      const canvas = canvasRef.current!
      const container = containerRef.current!
      
      // Set canvas size to match the displayed image
      const rect = container.getBoundingClientRect()
      canvas.width = rect.width
      canvas.height = rect.height
      
      setImageLoaded(true)
    }

    const img = videoRef.current
    img.addEventListener('load', updateCanvasSize)
    
    // Initial sizing
    setTimeout(updateCanvasSize, 100)

    return () => {
      img.removeEventListener('load', updateCanvasSize)
    }
  }, [])

  const handleFullscreen = () => {
    if (!containerRef.current) return

    if (!isFullscreen) {
      if (containerRef.current.requestFullscreen) {
        containerRef.current.requestFullscreen()
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen()
      }
    }
  }

  const handleImageError = () => {
    if (onError) {
      onError("Failed to load camera feed. Please check the connection.")
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#dc2626'
      case 'high': return '#ef4444'
      case 'medium': return '#f59e0b'
      case 'low': return '#3b82f6'
      default: return '#6b7280'
    }
  }

  return (
    <div className={`relative ${className}`}>
      <div 
        ref={containerRef}
        className="relative bg-slate-900 rounded-lg overflow-hidden group"
      >
        {/* Video Stream */}
        <img
          ref={videoRef}
          src={feedUrl}
          alt={`Live feed - ${cameraName}`}
          className="w-full h-auto max-h-96 object-contain"
          onError={handleImageError}
        />
        
        {/* Overlay Canvas */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ zIndex: 10 }}
        />
        
        {/* Control Overlay */}
        <div className="absolute top-2 right-2 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <Badge className="bg-red-500 text-white animate-pulse">
            ● LIVE
          </Badge>
          <button
            onClick={handleFullscreen}
            className="bg-black/50 text-white p-1 rounded hover:bg-black/70 transition-colors"
            title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          >
            {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </button>
        </div>
        
        {/* Performance Overlay */}
        {cameraStatus && showPerformance && (
          <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1">
                <Eye className="h-3 w-3" />
                {cameraStatus.fps} FPS
              </div>
              <div className="flex items-center gap-1">
                <AlertTriangle className="h-3 w-3" />
                {cameraStatus.detections_count} detections
              </div>
              {cameraStatus.status === 'error' && (
                <div className="flex items-center gap-1 text-red-400">
                  <AlertTriangle className="h-3 w-3" />
                  Error
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Detection Info Overlay */}
        {currentDetection && (
          <div className="absolute top-2 left-2 bg-red-600/90 text-white px-3 py-2 rounded text-sm font-medium">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              <span>
                {currentDetection.crime_type} - {(currentDetection.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="text-xs opacity-90 mt-1">
              {new Date(currentDetection.timestamp).toLocaleTimeString()}
            </div>
          </div>
        )}
      </div>
      
      {/* Status Bar */}
      <div className="mt-2 text-xs text-slate-400 text-center">
        Live detection active - Criminal activity will be automatically detected and reported
      </div>
    </div>
  )
}