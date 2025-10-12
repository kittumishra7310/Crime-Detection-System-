"use client"

import { useEffect, useState } from "react"
import { useAuth } from "@/hooks/useAuth"
import { useRouter } from "next/navigation"
import { Sidebar } from "@/components/Sidebar"
import { Navbar } from "@/components/Navbar"
import { DetectionCard } from "@/components/DetectionCard"
import { TrendChart } from "@/components/TrendChart"
import { CameraStatus } from "@/components/CameraStatus"
import { FileUpload } from "@/components/FileUpload"
import { LiveDetection } from "@/components/LiveDetection"
import { RealTimeAlerts } from "@/components/RealTimeAlerts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ActivityIcon, AlertTriangleIcon, ClockIcon, EyeIcon } from "@/components/icons"

interface Detection {
  id: string
  type: string
  severity: "low" | "medium" | "high"
  timestamp: string
  cameraId: string
  confidence: number
}

interface Stats {
  totalDetections: number
  alertsTriggered: number
  uptime: string
  activeCameras: number
}

export default function DashboardPage() {
  const { isAuthenticated, isLoading } = useAuth()
  const router = useRouter()
  const [detections, setDetections] = useState<Detection[]>([])
  const [stats, setStats] = useState<Stats>({
    totalDetections: 0,
    alertsTriggered: 0,
    uptime: "0h 0m",
    activeCameras: 0,
  })
  const [isLive, setIsLive] = useState(true)
  const [trendData, setTrendData] = useState([
    { time: "00:00", detections: 12, alerts: 2 },
    { time: "04:00", detections: 8, alerts: 1 },
    { time: "08:00", detections: 24, alerts: 4 },
    { time: "12:00", detections: 32, alerts: 3 },
    { time: "16:00", detections: 28, alerts: 5 },
    { time: "20:00", detections: 18, alerts: 2 },
  ])

  const [cameras, setCameras] = useState([
    {
      id: "CAM-001",
      name: "Front Entrance",
      location: "Building A - Main Door",
      status: "online" as const,
      lastHeartbeat: "2 min ago",
      resolution: "1080p",
    },
    {
      id: "CAM-002",
      name: "Parking Lot",
      location: "Outdoor - North Side",
      status: "online" as const,
      lastHeartbeat: "1 min ago",
      resolution: "4K",
    },
    {
      id: "CAM-003",
      name: "Server Room",
      location: "Building B - Floor 2",
      status: "warning" as const,
      lastHeartbeat: "15 min ago",
      resolution: "1080p",
    },
    {
      id: "CAM-004",
      name: "Loading Dock",
      location: "Building C - Rear",
      status: "offline" as const,
      lastHeartbeat: "2 hours ago",
      resolution: "720p",
    },
  ])

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/login")
    }
  }, [isAuthenticated, isLoading, router])

  useEffect(() => {
    // Mock data - replace with actual API calls
    const mockDetections: Detection[] = [
      {
        id: "1",
        type: "Person Detected",
        severity: "medium",
        timestamp: new Date().toISOString(),
        cameraId: "CAM-001",
        confidence: 0.92,
      },
      {
        id: "2",
        type: "Motion Alert",
        severity: "low",
        timestamp: new Date(Date.now() - 300000).toISOString(),
        cameraId: "CAM-002",
        confidence: 0.78,
      },
      {
        id: "3",
        type: "Unauthorized Access",
        severity: "high",
        timestamp: new Date(Date.now() - 600000).toISOString(),
        cameraId: "CAM-003",
        confidence: 0.95,
      },
    ]

    const mockStats: Stats = {
      totalDetections: 247,
      alertsTriggered: 12,
      uptime: "24h 15m",
      activeCameras: 8,
    }

    setDetections(mockDetections)
    setStats(mockStats)
  }, [])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-foreground">Loading...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return null
  }

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Navbar />
        <main className="flex-1 p-6 space-y-6">
          <div className="flex items-center gap-3">
            <div className={`h-3 w-3 rounded-full ${isLive ? "bg-green-500" : "bg-red-500"}`} />
            <span className="text-xl font-semibold text-foreground">
              {isLive ? "Live Detection Active" : "Detection Offline"}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="bg-card border-border">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Total Detections Today</CardTitle>
                <ActivityIcon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-foreground">{stats.totalDetections}</div>
                <p className="text-xs text-muted-foreground">+12% from yesterday</p>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Alerts Triggered</CardTitle>
                <AlertTriangleIcon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-foreground">{stats.alertsTriggered}</div>
                <p className="text-xs text-muted-foreground">-3% from yesterday</p>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">System Uptime</CardTitle>
                <ClockIcon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-foreground">{stats.uptime}</div>
                <p className="text-xs text-green-600">99.8% availability</p>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Active Cameras</CardTitle>
                <EyeIcon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-foreground">{stats.activeCameras}</div>
                <p className="text-xs text-muted-foreground">3 online, 1 warning</p>
              </CardContent>
            </Card>
          </div>

          {/* Detection Tools */}
          <Tabs defaultValue="overview" className="space-y-6">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="live-detection">Live Detection</TabsTrigger>
              <TabsTrigger value="file-upload">File Upload</TabsTrigger>
              <TabsTrigger value="alerts">Alerts</TabsTrigger>
            </TabsList>
            
            <TabsContent value="overview" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <TrendChart data={trendData} title="Detection Trends (24h)" />
                <CameraStatus cameras={cameras} />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="bg-card border-border">
                  <CardHeader>
                    <CardTitle className="text-foreground">System Status</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="aspect-video bg-muted rounded-lg flex items-center justify-center border border-border">
                      <div className="text-center">
                        <EyeIcon className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
                        <p className="text-muted-foreground font-medium">System Overview</p>
                        <p className="text-sm text-muted-foreground">All systems operational</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <div className="space-y-4">
                  <h2 className="text-xl font-semibold text-foreground">Recent Detections</h2>
                  {detections.map((detection) => (
                    <DetectionCard key={detection.id} detection={detection} />
                  ))}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="live-detection">
              <LiveDetection onDetectionAlert={(alert) => {
                // Handle new detection alerts
                console.log('New detection alert:', alert)
              }} />
            </TabsContent>

            <TabsContent value="file-upload">
              <FileUpload onDetectionComplete={(result) => {
                // Handle completed file detection
                console.log('File detection completed:', result)
              }} />
            </TabsContent>

            <TabsContent value="alerts">
              <RealTimeAlerts onNewAlert={(alert) => {
                // Handle new alerts
                console.log('New alert received:', alert)
              }} />
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </div>
  )
}
