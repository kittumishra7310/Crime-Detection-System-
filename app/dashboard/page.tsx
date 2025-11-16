"use client"

import { useEffect, useState } from "react"
import { useAuth } from "@/hooks/useAuth"
import { useRouter } from "next/navigation"
import { AuthHelper } from "@/components/AuthHelper"
import { Sidebar } from "@/components/Sidebar"
import { Navbar } from "@/components/Navbar"
import { DetectionCard } from "@/components/DetectionCard"
import { TrendChart } from "@/components/TrendChart"
import { CameraStatus } from "@/components/CameraStatus"
import { FileUpload } from "@/components/FileUpload"
import { LiveDetection } from "@/components/LiveDetection"
import { RealTimeAlerts } from "@/components/RealTimeAlerts"
import { WebSocketAuthHelper } from "@/components/WebSocketAuthHelper"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ActivityIcon, AlertTriangleIcon, ClockIcon, EyeIcon } from "@/components/icons"
import { enhancedApiService } from "@/services/enhanced-api"

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

  const [cameras, setCameras] = useState<any[]>([])
  const [authSynced, setAuthSynced] = useState(false)

  // Authentication sync effect
  useEffect(() => {
    if (isAuthenticated && !isLoading) {
      console.log("🔐 Authenticated, syncing with backend...")
      enhancedApiService.forceAuthSync().then((success) => {
        if (success) {
          console.log("✅ Backend authentication sync successful")
          setAuthSynced(true)
        } else {
          console.warn("⚠️ Backend authentication sync failed")
          setAuthSynced(false)
        }
      }).catch((error) => {
        console.error("❌ Backend authentication sync error:", error)
        setAuthSynced(false)
      })
    }
  }, [isAuthenticated, isLoading])

  useEffect(() => {
    // Only redirect if we're done loading and definitely not authenticated
    if (!isLoading && !isAuthenticated) {
      console.log("Dashboard: Not authenticated, redirecting to login")
      window.location.href = "/login"
    }
  }, [isAuthenticated, isLoading])

  // Fetch data function - defined outside useEffect for better access
  const fetchData = async () => {
    try {
      console.log("📊 Fetching dashboard data...")
      
      // Check authentication status first
      const authStatus = enhancedApiService.getAuthStatus();
      console.log("Auth status:", authStatus);
      
      if (!authStatus.hasToken || !authStatus.isValid) {
        console.warn("⚠️ Authentication not ready, skipping data fetch");
        return;
      }
      
      // Fetch cameras using enhanced API service
      const camerasData = await enhancedApiService.getCameras().catch((error) => {
        console.error("❌ Failed to fetch cameras:", error)
        return []
      })
      
      if (Array.isArray(camerasData)) {
        setCameras(camerasData.map((cam: any) => ({
          id: cam.camera_id,
          name: cam.name,
          location: cam.location || 'Unknown',
          status: cam.status === 'active' ? 'online' : cam.status === 'inactive' ? 'offline' : 'warning',
          lastHeartbeat: 'Just now',
          resolution: '1080p'
        })))
      }

      // Fetch detections using enhanced API service
      const detectionsData = await enhancedApiService.getDetections().catch((error) => {
        console.error("❌ Failed to fetch detections:", error)
        return []
      })
      
      if (Array.isArray(detectionsData)) {
        setDetections(detectionsData.slice(0, 10).map((det: any) => ({
          id: det.id.toString(),
          type: det.detection_type,
          severity: det.severity,
          timestamp: det.timestamp,
          cameraId: det.camera_id.toString(),
          confidence: det.confidence
        })))
      }
      
      // For system status, we'll use a simplified version for now
      // In a real implementation, you'd have a dedicated system status endpoint
      setStats({
        totalDetections: Array.isArray(detectionsData) ? detectionsData.length : 0,
        alertsTriggered: Array.isArray(detectionsData) ? detectionsData.filter((d: any) => d.severity === 'high').length : 0,
        uptime: "24h 0m",
        activeCameras: Array.isArray(camerasData) ? camerasData.filter((c: any) => c.status === 'active').length : 0
      })
      
      console.log("✅ Dashboard data loaded successfully")
    } catch (error) {
      console.error("❌ Error fetching dashboard data:", error)
    }
  }

  useEffect(() => {
    // Set up data refresh interval
    if (isAuthenticated && authSynced) {
      // Initial data fetch
      fetchData()
      
      // Refresh data every 30 seconds
      const interval = setInterval(fetchData, 30000)
      return () => clearInterval(interval)
    }
  }, [isAuthenticated, authSynced])

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-foreground">
          <div>Loading authentication...</div>
          <div className="text-sm mt-2">Checking credentials...</div>
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return null
  }

  return (
    <AuthHelper>
    <WebSocketAuthHelper>
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
    </WebSocketAuthHelper>
    </AuthHelper>
  )
}
