"use client"

import { useUser } from "@clerk/nextjs"
import { useEffect, useState } from "react"
import { apiService } from "@/services/api"

interface AuthSyncProps {
  children: React.ReactNode
  onSyncComplete?: () => void
  onSyncError?: (error: Error) => void
}

export function AuthSync({ children, onSyncComplete, onSyncError }: AuthSyncProps) {
  const { user, isLoaded } = useUser()
  const [isSynced, setIsSynced] = useState(false)
  const [syncError, setSyncError] = useState<Error | null>(null)

  useEffect(() => {
    const syncAuth = async () => {
      if (!isLoaded || !user) {
        setIsSynced(true) // No user to sync, consider it complete
        return
      }

      try {
        console.log("🔄 Syncing Clerk user with backend...")
        await apiService.syncClerkWithBackend()
        console.log("✅ Auth sync completed")
        setIsSynced(true)
        setSyncError(null)
        onSyncComplete?.()
      } catch (error) {
        console.error("❌ Auth sync failed:", error)
        setSyncError(error as Error)
        onSyncError?.(error as Error)
        // Still mark as synced to allow the app to continue
        setIsSynced(true)
      }
    }

    if (isLoaded) {
      syncAuth()
    }
  }, [user, isLoaded, onSyncComplete, onSyncError])

  if (!isLoaded || !isSynced) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-foreground">Syncing authentication...</p>
        </div>
      </div>
    )
  }

  if (syncError) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="text-destructive mb-4">⚠️ Authentication Sync Error</div>
          <p className="text-muted-foreground text-sm mb-4">{syncError.message}</p>
          <p className="text-muted-foreground text-xs">Some features may be limited</p>
          {children}
        </div>
      </div>
    )
  }

  return <>{children}</>
}