"use client"

import { useEffect } from "react"
import { useUser } from "@clerk/nextjs"
import { apiService } from "@/services/api"

export function useAuthSync() {
  const { user, isLoaded } = useUser()

  useEffect(() => {
    if (isLoaded && user) {
      // Sync Clerk user with backend when user is loaded
      const syncAuth = async () => {
        try {
          console.log("🔄 Syncing Clerk user with backend...")
          await apiService.syncClerkWithBackend()
          console.log("✅ Auth sync completed")
        } catch (error) {
          console.error("❌ Auth sync failed:", error)
        }
      }
      
      syncAuth()
    }
  }, [user, isLoaded])

  return null // This hook doesn't return anything, it just syncs auth
}