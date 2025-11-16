"use client"

import { useEffect, useState } from "react"
import { useAuth } from "@/hooks/useAuth"

export default function DebugAuthPage() {
  const { user, isLoading, isAuthenticated } = useAuth()
  const [storageData, setStorageData] = useState<any>({})

  useEffect(() => {
    const checkStorage = () => {
      const token = localStorage.getItem("token")
      const userStr = localStorage.getItem("user")
      let userData = null
      
      try {
        userData = userStr ? JSON.parse(userStr) : null
      } catch (e) {
        userData = { error: "Failed to parse user data" }
      }

      setStorageData({
        hasToken: !!token,
        tokenLength: token?.length || 0,
        tokenPreview: token?.substring(0, 50) + "...",
        hasUser: !!userStr,
        userData: userData,
      })
    }

    checkStorage()
    
    // Check every second
    const interval = setInterval(checkStorage, 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-slate-900 p-8 text-white">
      <h1 className="text-3xl font-bold mb-8">Authentication Debug Page</h1>

      <div className="space-y-6">
        <div className="bg-slate-800 p-6 rounded-lg">
          <h2 className="text-xl font-semibold mb-4">useAuth Hook State</h2>
          <div className="space-y-2">
            <p><strong>isLoading:</strong> {isLoading ? "true" : "false"}</p>
            <p><strong>isAuthenticated:</strong> {isAuthenticated ? "true" : "false"}</p>
            <p><strong>user:</strong> {user ? JSON.stringify(user, null, 2) : "null"}</p>
          </div>
        </div>

        <div className="bg-slate-800 p-6 rounded-lg">
          <h2 className="text-xl font-semibold mb-4">localStorage Data</h2>
          <div className="space-y-2">
            <p><strong>Has Token:</strong> {storageData.hasToken ? "YES" : "NO"}</p>
            <p><strong>Token Length:</strong> {storageData.tokenLength}</p>
            <p><strong>Token Preview:</strong> <code className="text-xs">{storageData.tokenPreview}</code></p>
            <p><strong>Has User:</strong> {storageData.hasUser ? "YES" : "NO"}</p>
            <p><strong>User Data:</strong></p>
            <pre className="bg-slate-900 p-4 rounded text-xs overflow-auto">
              {JSON.stringify(storageData.userData, null, 2)}
            </pre>
          </div>
        </div>

        <div className="bg-slate-800 p-6 rounded-lg">
          <h2 className="text-xl font-semibold mb-4">Actions</h2>
          <div className="space-x-4">
            <button
              onClick={() => window.location.href = "/login"}
              className="bg-blue-600 px-4 py-2 rounded hover:bg-blue-700"
            >
              Go to Login
            </button>
            <button
              onClick={() => window.location.href = "/dashboard"}
              className="bg-green-600 px-4 py-2 rounded hover:bg-green-700"
            >
              Go to Dashboard
            </button>
            <button
              onClick={() => {
                localStorage.clear()
                window.location.reload()
              }}
              className="bg-red-600 px-4 py-2 rounded hover:bg-red-700"
            >
              Clear Storage & Reload
            </button>
          </div>
        </div>

        <div className="bg-slate-800 p-6 rounded-lg">
          <h2 className="text-xl font-semibold mb-4">Console Logs</h2>
          <p className="text-sm text-slate-400">
            Open browser console (F12) to see detailed logs from useAuth and dashboard
          </p>
        </div>
      </div>
    </div>
  )
}
