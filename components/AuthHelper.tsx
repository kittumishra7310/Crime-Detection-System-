"use client"

import { useEffect, useState } from "react"
import { useUser } from "@clerk/nextjs"

interface AuthHelperProps {
  children: React.ReactNode
}

export function AuthHelper({ children }: AuthHelperProps) {
  const { user, isLoaded } = useUser()
  const [isSettingUp, setIsSettingUp] = useState(false)

  useEffect(() => {
    const setupAuth = async () => {
      if (!isLoaded || !user) return

      // Check if we already have a token
      const existingToken = localStorage.getItem('token')
      if (existingToken) {
        console.log("✅ Token already exists, skipping setup")
        return
      }

      setIsSettingUp(true)
      console.log("🔄 Setting up authentication for user:", user.id)

      try {
        // Check if backend is accessible
        try {
          const healthCheck = await fetch('http://localhost:8000/health', { method: 'GET' })
          if (!healthCheck.ok) {
            console.warn("⚠️ Backend health check failed")
          } else {
            console.log("✅ Backend is accessible")
          }
        } catch (healthError) {
          console.error("❌ Cannot connect to backend at http://localhost:8000")
          console.error("Make sure the backend server is running: python Backend/main.py")
          throw new Error("Backend server is not accessible")
        }

        // Create a deterministic username and password based on Clerk user ID
        const clerkUserId = user.id
        const email = user.emailAddresses?.[0]?.emailAddress || `clerk_${clerkUserId}@clerk.local`
        const username = user.username || `clerk_user_${clerkUserId.substring(0, 8)}`
        const password = 'clerk_auth_' + clerkUserId

        console.log("📧 Using credentials:", { username, email })

        // Try to login first (user might already exist)
        const loginResponse = await fetch('http://localhost:8000/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password })
        })

        let token = null
        let userData = null

        if (loginResponse.ok) {
          const data = await loginResponse.json()
          token = data.access_token
          userData = data
          console.log("✅ Login successful")
        } else {
          const loginError = await loginResponse.text()
          console.log("📝 User not found, attempting Clerk sync...", loginResponse.status)
          
          // Try to sync/register the Clerk user (creates or updates)
          const registerResponse = await fetch('http://localhost:8000/api/auth/clerk-sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              username,
              email,
              password,
              role: 'viewer'
            })
          })

          if (registerResponse.ok) {
            const data = await registerResponse.json()
            console.log("✅ Registration successful:", data)
            // Now try to login
            const loginRetryResponse = await fetch('http://localhost:8000/api/auth/login', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ username, password })
            })

            if (loginRetryResponse.ok) {
              const loginData = await loginRetryResponse.json()
              token = loginData.access_token
              userData = loginData
              console.log("✅ Registration and login successful")
            } else {
              const loginError = await loginRetryResponse.text()
              console.error("❌ Login after registration failed:", loginError)
            }
          } else {
            const registerError = await registerResponse.text()
            console.error("❌ Registration failed:", registerResponse.status, registerError)
            
            // If registration failed because user already exists (400), try login again
            if (registerResponse.status === 400) {
              console.log("🔄 User might already exist, trying login again...")
              const finalLoginResponse = await fetch('http://localhost:8000/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
              })
              
              if (finalLoginResponse.ok) {
                const loginData = await finalLoginResponse.json()
                token = loginData.access_token
                userData = loginData
                console.log("✅ Login successful after registration conflict")
              } else {
                console.error("❌ Final login attempt also failed")
                // User exists but password is wrong - need to reset
                console.error("⚠️ User exists but credentials don't match. You may need to clear the database or use different credentials.")
              }
            }
          }
        }

        if (token) {
          localStorage.setItem('token', token)
          localStorage.setItem('user', JSON.stringify(userData))
          console.log("✅ Token stored successfully")
          
          // Dispatch custom event to notify components
          window.dispatchEvent(new CustomEvent('authTokenUpdated', { detail: { token } }))
        } else {
          console.error("❌ Failed to obtain authentication token")
        }
      } catch (error) {
        console.error("❌ Authentication setup failed:", error)
      } finally {
        setIsSettingUp(false)
      }
    }

    setupAuth()
  }, [user, isLoaded])

  if (!isLoaded || isSettingUp) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-foreground">Setting up authentication...</p>
        </div>
      </div>
    )
  }

  return <>{children}</>
}