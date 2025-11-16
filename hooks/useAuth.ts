"use client"

import { useUser, useAuth as useClerkAuth } from "@clerk/nextjs"
import { useState, useEffect } from "react"

interface User {
  id: string
  username: string
  email: string
  role: string
  is_active: boolean
  created_at: string
}

export function useAuth() {
  const { user, isLoaded: isUserLoaded } = useUser()
  const { signOut } = useClerkAuth()
  const [userData, setUserData] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (isUserLoaded) {
      if (user) {
        // Map Clerk user to your existing user structure
        const mappedUser: User = {
          id: user.id,
          username: user.username || user.emailAddresses[0]?.emailAddress?.split('@')[0] || 'user',
          email: user.emailAddresses[0]?.emailAddress || '',
          role: 'viewer', // Default role, you can customize this
          is_active: true,
          created_at: user.createdAt?.toISOString() || new Date().toISOString(),
        }
        setUserData(mappedUser)
      } else {
        setUserData(null)
      }
      setIsLoading(false)
    }
  }, [user, isUserLoaded])

  const logout = async () => {
    await signOut()
    window.location.href = "/"
  }

  return {
    user: userData,
    isLoading,
    isAuthenticated: !!user,
    logout,
  }
}
