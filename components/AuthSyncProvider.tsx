"use client"

import { useAuthSync } from "@/hooks/useAuthSync"

export function AuthSyncProvider({ children }: { children: React.ReactNode }) {
  useAuthSync()
  return children
}