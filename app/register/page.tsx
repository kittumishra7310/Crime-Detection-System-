"use client"

import { SignUp } from "@clerk/nextjs"
import { ShieldIcon } from "@/components/icons"

export default function RegisterPage() {
  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <ShieldIcon className="h-12 w-12 text-blue-500" />
          </div>
          <h1 className="text-2xl font-bold text-slate-100 mb-2">Create Account</h1>
          <p className="text-slate-400">Register for Smart Surveillance System</p>
        </div>
        <SignUp 
          routing="hash"
          forceRedirectUrl="/dashboard"
          signInUrl="/login"
          appearance={{
            elements: {
              rootBox: "mx-auto",
              card: "bg-slate-800 border-slate-700",
              headerTitle: "text-slate-100",
              headerSubtitle: "text-slate-400",
              socialButtonsBlockButton: "bg-slate-700 border-slate-600 text-slate-100 hover:bg-slate-600",
              formFieldLabel: "text-slate-200",
              formFieldInput: "bg-slate-700 border-slate-600 text-slate-100 placeholder:text-slate-400",
              footerActionLink: "text-blue-400 hover:text-blue-300",
              dividerLine: "bg-slate-600",
              dividerText: "text-slate-400",
              formButtonPrimary: "bg-blue-600 hover:bg-blue-700 text-white",
            }
          }}
        />
      </div>
    </div>
  )
}
