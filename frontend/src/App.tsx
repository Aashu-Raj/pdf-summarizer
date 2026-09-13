import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"

import { Toaster } from "@/components/ui/sonner"
import { Spinner } from "@/components/ui/spinner"
import { AuthProvider, useAuth } from "@/lib/auth"
import { AuthPage } from "@/pages/AuthPage"
import { HomePage } from "@/pages/HomePage"

function ProtectedRoutes() {
  const { user, loading } = useAuth()

  if (loading) {
    return (
      <div className="flex min-h-svh items-center justify-center gap-2 text-sm text-muted-foreground">
        <Spinner /> Loading…
      </div>
    )
  }

  if (!user) {
    return <AuthPage />
  }

  return <HomePage />
}

export function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/*" element={<ProtectedRoutes />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <Toaster />
      </AuthProvider>
    </BrowserRouter>
  )
}

export default App
