'use client'

import { useAuth } from '@/contexts/AuthContext'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
// import { Loader2 } from 'lucide-react'

interface ProtectedRouteProps {
  children: React.ReactNode
}

export default function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { user, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth')
    }
  }, [user, loading, router])

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="relative mb-6">
            <div className="h-16 w-16 border-4 border-purple-500/30 border-t-purple-500 animate-spin rounded-full mx-auto" />
            <div className="absolute inset-0 h-16 w-16 border-4 border-blue-500/20 border-t-blue-500 animate-spin rounded-full mx-auto" style={{animationDelay: '0.5s'}} />
          </div>
          <p className="text-purple-400 text-lg font-semibold animate-pulse">
            Loading your session...
          </p>
        </div>
      </div>
    )
  }

  if (!user) {
    return null // Will redirect to auth page
  }

  return <>{children}</>
} 