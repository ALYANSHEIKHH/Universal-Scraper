'use client'

import { useState } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import ProtectedRoute from '@/components/ProtectedRoute'
import HeroSection from '@/components/HeroSection'
import ResultSummary from '@/components/ResultSummary'
import { LogOut, User, Sparkles } from 'lucide-react'

import { PredictionResult } from '@/types'
import './globals.css'

export default function Home() {
  const [summary, setSummary] = useState<PredictionResult | null>(null)
  const { user, logout } = useAuth()

  const handleLogout = () => {
    logout()
  }

  return (
    <ProtectedRoute>
      <main className="relative">
        {/* Header with User Info and Logout */}
        <header className="absolute top-0 left-0 right-0 z-50 p-6">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Sparkles className="w-8 h-8 text-purple-400" />
                <div className="absolute inset-0 w-8 h-8 bg-purple-400/20 rounded-full blur-xl animate-pulse" />
              </div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                Universal AI
              </h1>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-4 py-2 bg-gray-900/80 backdrop-blur-sm border border-gray-700/50 rounded-xl">
                <User className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-300 font-medium">
                  Welcome, {user?.name}
                </span>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 bg-gray-900/80 hover:bg-gray-800/80 backdrop-blur-sm border border-gray-700/50 hover:border-gray-600/50 rounded-xl transition-all duration-200 text-gray-300 hover:text-white"
              >
                <LogOut className="w-4 h-4" />
                <span className="text-sm font-medium">Logout</span>
              </button>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="pt-24 px-6">
          <HeroSection onSummary={(result: PredictionResult) => setSummary(result)} />
          {summary && <ResultSummary summary={summary} />}
        </div>
      </main>
    </ProtectedRoute>
  )
}
