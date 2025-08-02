'use client'

import { useState, useEffect } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { 
  User, 
  Mail, 
  Calendar, 
  Clock, 
  TrendingUp, 
  Edit, 
  Save, 
  X, 
  Shield, 
  Database, 
  Sparkles, 
  LogOut,
  Settings,
  Activity,
  Star,
  Zap
} from 'lucide-react'
import toast from 'react-hot-toast'

export default function UserProfile() {
  const { user, userStats, updateProfile, logout } = useAuth()
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [isUpdating, setIsUpdating] = useState(false)

  useEffect(() => {
    if (user) {
      setEditName(user.name)
    }
  }, [user])

  const handleUpdateProfile = async () => {
    if (!editName.trim()) {
      toast.error('Name cannot be empty')
      return
    }

    setIsUpdating(true)
try {
  const result = await updateProfile(editName.trim())
  if (result.success) {
    toast.success(result.message)
    setIsEditing(false)
  } else {
    toast.error(result.message)
  }
} catch {
  toast.error('Failed to update profile')
} finally {
  setIsUpdating(false)
}

  }

  const handleLogout = () => {
    logout()
  }

  if (!user) {
    return null
  }

  return (
    <div className="relative w-full min-h-screen bg-black text-white overflow-hidden">
      {/* Enhanced Background Effects */}
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="absolute top-[-20%] left-[-20%] w-[800px] h-[800px] bg-gradient-to-br from-purple-600/15 via-blue-600/10 to-cyan-600/8 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-[-20%] right-[-20%] w-[900px] h-[900px] bg-gradient-to-br from-emerald-600/12 via-teal-600/8 to-blue-600/10 rounded-full blur-3xl animate-pulse-slower" />
        <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] bg-gradient-to-br from-indigo-600/18 via-purple-600/12 to-pink-600/10 rounded-full blur-2xl animate-pulse" style={{transform: 'translate(-50%, -50%)'}} />
        
        {/* Grid pattern overlay */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:60px_60px] opacity-40" />
      </div>

      <div className="relative z-10 px-6 py-16">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-16">
            <div className="flex items-center justify-center gap-4 mb-8">
              <div className="relative">
                <User className="w-12 h-12 text-purple-400 animate-bounce" />
                <div className="absolute inset-0 w-12 h-12 bg-purple-400/20 rounded-full blur-xl animate-pulse" />
              </div>
              <h1 className="text-6xl font-black bg-gradient-to-r from-purple-400 via-blue-400 to-emerald-400 bg-clip-text text-transparent drop-shadow-2xl">
                User Profile
              </h1>
            </div>
            <p className="text-xl text-gray-300 font-medium">
              Manage your account and view your activity
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Profile Card */}
            <div className="lg:col-span-2">
              <div className="bg-gray-900/90 border border-gray-700/50 rounded-3xl p-8 shadow-2xl backdrop-blur-sm">
                <div className="flex items-center justify-between mb-8">
                  <h2 className="text-3xl font-bold text-white">Profile Information</h2>
                  <button
                    onClick={() => setIsEditing(!isEditing)}
                    className="flex items-center gap-2 px-4 py-2 bg-gray-800/60 hover:bg-gray-700/60 border border-gray-600/50 rounded-xl text-gray-300 hover:text-white transition-all duration-200"
                  >
                    {isEditing ? <X className="w-4 h-4" /> : <Edit className="w-4 h-4" />}
                    <span className="text-sm font-medium">
                      {isEditing ? 'Cancel' : 'Edit Profile'}
                    </span>
                  </button>
                </div>

                <div className="space-y-6">
                  {/* Name Field */}
                  <div className="space-y-3">
                    <label className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                      <User className="w-4 h-4" />
                      Full Name
                    </label>
                    {isEditing ? (
                      <div className="relative group">
                        <input
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          className="w-full px-4 py-3 bg-gray-800/80 border border-gray-600/50 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 backdrop-blur-sm"
                          placeholder="Enter your name"
                        />
                        <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/10 to-blue-500/10 opacity-0 group-focus-within:opacity-100 transition-opacity duration-300 pointer-events-none" />
                      </div>
                    ) : (
                      <div className="px-4 py-3 bg-gray-800/60 border border-gray-600/30 rounded-2xl text-white">
                        {user.name}
                      </div>
                    )}
                  </div>

                  {/* Email Field */}
                  <div className="space-y-3">
                    <label className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                      <Mail className="w-4 h-4" />
                      Email Address
                    </label>
                    <div className="px-4 py-3 bg-gray-800/60 border border-gray-600/30 rounded-2xl text-white">
                      {user.email}
                    </div>
                  </div>

                  {/* Created Date */}
                  <div className="space-y-3">
                    <label className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                      <Calendar className="w-4 h-4" />
                      Member Since
                    </label>
                    <div className="px-4 py-3 bg-gray-800/60 border border-gray-600/30 rounded-2xl text-white">
                      {user.created_at ? new Date(user.created_at).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric'
                      }) : 'N/A'}
                    </div>
                  </div>

                  {/* Last Login */}
                  <div className="space-y-3">
                    <label className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      Last Login
                    </label>
                    <div className="px-4 py-3 bg-gray-800/60 border border-gray-600/30 rounded-2xl text-white">
                      {user.last_login ? new Date(user.last_login).toLocaleString('en-US', {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      }) : 'N/A'}
                    </div>
                  </div>

                  {/* Save Button */}
                  {isEditing && (
                    <button
                      onClick={handleUpdateProfile}
                      disabled={isUpdating}
                      className="w-full flex items-center justify-center px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-bold rounded-2xl shadow-lg transition-all duration-200 disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-purple-400 transform hover:scale-105"
                    >
                      {isUpdating ? (
                        <>
                          <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                          Saving...
                        </>
                      ) : (
                        <>
                          <Save className="w-5 h-5 mr-2" />
                          Save Changes
                        </>
                      )}
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Stats Card */}
            <div className="space-y-6">
              {/* User Stats */}
              <div className="bg-gray-900/90 border border-gray-700/50 rounded-3xl p-6 shadow-2xl backdrop-blur-sm">
                <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-purple-400" />
                  Activity Stats
                </h3>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-gray-800/60 rounded-2xl border border-gray-600/30">
                    <div className="flex items-center gap-3">
                      <Activity className="w-5 h-5 text-blue-400" />
                      <span className="text-gray-300 font-medium">Total Logins</span>
                    </div>
                    <span className="text-white font-bold text-lg">
                      {userStats?.total_logins || 0}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between p-4 bg-gray-800/60 rounded-2xl border border-gray-600/30">
                    <div className="flex items-center gap-3">
                      <Star className="w-5 h-5 text-yellow-400" />
                      <span className="text-gray-300 font-medium">Account Status</span>
                    </div>
                    <span className="text-emerald-400 font-bold text-sm">Active</span>
                  </div>
                  
                  <div className="flex items-center justify-between p-4 bg-gray-800/60 rounded-2xl border border-gray-600/30">
                    <div className="flex items-center gap-3">
                      <Zap className="w-5 h-5 text-purple-400" />
                      <span className="text-gray-300 font-medium">AI Access</span>
                    </div>
                    <span className="text-blue-400 font-bold text-sm">Enabled</span>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="bg-gray-900/90 border border-gray-700/50 rounded-3xl p-6 shadow-2xl backdrop-blur-sm">
                <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                  <Settings className="w-5 h-5 text-purple-400" />
                  Quick Actions
                </h3>
                
                <div className="space-y-3">
                  <button
                    onClick={() => window.location.href = '/'}
                    className="w-full flex items-center gap-3 px-4 py-3 bg-gray-800/60 hover:bg-gray-700/60 border border-gray-600/50 rounded-xl text-gray-300 hover:text-white transition-all duration-200"
                  >
                    <Sparkles className="w-4 h-4" />
                    <span className="text-sm font-medium">Go to Classifier</span>
                  </button>
                  
                  <button
                    onClick={() => window.location.href = '/gallery'}
                    className="w-full flex items-center gap-3 px-4 py-3 bg-gray-800/60 hover:bg-gray-700/60 border border-gray-600/50 rounded-xl text-gray-300 hover:text-white transition-all duration-200"
                  >
                    <Database className="w-4 h-4" />
                    <span className="text-sm font-medium">View Gallery</span>
                  </button>
                  
                  <button
                    onClick={handleLogout}
                    className="w-full flex items-center gap-3 px-4 py-3 bg-red-900/60 hover:bg-red-800/60 border border-red-600/50 rounded-xl text-red-300 hover:text-white transition-all duration-200"
                  >
                    <LogOut className="w-4 h-4" />
                    <span className="text-sm font-medium">Sign Out</span>
                  </button>
                </div>
              </div>

              {/* Security Info */}
              <div className="bg-gray-900/90 border border-gray-700/50 rounded-3xl p-6 shadow-2xl backdrop-blur-sm">
                <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-emerald-400" />
                  Security
                </h3>
                
                <div className="space-y-3 text-sm text-gray-400">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                    <span>JWT Authentication</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                    <span>SQLite Database</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                    <span>Local Processing</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                    <span>Session Management</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 