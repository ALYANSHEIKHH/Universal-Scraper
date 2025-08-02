'use client'

import { useState } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { Eye, EyeOff, Mail, Lock, User,  Shield, Zap, Database,  Globe, TrendingUp, Clock, CheckCircle,  Star } from 'lucide-react'
import toast from 'react-hot-toast'

interface RegisterFormProps {
  onSwitchToLogin: () => void
}

export default function RegisterForm({ onSwitchToLogin }: RegisterFormProps) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const { register } = useAuth()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!name || !email || !password || !confirmPassword) {
      toast.error('Please fill in all fields')
      return
    }

    if (password !== confirmPassword) {
      toast.error('Passwords do not match')
      return
    }

    if (password.length < 6) {
      toast.error('Password must be at least 6 characters long')
      return
    }

    setIsLoading(true)
try {
  const result = await register(name, email, password)
  if (result.success) {
    toast.success(result.message)
  } else {
    toast.error(result.message)
  }
} catch {
  toast.error('An unexpected error occurred. Please try again.')
} finally {
  setIsLoading(false)
}
  }

  return (
    <div className="relative w-full min-h-screen flex items-center justify-center overflow-hidden bg-black text-white">
      {/* Enhanced Background Effects */}
      <div className="absolute inset-0 pointer-events-none z-0">
        {/* Multiple gradient orbs with different animations */}
        <div className="absolute top-[-20%] left-[-20%] w-[800px] h-[800px] bg-gradient-to-br from-purple-600/20 via-blue-600/15 to-cyan-600/10 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-[-20%] right-[-20%] w-[900px] h-[900px] bg-gradient-to-br from-emerald-600/15 via-teal-600/10 to-blue-600/15 rounded-full blur-3xl animate-pulse-slower" />
        <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] bg-gradient-to-br from-indigo-600/25 via-purple-600/20 to-pink-600/15 rounded-full blur-2xl animate-pulse" style={{transform: 'translate(-50%, -50%)'}} />
        
        {/* Floating geometric shapes with enhanced animations */}
        <div className="absolute left-1/4 top-1/3 w-40 h-40 bg-gradient-to-br from-purple-500/30 to-blue-500/20 opacity-50 rounded-full blur-2xl animate-float" />
        <div className="absolute right-1/4 bottom-1/4 w-24 h-24 bg-gradient-to-br from-emerald-500/25 to-cyan-500/20 opacity-40 rounded-full blur-2xl animate-float-reverse" />
        <div className="absolute left-1/3 bottom-1/3 w-20 h-20 bg-gradient-to-br from-pink-500/35 to-purple-500/25 opacity-45 rounded-full blur-xl animate-float" style={{animationDelay: '2s'}} />
        <div className="absolute right-1/3 top-1/4 w-16 h-16 bg-gradient-to-br from-yellow-500/30 to-orange-500/25 opacity-35 rounded-full blur-xl animate-float-reverse" style={{animationDelay: '3s'}} />
        
        {/* Enhanced grid pattern overlay */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:60px_60px] opacity-40" />
        
        {/* Animated particles */}
        <div className="absolute inset-0">
          {[...Array(15)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-white/20 rounded-full animate-pulse"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 3}s`,
                animationDuration: `${2 + Math.random() * 2}s`
              }}
            />
          ))}
        </div>
      </div>

      <div className="relative z-10 w-full max-w-md px-6">
        {/* Enhanced Header */}
        <div className="text-center mb-10">
          <div className="flex items-center justify-center gap-4 mb-8">
            <div className="relative">
              <Star className="w-12 h-12 text-purple-400 animate-bounce" />
              <div className="absolute inset-0 w-12 h-12 bg-purple-400/20 rounded-full blur-xl animate-pulse" />
            </div>
            <h1 className="text-5xl font-black bg-gradient-to-r from-purple-400 via-blue-400 to-emerald-400 bg-clip-text text-transparent drop-shadow-2xl">
              Join Us
            </h1>
          </div>
          <p className="text-xl text-gray-300 font-medium leading-relaxed">
            Create your account and start classifying images with <span className="text-purple-300 font-semibold">Universal AI</span>
          </p>
        </div>

        {/* Enhanced Feature Highlights */}
        <div className="grid grid-cols-1 gap-4 mb-10">
          <div className="group bg-gray-900/60 border border-gray-700/50 rounded-2xl p-5 backdrop-blur-sm hover:bg-gray-800/60 transition-all duration-300 hover:scale-105">
            <div className="flex items-center gap-4">
              <div className="relative">
                <Zap className="w-7 h-7 text-yellow-400 group-hover:animate-pulse" />
                <div className="absolute inset-0 w-7 h-7 bg-yellow-400/20 rounded-full blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              </div>
              <div>
                <h3 className="text-white font-bold text-lg mb-1">Instant Access</h3>
                <p className="text-gray-400 text-sm">Start classifying images immediately</p>
              </div>
            </div>
          </div>
          
          <div className="group bg-gray-900/60 border border-gray-700/50 rounded-2xl p-5 backdrop-blur-sm hover:bg-gray-800/60 transition-all duration-300 hover:scale-105">
            <div className="flex items-center gap-4">
              <div className="relative">
                <Globe className="w-7 h-7 text-blue-400 group-hover:animate-pulse" />
                <div className="absolute inset-0 w-7 h-7 bg-blue-400/20 rounded-full blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              </div>
              <div>
                <h3 className="text-white font-bold text-lg mb-1">Universal Support</h3>
                <p className="text-gray-400 text-sm">Any image type, any format</p>
              </div>
            </div>
          </div>
          
          <div className="group bg-gray-900/60 border border-gray-700/50 rounded-2xl p-5 backdrop-blur-sm hover:bg-gray-800/60 transition-all duration-300 hover:scale-105">
            <div className="flex items-center gap-4">
              <div className="relative">
                <Shield className="w-7 h-7 text-emerald-400 group-hover:animate-pulse" />
                <div className="absolute inset-0 w-7 h-7 bg-emerald-400/20 rounded-full blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              </div>
              <div>
                <h3 className="text-white font-bold text-lg mb-1">Privacy First</h3>
                <p className="text-gray-400 text-sm">Your data stays on your device</p>
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Registration Form */}
        <div className="bg-gray-900/90 border border-gray-700/50 rounded-3xl p-8 shadow-2xl backdrop-blur-sm">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Name Field */}
            <div className="space-y-3">
              <label htmlFor="name" className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <User className="w-4 h-4" />
                Full Name
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <User className="h-5 w-5 text-gray-400 group-focus-within:text-purple-400 transition-colors duration-200" />
                </div>
                <input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full pl-12 pr-4 py-4 bg-gray-800/80 border border-gray-600/50 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 backdrop-blur-sm group-hover:border-gray-500/50"
                  placeholder="Enter your full name"
                  required
                />
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/10 to-blue-500/10 opacity-0 group-focus-within:opacity-100 transition-opacity duration-300 pointer-events-none" />
              </div>
            </div>

            {/* Email Field */}
            <div className="space-y-3">
              <label htmlFor="email" className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <Mail className="w-4 h-4" />
                Email Address
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Mail className="h-5 w-5 text-gray-400 group-focus-within:text-purple-400 transition-colors duration-200" />
                </div>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full pl-12 pr-4 py-4 bg-gray-800/80 border border-gray-600/50 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 backdrop-blur-sm group-hover:border-gray-500/50"
                  placeholder="Enter your email address"
                  required
                />
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/10 to-blue-500/10 opacity-0 group-focus-within:opacity-100 transition-opacity duration-300 pointer-events-none" />
              </div>
            </div>

            {/* Password Field */}
            <div className="space-y-3">
              <label htmlFor="password" className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <Lock className="w-4 h-4" />
                Password
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-400 group-focus-within:text-purple-400 transition-colors duration-200" />
                </div>
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full pl-12 pr-12 py-4 bg-gray-800/80 border border-gray-600/50 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 backdrop-blur-sm group-hover:border-gray-500/50"
                  placeholder="Create a strong password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-gray-300 transition-colors duration-200"
                >
                  {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/10 to-blue-500/10 opacity-0 group-focus-within:opacity-100 transition-opacity duration-300 pointer-events-none" />
              </div>
              <p className="text-xs text-gray-400 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" />
                Must be at least 6 characters long
              </p>
            </div>

            {/* Confirm Password Field */}
            <div className="space-y-3">
              <label htmlFor="confirmPassword" className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <Lock className="w-4 h-4" />
                Confirm Password
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-400 group-focus-within:text-purple-400 transition-colors duration-200" />
                </div>
                <input
                  id="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full pl-12 pr-12 py-4 bg-gray-800/80 border border-gray-600/50 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 backdrop-blur-sm group-hover:border-gray-500/50"
                  placeholder="Confirm your password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-gray-300 transition-colors duration-200"
                >
                  {showConfirmPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/10 to-blue-500/10 opacity-0 group-focus-within:opacity-100 transition-opacity duration-300 pointer-events-none" />
              </div>
            </div>

            {/* Enhanced Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full group relative flex items-center justify-center px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-bold rounded-2xl shadow-xl transition-all duration-300 disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-purple-400 transform hover:scale-105 disabled:hover:scale-100"
            >
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/20 to-blue-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="relative flex items-center gap-3">
                {isLoading ? (
                  <>
                    <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Creating Account...</span>
                  </>
                ) : (
                  <>
                    <Star className="w-6 h-6 group-hover:animate-bounce" />
                    <span>Create Account</span>
                  </>
                )}
              </div>
            </button>
          </form>

          {/* Enhanced Switch to Login */}
          <div className="mt-8 text-center">
            <p className="text-gray-400 text-base">
              Already have an account?{' '}
              <button
                onClick={onSwitchToLogin}
                className="text-purple-400 hover:text-purple-300 font-bold transition-all duration-200 hover:underline"
              >
                Sign in here
              </button>
            </p>
          </div>

          {/* Enhanced Terms and Privacy */}
          <div className="mt-8 p-5 bg-gray-800/50 rounded-2xl border border-gray-600/30">
            <div className="flex items-center gap-2 mb-3">
              <Shield className="w-4 h-4 text-emerald-400" />
              <p className="text-sm text-gray-300 font-semibold">Secure Registration</p>
            </div>
            <div className="space-y-2 text-xs text-gray-400">
              <p>• Your data is encrypted and stored securely</p>
              <p>• We never share your personal information</p>
              <p>• Local processing ensures maximum privacy</p>
            </div>
          </div>
        </div>

        {/* Enhanced Footer Stats */}
        <div className="mt-10 text-center">
          <div className="flex items-center justify-center gap-6 text-sm text-gray-500">
            <div className="flex items-center gap-1">
              <TrendingUp className="w-3 h-3" />
              <span>99.9% Uptime</span>
            </div>
            <div className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              <span>24/7 Support</span>
            </div>
            <div className="flex items-center gap-1">
              <Database className="w-3 h-3" />
              <span>Secure Database</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 