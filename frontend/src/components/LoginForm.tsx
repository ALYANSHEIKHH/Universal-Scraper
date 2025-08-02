'use client'

import { useState, useEffect } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import {
  Eye, EyeOff, Mail, Lock, Sparkles, Shield,
  Zap, Database, Globe, TrendingUp, Clock,
  CheckCircle, AlertCircle
} from 'lucide-react'
import toast from 'react-hot-toast'

interface LoginFormProps {
  onSwitchToRegister: () => void
}

interface ParticleStyle {
  left: string
  top: string
  animationDelay: string
  animationDuration: string
}

interface FeatureItem {
  icon: React.ComponentType<{ className?: string }>
  title: string
  desc: string
  iconColor: string
  glowColor: string
}

export default function LoginForm({ onSwitchToRegister }: LoginFormProps) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [particles, setParticles] = useState<ParticleStyle[]>([])
  const { login } = useAuth()

  useEffect(() => {
    const styles = Array.from({ length: 15 }, () => ({
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
      animationDelay: `${Math.random() * 3}s`,
      animationDuration: `${2 + Math.random() * 2}s`
    }))
    setParticles(styles)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!email || !password) {
      toast.error('Please fill in all fields')
      return
    }

    setIsLoading(true)
    try {
      const result = await login(email, password)
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

  // Fixed: Use explicit color classes instead of dynamic ones
  const features: FeatureItem[] = [
    { 
      icon: Zap, 
      title: 'Lightning Fast', 
      desc: 'Instant AI classification with advanced algorithms', 
      iconColor: 'text-yellow-400',
      glowColor: 'bg-yellow-400/20'
    },
    { 
      icon: Globe, 
      title: 'Universal', 
      desc: 'Classify any type of image with precision', 
      iconColor: 'text-blue-400',
      glowColor: 'bg-blue-400/20'
    },
    { 
      icon: Shield, 
      title: 'Secure & Private', 
      desc: 'Local processing ensures your data stays private', 
      iconColor: 'text-emerald-400',
      glowColor: 'bg-emerald-400/20'
    }
  ]

  return (
    <div className="relative w-full min-h-screen flex items-center justify-center overflow-hidden bg-black text-white px-4 sm:px-6 lg:px-12">
      <style jsx>{`
        @keyframes pulse-slow {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 0.8; }
        }
        @keyframes pulse-slower {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 0.6; }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(180deg); }
        }
        @keyframes float-reverse {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(20px) rotate(-180deg); }
        }
        .animate-pulse-slow {
          animation: pulse-slow 4s ease-in-out infinite;
        }
        .animate-pulse-slower {
          animation: pulse-slower 6s ease-in-out infinite;
        }
        .animate-float {
          animation: float 6s ease-in-out infinite;
        }
        .animate-float-reverse {
          animation: float-reverse 8s ease-in-out infinite;
        }
        .delay-2000 {
          animation-delay: 2s;
        }
        .delay-3000 {
          animation-delay: 3s;
        }
      `}</style>
      
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="absolute top-[-20%] left-[-20%] w-[600px] sm:w-[700px] md:w-[800px] h-[600px] sm:h-[700px] md:h-[800px] bg-gradient-to-br from-purple-600/20 via-blue-600/15 to-cyan-600/10 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-[-20%] right-[-20%] w-[700px] md:w-[900px] h-[700px] md:h-[900px] bg-gradient-to-br from-emerald-600/15 via-teal-600/10 to-blue-600/15 rounded-full blur-3xl animate-pulse-slower" />
        <div className="absolute top-1/2 left-1/2 w-[300px] md:w-[400px] h-[300px] md:h-[400px] bg-gradient-to-br from-indigo-600/25 via-purple-600/20 to-pink-600/15 rounded-full blur-2xl animate-pulse transform -translate-x-1/2 -translate-y-1/2" />
        <div className="absolute left-1/4 top-1/3 w-32 sm:w-40 h-32 sm:h-40 bg-gradient-to-br from-purple-500/30 to-blue-500/20 opacity-50 rounded-full blur-2xl animate-float" />
        <div className="absolute right-1/4 bottom-1/4 w-20 sm:w-24 h-20 sm:h-24 bg-gradient-to-br from-emerald-500/25 to-cyan-500/20 opacity-40 rounded-full blur-2xl animate-float-reverse" />
        <div className="absolute left-1/3 bottom-1/3 w-16 sm:w-20 h-16 sm:h-20 bg-gradient-to-br from-pink-500/35 to-purple-500/25 opacity-45 rounded-full blur-xl animate-float delay-2000" />
        <div className="absolute right-1/3 top-1/4 w-14 sm:w-16 h-14 sm:h-16 bg-gradient-to-br from-yellow-500/30 to-orange-500/25 opacity-35 rounded-full blur-xl animate-float-reverse delay-3000" />
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:60px_60px] opacity-40" />
        <div className="absolute inset-0">
          {particles.map((style, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-white/20 rounded-full animate-pulse"
              style={style}
            />
          ))}
        </div>
      </div>

      <div className="relative z-10 w-full max-w-md sm:max-w-lg px-4 sm:px-6">
        {/* Header */}
        <div className="text-center  mb-10">
          <div className="flex items-center justify-center gap-4 mb-6 sm:mb-8">
            <div className="relative mt-3">
              <Sparkles className="w-10 sm:w-12 h-10 sm:h-12 text-purple-400 animate-bounce" />
              <div className="absolute inset-0 w-10 sm:w-12 h-10 sm:h-12 bg-purple-400/20 rounded-full blur-xl animate-pulse" />
            </div>
            <h1 className="text-4xl mr-11  sm:text-5xl font-black bg-gradient-to-r from-purple-400 via-blue-400 to-emerald-400 bg-clip-text text-transparent drop-shadow-2xl">
              Welcome
            </h1>
          </div>
          <p className="text-base sm:text-xl text-gray-300 font-medium leading-relaxed">
            Sign in to access your <span className="text-purple-300 font-semibold">Universal AI</span> Image Classifier
          </p>
        </div>

        {/* Features */}
        <div className="grid grid-cols-1 gap-4 mb-10">
          {features.map(({ icon: Icon, title, desc, iconColor, glowColor }, i) => (
            <div key={i} className="group bg-gray-900/60 border border-gray-700/50 rounded-2xl p-5 backdrop-blur-sm hover:bg-gray-800/60 transition-all duration-300 hover:scale-105">
              <div className="flex items-center gap-4">
                <div className="relative">
                  <Icon className={`w-6 h-6 sm:w-7 sm:h-7 ${iconColor} group-hover:animate-pulse`} />
                  <div className={`absolute inset-0 w-6 h-6 sm:w-7 sm:h-7 ${glowColor} rounded-full blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />
                </div>
                <div>
                  <h3 className="text-white font-bold text-base sm:text-lg mb-1">{title}</h3>
                  <p className="text-gray-400 text-sm">{desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Login Form */}
        <div className="bg-gray-900/90 border border-gray-700/50 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-sm">
          <form onSubmit={handleSubmit} className="space-y-6" noValidate>
            {/* Email */}
            <div className="space-y-3">
              <label htmlFor="email" className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <Mail className="w-4 h-4" /> Email Address
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
                  className="w-full pl-12 pr-4 py-3 sm:py-4 bg-gray-800/80 border border-gray-600/50 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 backdrop-blur-sm"
                  placeholder="Enter your email address"
                  aria-describedby="email-error"
                  required
                />
              </div>
            </div>

            {/* Password */}
            <div className="space-y-3">
              <label htmlFor="password" className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <Lock className="w-4 h-4" /> Password
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
                  className="w-full pl-12 pr-12 py-3 sm:py-4 bg-gray-800/80 border border-gray-600/50 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 backdrop-blur-sm"
                  placeholder="Enter your password"
                  aria-describedby="password-error"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-gray-300 transition-colors duration-200"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full group relative flex items-center justify-center px-6 sm:px-8 py-3 sm:py-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-bold rounded-2xl shadow-xl transition-all duration-300 disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-purple-400 transform hover:scale-105 disabled:hover:scale-100 disabled:cursor-not-allowed"
              aria-describedby="submit-status"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-3" />
                  <span>Signing In...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5 mr-3 group-hover:animate-bounce" />
                  <span>Sign In</span>
                </>
              )}
            </button>
          </form>

          {/* Switch */}
          <div className="mt-6 sm:mt-8 text-center text-sm sm:text-base">
            <p className="text-gray-400">
              Don&apos;t have an account?{' '}
              <button
                onClick={onSwitchToRegister}
                className="text-purple-400 hover:text-purple-300 font-bold transition-all duration-200 hover:underline focus:outline-none focus:ring-2 focus:ring-purple-400 rounded"
              >
                Create one now
              </button>
            </p>
          </div>

          {/* Demo Credentials */}
          <div className="mt-6 sm:mt-8 p-4 sm:p-5 bg-gray-800/50 rounded-2xl border border-gray-600/30 text-sm">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              <p className="text-gray-300 font-semibold">Demo Credentials</p>
            </div>
            <p className="text-gray-400">Email: <span className="text-white font-mono">demo@example.com</span></p>
            <p className="text-gray-400">Password: <span className="text-white font-mono">any password</span></p>
            <div className="mt-3 flex items-center gap-2 text-xs text-gray-500">
              <AlertCircle className="w-3 h-3" />
              <span>For demonstration purposes only</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-10 text-center text-sm">
          <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-6 text-gray-500">
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