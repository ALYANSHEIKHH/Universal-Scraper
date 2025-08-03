'use client'

import { createContext, useContext, useState, useEffect, ReactNode, useCallback, useRef } from 'react'
import { useRouter } from 'next/navigation'

interface User {
  id: string
  email: string
  name: string
  created_at?: string
  last_login?: string
}

interface UserStats {
  total_logins: number
  last_login: string
  created_at: string
}

interface AuthContextType {
  user: User | null
  userStats: UserStats | null
  login: (email: string, password: string) => Promise<{ success: boolean; message: string }>
  register: (name: string, email: string, password: string) => Promise<{ success: boolean; message: string }>
  logout: () => void
  loading: boolean
  updateProfile: (name: string) => Promise<{ success: boolean; message: string }>
  getUserStats: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

// ✅ FIXED: Use your actual backend URL consistently
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://alyan1-my-fastapi-backend.hf.space'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [userStats, setUserStats] = useState<UserStats | null>(null)
  const [loading, setLoading] = useState(true)
  const router = useRouter()
  const isMountedRef = useRef(true)

  // Cleanup function to prevent memory leaks
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  const getUserStats = useCallback(async (currentUser?: User) => {
    const token = localStorage.getItem('authToken')
    const userToCheck = currentUser || user
    
    if (!userToCheck || !token) return
    
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout
      
      // ✅ FIXED: Use correct backend URL and session-based auth
      const response = await fetch(`${API_BASE_URL}/api/auth/stats`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include', // ✅ FIXED: Include session cookies
        signal: controller.signal
      })
      
      clearTimeout(timeoutId)
      
      if (response.ok && isMountedRef.current) {
        const data = await response.json()
        setUserStats(data.stats)
      } else if (!response.ok) {
        console.warn('Failed to fetch user stats:', response.status, response.statusText)
      }
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.warn('User stats request timed out')
        } else {
          console.warn('Backend server not available, skipping user stats:', error.message)
        }
      }
      // Don't throw the error, just log it and continue
    }
  }, [user])

  useEffect(() => {
    const initializeAuth = () => {
      try {
        const token = localStorage.getItem('authToken')
        const userData = localStorage.getItem('userData')

        if (token && userData) {
          const parsedUser = JSON.parse(userData)
          setUser(parsedUser)
          // Pass the user data directly to avoid race condition
          getUserStats(parsedUser)
        }
      } catch (error) {
        console.error('Error parsing user data:', error)
        // Clean up invalid data
        localStorage.removeItem('authToken')
        localStorage.removeItem('userData')
      } finally {
        setLoading(false)
      }
    }

    initializeAuth()
  }, []) // Remove getUserStats from dependency array

  const login = async (email: string, password: string): Promise<{ success: boolean; message: string }> => {
    try {
      setLoading(true)
      
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 10000) // 10 second timeout for login
      
      // ✅ FIXED: Use correct backend URL and session-based auth
      const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include', // ✅ FIXED: Include session cookies
        body: JSON.stringify({ email, password }),
        signal: controller.signal
      })
      
      clearTimeout(timeoutId)
      const data = await response.json()

      if (response.ok) {
        const userData = {
          id: data.user.id,
          email: data.user.email,
          name: data.user.name,
          created_at: data.user.created_at,
          last_login: data.user.last_login,
        }

        setUser(userData)
        localStorage.setItem('authToken', data.token)
        localStorage.setItem('userData', JSON.stringify(userData))

        // Pass userData directly to avoid race condition
        await getUserStats(userData)
        router.push("/")

        return { success: true, message: 'Login successful! Welcome back.' }
      } else {
        return { success: false, message: data.detail || 'Login failed. Please check your credentials.' }
      }
    } catch (error) {
      console.error('Login error:', error)
      let errorMessage = 'Network error. Please check your connection.'
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Login request timed out. Please check your connection and try again.'
        } else if (error.message.includes('Failed to fetch')) {
          errorMessage = 'Cannot connect to server. Please check if the backend is running.'
        }
      }
      return { success: false, message: errorMessage }
    } finally {
      setLoading(false)
    }
  }

  const register = async (name: string, email: string, password: string): Promise<{ success: boolean; message: string }> => {
    try {
      setLoading(true)
      
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 10000) // 10 second timeout for register
      
      // ✅ FIXED: Use correct backend URL and session-based auth
      const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include', // ✅ FIXED: Include session cookies
        body: JSON.stringify({ name, email, password }),
        signal: controller.signal
      })
      
      clearTimeout(timeoutId)
      const data = await response.json()

      if (response.ok) {
        const userData = {
          id: data.user.id,
          email: data.user.email,
          name: data.user.name,
          created_at: data.user.created_at,
        }

        setUser(userData)
        localStorage.setItem('authToken', data.token)
        localStorage.setItem('userData', JSON.stringify(userData))
        
        // Pass userData directly to avoid race condition
        await getUserStats(userData)

        return { success: true, message: 'Registration successful! Welcome to Universal AI.' }
      } else {
        return { success: false, message: data.detail || 'Registration failed. Please try again.' }
      }
    } catch (error) {
      console.error('Registration error:', error)
      let errorMessage = 'Network error. Please check your connection.'
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Registration request timed out. Please check your connection and try again.'
        } else if (error.message.includes('Failed to fetch')) {
          errorMessage = 'Cannot connect to server. Please check if the backend is running.'
        }
      }
      return { success: false, message: errorMessage }
    } finally {
      setLoading(false)
    }
  }

  const updateProfile = async (name: string): Promise<{ success: boolean; message: string }> => {
    try {
      const token = localStorage.getItem('authToken')
      if (!token) {
        return { success: false, message: 'Authentication required.' }
      }

      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 8000) // 8 second timeout for profile update

      // ✅ FIXED: Use correct backend URL and session-based auth
      const response = await fetch(`${API_BASE_URL}/api/auth/profile`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include', // ✅ FIXED: Include session cookies
        body: JSON.stringify({ name }),
        signal: controller.signal
      })

      clearTimeout(timeoutId)
      const data = await response.json()
      
      if (response.ok && user) {
        const updatedUser = { ...user, name }
        setUser(updatedUser)
        localStorage.setItem('userData', JSON.stringify(updatedUser))
        return { success: true, message: 'Profile updated successfully!' }
      } else {
        return { success: false, message: data.detail || 'Failed to update profile.' }
      }
    } catch (error) {
      console.error('Profile update error:', error)
      let errorMessage = 'Network error. Please try again.'
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Profile update timed out. Please try again.'
        } else if (error.message.includes('Failed to fetch')) {
          errorMessage = 'Cannot connect to server. Please check if the backend is running.'
        }
      }
      return { success: false, message: errorMessage }
    }
  }

  const logout = async () => {
    try {
      const token = localStorage.getItem('authToken')
      if (token) {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 3000) // 3 second timeout for logout
        
        // ✅ FIXED: Use correct backend URL and session-based auth
        await fetch(`${API_BASE_URL}/api/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          credentials: 'include', // ✅ FIXED: Include session cookies
          signal: controller.signal
        })
        
        clearTimeout(timeoutId)
      }
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.warn('Logout request timed out - proceeding with local logout')
        } else {
          console.warn('Backend logout failed - proceeding with local logout:', error.message)
        }
      }
      // Don't throw the error, continue with local logout
    } finally {
      // Always perform local logout regardless of backend response
      setUser(null)
      setUserStats(null)
      localStorage.removeItem('authToken')
      localStorage.removeItem('userData')
      router.push('/auth')
    }
  }

  return (
    <AuthContext.Provider value={{
      user,
      userStats,
      login,
      register,
      logout,
      loading,
      updateProfile,
      getUserStats,
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}