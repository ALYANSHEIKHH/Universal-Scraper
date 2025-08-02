'use client'
import '../../src/app/globals.css'

import { useState } from 'react'
import Link from 'next/link'
import { Copy, RefreshCcw, CheckCircle2, BarChart3, Eye, Sparkles, Database } from 'lucide-react'
import { SummaryData } from '@/types'

// interface SummaryData {
//   [key: string]: number
// }

interface ResultSummaryProps {
  summary: SummaryData
}

export default function ResultSummary({ summary }: ResultSummaryProps) {
  const [copied, setCopied] = useState(false)

  if (!summary || Object.keys(summary).length === 0) return null

  const handleCopy = async () => {
    await navigator.clipboard.writeText(JSON.stringify(summary, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const totalImages = Object.values(summary).reduce((sum, count) => sum + count, 0)

  return (
    <div className="relative max-w-5xl mx-auto mt-16 px-6">
      {/* Background Effects */}
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="absolute top-1/2 left-1/2 w-[300px] h-[300px] bg-gradient-to-br from-purple-600/10 via-blue-600/5 to-cyan-600/10 rounded-full blur-3xl animate-pulse-slow" style={{transform: 'translate(-50%, -50%)'}} />
        <div className="absolute top-1/4 right-1/4 w-[200px] h-[200px] bg-gradient-to-br from-emerald-600/10 to-teal-600/5 rounded-full blur-2xl animate-pulse-slower" />
      </div>

      <div className="relative z-10 bg-gray-900/90 border border-gray-700/50 text-white rounded-3xl p-8 shadow-2xl backdrop-blur-sm">
        {/* Header Section */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-8">
          <div className="mb-6 lg:mb-0">
            <div className="flex items-center gap-3 mb-3">
              <div className="relative">
                <BarChart3 className="w-8 h-8 text-purple-400" />
                <div className="absolute inset-0 w-8 h-8 bg-purple-400/20 rounded-full blur-xl animate-pulse" />
              </div>
              <h2 className="text-3xl font-black bg-gradient-to-r from-purple-400 via-blue-400 to-emerald-400 bg-clip-text text-transparent">
                AI Classification Results
        </h2>
            </div>
            <p className="text-gray-300 text-lg font-medium">
              Successfully processed <span className="text-emerald-400 font-bold">{totalImages}</span> images with advanced AI
            </p>
          </div>

          {/* Action Tools */}
          <div className="flex items-center gap-3">
          <button
            onClick={handleCopy}
              className="group relative flex items-center gap-2 px-4 py-2.5 bg-gray-800/80 hover:bg-gray-700/80 text-white rounded-xl transition-all duration-200 shadow-lg backdrop-blur-sm border border-gray-600/50 hover:border-gray-500/50"
          >
              <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-gray-600/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
              <div className="relative flex items-center gap-2">
            {copied ? (
              <>
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    <span className="text-sm font-medium">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-4 h-4" />
                    <span className="text-sm font-medium">Copy JSON</span>
              </>
            )}
              </div>
          </button>
          <button
            onClick={() => window.location.reload()}
              className="group relative flex items-center gap-2 px-4 py-2.5 bg-gray-800/80 hover:bg-gray-700/80 text-white rounded-xl transition-all duration-200 shadow-lg backdrop-blur-sm border border-gray-600/50 hover:border-gray-500/50"
          >
              <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-gray-600/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
              <div className="relative flex items-center gap-2">
            <RefreshCcw className="w-4 h-4" />
                <span className="text-sm font-medium">Refresh</span>
              </div>
          </button>
        </div>
      </div>

        {/* Statistics Overview */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
          <div className="bg-gray-950/80 border border-gray-700/50 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <Database className="w-5 h-5 text-blue-400" />
              <span className="text-sm text-gray-400 font-medium">Total Categories</span>
            </div>
            <p className="text-2xl font-bold text-blue-400">{Object.keys(summary).length}</p>
          </div>
          <div className="bg-gray-950/80 border border-gray-700/50 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="w-5 h-5 text-emerald-400" />
              <span className="text-sm text-gray-400 font-medium">Total Images</span>
            </div>
            <p className="text-2xl font-bold text-emerald-400">{totalImages}</p>
          </div>
          <div className="bg-gray-950/80 border border-gray-700/50 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-5 h-5 text-purple-400" />
              <span className="text-sm text-gray-400 font-medium">Success Rate</span>
            </div>
            <p className="text-2xl font-bold text-purple-400">100%</p>
          </div>
        </div>

        {/* Classification Results */}
        <div className="space-y-4">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <Eye className="w-5 h-5 text-cyan-400" />
            Classification Breakdown
          </h3>
          
          {Object.entries(summary).map(([type, count], index) => (
            <div
            key={type}
              className="group relative bg-gray-950/80 border border-gray-700/50 rounded-xl p-5 hover:bg-gray-900/80 transition-all duration-300 shadow-lg backdrop-blur-sm hover:shadow-xl hover:scale-[1.02]"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-gray-600/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              
              <div className="relative flex flex-col lg:flex-row lg:items-center lg:justify-between">
                <div className="mb-4 lg:mb-0">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-3 h-3 bg-gradient-to-r from-purple-400 to-blue-400 rounded-full" />
                    <h4 className="text-lg font-bold capitalize text-white">
              {type}
                    </h4>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-2xl font-black text-emerald-400">
                      {count}
                    </span>
                    <span className="text-gray-400 font-medium">
                      image{count !== 1 ? 's' : ''} • {Math.round((count / totalImages) * 100)}% of total
              </span>
            </div>
                </div>
                
            <Link
              href={`/gallery/${encodeURIComponent(type)}`}
                  className="group/link relative inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold rounded-xl transition-all duration-200 shadow-lg hover:shadow-purple-500/25 transform hover:scale-105"
                >
                  <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-purple-500/20 to-transparent opacity-0 group-hover/link:opacity-100 transition-opacity duration-200" />
                  <div className="relative flex items-center gap-2">
                    <Eye className="w-4 h-4" />
                    <span>View Gallery</span>
                    <span className="text-sm opacity-80">→</span>
                  </div>
            </Link>
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="mt-8 pt-6 border-t border-gray-700/50">
          <div className="flex items-center justify-center gap-6 text-sm text-gray-500">
            <span className="flex items-center gap-1">
              <Sparkles className="w-3 h-3" />
              AI-Powered Classification
            </span>
            <span className="flex items-center gap-1">
              <Database className="w-3 h-3" />
              {totalImages} Images Processed
            </span>
            <span className="flex items-center gap-1">
              <BarChart3 className="w-3 h-3" />
              {Object.keys(summary).length} Categories
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
