'use client'

import { useState } from 'react'
import toast from 'react-hot-toast'
import {
  Loader2,
  UploadCloud,
  ImagePlus,
  FileArchive,
  Brain,
  BarChart3,
  Sparkles,
  Info,
  Zap,
  Shield,
  Cpu,
} from 'lucide-react'
import { SummaryData } from '@/types'

interface PredictionResult {
  prediction: string
  confidence: number
  probabilities: { [key: string]: number }
}

export default function HeroSection({
  onSummary,
}: {
  onSummary: (summary: SummaryData) => void
}) {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [showInfo, setShowInfo] = useState(false)

  // === Check for valid image URL ===
  const isImageUrl = (url: string) => /\.(jpg|jpeg|png|gif|bmp|webp|svg)(\?.*)?$/i.test(url)

  // === Scrape images from URL ===
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!url) return
    if (!isImageUrl(url)) {
      toast.error('Please provide a direct image URL (ending with .jpg, .png, etc.)')
      return
    }

    setLoading(true)
    toast.loading('Scraping images from URL...')

    try {
      const res = await fetch('http://localhost:8000/api/scrape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      })
      const data = await res.json()
      onSummary(data.summary)
      toast.success('Images scraped and classified with AI!')
    } catch {
      toast.error('Scrape failed. Check the URL or server.')
    } finally {
      toast.dismiss()
      setLoading(false)
    }
  }

  // === Upload multiple images ===
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    toast.loading('Uploading and classifying images with AI...')

    const formData = new FormData()
    Array.from(files).forEach(file => formData.append('files', file))

    try {
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      })

      const data = await res.json()
      onSummary(data.summary)
      toast.success('Images uploaded and classified with AI!')
    } catch {
      toast.error('Image upload failed.')
    } finally {
      toast.dismiss()
    }
  }

  // === Single Image AI Prediction ===
  const handleSinglePrediction = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    toast.loading('Analyzing image with AI...')
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        body: formData,
      })

      const data = await res.json()
      setPredictionResult(data)
      toast.success(
        `AI Prediction: ${data.prediction.toUpperCase()} (${Math.round(data.confidence * 100)}% confidence)`
      )
    } catch {
      toast.error('AI prediction failed.')
    } finally {
      toast.dismiss()
    }
  }

  // === Upload Kaggle zip dataset ===
  const handleKaggleZipUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !file.name.endsWith('.zip')) {
      toast.error('Please upload a valid .zip file.')
      return
    }

    toast.loading('Processing Kaggle dataset with AI...')
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('http://localhost:8000/api/upload-kaggle', {
        method: 'POST',
        body: formData,
      })

      const data = await res.json()
      onSummary(data.summary)
      toast.success('Kaggle dataset processed and classified with AI!')
    } catch {
      toast.error('Kaggle upload failed.')
    } finally {
      toast.dismiss()
    }
  }

  return (
    <section className="relative w-full min-h-screen flex items-center justify-center overflow-hidden bg-black text-white">
      {/* Enhanced Animated Background */}
      <div className="absolute inset-0 pointer-events-none z-0">
        {/* Main gradient orbs */}
        <div className="absolute top-[-20%] left-[-20%] w-[600px] h-[600px] bg-gradient-to-br from-purple-600/20 via-blue-600/15 to-cyan-600/10 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-[-20%] right-[-20%] w-[700px] h-[700px] bg-gradient-to-br from-emerald-600/15 via-teal-600/10 to-blue-600/15 rounded-full blur-3xl animate-pulse-slower" />
        <div className="absolute top-1/2 left-1/2 w-[300px] h-[300px] bg-gradient-to-br from-indigo-600/20 via-purple-600/15 to-pink-600/10 rounded-full blur-2xl animate-pulse" style={{transform: 'translate(-50%, -50%)'}} />
        
        {/* Floating geometric shapes */}
        <div className="absolute left-1/4 top-1/3 w-32 h-32 bg-gradient-to-br from-purple-500/30 to-blue-500/20 opacity-40 rounded-full blur-2xl animate-float" />
        <div className="absolute right-1/4 bottom-1/4 w-20 h-20 bg-gradient-to-br from-emerald-500/25 to-cyan-500/20 opacity-30 rounded-full blur-2xl animate-float-reverse" />
        <div className="absolute left-1/3 bottom-1/3 w-16 h-16 bg-gradient-to-br from-pink-500/30 to-purple-500/20 opacity-35 rounded-full blur-xl animate-float" style={{animationDelay: '2s'}} />
        
        {/* Grid pattern overlay */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:50px_50px] opacity-30" />
        
        {/* Subtle noise texture */}
        <div className="absolute inset-0 bg-noise opacity-5" />
      </div>

      {/* Content Container */}
      <div className="relative z-10 max-w-4xl w-full px-6 py-20 text-center flex flex-col items-center">
        {/* Header Section */}
        <div className="mb-12">
          <div className="flex items-center justify-center gap-3 mb-6">
            <div className="relative">
              <Sparkles className="w-10 h-10 text-purple-400 animate-bounce" />
              <div className="absolute inset-0 w-10 h-10 bg-purple-400/20 rounded-full blur-xl animate-pulse" />
            </div>
            <h1 className="text-6xl sm:text-7xl font-black bg-gradient-to-r from-purple-400 via-blue-400 to-emerald-400 bg-clip-text text-transparent drop-shadow-2xl tracking-tight leading-tight">
              Universal AI
        </h1>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent mb-4 tracking-tight">
            Image Classifier
          </h2>
          <div className="flex items-center justify-center gap-2 mb-6">
            <Cpu className="w-5 h-5 text-emerald-400" />
            <span className="text-emerald-400 font-semibold text-lg">Powered by Advanced AI</span>
            <Shield className="w-5 h-5 text-blue-400" />
          </div>
        </div>

        {/* Description */}
        <p className="text-gray-300 text-xl sm:text-2xl mb-12 max-w-3xl mx-auto font-medium leading-relaxed">
          State-of-the-art AI-powered image classification for any type of image. 
          <span className="text-purple-300 font-semibold block mt-2">Let artificial intelligence do the heavy lifting!</span>
        </p>

        {/* Feature Highlights
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-12 w-full max-w-2xl">
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 backdrop-blur-sm">
            <Zap className="w-8 h-8 text-yellow-400 mx-auto mb-2" />
            <h3 className="text-white font-semibold mb-1">Lightning Fast</h3>
            <p className="text-gray-400 text-sm">Instant AI classification</p>
          </div>
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 backdrop-blur-sm">
            <Database className="w-8 h-8 text-blue-400 mx-auto mb-2" />
            <h3 className="text-white font-semibold mb-1">Universal</h3>
            <p className="text-gray-400 text-sm">Any image type supported</p>
          </div>
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 backdrop-blur-sm">
            <Shield className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
            <h3 className="text-white font-semibold mb-1">Secure</h3>
            <p className="text-gray-400 text-sm">Local processing only</p>
          </div>
        </div> */}

        {/* Info Toggle */}
        <div className="flex items-center justify-center gap-2 mb-8">
          <button
            onClick={() => setShowInfo(!showInfo)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-900/80 hover:bg-gray-800/80 text-sm text-gray-200 border border-gray-700 transition-all duration-200 shadow-lg backdrop-blur-sm hover:border-gray-600"
            aria-label="Show info about this app"
          >
            <Info className="w-4 h-4" />
            {showInfo ? 'Hide Features' : 'What can I do?'}
          </button>
        </div>

        {/* Info Panel */}
        {showInfo && (
          <div className="mb-10 bg-gray-900/90 border border-gray-700 rounded-2xl p-6 text-left text-gray-200 shadow-2xl max-w-2xl mx-auto backdrop-blur-sm animate-fade-in">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-purple-400" />
              Powerful Features
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <div className="w-1.5 h-1.5 bg-purple-400 rounded-full mt-2 flex-shrink-0" />
                  <span>Paste a dataset URL to scrape and classify images automatically</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0" />
                  <span>Upload one or more images for instant AI classification</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full mt-2 flex-shrink-0" />
                  <span>Upload a ZIP file (e.g., Kaggle dataset) for batch analysis</span>
                </li>
              </ul>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0" />
                  <span>Classify a single image and see the AI&rsquo;s confidence breakdown</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="w-1.5 h-1.5 bg-yellow-400 rounded-full mt-2 flex-shrink-0" />
                  <span>All processing is local and secure—your images are never stored permanently</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="w-1.5 h-1.5 bg-pink-400 rounded-full mt-2 flex-shrink-0" />
                  <span>Advanced AI models provide high-accuracy classification</span>
                </li>
              </ul>
            </div>
          </div>
        )}

        {/* URL Scraping Form */}
        <form
          onSubmit={handleSubmit}
          className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-10 w-full max-w-3xl"
        >
          <div className="relative w-full sm:w-2/3">
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
              placeholder="Paste your image dataset URL here..."
              className="w-full px-6 py-4 rounded-xl bg-gray-900/80 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200 shadow-xl backdrop-blur-sm text-lg"
            required
              aria-label="Dataset URL"
          />
            <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-purple-500/20 to-blue-500/20 opacity-0 hover:opacity-100 transition-opacity duration-200 pointer-events-none" />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center justify-center px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-bold rounded-xl shadow-2xl transition-all duration-200 disabled:opacity-50 text-lg focus:outline-none focus:ring-2 focus:ring-purple-400 transform hover:scale-105"
            aria-label="Scrape images from URL"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin mr-2" />
                Processing...
              </>
            ) : (
              <>
                <UploadCloud className="w-5 h-5 mr-2" />
                Scrape Images
              </>
            )}
          </button>
        </form>

        {/* Action Buttons Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-12 w-full max-w-4xl">
          {/* Single Image Prediction */}
          <label htmlFor="single-predict" className="group cursor-pointer">
            <div className="relative p-4 rounded-xl border border-emerald-600/50 bg-emerald-600/10 hover:bg-emerald-600/20 text-emerald-400 font-semibold transition-all duration-200 shadow-lg hover:shadow-emerald-500/25 hover:scale-105 backdrop-blur-sm">
              <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-emerald-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
              <div className="relative flex items-center justify-center gap-2">
                <Brain className="w-5 h-5 group-hover:animate-pulse" />
                <span className="text-sm">Classify Single Image</span>
              </div>
            </div>
          </label>
          <input
            id="single-predict"
            type="file"
            accept="image/*"
            onChange={handleSinglePrediction}
            className="hidden"
            aria-label="Upload single image for prediction"
          />

          {/* Upload Images */}
          <label htmlFor="upload-images" className="group cursor-pointer">
            <div className="relative p-4 rounded-xl border border-gray-600/50 bg-gray-900/50 hover:bg-gray-800/50 text-gray-300 font-semibold transition-all duration-200 shadow-lg hover:shadow-gray-500/25 hover:scale-105 backdrop-blur-sm">
              <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-gray-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
              <div className="relative flex items-center justify-center gap-2">
                <ImagePlus className="w-5 h-5" />
                <span className="text-sm">Upload Images</span>
              </div>
            </div>
          </label>
          <input
            id="upload-images"
            type="file"
            multiple
            accept="image/*"
            onChange={handleUpload}
            className="hidden"
            aria-label="Upload multiple images"
          />

          {/* Upload Kaggle Zip */}
          <label htmlFor="upload-kaggle" className="group cursor-pointer">
            <div className="relative p-4 rounded-xl border border-gray-600/50 bg-gray-900/50 hover:bg-gray-800/50 text-gray-300 font-semibold transition-all duration-200 shadow-lg hover:shadow-gray-500/25 hover:scale-105 backdrop-blur-sm">
              <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-gray-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
              <div className="relative flex items-center justify-center gap-2">
                <FileArchive className="w-5 h-5" />
                <span className="text-sm">Upload Kaggle ZIP</span>
              </div>
            </div>
          </label>
          <input
            id="upload-kaggle"
            type="file"
            accept=".zip"
            onChange={handleKaggleZipUpload}
            className="hidden"
            aria-label="Upload Kaggle ZIP dataset"
          />

          {/* Model Info Button */}
          <button
            onClick={async () => {
              try {
  const res = await fetch('http://localhost:8000/api/model-info')
  const data = await res.json()
  toast.success(`Model: ${data.model_type} | Device: ${data.device}`)
} catch {
  toast.error('Failed to fetch model info')
}

            }}
            className="group relative p-4 rounded-xl border border-blue-600/50 bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 font-semibold transition-all duration-200 shadow-lg hover:shadow-blue-500/25 hover:scale-105 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
            aria-label="Show model info"
          >
            <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-blue-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
            <div className="relative flex items-center justify-center gap-2">
              <BarChart3 className="w-5 h-5" />
              <span className="text-sm">Model Info</span>
            </div>
          </button>
        </div>

        {/* AI Prediction Results */}
        {predictionResult && (
          <div className="bg-gray-900/90 border border-gray-700 rounded-2xl p-8 mb-10 text-left shadow-2xl animate-fade-in backdrop-blur-sm max-w-4xl w-full">
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
              <Brain className="w-6 h-6 text-emerald-400 animate-pulse" />
              AI Prediction Results
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Main Prediction */}
              <div className="bg-gray-950/80 rounded-xl p-6 border border-gray-800">
                <p className="text-sm text-gray-400 mb-2 font-medium">Predicted Class</p>
                <p className="text-2xl font-bold text-emerald-400 capitalize mb-2">
                  {predictionResult.prediction}
                </p>
                <div className="flex items-center gap-2">
                  <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all duration-500"
                      style={{ width: `${predictionResult.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-300 font-semibold min-w-[60px]">
                    {Math.round(predictionResult.confidence * 100)}%
                  </span>
                </div>
              </div>
              {/* Probability Breakdown */}
              <div className="bg-gray-950/80 rounded-xl p-6 border border-gray-800">
                <p className="text-sm text-gray-400 mb-4 font-medium">Probability Distribution</p>
                <div className="space-y-3">
                {Object.entries(predictionResult.probabilities).map(([type, prob]) => (
                    <div key={type} className="space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-300 capitalize font-medium">{type}</span>
                        <span className="text-xs text-gray-400 font-mono">
                          {Math.round(prob * 100)}%
                        </span>
                      </div>
                      <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-purple-500 to-blue-500 transition-all duration-500"
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                  </div>
              </div>
            </div>
            <button
              onClick={() => setPredictionResult(null)}
              className="mt-6 text-sm text-gray-400 hover:text-gray-300 transition-colors duration-200 flex items-center gap-2"
            >
              <span className="w-4 h-4 rounded-full border border-gray-600 flex items-center justify-center">✕</span>
              Clear Results
            </button>
          </div>
        )}

        {/* Footer */}
        <div className="text-center">
          <p className="text-sm text-gray-500 italic mb-4">
          🔒 Your images are processed securely with state-of-the-art AI models and never stored permanently.
        </p>
          <div className="flex items-center justify-center gap-4 text-xs text-gray-600">
            <span className="flex items-center gap-1">
              <Cpu className="w-3 h-3" />
              Local AI Processing
            </span>
            <span className="flex items-center gap-1">
              <Shield className="w-3 h-3" />
              Privacy First
            </span>
            <span className="flex items-center gap-1">
              <Zap className="w-3 h-3" />
              Lightning Fast
            </span>
          </div>
        </div>
      </div>
    </section>
  )
}