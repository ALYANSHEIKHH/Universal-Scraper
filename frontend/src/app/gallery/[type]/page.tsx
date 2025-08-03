'use client'

import { useParams, useRouter } from 'next/navigation'
import Image from 'next/image'
import { useEffect, useState } from 'react'
import { PhotoProvider, PhotoView } from 'react-photo-view'
import 'react-photo-view/dist/react-photo-view.css'
import {
  ArrowLeft,
  // Image as ImageIcon, 
  // Loader2, 
  Sparkles,
  Eye,
  Grid3X3,
  Calendar,
  Database,
  LogOut,
  User,
  Filter,
  Search,
  Download,
  Share2,
  Heart,
  Star,
  Zap,
  Shield,
  Globe,
  Clock,
  TrendingUp,
  BarChart3,
  // Settings,
  RefreshCw,
  Maximize2,
  // Info,
  // ChevronLeft,
  // ChevronRight
} from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'
import ProtectedRoute from '@/components/ProtectedRoute'

// Reusable function to download an image
async function handleDownload(url: string, filename?: string) {
  try {
    // First try with fetch and proper CORS handling
    const response = await fetch(url, {
      mode: 'cors',
      headers: {
        'Origin': window.location.origin
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const blob = await response.blob();
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename || url.split('/').pop() || 'image.jpg';
    document.body.appendChild(link);
    link.click();

    setTimeout(() => {
      URL.revokeObjectURL(link.href);
      link.remove();
    }, 100);

  } catch (err) {
    console.error('Primary download method failed:', err);

    // Fallback: Try direct link download (may not work with CORS but worth trying)
    try {
      const link = document.createElement('a');
      link.href = url;
      link.download = filename || url.split('/').pop() || 'image.jpg';
      link.target = '_blank';
      link.rel = 'noopener';
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (fallbackErr) {
      console.error('Fallback download failed:', fallbackErr);

      // Final fallback: Open in new tab
      const newWindow = window.open(url, '_blank');
      if (!newWindow) {
        alert('Download failed due to CORS restrictions. Please right-click the image and select "Save image as..." to download manually.');
      } else {
        alert('Download initiated. If the image opens in a new tab, right-click and select "Save image as..." to download.');
      }
    }
  }
}

// Reusable function to share an image using the Web Share API
async function handleShare(url: string, filename?: string) {
  if (navigator.share) {
    try {
      await navigator.share({
        title: filename || 'AI Gallery Image',
        text: 'Check out this AI-classified image!',
        url,
      });
    } catch (err) {
      // User cancelled or error - silently handle
      console.log('Share cancelled or failed:', err);
    }
  } else {
    // fallback: copy link
    try {
      await navigator.clipboard.writeText(url);
      alert('Link copied to clipboard!');
    } catch (err) {
      console.error('Clipboard access failed:', err);
      // Fallback for clipboard failure
      const textarea = document.createElement('textarea');
      textarea.value = url;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      alert('Link copied to clipboard!');
    }
  }
}

// Function to download all images
async function handleBulkDownload(images: Array<{ url: string; filename: string }>, type: string) {
  if (images.length === 0) {
    alert('No images to download.');
    return;
  }

  const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  // Show progress to user
  const totalImages = images.length;
  let downloadedCount = 0;

  for (let i = 0; i < images.length; i++) {
    const img = images[i];
    const filename = img.filename || `${type}_image_${i + 1}.jpg`;

    try {
      await handleDownload(img.url, filename);
      downloadedCount++;

      // Update user on progress (you could replace this with a toast notification)
      if (i === images.length - 1) {
        console.log(`Download complete: ${downloadedCount}/${totalImages} images`);
      }

      // Add delay between downloads to prevent overwhelming the browser
      if (i < images.length - 1) {
        await delay(1000); // Increased delay to prevent CORS issues
      }
    } catch (error) {
      console.error(`Failed to download image ${i + 1}:`, error);
    }
  }

  alert(`Bulk download initiated for ${totalImages} images. Check your downloads folder.`);
}

// Function to share gallery
async function handleBulkShare(images: Array<{ url: string; filename: string }>, type: string) {
  const galleryText = `Check out this AI-classified ${type} gallery with ${images.length} images!`;
  const firstImageUrl = images[0]?.url;

  if (navigator.share && firstImageUrl) {
    try {
      await navigator.share({
        title: `${type.charAt(0).toUpperCase() + type.slice(1)} Gallery`,
        text: galleryText,
        url: firstImageUrl,
      });
    } catch (err) {
      console.log('Share cancelled or failed:', err);
    }
  } else {
    // Fallback: copy gallery info to clipboard
    const galleryInfo = `${galleryText}\n\nImages:\n${images.map((img, i) => `${i + 1}. ${img.url}`).join('\n')}`;
    try {
      await navigator.clipboard.writeText(galleryInfo);
      alert('Gallery information copied to clipboard!');
    } catch (err) {
      console.error('Clipboard access failed:', err);
      alert('Unable to share gallery. Please try individual image sharing.');
    }
  }
}

export default function GalleryPage() {
  const params = useParams();
  const type = typeof params?.type === 'string' ? params.type : '';

  const router = useRouter()
  const { user, logout } = useAuth()

  const [images, setImages] = useState<{ url: string; filename: string; analysis?: string | null }[]>([])
  const [loading, setLoading] = useState(true)
  // const [selectedImage, setSelectedImage] = useState<number | null>(null)
  const [viewMode, setViewMode] = useState<'grid' | 'masonry' | 'carousel'>('grid')
  const [searchTerm, setSearchTerm] = useState('')
  const [sortBy, setSortBy] = useState<'date' | 'name' | 'size'>('date')
  const [showFilters, setShowFilters] = useState(false)
  const [favorites, setFavorites] = useState<Set<number>>(new Set())

  // ✅ BEST PRACTICE: Define outside useEffect to avoid recreation
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'https://alyan1-my-fastapi-backend.hf.space';

  useEffect(() => {
    if (!type || !type.trim()) return
    setLoading(true)

    fetch(`${backendUrl}/api/images/${type}`, {
      credentials: 'include', // ✅ FIXED: Include session cookies for authentication
    })
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then(data => setImages(data.images || []))
      .catch(err => {
        console.error('Image fetch error:', err);
        setImages([]);
      })
      .finally(() => setLoading(false))
  }, [type]) // Don't forget to add the dependency array!
  const handleLogout = () => {
    logout()
  }

  const toggleFavorite = (index: number) => {
    const newFavorites = new Set(favorites)
    if (newFavorites.has(index)) {
      newFavorites.delete(index)
    } else {
      newFavorites.add(index)
    }
    setFavorites(newFavorites)
  }

  const filteredImages = images.filter((img, index) =>
    searchTerm === '' || `Image ${index + 1}`.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-black text-white relative overflow-hidden">
        {/* Enhanced Background Effects */}
        <div className="absolute inset-0 pointer-events-none z-0">
          {/* Multiple gradient orbs with different animations */}
          <div className="absolute top-[-20%] left-[-20%] w-[800px] h-[800px] bg-gradient-to-br from-purple-600/15 via-blue-600/10 to-cyan-600/8 rounded-full blur-3xl animate-pulse-slow" />
          <div className="absolute bottom-[-20%] right-[-20%] w-[900px] h-[900px] bg-gradient-to-br from-emerald-600/12 via-teal-600/8 to-blue-600/10 rounded-full blur-3xl animate-pulse-slower" />
          <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] bg-gradient-to-br from-indigo-600/18 via-purple-600/12 to-pink-600/10 rounded-full blur-2xl animate-pulse" style={{ transform: 'translate(-50%, -50%)' }} />

          {/* Floating geometric shapes with enhanced animations */}
          <div className="absolute left-1/4 top-1/3 w-40 h-40 bg-gradient-to-br from-purple-500/25 to-blue-500/15 opacity-50 rounded-full blur-2xl animate-float" />
          <div className="absolute right-1/4 bottom-1/4 w-24 h-24 bg-gradient-to-br from-emerald-500/20 to-cyan-500/15 opacity-40 rounded-full blur-2xl animate-float-reverse" />
          <div className="absolute left-1/3 bottom-1/3 w-20 h-20 bg-gradient-to-br from-pink-500/30 to-purple-500/20 opacity-45 rounded-full blur-xl animate-float" style={{ animationDelay: '2s' }} />
          <div className="absolute right-1/3 top-1/4 w-16 h-16 bg-gradient-to-br from-yellow-500/25 to-orange-500/20 opacity-35 rounded-full blur-xl animate-float-reverse" style={{ animationDelay: '3s' }} />

          {/* Enhanced grid pattern overlay */}
          <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:60px_60px] opacity-40" />

          {/* Animated particles */}
          <div className="absolute inset-0">
            {[...Array(20)].map((_, i) => (
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

        <div className="relative z-10 px-6 py-16">
          <div className="max-w-7xl mx-auto">
            {/* Enhanced Navigation Header */}
            <div className="flex items-center justify-between mb-12">
              <button
                onClick={() => router.back()}
                className="group relative flex items-center gap-3 px-6 py-3 bg-gray-900/80 hover:bg-gray-800/80 text-white rounded-2xl transition-all duration-300 shadow-xl backdrop-blur-sm border border-gray-700/50 hover:border-gray-600/50 hover:scale-105"
              >
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-gray-600/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                <div className="relative flex items-center gap-3">
                  <ArrowLeft className="w-5 h-5 group-hover:animate-bounce" />
                  <span className="text-sm font-semibold">Back to Results</span>
                </div>
              </button>

              <div className="flex items-center gap-6">
                {/* Enhanced Stats Display */}
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2 px-4 py-2 bg-gray-900/60 backdrop-blur-sm border border-gray-700/40 rounded-xl">
                    <Database className="w-4 h-4 text-blue-400" />
                    <span className="text-sm text-gray-300 font-medium">{images.length} images</span>
                  </div>
                  <div className="flex items-center gap-2 px-4 py-2 bg-gray-900/60 backdrop-blur-sm border border-gray-700/40 rounded-xl">
                    <Calendar className="w-4 h-4 text-emerald-400" />
                    <span className="text-sm text-gray-300 font-medium">{new Date().toLocaleDateString()}</span>
                  </div>
                  <div className="flex items-center gap-2 px-4 py-2 bg-gray-900/60 backdrop-blur-sm border border-gray-700/40 rounded-xl">
                    <TrendingUp className="w-4 h-4 text-purple-400" />
                    <span className="text-sm text-gray-300 font-medium">{favorites.size} favorites</span>
                  </div>
                </div>

                {/* Enhanced User Profile */}
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2 px-4 py-2 bg-gray-900/80 backdrop-blur-sm border border-gray-700/50 rounded-xl">
                    <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center">
                      <User className="w-4 h-4 text-white" />
                    </div>
                    <span className="text-sm text-gray-300 font-medium">
                      {user?.name}
                    </span>
                  </div>

                  <button
                    onClick={handleLogout}
                    className="flex items-center gap-2 px-4 py-2 bg-gray-900/80 hover:bg-red-900/80 backdrop-blur-sm border border-gray-700/50 hover:border-red-600/50 rounded-xl transition-all duration-300 text-gray-300 hover:text-white hover:scale-105"
                  >
                    <LogOut className="w-4 h-4" />
                    <span className="text-sm font-medium">Logout</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Enhanced Header with Advanced Features */}
            <header className="text-center mb-16">
              <div className="flex items-center justify-center gap-4 mb-8">
                <div className="relative">
                  <Sparkles className="w-12 h-12 text-purple-400 animate-bounce" />
                  <div className="absolute inset-0 w-12 h-12 bg-purple-400/20 rounded-full blur-xl animate-pulse" />
                </div>
                <h1 className="text-7xl sm:text-8xl font-black bg-gradient-to-r from-purple-400 via-blue-400 to-emerald-400 bg-clip-text text-transparent drop-shadow-2xl tracking-tight leading-tight">
                  {type ? type.charAt(0).toUpperCase() + type.slice(1) : type} Gallery
                </h1>
              </div>

              <p className="text-2xl text-gray-300 font-medium max-w-3xl mx-auto leading-relaxed mb-8">
                Explore the AI-classified images in the <span className="text-purple-300 font-semibold">{type}</span> category.
                Each image has been processed with advanced machine learning algorithms for precise classification.
              </p>

              {/* Enhanced Gallery Controls */}
              <div className="flex items-center justify-center gap-6 mb-8">
                {/* View Mode Toggle */}
                <div className="flex items-center gap-2 bg-gray-900/60 backdrop-blur-sm border border-gray-700/40 rounded-xl p-1">
                  {(['grid', 'masonry', 'carousel'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setViewMode(mode)}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${viewMode === mode
                          ? 'bg-purple-600 text-white shadow-lg'
                          : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                        }`}
                    >
                      {mode === 'grid' && <Grid3X3 className="w-4 h-4 inline mr-2" />}
                      {mode === 'masonry' && <BarChart3 className="w-4 h-4 inline mr-2" />}
                      {mode === 'carousel' && <Maximize2 className="w-4 h-4 inline mr-2" />}
                      {mode.charAt(0).toUpperCase() + mode.slice(1)}
                    </button>
                  ))}
                </div>

                {/* Search Bar */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search images..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-gray-900/60 backdrop-blur-sm border border-gray-700/40 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200 w-64"
                  />
                </div>

                {/* Filter Toggle */}
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-900/60 backdrop-blur-sm border border-gray-700/40 rounded-xl text-gray-300 hover:text-white transition-all duration-200"
                >
                  <Filter className="w-4 h-4" />
                  <span className="text-sm font-medium">Filters</span>
                </button>
              </div>

              {/* Advanced Filters Panel */}
              {showFilters && (
                <div className="bg-gray-900/80 backdrop-blur-sm border border-gray-700/50 rounded-2xl p-6 mb-8 animate-fade-in">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                      <label className="text-sm text-gray-400 font-medium mb-2 block">Sort By</label>
                      <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value as 'date' | 'name' | 'size')}
                        className="w-full px-3 py-2 bg-gray-800/60 border border-gray-600/40 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                      >
                        <option value="date">Date Added</option>
                        <option value="name">Name</option>
                        <option value="size">File Size</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400 font-medium mb-2 block">View Options</label>
                      <div className="flex items-center gap-4">
                        <label className="flex items-center gap-2 text-sm text-gray-300">
                          <input type="checkbox" className="rounded border-gray-600 bg-gray-800" />
                          Show Favorites Only
                        </label>
                      </div>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400 font-medium mb-2 block">Actions</label>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleBulkDownload(filteredImages, type)}
                          className="flex items-center gap-1 px-3 py-1 bg-blue-600/60 hover:bg-blue-600/80 rounded-lg text-sm transition-all duration-200"
                        >
                          <Download className="w-3 h-3" />
                          Export All
                        </button>
                        <button
                          onClick={() => handleBulkShare(filteredImages, type)}
                          className="flex items-center gap-1 px-3 py-1 bg-green-600/60 hover:bg-green-600/80 rounded-lg text-sm transition-all duration-200"
                        >
                          <Share2 className="w-3 h-3" />
                          Share Gallery
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </header>

            {/* Enhanced Loading State */}
            {loading ? (
              <div className="flex flex-col items-center justify-center mt-24 space-y-8">
                <div className="relative">
                  <div className="h-24 w-24 border-4 border-purple-500/30 border-t-purple-500 animate-spin rounded-full" />
                  <div className="absolute inset-0 h-24 w-24 border-4 border-blue-500/20 border-t-blue-500 animate-spin rounded-full" style={{ animationDelay: '0.5s' }} />
                  <div className="absolute inset-0 h-24 w-24 border-4 border-emerald-500/15 border-t-emerald-500 animate-spin rounded-full" style={{ animationDelay: '1s' }} />
                </div>
                <div className="text-center">
                  <p className="text-purple-400 text-2xl font-semibold animate-pulse mb-3">
                    Loading {type} images...
                  </p>
                  <p className="text-gray-500 text-lg">Processing AI-classified content with advanced algorithms</p>
                  <div className="flex items-center justify-center gap-2 mt-4">
                    <Zap className="w-4 h-4 text-yellow-400 animate-pulse" />
                    <span className="text-sm text-gray-400">Optimizing for performance</span>
                  </div>
                </div>
              </div>
            ) : images.length === 0 ? (
              /* Enhanced Empty State */
              <div className="text-center mt-24">
                <div className="relative mx-auto w-40 h-40 mb-8">
                  <div className="absolute inset-0 bg-gradient-to-br from-gray-800 to-gray-900 rounded-full blur-2xl opacity-60" />
                  <div className="relative w-full h-full bg-gray-900/80 border border-gray-700/50 rounded-full flex items-center justify-center text-8xl backdrop-blur-sm">
                    🖼️
                  </div>
                </div>
                <h3 className="text-4xl font-bold text-white mb-6">No Images Found</h3>
                <p className="text-gray-400 text-xl mb-10 max-w-2xl mx-auto">
                  No images were found in the <span className="text-purple-300 font-semibold">{type}</span> category.
                  Try a different category or check back later for new AI-classified content.
                </p>
                <div className="flex items-center justify-center gap-4">
                  <button
                    onClick={() => router.push('/')}
                    className="inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold rounded-2xl transition-all duration-300 shadow-xl hover:shadow-purple-500/25 transform hover:scale-105"
                  >
                    <Sparkles className="w-5 h-5" />
                    Back to Classifier
                  </button>
                  <button
                    onClick={() => window.location.reload()}
                    className="inline-flex items-center gap-3 px-8 py-4 bg-gray-800/80 hover:bg-gray-700/80 text-white font-semibold rounded-2xl transition-all duration-300 border border-gray-600/50 hover:border-gray-500/50"
                  >
                    <RefreshCw className="w-5 h-5" />
                    Refresh
                  </button>
                </div>
              </div>
            ) : (
              /* Enhanced Image Gallery */
              <PhotoProvider>
                <section className={`grid gap-8 ${viewMode === 'grid'
                    ? 'grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5'
                    : viewMode === 'masonry'
                      ? 'columns-1 sm:columns-2 md:columns-3 xl:columns-4 2xl:columns-5'
                      : 'grid-cols-1'
                  }`}>
                  {filteredImages.map((img, index) => (
                    <PhotoView key={index} src={img.url}>
                      <div className={`group relative bg-gray-900/80 backdrop-blur-sm border border-gray-700/50 rounded-3xl overflow-hidden shadow-2xl hover:shadow-3xl transition-all duration-500 hover:scale-105 ${viewMode === 'masonry' ? 'break-inside-avoid mb-8' : ''
                        }`}>
                        {/* Enhanced Image Display with Error Handling */}
                        {/* const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'https://alyan1-my-fastapi-backend.hf.space'; */}

                        <div className="relative w-full h-64 aspect-square overflow-hidden">
                          <Image
                            src={`${backendUrl}${img.url}`}
                            width={600}
                            height={400}
                            alt={`${type} image ${index + 1}`}
                            className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                            loading="lazy"
                            onError={(e) => {
                              console.error('Image failed to load:', `${backendUrl}${img.url}`);
                              // Optional: Set a fallback image
                              e.currentTarget.src = '/placeholder-image.jpg';
                            }}
                          />
                          {/* Analysis Text Display */}
                          {img.analysis && (
                            <div className="absolute bottom-0 left-0 right-0 bg-black/80 text-white text-xs p-2 max-h-32 overflow-y-auto">
                              <pre className="whitespace-pre-wrap break-words">{img.analysis}</pre>
                            </div>
                          )}
                          {/* Enhanced Overlay Effects */}
                          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                          {/* Enhanced Image Info */}
                          <div className="absolute bottom-0 left-0 right-0 p-6 transform translate-y-full group-hover:translate-y-0 transition-transform duration-500">
                            <div className="flex items-center justify-between text-white mb-3">
                              <div>
                                <p className="text-lg font-bold capitalize">{type}</p>
                                <p className="text-sm text-gray-300">Image #{index + 1}</p>
                              </div>
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    toggleFavorite(index)
                                  }}
                                  className={`w-10 h-10 rounded-full flex items-center justify-center backdrop-blur-sm transition-all duration-200 ${favorites.has(index)
                                      ? 'bg-red-500/80 text-white'
                                      : 'bg-white/20 text-gray-300 hover:bg-red-500/60 hover:text-white'
                                    }`}
                                >
                                  <Heart className={`w-5 h-5 ${favorites.has(index) ? 'fill-current' : ''}`} />
                                </button>
                                <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center backdrop-blur-sm">
                                  <Eye className="w-5 h-5" />
                                </div>
                              </div>
                            </div>
                            {/* Enhanced Action Buttons */}
                            <div className="flex items-center gap-2">
                              <button
                                className="flex items-center gap-1 px-3 py-1.5 bg-blue-600/80 hover:bg-blue-600 rounded-lg text-xs font-medium transition-all duration-200"
                                onClick={e => {
                                  e.stopPropagation();
                                  const filename = img.filename || `${type}_image_${index + 1}.jpg`;
                                  handleDownload(img.url, filename);
                                }}
                              >
                                <Download className="w-3 h-3" />
                                Download
                              </button>
                              <button
                                className="flex items-center gap-1 px-3 py-1.5 bg-green-600/80 hover:bg-green-600 rounded-lg text-xs font-medium transition-all duration-200"
                                onClick={e => {
                                  e.stopPropagation();
                                  const filename = img.filename || `${type}_image_${index + 1}.jpg`;
                                  handleShare(img.url, filename);
                                }}
                              >
                                <Share2 className="w-3 h-3" />
                                Share
                              </button>
                              <button className="flex items-center gap-1 px-3 py-1.5 bg-purple-600/80 hover:bg-purple-600 rounded-lg text-xs font-medium transition-all duration-200">
                                <Star className="w-3 h-3" />
                                Rate
                              </button>
                            </div>
                          </div>
                          {/* Enhanced Corner Badge */}
                          <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-sm rounded-xl px-3 py-1.5 border border-gray-600/50">
                            <span className="text-sm font-bold text-white">#{index + 1}</span>
                          </div>
                          {/* Enhanced Status Indicators */}
                          <div className="absolute top-4 right-4 flex items-center gap-2">
                            <div className="w-3 h-3 bg-emerald-500 rounded-full animate-pulse" />
                            <span className="text-xs text-emerald-400 font-medium">AI Verified</span>
                          </div>
                        </div>
                        {/* Enhanced Hover Glow Effect */}
                        <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
                        {/* Enhanced Border Glow */}
                        <div className="absolute inset-0 rounded-3xl border-2 border-transparent group-hover:border-purple-500/50 transition-all duration-500" />
                      </div>
                    </PhotoView>
                  ))}
                </section>
              </PhotoProvider>
            )}

            {/* Enhanced Footer with Advanced Stats */}
            <footer className="mt-20">
              <div className="bg-gray-900/80 backdrop-blur-sm border border-gray-700/50 rounded-3xl p-8">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
                  <div className="text-center">
                    <Sparkles className="w-8 h-8 text-purple-400 mx-auto mb-2" />
                    <h4 className="text-white font-semibold mb-1">AI-Powered</h4>
                    <p className="text-gray-400 text-sm">Advanced classification</p>
                  </div>
                  <div className="text-center">
                    <Database className="w-8 h-8 text-blue-400 mx-auto mb-2" />
                    <h4 className="text-white font-semibold mb-1">{images.length} Images</h4>
                    <p className="text-gray-400 text-sm">Total processed</p>
                  </div>
                  <div className="text-center">
                    <Shield className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
                    <h4 className="text-white font-semibold mb-1">Secure</h4>
                    <p className="text-gray-400 text-sm">Local processing</p>
                  </div>
                  <div className="text-center">
                    <Globe className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
                    <h4 className="text-white font-semibold mb-1">Universal</h4>
                    <p className="text-gray-400 text-sm">Any image type</p>
                  </div>
                </div>

                <div className="flex items-center justify-center gap-6 text-sm text-gray-400 mb-4">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    Last updated: {new Date().toLocaleString()}
                  </span>
                  <span className="flex items-center gap-1">
                    <TrendingUp className="w-3 h-3" />
                    {favorites.size} favorites
                  </span>
                  <span className="flex items-center gap-1">
                    <BarChart3 className="w-3 h-3" />
                    {Object.keys(images).length} categories
                  </span>
                </div>

                <div className="text-center">
                  <p className="text-gray-500 text-xs">
                    Built with 💙 for AI innovation · UniversalAI © {new Date().getFullYear()}
                  </p>
                </div>
              </div>
            </footer>
          </div>
        </div>
      </div>
    </ProtectedRoute>
  )
}