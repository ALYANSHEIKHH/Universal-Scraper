// import { useState } from 'react'
// import toast from 'react-hot-toast'
// import { Loader2, UploadCloud, ImagePlus, FileArchive, Brain, BarChart3 } from 'lucide-react'
// import { useRouter } from 'next/navigation'
// import '../../src/app/globals.css'
// import GalleryGrid from './GalleryGrid'
// import DashboardStats from './DashboardStats'
// import ImageCategoryChart from './ImageCategoryChart'

// interface PredictionResult {
//   prediction: string
//   confidence: number
//   probabilities: { [key: string]: number }
// }

// interface SummaryData {
//   total_images: number
//   processed_images: number
//   classifications: Record<string, number>
//   confidence_avg: number
// }

// interface DataIngestionPanelProps {
//   onSummary: (summary: SummaryData) => void
// }

// export default function DataIngestionPanel({ onSummary }: DataIngestionPanelProps) {
//   const [url, setUrl] = useState('')
//   const [loading, setLoading] = useState(false)
//   const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
//   const [uploadLoading, setUploadLoading] = useState(false)
//   const [urlError, setUrlError] = useState('')
//   const router = useRouter()
//   const [images, setImages] = useState([])
//   const [summary, setSummary] = useState(null)

//   const validateUrl = (url: string) => {
//     try {
//       new URL(url)
//       setUrlError('')
//       return true
//     } catch {
//       setUrlError('Please enter a valid URL')
//       return false
//     }
//   }

//   const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
//     const value = e.target.value
//     setUrl(value)
//     if (value) validateUrl(value)
//   }

//   // === Scrape images from URL ===
//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault()
//     if (!url) return
    
//     if (!validateUrl(url)) {
//       toast.error('Please enter a valid URL')
//       return
//     }
    
//     setLoading(true)
//     toast.loading('Scraping images from URL...')
    
//     try {
//       const res = await fetch('http://localhost:8000/api/scrape', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ url }),
//       })
      
//       if (!res.ok) {
//         throw new Error(`HTTP ${res.status}: ${res.statusText}`)
//       }
      
//       const data = await res.json()
//       onSummary(data.summary)
//       toast.success('Images scraped and classified!')

//       // Store in localStorage and redirect
//       localStorage.setItem('dashboard_summary', JSON.stringify(data.summary))
//       localStorage.setItem('dashboard_images', JSON.stringify(data.images))
//       router.push('/dashboard')
//     } catch (err) {
//       const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred'
//       toast.error(`Scrape failed: ${errorMessage}`)
//     } finally {
//       toast.dismiss()
//       setLoading(false)
//     }
//   }

//   // === Upload multiple images ===
//   const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
//     const files = e.target.files
//     if (!files || files.length === 0) return
    
//     setUploadLoading(true)
//     toast.loading(`Uploading ${files.length} images...`)
    
//     const formData = new FormData()
//     Array.from(files).forEach(file => formData.append('files', file))
//     try {
//       const res = await fetch('http://localhost:8000/api/upload', {
//         method: 'POST',
//         body: formData,
//       })
//       const data = await res.json()
//       onSummary(data.summary)
//       toast.success('Images uploaded and classified!')
//     } catch (err) {
//       toast.error('Image upload failed.')
//     } finally {
//       setUploadLoading(false)
//       toast.dismiss()
//     }
//   }

//   // === Single Image Prediction ===
//   const handleSinglePrediction = async (e: React.ChangeEvent<HTMLInputElement>) => {
//     const file = e.target.files?.[0]
//     if (!file) return
//     toast.loading('Classifying image...')
//     const formData = new FormData()
//     formData.append('file', file)
//     try {
//       const res = await fetch('http://localhost:8000/api/predict', {
//         method: 'POST',
//         body: formData,
//       })
//       const data = await res.json()
//       setPredictionResult(data)
//       toast.success(`Prediction: ${data.prediction.toUpperCase()} (${Math.round(data.confidence * 100)}%)`)
//     } catch (err) {
//       toast.error('Prediction failed.')
//     } finally {
//       toast.dismiss()
//     }
//   }

//   // === Upload ZIP dataset ===
//   const handleZipUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
//     const file = e.target.files?.[0]
//     if (!file || !file.name.endsWith('.zip')) {
//       toast.error('Please upload a valid .zip file.')
//       return
//     }
//     toast.loading('Processing ZIP dataset...')
//     const formData = new FormData()
//     formData.append('file', file)
//     try {
//       const res = await fetch('http://localhost:8000/api/upload-kaggle', {
//         method: 'POST',
//         body: formData,
//       })
//       const data = await res.json()
//       onSummary(data.summary)
//       toast.success('ZIP dataset processed and classified!')
//     } catch (err) {
//       toast.error('ZIP upload failed.')
//     } finally {
//       toast.dismiss()
//     }
//   }

//   const handleDashboardClick = () => {
//     router.push('/dashboard')
//   }

//   return (
//     <section className="w-full py-24 bg-gradient-to-br from-gray-50 via-gray-100 to-gray-50 dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950 text-zinc-900 dark:text-zinc-100 border-b border-zinc-200 dark:border-zinc-800">
//       <div className="max-w-4xl mx-auto px-6 text-center">
//         <h1 className="text-4xl sm:text-5xl font-extrabold bg-gradient-to-r from-indigo-400 to-blue-500 bg-clip-text text-transparent mb-4 tracking-tight">
//           🧠 Universal AI Image Classifier
//         </h1>
//         <p className="text-zinc-500 dark:text-zinc-400 text-lg sm:text-xl mb-10">
//           Upload, scrape, or analyze any image. The AI will sort it into the right category!
//         </p>
//         <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-8">
//           <input
//             type="url"
//             value={url}
//             onChange={handleUrlChange}
//             placeholder="Paste any image dataset URL"
//             className="w-full sm:w-2/3 px-5 py-3 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-700 text-sm placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all"
//             required
//           />
//           <button
//             type="submit"
//             disabled={loading}
//             className="inline-flex items-center justify-center px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg transition disabled:opacity-50"
//           >
//             {loading ? (
//               <>
//                 <Loader2 className="w-5 h-5 animate-spin mr-2" />
//                 Processing...
//               </>
//             ) : (
//               <>
//                 <UploadCloud className="w-5 h-5 mr-2" />
//                 Scrape Images
//               </>
//             )}
//           </button>
//         </div>
        
//         <div className="flex flex-wrap justify-center gap-4 mb-8">
//           <button onClick={handleDashboardClick} className="secondary-button">
//             <BarChart3 className="w-5 h-5 mr-2" />
//             View Dashboard
//           </button>
//         </div>
//         <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
//           {/* Single Image Prediction */}
//           <label htmlFor="single-predict" className="inline-flex items-center cursor-pointer px-4 py-3 rounded-lg border border-emerald-600 bg-emerald-600/10 hover:bg-emerald-600/20 text-sm text-emerald-400 transition group">
//             <Brain className="w-5 h-5 mr-2 group-hover:animate-pulse" />
//             Classify Single Image
//           </label>
//           <input
//             id="single-predict"
//             type="file"
//             accept="image/*"
//             onChange={handleSinglePrediction}
//             className="hidden"
//           />
//           {/* Upload Images */}
//           <label htmlFor="upload-images" className="inline-flex items-center cursor-pointer px-4 py-3 rounded-lg border border-zinc-700 bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-sm text-zinc-700 dark:text-zinc-300 transition">
//             <ImagePlus className="w-5 h-5 mr-2" />
//             Upload Images
//           </label>
//           <input
//             id="upload-images"
//             type="file"
//             multiple
//             accept="image/*"
//             onChange={handleUpload}
//             className="hidden"
//           />
//           {/* Upload ZIP */}
//           <label htmlFor="upload-zip" className="inline-flex items-center cursor-pointer px-4 py-3 rounded-lg border border-zinc-700 bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-sm text-zinc-700 dark:text-zinc-300 transition">
//             <FileArchive className="w-5 h-5 mr-2" />
//             Upload ZIP
//           </label>
//           <input
//             id="upload-zip"
//             type="file"
//             accept=".zip"
//             onChange={handleZipUpload}
//             className="hidden"
//           />
//           {/* Model Info Button */}
//           <button
//             onClick={async () => {
//               try {
//                 const res = await fetch('http://localhost:8000/api/model-info')
//                 const data = await res.json()
//                 toast.success(`Model: ${data.model_info?.architecture || data.model_info?.model || 'Unknown'} | Device: ${data.model_info?.device}`)
//               } catch (err) {
//                 toast.error('Failed to fetch model info')
//               }
//             }}
//             className="inline-flex items-center justify-center px-4 py-3 rounded-lg border border-blue-600 bg-blue-600/10 hover:bg-blue-600/20 text-sm text-blue-400 transition"
//           >
//             <BarChart3 className="w-5 h-5 mr-2" />
//             Model Info
//           </button>
//         </div>
//         {/* AI Prediction Results */}
//         {predictionResult && (
//           <div className="bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-700 rounded-xl p-6 mb-8 text-left">
//             <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
//               <Brain className="w-5 h-5 text-emerald-400" />
//               AI Prediction Results
//             </h3>
//             <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//               {/* Main Prediction */}
//               <div className="bg-zinc-50 dark:bg-zinc-900 rounded-lg p-4">
//                 <p className="text-sm text-zinc-400 mb-1">Predicted Class</p>
//                 <p className="text-xl font-bold text-emerald-400 capitalize">
//                   {predictionResult.prediction}
//                 </p>
//                 <p className="text-sm text-zinc-700 dark:text-zinc-300 mt-1">
//                   Confidence: {Math.round(predictionResult.confidence * 100)}%
//                 </p>
//               </div>
//               {/* Probability Breakdown */}
//               <div className="bg-zinc-50 dark:bg-zinc-900 rounded-lg p-4">
//                 <p className="text-sm text-zinc-400 mb-3">Probability Distribution</p>
//                 {Object.entries(predictionResult.probabilities).map(([type, prob]) => (
//                   <div key={type} className="flex justify-between items-center mb-1">
//                     <span className="text-sm text-zinc-700 dark:text-zinc-300 capitalize">{type}</span>
//                     <div className="flex items-center gap-2">
//                       <span className="text-xs font-mono">{(prob * 100).toFixed(1)}%</span>
//                     </div>
//                   </div>
//                 ))}
//               </div>
//             </div>
//           </div>
//         )}
//       </div>
//     </section>
//   )
// }
