import React from 'react'
import '../../src/app/globals.css'
import Image from 'next/image'

interface ImageData {
  url: string
  filename: string
  category: string
  confidence: number
}

interface GalleryGridProps {
  images: ImageData[]
  onImageClick?: (img: ImageData) => void
}

const GalleryGrid: React.FC<GalleryGridProps> = ({ images, onImageClick }) => {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
      {images.map(img => (
        <div
          key={img.filename}
          className="relative group rounded-lg overflow-hidden shadow hover:shadow-lg transition cursor-pointer"
          onClick={() => onImageClick && onImageClick(img)}
        >
          <Image src={img.url} alt={img.filename} className="w-full h-40 object-cover" />
          <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-xs px-2 py-1 flex justify-between items-center">
            <span className="capitalize">{img.category}</span>
            <span>{Math.round(img.confidence * 100)}%</span>
          </div>
        </div>
      ))}
    </div>
  )
}

export default GalleryGrid 