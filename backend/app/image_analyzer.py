import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance, ImageFilter
import os
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import colorsys

logger = logging.getLogger("ImageAnalyzer")

class ImageAnalyzer:
    """
    Advanced image analysis system for medical images
    """
    
    def __init__(self, output_dir: str = "analysis_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_image(self, image_path: str) -> Dict:
        """
        Comprehensive image analysis
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB for analysis
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            analysis_results = {
                "basic_stats": self._get_basic_stats(image_rgb),
                "color_analysis": self._analyze_colors(image_rgb),
                "texture_analysis": self._analyze_texture(image_rgb),
                "edge_analysis": self._analyze_edges(image_rgb),
                "histogram_data": self._get_histogram_data(image_rgb),
                "dominant_colors": self._extract_dominant_colors(image_rgb),
                "image_quality": self._assess_image_quality(image_rgb),
                "medical_features": self._extract_medical_features(image_rgb)
            }
            
            # Generate visualizations
            viz_paths = self._generate_visualizations(image_rgb, analysis_results, image_path)
            analysis_results["visualizations"] = viz_paths
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Image analysis failed for {image_path}: {e}")
            return {"error": str(e)}
    
    def _get_basic_stats(self, image: np.ndarray) -> Dict:
        """Get basic image statistics"""
        height, width, channels = image.shape
        return {
            "dimensions": {"width": width, "height": height},
            "aspect_ratio": width / height,
            "total_pixels": width * height,
            "channels": channels,
            "dtype": str(image.dtype),
            "memory_size_mb": image.nbytes / (1024 * 1024)
        }
    
    def _analyze_colors(self, image: np.ndarray) -> Dict:
        """Analyze color characteristics"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Calculate color statistics
        rgb_means = np.mean(image, axis=(0, 1))
        rgb_stds = np.std(image, axis=(0, 1))
        hsv_means = np.mean(hsv, axis=(0, 1))
        
        # Calculate color diversity
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        color_diversity = unique_colors / (image.shape[0] * image.shape[1])
        
        return {
            "rgb_means": rgb_means.tolist(),
            "rgb_stds": rgb_stds.tolist(),
            "hsv_means": hsv_means.tolist(),
            "color_diversity": color_diversity,
            "unique_colors": unique_colors,
            "brightness": np.mean(hsv[:, :, 2]),
            "saturation": np.mean(hsv[:, :, 1])
        }
    
    def _analyze_texture(self, image: np.ndarray) -> Dict:
        """Analyze texture characteristics"""
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate GLCM-like features
        # Local Binary Pattern approximation
        lbp = self._calculate_lbp(gray)
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            "texture_variance": np.var(gray),
            "texture_entropy": self._calculate_entropy(gray),
            "gradient_mean": np.mean(gradient_magnitude),
            "gradient_std": np.std(gradient_magnitude),
            "lbp_uniformity": self._calculate_lbp_uniformity(lbp)
        }
    
    def _analyze_edges(self, image: np.ndarray) -> Dict:
        """Analyze edge characteristics"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            "edge_density": edge_density,
            "num_contours": len(contours),
            "total_edge_length": np.sum(edges > 0),
            "edge_complexity": len(contours) / (image.shape[0] * image.shape[1])
        }
    
    def _get_histogram_data(self, image: np.ndarray) -> Dict:
        """Get histogram data for all channels"""
        histograms = {}
        for i, channel in enumerate(['red', 'green', 'blue']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            histograms[channel] = hist.flatten().tolist()
        
        # Grayscale histogram
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        histograms['grayscale'] = gray_hist.flatten().tolist()
        
        return histograms
    
    def _extract_dominant_colors(self, image: np.ndarray, n_colors: int = 5) -> List[Dict]:
        """Extract dominant colors using K-means clustering"""
        # Reshape image for clustering
        pixels = image.reshape(-1, 3)
        
        # Use K-means to find dominant colors
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors and their percentages
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        dominant_colors = []
        for i, color in enumerate(colors):
            percentage = np.sum(labels == i) / len(labels)
            dominant_colors.append({
                "rgb": color.tolist(),
                "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                "percentage": percentage,
                "hsv": colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
            })
        
        # Sort by percentage
        dominant_colors.sort(key=lambda x: x["percentage"], reverse=True)
        return dominant_colors
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict:
        """Assess image quality metrics"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate noise level
        noise = self._estimate_noise(gray)
        
        # Calculate contrast
        contrast = gray.std()
        
        return {
            "sharpness": sharpness,
            "noise_level": noise,
            "contrast": contrast,
            "brightness": gray.mean(),
            "dynamic_range": gray.max() - gray.min()
        }
    
    def _extract_medical_features(self, image: np.ndarray) -> Dict:
        """Extract features relevant to medical image analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate tissue-like regions (assuming medical images)
        # This is a simplified approach - in practice, more sophisticated methods would be used
        
        # Find regions with medical image characteristics
        # (e.g., consistent texture, moderate contrast)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        tissue_mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
        
        return {
            "tissue_coverage": np.sum(tissue_mask > 0) / tissue_mask.size,
            "image_consistency": np.std(gray) / np.mean(gray),
            "feature_density": self._calculate_feature_density(gray)
        }
    
    def _generate_visualizations(self, image: np.ndarray, analysis: Dict, original_path: str) -> Dict:
        """Generate various visualizations"""
        viz_paths = {}
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Color histogram
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Image Analysis Visualizations', fontsize=16)
        
        # RGB histograms
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            axes[0, 0].hist(image[:, :, i].flatten(), bins=50, alpha=0.7, 
                           color=color, label=color.capitalize())
        axes[0, 0].set_title('RGB Histograms')
        axes[0, 0].set_xlabel('Pixel Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Dominant colors
        dominant_colors = analysis['dominant_colors']
        color_patches = []
        color_labels = []
        for i, color_info in enumerate(dominant_colors[:5]):
            color_patches.append(plt.Rectangle((0, 0), 1, 1, 
                                             facecolor=color_info['hex']))
            color_labels.append(f"{color_info['percentage']:.1%}")
        
        axes[0, 1].add_patch(plt.Rectangle((0, 0), 5, 1, facecolor='white'))
        for i, patch in enumerate(color_patches):
            axes[0, 1].add_patch(plt.Rectangle((i, 0), 1, 1, 
                                             facecolor=patch.get_facecolor()))
        axes[0, 1].set_xlim(0, 5)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title('Dominant Colors')
        axes[0, 1].set_xticks(range(5))
        axes[0, 1].set_xticklabels(color_labels, rotation=45)
        
        # Original image
        axes[1, 0].imshow(image)
        axes[1, 0].set_title('Original Image')
        axes[1, 0].axis('off')
        
        # Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        axes[1, 1].imshow(edges, cmap='gray')
        axes[1, 1].set_title('Edge Detection')
        axes[1, 1].axis('off')
        
        # Use subplots_adjust instead of tight_layout for better compatibility
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
        
        # Save visualization
        viz_filename = f"analysis_{Path(original_path).stem}.png"
        viz_path = self.output_dir / viz_filename
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        viz_paths["main_analysis"] = str(viz_path)
        
        # 2. Quality metrics visualization
        quality_fig, quality_ax = plt.subplots(figsize=(10, 6))
        quality_metrics = analysis['image_quality']
        metrics_names = list(quality_metrics.keys())
        metrics_values = list(quality_metrics.values())
        
        bars = quality_ax.bar(metrics_names, metrics_values, color='skyblue')
        quality_ax.set_title('Image Quality Metrics')
        quality_ax.set_ylabel('Value')
        quality_ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            quality_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.2f}', ha='center', va='bottom')
        
        # Use subplots_adjust for quality metrics as well
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
        quality_viz_path = self.output_dir / f"quality_{Path(original_path).stem}.png"
        plt.savefig(quality_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        viz_paths["quality_metrics"] = str(quality_viz_path)
        
        return viz_paths
    
    # Helper methods
    def _calculate_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern"""
        lbp = np.zeros_like(gray_image)
        for i in range(1, gray_image.shape[0]-1):
            for j in range(1, gray_image.shape[1]-1):
                center = gray_image[i, j]
                code = 0
                code |= (gray_image[i-1, j-1] > center) << 7
                code |= (gray_image[i-1, j] > center) << 6
                code |= (gray_image[i-1, j+1] > center) << 5
                code |= (gray_image[i, j+1] > center) << 4
                code |= (gray_image[i+1, j+1] > center) << 3
                code |= (gray_image[i+1, j] > center) << 2
                code |= (gray_image[i+1, j-1] > center) << 1
                code |= (gray_image[i, j-1] > center) << 0
                lbp[i, j] = code
        return lbp
    
    def _calculate_lbp_uniformity(self, lbp: np.ndarray) -> float:
        """Calculate LBP uniformity"""
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        return np.sum(hist > 0) / 256
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist[hist > 0]
        prob = hist / hist.sum()
        return -np.sum(prob * np.log2(prob))
    
    def _estimate_noise(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in image"""
        # Simple noise estimation using high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray_image, -1, kernel)
        return np.std(filtered)
    
    def _calculate_feature_density(self, gray_image: np.ndarray) -> float:
        """Calculate feature density"""
        # Use gradient magnitude as feature indicator
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Count pixels with significant gradient
        threshold = np.mean(gradient_magnitude) + np.std(gradient_magnitude)
        feature_pixels = np.sum(gradient_magnitude > threshold)
        
        return feature_pixels / gradient_magnitude.size
