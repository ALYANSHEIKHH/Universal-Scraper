# scraper_integration.py
"""
🔧 Integration code to upgrade your existing image scraper
Replace your existing prediction functions with these enhanced versions
"""

import logging
from typing import Dict, Optional
from enhanced_classifier import EnhancedImageClassifier
import time

# Initialize the enhanced classifier (singleton pattern)
_enhanced_classifier = None

def get_enhanced_classifier():
    """Get or create the enhanced classifier instance."""
    global _enhanced_classifier
    if _enhanced_classifier is None:
        _enhanced_classifier = EnhancedImageClassifier()
    return _enhanced_classifier

# === Updated Prediction Functions ===

def predict_cancer_type(image_data) -> dict:
    """
    🔬 UPDATED: Enhanced cancer type prediction using OCR + CLIP
    This replaces your existing predict_cancer_type function
    """
    try:
        logger = logging.getLogger("CancerAI")
        start_time = time.time()
        
        # Use enhanced classifier
        classifier = get_enhanced_classifier()
        result = classifier.classify_image(image_data)
        
        # Map enhanced categories back to your cancer categories if needed
        enhanced_category = result['category']
        
        # Map to your existing cancer labels or keep as-is
        cancer_category = map_to_cancer_categories(enhanced_category)
        
        # Format response to match your existing API structure
        response = {
            "prediction": cancer_category,
            "confidence": result['confidence'],
            "probabilities": create_probability_dict(result),
            "raw_prediction": enhanced_category,
            "method_used": result['method_used'],
            "ocr_text": result['ocr_text'],
            "decision_path": result['decision_path'],
            "processing_time": result['processing_time']
        }
        
        # Enhanced logging
        method_desc = {
            'ocr_keywords': 'OCR + Keywords',
            'clip_visual': 'CLIP Visual',
            'none': 'Fallback'
        }
        
        logger.info(f"🧠 Enhanced prediction: {cancer_category} ({result['confidence']:.2%}) via {method_desc.get(result['method_used'], 'Unknown')}")
        
        if result['ocr_text']:
            logger.info(f"📝 OCR extracted: '{result['ocr_text'][:100]}...'")
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced prediction error: {e}")
        # Fallback to uncategorized
        return {
            "prediction": "unsorted", 
            "confidence": 0.0, 
            "probabilities": {},
            "method_used": "error",
            "ocr_text": "",
            "decision_path": [f"Error: {str(e)}"],
            "processing_time": 0.0
        }

def predict_image_clip(image_data, class_names=None, top_n=3) -> dict:
    """
    🤖 UPDATED: Universal CLIP prediction with OCR fallback
    This replaces your existing predict_image_clip function
    """
    try:
        logger = logging.getLogger("CancerAI")
        
        # Use enhanced classifier for better accuracy
        classifier = get_enhanced_classifier()
        result = classifier.classify_image(image_data)
        
        # Format response to match existing CLIP API
        response = {
            "prediction": result['category'],
            "confidence": result['confidence'],
            "top_classes": [result['category']],  # Could be expanded
            "top_confidences": [result['confidence']],
            "probabilities": create_probability_dict(result),
            "model": f"Enhanced ({result['method_used']})",
            "method_used": result['method_used'],
            "ocr_text": result['ocr_text'],
            "decision_path": result['decision_path']
        }
        
        logger.info(f"🤖 Enhanced CLIP: {result['category']} ({result['confidence']:.2%}) via {result['method_used']}")
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced CLIP error: {e}")
        return {
            "prediction": "other", 
            "confidence": 0.0, 
            "probabilities": {}, 
            "model": "Enhanced (error)",
            "method_used": "error",
            "ocr_text": "",
            "decision_path": [f"Error: {str(e)}"]
        }

# === Helper Functions ===

def map_to_cancer_categories(enhanced_category: str) -> str:
    """
    Map enhanced categories back to your existing cancer classification system.
    Modify this based on your specific needs.
    """
    # Example mapping - adjust based on your actual categories
    category_mapping = {
        "medical": "medical_scan",  # Could be any of your cancer types
        "educational": "research_document", 
        "technology": "equipment_image",
        "id_card": "patient_id",
        "random_photo": "unsorted",
        "uncategorized": "unsorted"
    }
    
    return category_mapping.get(enhanced_category, "unsorted")

def create_probability_dict(result: Dict) -> Dict:
    """Create probabilities dictionary for API compatibility."""
    # If you need full probability distribution, you could expand this
    category = result['category']
    confidence = result['confidence']
    
    # Create basic probability dict
    probabilities = {category: confidence}
    
    # Add some reasonable estimates for other categories
    remaining_prob = 1.0 - confidence
    other_categories = ['medical', 'educational', 'technology', 'finance', 'id_card', 'random_photo', 'uncategorized']
    
    for cat in other_categories:
        if cat != category:
            probabilities[cat] = remaining_prob / len(other_categories)
    
    return probabilities

# === Batch Processing Functions ===

def process_images_batch(image_data_list: list) -> list:
    """
    🚀 Process multiple images efficiently with enhanced classification
    """
    classifier = get_enhanced_classifier()
    results = []
    
    logger = logging.getLogger("CancerAI")
    logger.info(f"🔄 Processing batch of {len(image_data_list)} images")
    
    for i, image_data in enumerate(image_data_list):
        try:
            result = classifier.classify_image(image_data)
            results.append(result)
            
            if i % 10 == 0:
                logger.info(f"📊 Batch progress: {i+1}/{len(image_data_list)} images processed")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to process image {i+1}: {e}")
            results.append({
                'category': 'uncategorized',
                'confidence': 0.0,
                'method_used': 'error',
                'ocr_text': '',
                'decision_path': [f"Error: {str(e)}"],
                'processing_time': 0.0
            })
    
    logger.info(f"✅ Batch processing complete: {len(results)} results")
    return results

# === Analytics & Monitoring Functions ===

def get_classification_stats() -> Dict:
    """
    📊 Get statistics about classification performance
    """
    try:
        # This would ideally connect to your database to get real stats
        # For now, return basic info
        stats = {
            "classifier_status": "active",
            "ocr_available": True,
            "clip_available": _enhanced_classifier.clip_model is not None if _enhanced_classifier else False,
            "supported_categories": len(_enhanced_classifier.category_keywords) if _enhanced_classifier else 0,
            "clip_categories": len(_enhanced_classifier.clip_categories) if _enhanced_classifier else 0
        }
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}

def validate_classification_setup() -> Dict:
    """
    🔍 Validate that the enhanced classification system is properly configured
    """
    issues = []
    warnings = []
    
    try:
        # Check if classifier can be initialized
        classifier = get_enhanced_classifier()
        
        # Check OCR availability
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
        except Exception as e:
            issues.append(f"Tesseract OCR not available: {e}")
        
        # Check CLIP model
        if classifier.clip_model is None:
            issues.append("CLIP model failed to load")
        
        # Check configuration
        if len(classifier.category_keywords) == 0:
            issues.append("No keyword categories configured")
            
        if len(classifier.clip_categories) == 0:
            issues.append("No CLIP categories configured")
        
        # Check device availability
        import torch
        if not torch.cuda.is_available():
            warnings.append("CUDA not available, using CPU (slower)")
        
        return {
            "status": "healthy" if len(issues) == 0 else "issues_found",
            "issues": issues,
            "warnings": warnings,
            "ocr_available": len([i for i in issues if "OCR" in i]) == 0,
            "clip_available": classifier.clip_model is not None,
            "total_categories": len(classifier.category_keywords) + len(classifier.clip_categories)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "issues": [f"System validation failed: {e}"],
            "warnings": [],
            "ocr_available": False,
            "clip_available": False,
            "total_categories": 0
        }

# === Integration with Your Existing API ===

def enhance_existing_endpoints():
    """
    🔧 Instructions for integrating with your existing FastAPI endpoints
    
    Replace these functions in your main application:
    
    1. Replace predict_cancer_type() calls with the enhanced version above
    2. Replace predict_image_clip() calls with the enhanced version above
    3. Add new endpoints for enhanced features
    """
    
    integration_guide = """
    
    === INTEGRATION STEPS ===
    
    1. Install Required Dependencies:
       pip install pytesseract open-clip-torch pillow
       
       # Install Tesseract OCR system package:
       # Ubuntu/Debian: sudo apt-get install tesseract-ocr
       # macOS: brew install tesseract
       # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
    
    2. Add to your main.py imports:
       from enhanced_image_classifier import EnhancedImageClassifier
       from scraper_integration import (
           predict_cancer_type, predict_image_clip, 
           get_classification_stats, validate_classification_setup
       )
    
    3. Replace your existing functions:
       - Your predict_cancer_type() -> Use enhanced version
       - Your predict_image_clip() -> Use enhanced version
    
    4. Add new API endpoints:
    """
    
    return integration_guide

# === New API Endpoints to Add ===

def create_enhanced_api_endpoints():
    """
    🚀 New API endpoints you can add for enhanced functionality
    Copy these into your main FastAPI application
    """
    
    endpoint_code = '''
    
    # Add these new endpoints to your FastAPI app:
    
    @app.get("/api/classifier/status")
    async def get_classifier_status():
        """Get status of the enhanced classification system"""
        return get_classification_stats()
    
    @app.get("/api/classifier/validate")
    async def validate_classifier():
        """Validate classification system setup"""
        return validate_classification_setup()
    
    @app.post("/api/classify/enhanced")
    async def classify_image_enhanced(file: UploadFile = File(...)):
        """Enhanced image classification with detailed decision path"""
        try:
            content = await file.read()
            if len(content) > MAX_IMAGE_SIZE:
                raise HTTPException(status_code=413, detail="Image too large")
            
            classifier = get_enhanced_classifier()
            result = classifier.classify_image(content)
            
            return {
                "filename": file.filename,
                "category": result['category'],
                "confidence": result['confidence'],
                "method_used": result['method_used'],
                "ocr_text": result['ocr_text'][:200] + "..." if len(result['ocr_text']) > 200 else result['ocr_text'],
                "decision_path": result['decision_path'],
                "processing_time": result['processing_time'],
                "clip_available": result['clip_available']
            }
            
        except Exception as e:
            logger.error(f"Enhanced classification failed: {e}")
            raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    
    @app.post("/api/classify/batch-enhanced")
    async def classify_batch_enhanced(files: List[UploadFile] = File(...)):
        """Enhanced batch classification with detailed results"""
        try:
            results = []
            classifier = get_enhanced_classifier()
            
            for file in files:
                try:
                    content = await file.read()
                    if len(content) > MAX_IMAGE_SIZE:
                        continue
                    
                    result = classifier.classify_image(content)
                    results.append({
                        "filename": file.filename,
                        "category": result['category'],
                        "confidence": result['confidence'],
                        "method_used": result['method_used'],
                        "processing_time": result['processing_time']
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to process {file.filename}: {e}")
                    continue
            
            return {
                "total_files": len(files),
                "processed_files": len(results),
                "results": results,
                "success_rate": len(results) / len(files) if files else 0
            }
            
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")
    
    @app.get("/api/classifier/categories")
    async def get_available_categories():
        """Get all available classification categories"""
        try:
            classifier = get_enhanced_classifier()
            return {
                "ocr_categories": list(classifier.category_keywords.keys()),
                "clip_categories": classifier.clip_categories,
                "total_categories": len(classifier.category_keywords) + len(classifier.clip_categories)
            }
        except Exception as e:
            return {"error": str(e)}
    
    @app.post("/api/classifier/test")
    async def test_classification_methods(file: UploadFile = File(...)):
        """Test both OCR and CLIP methods separately for comparison"""
        try:
            content = await file.read()
            classifier = get_enhanced_classifier()
            
            # Test OCR only
            ocr_text, ocr_success = classifier.extract_text_with_ocr(content)
            ocr_result = None
            if ocr_success:
                ocr_category, ocr_confidence = classifier.classify_by_keywords(ocr_text)
                ocr_result = {"category": ocr_category, "confidence": ocr_confidence}
            
            # Test CLIP only
            clip_category, clip_confidence = classifier.classify_with_clip(content)
            clip_result = {"category": clip_category, "confidence": clip_confidence}
            
            # Full enhanced result
            full_result = classifier.classify_image(content)
            
            return {
                "filename": file.filename,
                "ocr_test": {
                    "success": ocr_success,
                    "extracted_text": ocr_text[:200] + "..." if len(ocr_text) > 200 else ocr_text,
                    "classification": ocr_result
                },
                "clip_test": clip_result,
                "enhanced_result": {
                    "final_category": full_result['category'],
                    "final_confidence": full_result['confidence'],
                    "method_used": full_result['method_used'],
                    "decision_path": full_result['decision_path']
                }
            }
            
        except Exception as e:
            logger.error(f"Classification test failed: {e}")
            raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")
    '''
    
    return endpoint_code

# === Performance Monitoring ===

class ClassificationMonitor:
    """
    📊 Monitor classification performance and accuracy
    """
    
    def __init__(self):
        self.stats = {
            'total_classifications': 0,
            'ocr_successful': 0,
            'clip_fallbacks': 0,
            'errors': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0,
            'category_distribution': {}
        }
    
    def record_classification(self, result: Dict):
        """Record a classification result for monitoring"""
        try:
            self.stats['total_classifications'] += 1
            
            if result['method_used'] == 'ocr_keywords':
                self.stats['ocr_successful'] += 1
            elif result['method_used'] == 'clip_visual':
                self.stats['clip_fallbacks'] += 1
            elif result['method_used'] == 'error':
                self.stats['errors'] += 1
            
            # Update averages
            total = self.stats['total_classifications']
            self.stats['average_confidence'] = (
                (self.stats['average_confidence'] * (total - 1) + result['confidence']) / total
            )
            self.stats['average_processing_time'] = (
                (self.stats['average_processing_time'] * (total - 1) + result['processing_time']) / total
            )
            
            # Update category distribution
            category = result['category']
            self.stats['category_distribution'][category] = (
                self.stats['category_distribution'].get(category, 0) + 1
            )
            
        except Exception as e:
            logging.getLogger("ClassificationMonitor").warning(f"Failed to record stats: {e}")
    
    def get_stats(self) -> Dict:
        """Get current monitoring statistics"""
        total = self.stats['total_classifications']
        if total == 0:
            return self.stats
        
        # Calculate percentages
        enhanced_stats = self.stats.copy()
        enhanced_stats.update({
            'ocr_success_rate': self.stats['ocr_successful'] / total,
            'clip_fallback_rate': self.stats['clip_fallbacks'] / total,
            'error_rate': self.stats['errors'] / total,
            'most_common_category': max(self.stats['category_distribution'].items(), 
                                      key=lambda x: x[1])[0] if self.stats['category_distribution'] else None
        })
        
        return enhanced_stats
    
    def reset_stats(self):
        """Reset monitoring statistics"""
        self.__init__()

# Global monitor instance
classification_monitor = ClassificationMonitor()

# === Export Functions ===
__all__ = [
    'predict_cancer_type',
    'predict_image_clip', 
    'get_enhanced_classifier',
    'process_images_batch',
    'get_classification_stats',
    'validate_classification_setup',
    'enhance_existing_endpoints',
    'create_enhanced_api_endpoints',
    'ClassificationMonitor',
    'classification_monitor'
]