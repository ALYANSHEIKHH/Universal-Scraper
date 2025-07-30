# # import torch
# # import torch.nn as nn
# # import torchvision.transforms as transforms
# # from torchvision import models
# # import numpy as np
# # from PIL import Image
# # import logging
# # from typing import Dict, List, Tuple, Optional
# # import json
# # from pathlib import Path
# # import cv2
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.feature_extraction import image as skimage
# # import joblib

# # logger = logging.getLogger("EnhancedClassifier")

# # class EnhancedImageClassifier:
# #     """
# #     Enhanced image classification system with multiple analysis methods
# #     """
    
# #     def __init__(self, model_path: str = "models/cancer_classifier.pth", 
# #                  feature_model_path: str = "models/feature_classifier.pkl"):
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         self.model_path = model_path
# #         self.feature_model_path = feature_model_path
        
# #         # Initialize models
# #         self.deep_model = self._load_deep_model()
# #         self.feature_model = self._load_feature_model()
        
# #         # Image transforms
# #         self.transform = transforms.Compose([
# #             transforms.Resize((256, 256)),
# #             transforms.CenterCrop(224),
# #             transforms.ToTensor(),
# #             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
# #                                std=[0.229, 0.224, 0.225])
# #         ])
        
# #         # Cancer types and their descriptions
# #         self.cancer_types = {
# #             "lung": {
# #                 "name": "Lung Cancer",
# #                 "description": "Malignant growth in lung tissue",
# #                 "characteristics": ["nodules", "masses", "consolidation", "effusion"],
# #                 "severity_levels": ["benign", "malignant", "metastatic"]
# #             },
# #             "skin": {
# #                 "name": "Skin Cancer",
# #                 "description": "Abnormal growth of skin cells",
# #                 "characteristics": ["asymmetry", "irregular borders", "color variation", "diameter"],
# #                 "severity_levels": ["melanoma", "basal_cell", "squamous_cell", "benign"]
# #             },
# #             "breast": {
# #                 "name": "Breast Cancer",
# #                 "description": "Malignant tumor in breast tissue",
# #                 "characteristics": ["calcifications", "masses", "architectural_distortion", "asymmetry"],
# #                 "severity_levels": ["benign", "malignant", "invasive", "in_situ"]
# #             },
# #             "unsorted": {
# #                 "name": "Unclassified",
# #                 "description": "Image requires further analysis",
# #                 "characteristics": ["unclear", "low_quality", "non_medical"],
# #                 "severity_levels": ["unknown"]
# #             }
# #         }
    
# #     def _load_deep_model(self) -> nn.Module:
# #         """Load the deep learning model"""
# #         try:
# #             model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# #             model.fc = nn.Linear(model.fc.in_features, len(self.cancer_types))
            
# #             if Path(self.model_path).exists():
# #                 checkpoint = torch.load(self.model_path, map_location=self.device)
# #                 state_dict = checkpoint.get('model_state_dict', checkpoint)
# #                 model.load_state_dict(state_dict)
# #                 logger.info(f"✅ Loaded deep model from {self.model_path}")
# #             else:
# #                 logger.warning(f"⚠️ No pre-trained model found at {self.model_path}")
            
# #             model.to(self.device)
# #             model.eval()
# #             return model
            
# #         except Exception as e:
# #             logger.error(f"Failed to load deep model: {e}")
# #             return None
    
# #     def _load_feature_model(self) -> Optional[RandomForestClassifier]:
# #         """Load the feature-based classifier"""
# #         try:
# #             if Path(self.feature_model_path).exists():
# #                 model = joblib.load(self.feature_model_path)
# #                 logger.info(f"✅ Loaded feature model from {self.feature_model_path}")
# #                 return model
# #             else:
# #                 logger.warning(f"⚠️ No feature model found at {self.feature_model_path}")
# #                 return None
# #         except Exception as e:
# #             logger.error(f"Failed to load feature model: {e}")
# #             return None
    
# #     def classify_image(self, image_path: str) -> Dict:
# #         """
# #         Comprehensive image classification with multiple methods
# #         """
# #         try:
# #             # Load and preprocess image
# #             image = Image.open(image_path).convert("RGB")
            
# #             # Deep learning classification
# #             deep_prediction = self._deep_classify(image)
            
# #             # Feature-based classification
# #             feature_prediction = self._feature_classify(image_path)
            
# #             # Rule-based analysis
# #             rule_analysis = self._rule_based_analysis(image_path)
            
# #             # Combine predictions
# #             final_prediction = self._combine_predictions(
# #                 deep_prediction, feature_prediction, rule_analysis
# #             )
            
# #             # Add detailed information
# #             cancer_info = self.cancer_types.get(final_prediction["prediction"], 
# #                                               self.cancer_types["unsorted"])
            
# #             result = {
# #                 "prediction": final_prediction["prediction"],
# #                 "confidence": final_prediction["confidence"],
# #                 "probabilities": final_prediction["probabilities"],
# #                 "cancer_info": cancer_info,
# #                 "analysis_methods": {
# #                     "deep_learning": deep_prediction,
# #                     "feature_based": feature_prediction,
# #                     "rule_based": rule_analysis
# #                 },
# #                 "recommendations": self._generate_recommendations(final_prediction),
# #                 "processing_time": final_prediction.get("processing_time", 0)
# #             }
            
# #             return result
            
# #         except Exception as e:
# #             logger.error(f"Classification failed for {image_path}: {e}")
# #             return {
# #                 "prediction": "unsorted",
# #                 "confidence": 0.0,
# #                 "probabilities": {},
# #                 "error": str(e)
# #             }
    
# #     def _deep_classify(self, image: Image.Image) -> Dict:
# #         """Deep learning classification"""
# #         if self.deep_model is None:
# #             return {"prediction": "unsorted", "confidence": 0.0, "probabilities": {}}
        
# #         try:
# #             # Preprocess image
# #             input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
# #             # Get predictions
# #             with torch.no_grad():
# #                 outputs = self.deep_model(input_tensor)
# #                 probabilities = torch.softmax(outputs, dim=1)
# #                 confidence, predicted = torch.max(probabilities, 1)
            
# #             # Convert to class names
# #             class_names = list(self.cancer_types.keys())
# #             prediction = class_names[predicted.item()]
            
# #             # Create probability dict
# #             prob_dict = {class_names[i]: prob.item() 
# #                         for i, prob in enumerate(probabilities[0])}
            
# #             return {
# #                 "prediction": prediction,
# #                 "confidence": confidence.item(),
# #                 "probabilities": prob_dict,
# #                 "method": "deep_learning"
# #             }
            
# #         except Exception as e:
# #             logger.error(f"Deep classification failed: {e}")
# #             return {"prediction": "unsorted", "confidence": 0.0, "probabilities": {}}
    
# #     def _feature_classify(self, image_path: str) -> Dict:
# #         """Feature-based classification using traditional ML"""
# #         if self.feature_model is None:
# #             return {"prediction": "unsorted", "confidence": 0.0, "probabilities": {}}
        
# #         try:
# #             # Extract features
# #             features = self._extract_image_features(image_path)
            
# #             # Make prediction
# #             prediction = self.feature_model.predict([features])[0]
# #             probabilities = self.feature_model.predict_proba([features])[0]
            
# #             # Convert to class names
# #             class_names = list(self.cancer_types.keys())
# #             prediction_name = class_names[prediction]
            
# #             # Create probability dict
# #             prob_dict = {class_names[i]: prob for i, prob in enumerate(probabilities)}
            
# #             return {
# #                 "prediction": prediction_name,
# #                 "confidence": max(probabilities),
# #                 "probabilities": prob_dict,
# #                 "method": "feature_based"
# #             }
            
# #         except Exception as e:
# #             logger.error(f"Feature classification failed: {e}")
# #             return {"prediction": "unsorted", "confidence": 0.0, "probabilities": {}}
    
# #     def _extract_image_features(self, image_path: str) -> List[float]:
# #         """Extract traditional image features"""
# #         try:
# #             # Load image
# #             image = cv2.imread(image_path)
# #             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
# #             features = []
            
# #             # Color features
# #             hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# #             features.extend([
# #                 np.mean(hsv[:, :, 0]),  # Hue mean
# #                 np.std(hsv[:, :, 0]),   # Hue std
# #                 np.mean(hsv[:, :, 1]),  # Saturation mean
# #                 np.std(hsv[:, :, 1]),   # Saturation std
# #                 np.mean(hsv[:, :, 2]),  # Value mean
# #                 np.std(hsv[:, :, 2])    # Value std
# #             ])
            
# #             # Texture features
# #             # Local Binary Pattern
# #             lbp = self._calculate_lbp(gray)
# #             lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
# #             features.extend(lbp_hist[:16])  # First 16 bins
            
# #             # Gradient features
# #             grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
# #             grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
# #             gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
# #             features.extend([
# #                 np.mean(gradient_magnitude),
# #                 np.std(gradient_magnitude),
# #                 np.max(gradient_magnitude)
# #             ])
            
# #             # Shape features
# #             edges = cv2.Canny(gray, 50, 150)
# #             contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
# #             if contours:
# #                 areas = [cv2.contourArea(c) for c in contours]
# #                 features.extend([
# #                     len(contours),
# #                     np.mean(areas),
# #                     np.std(areas),
# #                     np.max(areas)
# #                 ])
# #             else:
# #                 features.extend([0, 0, 0, 0])
            
# #             # Quality features
# #             laplacian = cv2.Laplacian(gray, cv2.CV_64F)
# #             features.extend([
# #                 laplacian.var(),  # Sharpness
# #                 gray.std(),       # Contrast
# #                 gray.mean()       # Brightness
# #             ])
            
# #             return features
            
# #         except Exception as e:
# #             logger.error(f"Feature extraction failed: {e}")
# #             return [0.0] * 50  # Return zeros if extraction fails
    
# #     def _rule_based_analysis(self, image_path: str) -> Dict:
# #         """Rule-based analysis using image characteristics"""
# #         try:
# #             image = cv2.imread(image_path)
# #             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #             hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
# #             # Calculate various metrics
# #             brightness = np.mean(hsv[:, :, 2])
# #             saturation = np.mean(hsv[:, :, 1])
# #             contrast = gray.std()
# #             sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
# #             # Rule-based classification
# #             rules = {
# #                 "lung": {
# #                     "conditions": [
# #                         brightness > 100,  # Generally bright
# #                         contrast > 30,     # Good contrast
# #                         sharpness > 100    # Sharp features
# #                     ],
# #                     "weight": 0.3
# #                 },
# #                 "skin": {
# #                     "conditions": [
# #                         saturation > 50,   # Colorful
# #                         contrast > 20,     # Moderate contrast
# #                         brightness > 80    # Bright
# #                     ],
# #                     "weight": 0.3
# #                 },
# #                 "breast": {
# #                     "conditions": [
# #                         brightness > 90,   # Bright
# #                         contrast > 25,     # Good contrast
# #                         sharpness > 80     # Sharp
# #                     ],
# #                     "weight": 0.3
# #                 }
# #             }
            
# #             # Calculate scores
# #             scores = {}
# #             for cancer_type, rule in rules.items():
# #                 score = sum(rule["conditions"]) / len(rule["conditions"]) * rule["weight"]
# #                 scores[cancer_type] = score
            
# #             # Find best match
# #             if scores:
# #                 best_type = max(scores, key=scores.get)
# #                 confidence = scores[best_type]
# #             else:
# #                 best_type = "unsorted"
# #                 confidence = 0.0
            
# #             return {
# #                 "prediction": best_type,
# #                 "confidence": confidence,
# #                 "scores": scores,
# #                 "method": "rule_based"
# #             }
            
# #         except Exception as e:
# #             logger.error(f"Rule-based analysis failed: {e}")
# #             return {"prediction": "unsorted", "confidence": 0.0, "scores": {}}
    
# #     def _combine_predictions(self, deep_pred: Dict, feature_pred: Dict, 
# #                            rule_pred: Dict) -> Dict:
# #         """Combine predictions from different methods"""
# #         # Weighted combination
# #         weights = {
# #             "deep_learning": 0.6,
# #             "feature_based": 0.3,
# #             "rule_based": 0.1
# #         }
        
# #         # Initialize combined probabilities
# #         combined_probs = {cancer_type: 0.0 for cancer_type in self.cancer_types.keys()}
        
# #         # Deep learning contribution
# #         if deep_pred["probabilities"]:
# #             for cancer_type, prob in deep_pred["probabilities"].items():
# #                 combined_probs[cancer_type] += prob * weights["deep_learning"]
        
# #         # Feature-based contribution
# #         if feature_pred["probabilities"]:
# #             for cancer_type, prob in feature_pred["probabilities"].items():
# #                 combined_probs[cancer_type] += prob * weights["feature_based"]
        
# #         # Rule-based contribution
# #         if rule_pred["scores"]:
# #             total_score = sum(rule_pred["scores"].values())
# #             if total_score > 0:
# #                 for cancer_type, score in rule_pred["scores"].items():
# #                     combined_probs[cancer_type] += (score / total_score) * weights["rule_based"]
        
# #         # Find best prediction
# #         best_type = max(combined_probs, key=combined_probs.get)
# #         confidence = combined_probs[best_type]
        
# #         return {
# #             "prediction": best_type,
# #             "confidence": confidence,
# #             "probabilities": combined_probs
# #         }
    
# #     def _generate_recommendations(self, prediction: Dict) -> List[str]:
# #         """Generate recommendations based on classification"""
# #         recommendations = []
        
# #         cancer_type = prediction["prediction"]
# #         confidence = prediction["confidence"]
        
# #         if cancer_type == "unsorted":
# #             recommendations.extend([
# #                 "Image quality may be insufficient for accurate classification",
# #                 "Consider re-scanning with higher resolution",
# #                 "Manual review by medical professional recommended"
# #             ])
# #         else:
# #             cancer_info = self.cancer_types[cancer_type]
            
# #             if confidence > 0.8:
# #                 recommendations.extend([
# #                     f"High confidence classification as {cancer_info['name']}",
# #                     "Consider immediate medical consultation",
# #                     "Additional imaging may be beneficial for confirmation"
# #                 ])
# #             elif confidence > 0.6:
# #                 recommendations.extend([
# #                     f"Moderate confidence classification as {cancer_info['name']}",
# #                     "Medical consultation recommended",
# #                     "Consider additional diagnostic tests"
# #                 ])
# #             else:
# #                 recommendations.extend([
# #                     f"Low confidence classification as {cancer_info['name']}",
# #                     "Manual review strongly recommended",
# #                     "Consider alternative imaging modalities"
# #                 ])
        
# #         return recommendations
    
# #     def _calculate_lbp(self, gray_image: np.ndarray) -> np.ndarray:
# #         """Calculate Local Binary Pattern"""
# #         lbp = np.zeros_like(gray_image)
# #         for i in range(1, gray_image.shape[0]-1):
# #             for j in range(1, gray_image.shape[1]-1):
# #                 center = gray_image[i, j]
# #                 code = 0
# #                 code |= (gray_image[i-1, j-1] > center) << 7
# #                 code |= (gray_image[i-1, j] > center) << 6
# #                 code |= (gray_image[i-1, j+1] > center) << 5
# #                 code |= (gray_image[i, j+1] > center) << 4
# #                 code |= (gray_image[i+1, j+1] > center) << 3
# #                 code |= (gray_image[i+1, j] > center) << 2
# #                 code |= (gray_image[i+1, j-1] > center) << 1
# #                 code |= (gray_image[i, j-1] > center) << 0
# #                 lbp[i, j] = code
# #         return lbp








# enhanced_image_classifier.py
"""
🧠 Enhanced Image Classification System
Combines OCR text extraction with CLIP visual understanding for accurate categorization
"""

import logging
import time
import torch
import re
from typing import Tuple, Optional, Dict
from PIL import Image
from io import BytesIO
import pytesseract
import open_clip

from app.config import CATEGORY_KEYWORDS, CLIP_CATEGORIES, OCR_CONFIG, CLIP_CONFIG

logger = logging.getLogger("EnhancedClassifier")

class EnhancedImageClassifier:
    def __init__(self, config_module=None):
        if config_module is None:
            self.category_keywords = CATEGORY_KEYWORDS
            self.clip_categories = CLIP_CATEGORIES
            self.ocr_config = OCR_CONFIG
            self.clip_config = CLIP_CONFIG
        else:
            self.category_keywords = config_module.CATEGORY_KEYWORDS
            self.clip_categories = config_module.CLIP_CATEGORIES
            self.ocr_config = config_module.OCR_CONFIG
            self.clip_config = config_module.CLIP_CONFIG

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 Loading CLIP model on device: {self.device}")

        try:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                self.clip_config['model_name'], 
                pretrained=self.clip_config['pretrained']
            )
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            self.clip_tokenizer = open_clip.get_tokenizer(self.clip_config['model_name'])
            self._prepare_clip_embeddings()
            logger.info("✅ CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP model: {e}")
            self.clip_model = None

    def _prepare_clip_embeddings(self):
        try:
            text_prompts = [f"a photo of {category}" for category in self.clip_categories]
            text_tokens = self.clip_tokenizer(text_prompts).to(self.device)
            with torch.no_grad():
                self.text_embeddings = self.clip_model.encode_text(text_tokens)
                self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=-1, keepdim=True)
            logger.info(f"✅ Pre-computed embeddings for {len(self.clip_categories)} CLIP categories")
        except Exception as e:
            logger.error(f"❌ Failed to prepare CLIP embeddings: {e}")
            self.text_embeddings = None

    def extract_text_with_ocr(self, image_data: bytes) -> Tuple[str, bool]:
        try:
            image = Image.open(BytesIO(image_data)).convert('RGB')
            if self.ocr_config.get('enhance_image', True):
                image = image.convert('L')
                if image.size[0] < 300 or image.size[1] < 300:
                    scale_factor = max(300 / image.size[0], 300 / image.size[1])
                    new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
            custom_config = self.ocr_config.get('tesseract_config', '--oem 3 --psm 6')
            extracted_text = pytesseract.image_to_string(image, config=custom_config)
            cleaned_text = self._clean_extracted_text(extracted_text)
            if len(cleaned_text.strip()) >= self.ocr_config.get('min_text_length', 3):
                logger.info(f"✅ OCR extracted text: '{cleaned_text[:100]}...' ({len(cleaned_text)} chars)")
                return cleaned_text, True
            else:
                logger.info("⚠️ OCR extracted insufficient text")
                return "", False
        except Exception as e:
            logger.warning(f"❌ OCR extraction failed: {e}")
            return "", False

    def _clean_extracted_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\-.,!?@#$%&*()+=]', ' ', text)
        words = text.split()
        cleaned_words = [word for word in words if len(word) >= 2 or word.isdigit()]
        return ' '.join(cleaned_words)

    def classify_by_keywords(self, text: str) -> Tuple[Optional[str], float]:
        if not text:
            return None, 0.0

        text_lower = text.lower()
        category_scores = {}

        for category, keywords in self.category_keywords.items():
            score = 0
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    keyword_weight = len(keyword) / 10.0
                    occurrences = text_lower.count(keyword.lower())
                    score += occurrences * keyword_weight
                    matched_keywords.append(keyword)
            if score > 0:
                category_scores[category] = {
                    'score': score,
                    'keywords': matched_keywords
                }

        if not category_scores:
            return None, 0.0

        best_category = max(category_scores.keys(), key=lambda k: category_scores[k]['score'])
        best_score = category_scores[best_category]['score']
        matched_kw = category_scores[best_category]['keywords']
        confidence = min(0.95, best_score / 5.0)

        logger.info(f"🎯 Keyword match: '{best_category}' (confidence: {confidence:.2%}) - matched: {matched_kw}")
        return best_category, confidence

    def classify_with_clip(self, image_data: bytes) -> Tuple[Optional[str], float]:
        if self.clip_model is None or self.text_embeddings is None:
            logger.warning("⚠️ CLIP model not available")
            return None, 0.0

        try:
            image = Image.open(BytesIO(image_data)).convert('RGB')
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_embedding = self.clip_model.encode_image(image_tensor)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                similarities = (image_embedding @ self.text_embeddings.T).squeeze(0)
                probabilities = torch.softmax(similarities * self.clip_config.get("temperature", 100), dim=0)
                best_idx = similarities.argmax().item()
                best_category = self.clip_categories[best_idx]
                confidence = probabilities[best_idx].item()
                top3_indices = similarities.argsort(descending=True)[:3]
                top3_predictions = [(self.clip_categories[i], probabilities[i].item()) for i in top3_indices]
                logger.info(f"🤖 CLIP predictions: {top3_predictions}")
                logger.info(f"🎯 CLIP best match: '{best_category}' (confidence: {confidence:.2%})")
                return best_category, confidence
        except Exception as e:
            logger.error(f"❌ CLIP classification failed: {e}")
            return None, 0.0

    def classify_image(self, image_data: bytes) -> Dict:
        start_time = time.time()
        result = {
            'category': 'uncategorized',
            'confidence': 0.0,
            'method_used': 'none',
            'ocr_text': '',
            'decision_path': [],
            'processing_time': 0.0,
            'clip_available': self.clip_model is not None
        }

        logger.info("🔄 Starting enhanced image classification...")
        result['decision_path'].append("Step 1: Attempting OCR text extraction")
        ocr_text, ocr_success = self.extract_text_with_ocr(image_data)
        result['ocr_text'] = ocr_text

        if ocr_success:
            result['decision_path'].append("Step 1.1: OCR successful, attempting keyword matching")
            category, confidence = self.classify_by_keywords(ocr_text)
            if category and confidence >= self.ocr_config.get('min_confidence', 0.3):
                result['category'] = category
                result['confidence'] = confidence
                result['method_used'] = 'ocr_keywords'
                result['decision_path'].append(f"Step 1.2: ✅ OCR classification successful: '{category}' ({confidence:.2%})")
                logger.info(f"✅ Classification complete via OCR: '{category}' ({confidence:.2%})")
            else:
                result['decision_path'].append("Step 1.2: ⚠️ OCR found text but no strong keyword matches")
                logger.info("⚠️ OCR found text but no strong keyword matches, trying CLIP...")
        else:
            result['decision_path'].append("Step 1.1: ❌ OCR failed or insufficient text found")
            logger.info("❌ OCR failed or insufficient text, trying CLIP...")

        if result['method_used'] == 'none':
            result['decision_path'].append("Step 2: Falling back to CLIP visual classification")
            clip_category, clip_confidence = self.classify_with_clip(image_data)
            if clip_category and clip_confidence >= self.clip_config.get('min_confidence', 0.2):
                result['category'] = clip_category
                result['confidence'] = clip_confidence
                result['method_used'] = 'clip_visual'
                result['decision_path'].append(f"Step 2.1: ✅ CLIP classification successful: '{clip_category}' ({clip_confidence:.2%})")
                logger.info(f"✅ Classification complete via CLIP: '{clip_category}' ({clip_confidence:.2%})")
            else:
                result['decision_path'].append("Step 2.1: ❌ CLIP classification failed or low confidence")
                logger.info("❌ Both OCR and CLIP failed, classifying as 'uncategorized'")

        result['processing_time'] = time.time() - start_time
        if result['category'] == 'uncategorized':
            result['decision_path'].append("Final: Classified as 'uncategorized' - no method succeeded")

        logger.info(f"🎝️ Final classification: '{result['category']}' via {result['method_used']} ({result['confidence']:.2%}) in {result['processing_time']:.2f}s")
        return result

def classify_image_enhanced(image_data: bytes, config_module=None) -> Dict:
    classifier = EnhancedImageClassifier(config_module)
    return classifier.classify_image(image_data)






















###################################################



# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models
# import numpy as np
# from PIL import Image
# import logging
# from typing import Dict, List, Tuple, Optional
# import json
# from pathlib import Path
# import cv2
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction import image as skimage
# import joblib

# logger = logging.getLogger("EnhancedClassifier")

# class EnhancedImageClassifier:
#     """
#     Enhanced image classification system with multiple analysis methods
#     """
    
#     def __init__(self, model_path: str = "models/cancer_classifier.pth", 
#                  feature_model_path: str = "models/feature_classifier.pkl"):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_path = model_path
#         self.feature_model_path = feature_model_path
        
#         # Initialize models
#         self.deep_model = self._load_deep_model()
#         self.feature_model = self._load_feature_model()
        
#         # Image transforms
#         self.transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                std=[0.229, 0.224, 0.225])
#         ])
        
#         # Cancer types and their descriptions
#         self.cancer_types = {
#             "lung": {
#                 "name": "Lung Cancer",
#                 "description": "Malignant growth in lung tissue",
#                 "characteristics": ["nodules", "masses", "consolidation", "effusion"],
#                 "severity_levels": ["benign", "malignant", "metastatic"]
#             },
#             "skin": {
#                 "name": "Skin Cancer",
#                 "description": "Abnormal growth of skin cells",
#                 "characteristics": ["asymmetry", "irregular borders", "color variation", "diameter"],
#                 "severity_levels": ["melanoma", "basal_cell", "squamous_cell", "benign"]
#             },
#             "breast": {
#                 "name": "Breast Cancer",
#                 "description": "Malignant tumor in breast tissue",
#                 "characteristics": ["calcifications", "masses", "architectural_distortion", "asymmetry"],
#                 "severity_levels": ["benign", "malignant", "invasive", "in_situ"]
#             },
#             "unsorted": {
#                 "name": "Unclassified",
#                 "description": "Image requires further analysis",
#                 "characteristics": ["unclear", "low_quality", "non_medical"],
#                 "severity_levels": ["unknown"]
#             }
#         }
    
#     def _load_deep_model(self) -> nn.Module:
#         """Load the deep learning model"""
#         try:
#             model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#             model.fc = nn.Linear(model.fc.in_features, len(self.cancer_types))
            
#             if Path(self.model_path).exists():
#                 checkpoint = torch.load(self.model_path, map_location=self.device)
#                 state_dict = checkpoint.get('model_state_dict', checkpoint)
#                 model.load_state_dict(state_dict)
#                 logger.info(f"✅ Loaded deep model from {self.model_path}")
#             else:
#                 logger.warning(f"⚠️ No pre-trained model found at {self.model_path}")
            
#             model.to(self.device)
#             model.eval()
#             return model
            
#         except Exception as e:
#             logger.error(f"Failed to load deep model: {e}")
#             return None
    
#     def _load_feature_model(self) -> Optional[RandomForestClassifier]:
#         """Load the feature-based classifier"""
#         try:
#             if Path(self.feature_model_path).exists():
#                 model = joblib.load(self.feature_model_path)
#                 logger.info(f"✅ Loaded feature model from {self.feature_model_path}")
#                 return model
#             else:
#                 logger.warning(f"⚠️ No feature model found at {self.feature_model_path}")
#                 return None
#         except Exception as e:
#             logger.error(f"Failed to load feature model: {e}")
#             return None
    
#     def classify_image(self, image_path: str) -> Dict:
#         """
#         Comprehensive image classification with multiple methods
#         """
#         try:
#             # Load and preprocess image
#             image = Image.open(image_path).convert("RGB")
            
#             # Deep learning classification
#             deep_prediction = self._deep_classify(image)
            
#             # Feature-based classification
#             feature_prediction = self._feature_classify(image_path)
            
#             # Rule-based analysis
#             rule_analysis = self._rule_based_analysis(image_path)
            
#             # Combine predictions
#             final_prediction = self._combine_predictions(
#                 deep_prediction, feature_prediction, rule_analysis
#             )
            
#             # Add detailed information
#             cancer_info = self.cancer_types.get(final_prediction["prediction"], 
#                                               self.cancer_types["unsorted"])
            
#             result = {
#                 "prediction": final_prediction["prediction"],
#                 "confidence": final_prediction["confidence"],
#                 "probabilities": final_prediction["probabilities"],
#                 "cancer_info": cancer_info,
#                 "analysis_methods": {
#                     "deep_learning": deep_prediction,
#                     "feature_based": feature_prediction,
#                     "rule_based": rule_analysis
#                 },
#                 "recommendations": self._generate_recommendations(final_prediction),
#                 "processing_time": final_prediction.get("processing_time", 0)
#             }
            
#             return result
            
#         except Exception as e:
#             logger.error(f"Classification failed for {image_path}: {e}")
#             return {
#                 "prediction": "unsorted",
#                 "confidence": 0.0,
#                 "probabilities": {},
#                 "error": str(e)
#             }
    
#     def _deep_classify(self, image: Image.Image) -> Dict:
#         """Deep learning classification"""
#         if self.deep_model is None:
#             return {"prediction": "unsorted", "confidence": 0.0, "probabilities": {}}
        
#         try:
#             # Preprocess image
#             input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
#             # Get predictions
#             with torch.no_grad():
#                 outputs = self.deep_model(input_tensor)
#                 probabilities = torch.softmax(outputs, dim=1)
#                 confidence, predicted = torch.max(probabilities, 1)
            
#             # Convert to class names
#             class_names = list(self.cancer_types.keys())
#             prediction = class_names[predicted.item()]
            
#             # Create probability dict
#             prob_dict = {class_names[i]: prob.item() 
#                         for i, prob in enumerate(probabilities[0])}
            
#             return {
#                 "prediction": prediction,
#                 "confidence": confidence.item(),
#                 "probabilities": prob_dict,
#                 "method": "deep_learning"
#             }
            
#         except Exception as e:
#             logger.error(f"Deep classification failed: {e}")
#             return {"prediction": "unsorted", "confidence": 0.0, "probabilities": {}}
    
#     def _feature_classify(self, image_path: str) -> Dict:
#         """Feature-based classification using traditional ML"""
#         if self.feature_model is None:
#             return {"prediction": "unsorted", "confidence": 0.0, "probabilities": {}}
        
#         try:
#             # Extract features
#             features = self._extract_image_features(image_path)
            
#             # Make prediction
#             prediction = self.feature_model.predict([features])[0]
#             probabilities = self.feature_model.predict_proba([features])[0]
            
#             # Convert to class names
#             class_names = list(self.cancer_types.keys())
#             prediction_name = class_names[prediction]
            
#             # Create probability dict
#             prob_dict = {class_names[i]: prob for i, prob in enumerate(probabilities)}
            
#             return {
#                 "prediction": prediction_name,
#                 "confidence": max(probabilities),
#                 "probabilities": prob_dict,
#                 "method": "feature_based"
#             }
            
#         except Exception as e:
#             logger.error(f"Feature classification failed: {e}")
#             return {"prediction": "unsorted", "confidence": 0.0, "probabilities": {}}
    
#     def _extract_image_features(self, image_path: str) -> List[float]:
#         """Extract traditional image features"""
#         try:
#             # Load image
#             image = cv2.imread(image_path)
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
#             features = []
            
#             # Color features
#             hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#             features.extend([
#                 np.mean(hsv[:, :, 0]),  # Hue mean
#                 np.std(hsv[:, :, 0]),   # Hue std
#                 np.mean(hsv[:, :, 1]),  # Saturation mean
#                 np.std(hsv[:, :, 1]),   # Saturation std
#                 np.mean(hsv[:, :, 2]),  # Value mean
#                 np.std(hsv[:, :, 2])    # Value std
#             ])
            
#             # Texture features
#             # Local Binary Pattern
#             lbp = self._calculate_lbp(gray)
#             lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
#             features.extend(lbp_hist[:16])  # First 16 bins
            
#             # Gradient features
#             grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#             grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#             gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
#             features.extend([
#                 np.mean(gradient_magnitude),
#                 np.std(gradient_magnitude),
#                 np.max(gradient_magnitude)
#             ])
            
#             # Shape features
#             edges = cv2.Canny(gray, 50, 150)
#             contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             if contours:
#                 areas = [cv2.contourArea(c) for c in contours]
#                 features.extend([
#                     len(contours),
#                     np.mean(areas),
#                     np.std(areas),
#                     np.max(areas)
#                 ])
#             else:
#                 features.extend([0, 0, 0, 0])
            
#             # Quality features
#             laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#             features.extend([
#                 laplacian.var(),  # Sharpness
#                 gray.std(),       # Contrast
#                 gray.mean()       # Brightness
#             ])
            
#             return features
            
#         except Exception as e:
#             logger.error(f"Feature extraction failed: {e}")
#             return [0.0] * 50  # Return zeros if extraction fails
    
#     def _rule_based_analysis(self, image_path: str) -> Dict:
#         """Rule-based analysis using image characteristics"""
#         try:
#             image = cv2.imread(image_path)
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
#             # Calculate various metrics
#             brightness = np.mean(hsv[:, :, 2])
#             saturation = np.mean(hsv[:, :, 1])
#             contrast = gray.std()
#             sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
#             # Rule-based classification
#             rules = {
#                 "lung": {
#                     "conditions": [
#                         brightness > 100,  # Generally bright
#                         contrast > 30,     # Good contrast
#                         sharpness > 100    # Sharp features
#                     ],
#                     "weight": 0.3
#                 },
#                 "skin": {
#                     "conditions": [
#                         saturation > 50,   # Colorful
#                         contrast > 20,     # Moderate contrast
#                         brightness > 80    # Bright
#                     ],
#                     "weight": 0.3
#                 },
#                 "breast": {
#                     "conditions": [
#                         brightness > 90,   # Bright
#                         contrast > 25,     # Good contrast
#                         sharpness > 80     # Sharp
#                     ],
#                     "weight": 0.3
#                 }
#             }
            
#             # Calculate scores
#             scores = {}
#             for cancer_type, rule in rules.items():
#                 score = sum(rule["conditions"]) / len(rule["conditions"]) * rule["weight"]
#                 scores[cancer_type] = score
            
#             # Find best match
#             if scores:
#                 best_type = max(scores, key=scores.get)
#                 confidence = scores[best_type]
#             else:
#                 best_type = "unsorted"
#                 confidence = 0.0
            
#             return {
#                 "prediction": best_type,
#                 "confidence": confidence,
#                 "scores": scores,
#                 "method": "rule_based"
#             }
            
#         except Exception as e:
#             logger.error(f"Rule-based analysis failed: {e}")
#             return {"prediction": "unsorted", "confidence": 0.0, "scores": {}}
    
#     def _combine_predictions(self, deep_pred: Dict, feature_pred: Dict, 
#                            rule_pred: Dict) -> Dict:
#         """Combine predictions from different methods"""
#         # Weighted combination
#         weights = {
#             "deep_learning": 0.6,
#             "feature_based": 0.3,
#             "rule_based": 0.1
#         }
        
#         # Initialize combined probabilities
#         combined_probs = {cancer_type: 0.0 for cancer_type in self.cancer_types.keys()}
        
#         # Deep learning contribution
#         if deep_pred["probabilities"]:
#             for cancer_type, prob in deep_pred["probabilities"].items():
#                 combined_probs[cancer_type] += prob * weights["deep_learning"]
        
#         # Feature-based contribution
#         if feature_pred["probabilities"]:
#             for cancer_type, prob in feature_pred["probabilities"].items():
#                 combined_probs[cancer_type] += prob * weights["feature_based"]
        
#         # Rule-based contribution
#         if rule_pred["scores"]:
#             total_score = sum(rule_pred["scores"].values())
#             if total_score > 0:
#                 for cancer_type, score in rule_pred["scores"].items():
#                     combined_probs[cancer_type] += (score / total_score) * weights["rule_based"]
        
#         # Find best prediction
#         best_type = max(combined_probs, key=combined_probs.get)
#         confidence = combined_probs[best_type]
        
#         return {
#             "prediction": best_type,
#             "confidence": confidence,
#             "probabilities": combined_probs
#         }
    
#     def _generate_recommendations(self, prediction: Dict) -> List[str]:
#         """Generate recommendations based on classification"""
#         recommendations = []
        
#         cancer_type = prediction["prediction"]
#         confidence = prediction["confidence"]
        
#         if cancer_type == "unsorted":
#             recommendations.extend([
#                 "Image quality may be insufficient for accurate classification",
#                 "Consider re-scanning with higher resolution",
#                 "Manual review by medical professional recommended"
#             ])
#         else:
#             cancer_info = self.cancer_types[cancer_type]
            
#             if confidence > 0.8:
#                 recommendations.extend([
#                     f"High confidence classification as {cancer_info['name']}",
#                     "Consider immediate medical consultation",
#                     "Additional imaging may be beneficial for confirmation"
#                 ])
#             elif confidence > 0.6:
#                 recommendations.extend([
#                     f"Moderate confidence classification as {cancer_info['name']}",
#                     "Medical consultation recommended",
#                     "Consider additional diagnostic tests"
#                 ])
#             else:
#                 recommendations.extend([
#                     f"Low confidence classification as {cancer_info['name']}",
#                     "Manual review strongly recommended",
#                     "Consider alternative imaging modalities"
#                 ])
        
#         return recommendations
    
#     def _calculate_lbp(self, gray_image: np.ndarray) -> np.ndarray:
#         """Calculate Local Binary Pattern"""
#         lbp = np.zeros_like(gray_image)
#         for i in range(1, gray_image.shape[0]-1):
#             for j in range(1, gray_image.shape[1]-1):
#                 center = gray_image[i, j]
#                 code = 0
#                 code |= (gray_image[i-1, j-1] > center) << 7
#                 code |= (gray_image[i-1, j] > center) << 6
#                 code |= (gray_image[i-1, j+1] > center) << 5
#                 code |= (gray_image[i, j+1] > center) << 4
#                 code |= (gray_image[i+1, j+1] > center) << 3
#                 code |= (gray_image[i+1, j] > center) << 2
#                 code |= (gray_image[i+1, j-1] > center) << 1
#                 code |= (gray_image[i, j-1] > center) << 0
#                 lbp[i, j] = code
#         return lbp