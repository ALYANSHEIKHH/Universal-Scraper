"""
🔧 Configuration file for Cancer Image Scraper with Enhanced Classification
"""

import os

SECRET_KEY = os.getenv("SECRET_KEY", "default-secret")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))

# === System Constants ===
MAX_TOTAL_SIZE_BYTES = 500 * 1024 * 1024   # 500MB
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024    # 10MB
MAX_IMAGE_COUNT = 1000
IMAGE_CLEANUP_AGE_HOURS = 24
IMAGE_FOLDER = "classified_images"

# Original Cancer Labels
LABELS = ["breast_cancer", "lung_cancer", "skin_cancer", "unsorted"]

# === Enhanced OCR + Keyword Categories ===
CATEGORY_KEYWORDS = {
    "medical": [
        "medical", "health", "doctor", "physician", "nurse", "hospital", "clinic", "patient",
        "treatment", "diagnosis", "medicine", "medication", "prescription", "surgery", "operation",
        "therapy", "rehabilitation", "recovery", "healing", "disease", "illness", "condition",
        "symptom", "infection", "virus", "bacteria", "vaccine", "immunization", "blood", "heart",
        "brain", "cancer", "tumor", "malignant", "benign", "biopsy", "oncology", "chemotherapy",
        "radiation", "mammogram", "x-ray", "mri", "ct scan", "ultrasound", "chart", "record", 
        "report", "scan", "lab", "test", "result", "consultation", "referral", "insurance", 
        "pathology"
    ],

    "educational": [
        "university", "college", "school", "education", "academic", "research", "student", 
        "teacher", "professor", "classroom", "lecture", "tutorial", "assignment", "exam", 
        "syllabus", "curriculum", "degree", "certificate", "graduation", "marksheet", "grades", 
        "result", "subject", "transcript", "percentage", "roll number", "gpa", "study", "textbook", 
        "learning", "homework", "quiz", "science", "mathematics", "history", "biology", "chemistry",
        "physics", "philosophy", "journal", "abstract", "conclusion", "methodology", "bibliography",
        "publication"
    ],

    "sports": [
        "sport", "game", "match", "tournament", "championship", "league", "team", "player", 
        "athlete", "coach", "training", "exercise", "fitness", "score", "goal", "point", "win", 
        "victory", "defeat", "competition", "football", "soccer", "basketball", "baseball", 
        "tennis", "golf", "swimming", "running", "cycling", "boxing", "wrestling", "volleyball", 
        "hockey", "cricket", "rugby", "skiing", "surfing", "climbing", "yoga", "gym", "marathon", 
        "triathlon", "olympics", "fifa", "nba", "nfl", "mlb"
    ],

    "technology": [
        "technology", "computer", "software", "hardware", "internet", "web", "app", "application", 
        "program", "code", "programming", "development", "system", "network", "server", "database", 
        "cloud", "ai", "artificial intelligence", "machine learning", "analytics", "digital", 
        "device", "google", "microsoft", "apple", "amazon", "facebook", "linkedin", "github", 
        "stackoverflow", "android", "ios", "windows", "linux", "mac", "laptop", "tablet", "api", 
        "sdk", "framework"
    ],

    "finance": [
        "finance", "financial", "money", "bank", "investment", "stock", "share", "portfolio", 
        "market", "trading", "profit", "loss", "income", "expense", "budget", "cost", "price", 
        "payment", "loan", "credit", "debt", "mortgage", "tax", "accounting", "audit", "receipt", 
        "invoice", "bill", "contract", "statement", "balance", "earnings", "dividend", "currency", 
        "bitcoin", "forex"
    ],

    "id_card": [
        "id", "identification", "card", "license", "passport", "driver", "permit", "official", 
        "government", "expires", "expiry", "birth", "number", "ssn", "address", "phone", "email", 
        "photo", "signature", "barcode", "chip", "hologram", "valid", "verify"
    ],

    "random_photo": [
        "photo", "picture", "image", "selfie", "portrait", "landscape", "nature", "friends", 
        "vacation", "travel", "holiday", "party", "celebration", "sunset", "sunrise", "beach", 
        "mountain", "forest", "building", "architecture", "animal", "food", "restaurant", "cafe"
    ],

    "breast_cancer": [
        "breast", "mammography", "mammogram", "breast cancer", "mastectomy", "lumpectomy", 
        "ductal", "lobular", "her2", "estrogen", "progesterone"
    ],

    "lung_cancer": [
        "lung", "pulmonary", "respiratory", "lung cancer", "chest x-ray", "pneumonia", "bronchus", 
        "alveoli", "smoking", "tobacco"
    ],

    "skin_cancer": [
        "skin", "dermatology", "melanoma", "mole", "lesion", "skin cancer", "dermatologist", 
        "pigmentation", "freckle"
    ],

    # Optional: New explicit marksheet category (maps to educational)
    "marksheet": [
        "marksheet", "grades", "transcript", "result", "percentage", "subject", "total", "gpa", 
        "pass", "failed"
    ]
}

# === CLIP Visual Categories (Descriptive Prompts) ===
CLIP_CATEGORIES = [
    "a scanned medical report or x-ray image",
    "a marksheet or educational certificate document",
    "a photo of a sports game or athlete",
    "a screenshot of a technology app or dashboard",
    "a scanned financial document or bank report",
    "an identification card or government-issued ID",
    "a photo of nature, people, or random objects",
    "a printed academic or text document",
    "a diagram, chart, or labeled infographic",
    "a digital app screen or UI screenshot",
    "a logo or corporate brand design",
    "an artwork, sketch, or digital drawing"
]

# === OCR Configuration ===
OCR_CONFIG = {
    'tesseract_config': '--oem 3 --psm 6',
    'enhance_image': True,
    'min_image_size': 300,
    'min_text_length': 5,
    'min_confidence': 0.3,
    'timeout_seconds': 10
}

# === CLIP Model Settings ===
CLIP_CONFIG = {
    'model_name': 'ViT-B-32',
    'pretrained': 'openai',
    'min_confidence': 0.25,
    'temperature': 100,
    'batch_size': 1,
    'device': 'auto'
}

# === Logging & System Behavior ===
SYSTEM_CONFIG = {
    'log_level': 'INFO',
    'log_decisions': True,
    'log_processing_time': True,
    'default_category': 'unsorted',
    'enable_ocr': True,
    'enable_clip': True,
    'max_image_size_mb': MAX_IMAGE_SIZE_BYTES / (1024 * 1024),
    'max_processing_time': 30,
    'cache_clip_embeddings': True,
    'cache_ocr_results': False
}

# === Category Mapping: Enhanced → Original Folder ===
ENHANCED_TO_ORIGINAL_MAPPING = {
    "medical": "unsorted",
    "educational": "unsorted",
    "sports": "unsorted",
    "technology": "unsorted",
    "finance": "unsorted",
    "id_card": "unsorted",
    "random_photo": "unsorted",
    "marksheet": "unsorted",
    "breast_cancer": "breast_cancer",
    "lung_cancer": "lung_cancer",
    "skin_cancer": "skin_cancer",
    "uncategorized": "unsorted"
}

# === Confidence Thresholds for Logging or Debugging ===
CONFIDENCE_THRESHOLDS = {
    'ocr_keywords': {
        'high': 0.7,
        'medium': 0.4,
        'low': 0.2
    },
    'clip_visual': {
        'high': 0.6,
        'medium': 0.35,
        'low': 0.15
    }
}

# === Environment-aware Configuration ===
def get_config():
    env = os.getenv('ENVIRONMENT', 'development')
    if env == 'production':
        return {
            **globals(),
            'MAX_TOTAL_SIZE_BYTES': 1024 * 1024 * 1024,  # 1GB
            'MAX_IMAGE_COUNT': 5000,
            'IMAGE_CLEANUP_AGE_HOURS': 48,
            'SYSTEM_CONFIG': {
                **SYSTEM_CONFIG,
                'log_level': 'WARNING',
                'cache_clip_embeddings': True
            }
        }
    return globals()

# Access point
config = get_config()
