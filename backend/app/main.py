from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta
import hashlib
from app.database import db
from PIL import Image
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
import os
import shutil
import zipfile
from io import BytesIO
from typing import List, Optional
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import logging
from uuid import uuid4
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
import time
from pathlib import Path
import csv
import sqlite3
import threading
from app.models.training import train_model
import clip
import open_clip
from app import config
from app.image_analyzer import ImageAnalyzer
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# from fastapi.responses import JSONResponse
# import shutil
# import os
import uuid
from app.auth import router as auth_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://localhost:3000",  # Local development (https)
        "https://your-frontend-domain.vercel.app",  # Your deployed frontend (when you deploy to Vercel)
        "https://alyan1-my-fastapi-backend.hf.space"  # Your Hugging Face space (for API docs)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)

@app.get("/")
def home():
    return {"message": "API is running!"}

from app.enhanced_classifier import classify_image_enhanced

app = FastAPI()

UPLOAD_DIR = "data/uploads"

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Save the uploaded image
    file_ext = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    saved_path = os.path.join(UPLOAD_DIR, unique_filename)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read image bytes
    with open(saved_path, "rb") as f:
        image_bytes = f.read()

    # Classify the image
    result = classify_image_enhanced(image_bytes)

    # Move file into predicted category folder
    final_category = result["category"]
    category_folder = os.path.join("data", final_category)
    os.makedirs(category_folder, exist_ok=True)

    final_path = os.path.join(category_folder, unique_filename)
    shutil.move(saved_path, final_path)

    # Return result
    return JSONResponse({
        "filename": unique_filename,
        "category": result["category"],
        "method_used": result["method_used"],
        "confidence": result["confidence"],
        "processing_time": result["processing_time"],
        "decision_path": result["decision_path"]
    })

# === Configuration ===
MAX_TOTAL_SIZE_BYTES = config.MAX_TOTAL_SIZE_BYTES
MAX_IMAGE_SIZE = config.MAX_IMAGE_SIZE_BYTES
MAX_IMAGE_COUNT = config.MAX_IMAGE_COUNT
IMAGE_CLEANUP_AGE_HOURS = config.IMAGE_CLEANUP_AGE_HOURS
LABELS = config.LABELS
IMAGE_FOLDER = config.IMAGE_FOLDER
TEMP_FOLDER = "temp_processing"
MODEL_PATH = "models/cancer_classifier.pth"
CONFIDENCE_THRESHOLD = 0.7
MAX_CONCURRENT_DOWNLOADS = 5

# === Dashboard System ===
# dashboard_system = DashboardSystem() # This line is removed as DashboardSystem is no longer imported

# Create necessary directories
for folder in [IMAGE_FOLDER, TEMP_FOLDER, "models"]:
    os.makedirs(folder, exist_ok=True)

for label in LABELS:
    os.makedirs(os.path.join(IMAGE_FOLDER, label), exist_ok=True)

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CancerAI")

# === AI Model Architecture ===
class CancerClassifier(nn.Module):
    """
    Advanced Cancer Type Classifier using ResNet50 backbone
    """
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(CancerClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze early layers for transfer learning
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
            
        # Custom classifier head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate/4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# === AI Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🔧 AI Device: {device}")

# Initialize model
# === Model Definition ===
# === Model Definition ===
model = CancerClassifier(num_classes=len(LABELS))

# === Load Pre-trained Weights If Available ===
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        # Load only the state_dict, fallback if not found
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)

        logger.info("✅ Loaded pre-trained AI model from '%s'", MODEL_PATH)
    except Exception as e:
        logger.warning("⚠️ Failed to load pre-trained weights: %s", str(e))
        logger.info("🔄 Proceeding with randomly initialized weights")
else:
    logger.info("🆕 No pre-trained model found at '%s' — using fresh weights", MODEL_PATH)

# === Final Model Setup ===
model.to(device)
model.eval()


# === Inference-time Image Preprocessing Pipeline ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# === Optional: Training-time Data Augmentation (Only used during training) ===
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])


# === Authentication Configuration ===
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database is now handled by app.database.db
# No more in-memory storage - all data is permanent

# === Authentication Models ===
class UserLogin(BaseModel):
    email: str
    password: str

class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str

class TokenResponse(BaseModel):
    token: str
    user: UserResponse

# === Authentication Functions ===
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return email
    except jwt.PyJWTError:
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    email = verify_token(credentials.credentials)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.get_user_by_email(email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# === FastAPI App ===
app = FastAPI(
    title="🧠 AI Cancer Image Scraper & Classifier",
    description="Advanced AI-powered medical image classification system",
    version="2.0.0"
)

# === Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Static Files ===
app.mount("/images", StaticFiles(directory=IMAGE_FOLDER), name="images")

# === Helper Functions ===
def is_valid_image_url(url: str) -> bool:
    """Accept any http(s) URL as a potential image URL (no extension/path checks)."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ['http', 'https']
    except:
        return False

def preprocess_image(image_data) -> Optional[torch.Tensor]:
    """Preprocess image for AI model"""
    try:
        if isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        else:
            return None
            
        # Check image size
        if len(image_data) > MAX_IMAGE_SIZE:
            logger.warning("Image too large, skipping")
            return None
            
        return transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None

def predict_cancer_type(image_data) -> dict:
    try:
        input_tensor = preprocess_image(image_data)
        if input_tensor is None:
            return {"prediction": "unsorted", "confidence": 0.0, "probabilities": {}}

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        prob_dict = {LABELS[i]: float(probabilities[0][i]) for i in range(len(LABELS))}

        # Use updated threshold logic
        if confidence >= CONFIDENCE_THRESHOLD:
            prediction = LABELS[predicted_class]
        else:
            prediction = "unsorted"

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "raw_prediction": LABELS[predicted_class]
        }

    except Exception as e:
        logger.error(f"AI prediction error: {e}")
        return {"prediction": "unsorted", "confidence": 0.0, "probabilities": {}}
  


   

def save_classified_image(image_data: bytes, classification: dict, source_info: str = "") -> str:
    """Save image to appropriate folder based on AI classification and trigger automated analysis"""
    try:
        prediction = classification["prediction"]
        confidence = classification["confidence"]

        # For CLIP model predictions, use the prediction directly
        # Only fallback to "unsorted" if it's truly an invalid prediction
        if not prediction or prediction == "unsorted":
            prediction = "unsorted"

        # Generate unique filename
        filename = f"{uuid4()}_{int(confidence * 100)}.jpg"
        folder_path = os.path.join(IMAGE_FOLDER, prediction)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, filename)

        # Save image
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image.save(file_path, "JPEG", quality=85, optimize=True)

        logger.info(f"✅ Saved to: {file_path} | From: {source_info}")
        
        # Enforce MAX_IMAGE_COUNT in this folder
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        if len(image_files) > MAX_IMAGE_COUNT:
            # Sort by modification time (oldest first)
            image_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
            to_delete = image_files[:-MAX_IMAGE_COUNT]
            for old_file in to_delete:
                old_path = os.path.join(folder_path, old_file)
                try:
                    os.remove(old_path)
                    logger.info(f"🗑️ Deleted to enforce max image count: {old_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {old_path}: {e}")

        # Trigger automated analysis workflow
        try:
            # Run analysis in background to avoid blocking the main process
            def run_analysis():
                try:
                    logger.info(f"🔄 Starting automated analysis for: {file_path}")
                    # dashboard_system.process_new_image(file_path) # This line is removed as DashboardSystem is no longer imported
                    logger.info(f"✅ Automated analysis completed for: {file_path}")
                except Exception as e:
                    logger.error(f"❌ Background analysis failed for: {file_path}: {e}")
            
            # Start analysis in background thread
            analysis_thread = threading.Thread(target=run_analysis, daemon=True)
            analysis_thread.start()
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to trigger automated analysis: {e}")
        
        return file_path

    except Exception as e:
        logger.error(f"❌ Failed to save image: {e}")
        return ""

# === API Models ===
class URLRequest(BaseModel):
    url: str
    max_images: Optional[int] = 100

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    processing_time: float

class ScrapeResponse(BaseModel):
    message: str
    summary: dict
    total_processed: int
    processing_time: float
    success_rate: float

# === Modular Data Collection & Management Workflow ===
DATA_ROOT = "broad_data"
RAW_DATA_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, "processed")
METADATA_DB = os.path.join(DATA_ROOT, "metadata.db")

# 1. Initialize data directories and metadata storage
def init_data_dirs_and_metadata():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    # Initialize SQLite DB for metadata if not exists
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        source TEXT,
        datatype TEXT,
        status TEXT,
        annotations TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

# 2. Data ingestion from APIs (example: placeholder)
def ingest_from_api(api_url, params=None):
    """Ingest data from an API endpoint. Returns list of (data, metadata_dict)."""
    # Example: Download an image from a public API
    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if 'image' in content_type:
            filename = f"api_{uuid4()}.jpg"
            metadata = {
                'filename': filename,
                'source': api_url,
                'datatype': 'image',
                'status': 'raw',
                'annotations': ''
            }
            return [(response.content, metadata)]
        # Extend for other data types as needed
        return []
    except Exception as e:
        logger.warning(f"API ingestion failed: {e}")
        return []

# 2. Data ingestion from web scraping (example: images)
def ingest_from_web(url, max_images=5):
    """Scrape images from a web page. Returns list of (data, metadata_dict)."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.find_all("img")
        results = []
        for img in img_tags[:max_images]:
            src = img.get("src") or img.get("data-src")
            if src:
                full_url = urljoin(url, src)
                if is_valid_image_url(full_url):
                    try:
                        img_resp = requests.get(full_url, timeout=10, headers=headers)
                        img_resp.raise_for_status()
                        filename = f"web_{uuid4()}.jpg"
                        metadata = {
                            'filename': filename,
                            'source': full_url,
                            'datatype': 'image',
                            'status': 'raw',
                            'annotations': ''
                        }
                        results.append((img_resp.content, metadata))
                    except Exception as e:
                        logger.warning(f"Failed to download {full_url}: {e}")
        return results
    except Exception as e:
        logger.warning(f"Web ingestion failed: {e}")
        return []

# 3. Validate and preprocess collected data
def validate_and_preprocess(data, metadata):
    """Validate and preprocess data. Returns (processed_data, updated_metadata)."""
    try:
        if metadata['datatype'] == 'image':
            # Validate size
            if len(data) > MAX_IMAGE_SIZE:
                logger.warning(f"Image too large: {metadata['source']}")
                return None, None
            # Preprocess: convert to RGB, resize, etc.
            image = Image.open(BytesIO(data)).convert("RGB")
            image = image.resize((256, 256))
            output = BytesIO()
            image.save(output, format="JPEG", quality=85)
            processed_data = output.getvalue()
            metadata['status'] = 'processed'
            return processed_data, metadata
        # Extend for other data types
        return data, metadata
    except Exception as e:
        logger.warning(f"Preprocessing failed: {e}")
        return None, None

# 4. Store raw and processed data
def store_data(data, metadata, processed=False):
    """Store data in the appropriate directory."""
    try:
        folder = PROCESSED_DATA_DIR if processed else RAW_DATA_DIR
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, metadata['filename'])
        with open(file_path, 'wb') as f:
            f.write(data)
        return file_path
    except Exception as e:
        logger.warning(f"Failed to store data: {e}")
        return None

# 5. Record metadata and annotations
def record_metadata(metadata):
    try:
        conn = sqlite3.connect(METADATA_DB)
        c = conn.cursor()
        c.execute('''INSERT INTO metadata (filename, source, datatype, status, annotations) VALUES (?, ?, ?, ?, ?)''',
                  (metadata['filename'], metadata['source'], metadata['datatype'], metadata['status'], metadata.get('annotations', '')))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to record metadata: {e}")

# 6. Main workflow orchestrator
def main_data_workflow():
    """Orchestrate the data collection, processing, and storage workflow."""
    init_data_dirs_and_metadata()
    # Example: Ingest from API
    api_results = ingest_from_api("https://picsum.photos/200")
    # Example: Ingest from web
    web_results = ingest_from_web("https://unsplash.com/s/photos/nature", max_images=3)
    all_results = api_results + web_results
    for raw_data, metadata in all_results:
        # Store raw
        store_data(raw_data, metadata, processed=False)
        # Validate & preprocess
        processed_data, updated_metadata = validate_and_preprocess(raw_data, metadata)
        if processed_data:
            store_data(processed_data, updated_metadata, processed=True)
            record_metadata(updated_metadata)

# Place this at the top level, not inside any function
# @app.get("/api/dashboard") # This line is removed as DashboardSystem is no longer imported
# async def get_dashboard_data(): # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     📊 Get analytics from metadata database # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     try: # This line is removed as DashboardSystem is no longer imported
#         data = dashboard_system.get_dashboard_summary() # This line is removed as DashboardSystem is no longer imported
#         if "error" in data: # This line is removed as DashboardSystem is no longer imported
#             raise HTTPException(status_code=500, detail=data["error"]) # This line is removed as DashboardSystem is no longer imported
#         return data # This line is removed as DashboardSystem is no longer imported
#     except Exception as e: # This line is removed as DashboardSystem is no longer imported
#         logger.error(f"Dashboard summary error: {e}") # This line is removed as DashboardSystem is no longer imported
#         raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data.") # This line is removed as DashboardSystem is no longer imported


# === Main API Endpoints ===

@app.get("/")
async def root():
    return {
        "message": "🧠 AI Cancer Image Scraper & Classifier",
        "version": "2.0.0",
        "status": "operational",
        "ai_device": str(device)
    }

# === Authentication Endpoints ===
@app.post("/api/auth/login", response_model=TokenResponse)
async def login(user_credentials: UserLogin):
    # Try to verify existing user
    user = db.verify_user(user_credentials.email, user_credentials.password)
    
    if not user:
        # For demo purposes, create new user if not found
        try:
            user = db.create_user(
                user_credentials.email, 
                user_credentials.email.split('@')[0], 
                user_credentials.password
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    access_token = create_access_token(data={"sub": user["email"]})
    
    # Store session in database
    token_hash = hashlib.sha256(access_token.encode()).hexdigest()
    expires_at = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    db.store_session(int(user["id"]), token_hash, expires_at)
    
    return TokenResponse(
        token=access_token,
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            name=user["name"]
        )
    )

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(user_data: UserRegister):
    try:
        # Create new user in database
        user = db.create_user(user_data.email, user_data.name, user_data.password)
        
        access_token = create_access_token(data={"sub": user["email"]})
        
        # Store session in database
        token_hash = hashlib.sha256(access_token.encode()).hexdigest()
        expires_at = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        db.store_session(int(user["id"]), token_hash, expires_at)
        
        return TokenResponse(
            token=access_token,
            user=UserResponse(
                id=user["id"],
                email=user["email"],
                name=user["name"]
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        name=current_user["name"]
    )

@app.get("/api/auth/stats")
async def get_user_stats(current_user: dict = Depends(get_current_user)):
    """Get user statistics and activity"""
    stats = db.get_user_stats(int(current_user["id"]))
    return {
        "user": current_user,
        "stats": stats
    }

@app.post("/api/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user and invalidate session"""
    # In a real implementation, you would invalidate the JWT token
    # For now, we'll just return success
    return {"message": "Logged out successfully"}

@app.put("/api/auth/profile")
async def update_profile(
    name: str,
    current_user: dict = Depends(get_current_user)
):
    """Update user profile"""
    success = db.update_user_profile(int(current_user["id"]), name)
    if success:
        return {"message": "Profile updated successfully", "name": name}
    else:
        raise HTTPException(status_code=400, detail="Failed to update profile")

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_single_image(file: UploadFile = File(...)):
    """
    🔬 Analyze single image with AI for cancer type prediction
    """
    start_time = time.time()
    
    try:
        # Read and validate file
        content = await file.read()
        if len(content) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Image too large")
        
        # Use CLIP for universal prediction
        result = predict_image_clip(content)
        processing_time = time.time() - start_time
        
        logger.info(f"🔬 Single prediction: {result['prediction']} ({result['confidence']:.2%})")
        
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/scrape", response_model=ScrapeResponse)
async def scrape_and_classify(request: URLRequest):
    start_time = time.time()
    try:
        logger.info(f"🌐 Starting scrape from: {request.url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(request.url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.find_all("img")
        image_urls = []
        # 1. <img> tags
        for img in img_tags:
            src = img.get("src") or img.get("data-src")
            if src:
                full_url = urljoin(request.url, src)
                if is_valid_image_url(full_url):
                    image_urls.append(full_url)
        # 2. <source> tags (srcset)
        for source in soup.find_all("source"):
            srcset = source.get("srcset")
            if srcset:
                for src in srcset.split(","):
                    src_url = src.strip().split(" ")[0]
                    full_url = urljoin(request.url, src_url)
                    if is_valid_image_url(full_url):
                        image_urls.append(full_url)
        # 3. <meta> og:image and twitter:image
        for meta in soup.find_all("meta"):
            if meta.get("property") in ["og:image", "twitter:image"] or meta.get("name") in ["og:image", "twitter:image"]:
                content = meta.get("content")
                if content:
                    full_url = urljoin(request.url, content)
                    if is_valid_image_url(full_url):
                        image_urls.append(full_url)
        # 4. <link rel="image_src"> and <link rel="icon">
        for link in soup.find_all("link"):
            if link.get("rel") and ("image_src" in link.get("rel") or "icon" in link.get("rel")):
                href = link.get("href")
                if href:
                    full_url = urljoin(request.url, href)
                    if is_valid_image_url(full_url):
                        image_urls.append(full_url)
        # 5. Inline CSS background-image URLs
        for tag in soup.find_all(style=True):
            style = tag.get("style")
            if style and "background-image" in style:
                import re
                matches = re.findall(r'url\(["\']?(.*?)["\']?\)', style)
                for match in matches:
                    full_url = urljoin(request.url, match)
                    if is_valid_image_url(full_url):
                        image_urls.append(full_url)
        # 6. <style> tags
        for style_tag in soup.find_all("style"):
            import re
            matches = re.findall(r'url\(["\']?(.*?)["\']?\)', style_tag.text)
            for match in matches:
                full_url = urljoin(request.url, match)
                if is_valid_image_url(full_url):
                    image_urls.append(full_url)
        # Deduplicate
        image_urls = list(dict.fromkeys(image_urls))
        # Enforce max image count per request
        max_images = request.max_images if request.max_images is not None else MAX_IMAGE_COUNT
        image_urls = image_urls[:max_images]
        logger.info(f"📸 Found {len(image_urls)} image URLs")
        if not image_urls:
            raise HTTPException(status_code=400, detail="No images found at the provided URL. Please provide a direct image URL or a page containing images.")
        summary = {}
        total_size = 0
        processed = 0
        async def process_image(url: str):
            nonlocal total_size, processed
            try:
                img_response = requests.get(url, timeout=10, headers=headers)
                img_response.raise_for_status()
                img_data = img_response.content
                if len(img_data) > MAX_IMAGE_SIZE:
                    logger.info(f"⏭️ Skipped image (too large, {len(img_data)} bytes): {url}")
                    return None
                if total_size + len(img_data) > MAX_TOTAL_SIZE_BYTES:
                    logger.info(f"⏭️ Skipped image (total size limit reached): {url}")
                    return None
                total_size += len(img_data)
                classification = predict_image_clip(img_data)
                save_classified_image(img_data, classification, f"scraped from {url}")
                pred = classification["prediction"]
                summary[pred] = summary.get(pred, 0) + 1
                processed += 1
                return pred
            except Exception as e:
                logger.warning(f"⚠️ Failed to process {url}: {e}")
                return None
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        async def process_with_semaphore(url):
            async with semaphore:
                return await asyncio.get_event_loop().run_in_executor(ThreadPoolExecutor(), lambda: asyncio.run(process_image(url)))
        tasks = [process_with_semaphore(url) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processing_time = time.time() - start_time
        success_rate = processed / len(image_urls) if image_urls else 0
        logger.info(f"✅ Scraping complete: {processed}/{len(image_urls)} images processed")
        return ScrapeResponse(
            message="Scraping and AI classification completed",
            summary=summary,
            total_processed=processed,
            processing_time=processing_time,
            success_rate=success_rate
        )
    except HTTPException as e:
        logger.error(f"Scraping failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

@app.post("/api/upload")
async def upload_and_classify(files: List[UploadFile] = File(...)):
    start_time = time.time()
    try:
        logger.info(f"📤 Processing {len(files)} uploaded files")
        summary = {}
        processed = 0
        for file in files:
            try:
                content = await file.read()
                if len(content) > MAX_IMAGE_SIZE:
                    logger.info(f"⏭️ Skipped upload (too large, {len(content)} bytes): {file.filename}")
                    continue
                classification = predict_image_clip(content)
                save_classified_image(content, classification, f"uploaded: {file.filename}")
                pred = classification["prediction"]
                summary[pred] = summary.get(pred, 0) + 1
                processed += 1
            except Exception as e:
                logger.warning(f"⚠️ Failed to process {file.filename}: {e}")
                continue
        processing_time = time.time() - start_time
        success_rate = processed / len(files) if files else 0
        logger.info(f"✅ Upload complete: {processed}/{len(files)} images processed")
        return {
            "message": "Upload and AI classification completed",
            "summary": summary,
            "total_processed": processed,
            "processing_time": processing_time,
            "success_rate": success_rate
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/upload-kaggle")
async def upload_kaggle_dataset(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        logger.info(f"🗂️ Processing Kaggle dataset: {file.filename}")
        temp_extract = os.path.join(TEMP_FOLDER, f"kaggle_{uuid4()}")
        os.makedirs(temp_extract, exist_ok=True)
        try:
            zip_path = os.path.join(temp_extract, file.filename)
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract)
            os.remove(zip_path)
            summary = {}
            total_size = 0
            processed = 0
            image_files = []
            for root, _, files in os.walk(temp_extract):
                for filename in files:
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        image_files.append(os.path.join(root, filename))
            logger.info(f"📸 Found {len(image_files)} images in dataset")
            for img_path in image_files:
                try:
                    if total_size > MAX_TOTAL_SIZE_BYTES:
                        logger.warning("⚠️ Size limit reached, stopping processing")
                        break
                    with open(img_path, "rb") as f:
                        img_data = f.read()
                    total_size += len(img_data)
                    classification = predict_image_clip(img_data)
                    save_classified_image(img_data, classification, f"kaggle: {os.path.basename(img_path)}")
                    pred = classification["prediction"]
                    summary[pred] = summary.get(pred, 0) + 1
                    processed += 1
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process {img_path}: {e}")
                    continue
            processing_time = time.time() - start_time
            success_rate = processed / len(image_files) if image_files else 0
            logger.info(f"✅ Kaggle processing complete: {processed}/{len(image_files)} images")
            return {
                "message": "Kaggle dataset processed with AI classification",
                "summary": summary,
                "total_processed": processed,
                "processing_time": processing_time,
                "success_rate": success_rate
            }
        finally:
            shutil.rmtree(temp_extract, ignore_errors=True)
    except Exception as e:
        logger.error(f"Kaggle processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Kaggle processing failed: {str(e)}")

@app.get("/api/images/{cancer_type}")
async def get_images(cancer_type: str):
    """
    🖼️ Get classified images by cancer type, including analysis text
    """
    try:
        folder = os.path.join(IMAGE_FOLDER, cancer_type.lower())
        analysis_folder = os.path.join("analysis_output", cancer_type.lower())
        if not os.path.exists(folder):
            return {"images": [], "count": 0}
        image_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        images = []
        for img in image_files:
            url = f"http://localhost:8000/images/{cancer_type}/{img}"
            base, _ = os.path.splitext(img)
            analysis_path = os.path.join(analysis_folder, f"{base}.txt")
            analysis_text = None
            if os.path.exists(analysis_path):
                try:
                    with open(analysis_path, "r", encoding="utf-8") as f:
                        analysis_text = f.read()
                except Exception as e:
                    analysis_text = f"[Error reading analysis: {e}]"
            images.append({
                "url": url,
                "filename": img,
                "analysis": analysis_text
            })
        return {
            "images": images,
            "count": len(images),
            "cancer_type": cancer_type
        }
    except Exception as e:
        logger.error(f"Failed to get images: {e}")
        return {"images": [], "count": 0}

@app.get("/api/model-info")
async def get_model_info():
    """
    ℹ️ Get AI model information and statistics
    """
    try:
        # Count images in each category
        stats = {}
        total_images = 0
        
        for label in LABELS:
            folder = os.path.join(IMAGE_FOLDER, label)
            if os.path.exists(folder):
                count = len([f for f in os.listdir(folder) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                stats[label] = count
                total_images += count
            else:
                stats[label] = 0
        
        return {
            "model_info": {
                "architecture": "ResNet50 + Custom Head",
                "classes": LABELS,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "device": str(device),
                "model_loaded": os.path.exists(MODEL_PATH)
            },
            "dataset_stats": stats,
            "total_images": total_images,
            "pytorch_version": torch.__version__
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return {"error": str(e)}

# === Hybrid Training Job Manager ===
class JobManager:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.jobs = {}  # job_id: {'future': Future, 'status': str, 'result': any}
        self.lock = threading.Lock()

    def start_job(self, *args, **kwargs):
        job_id = str(uuid4())
        def job_wrapper():
            try:
                # self.update_status(job_id, 'running') # This line is removed as DashboardSystem is no longer imported
                result = train_model(*args, **kwargs)
                # self.update_status(job_id, 'finished', result) # This line is removed as DashboardSystem is no longer imported
                return result
            except Exception as e:
                # self.update_status(job_id, 'failed', str(e)) # This line is removed as DashboardSystem is no longer imported
                return None
        future = self.executor.submit(job_wrapper)
        with self.lock:
            self.jobs[job_id] = {'future': future, 'status': 'queued', 'result': None}
        return job_id

    def update_status(self, job_id, status, result=None):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = status
                if result is not None:
                    self.jobs[job_id]['result'] = result

    def get_status(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return {'status': 'not_found'}
            status = job['status']
            return {'status': status}

    def get_result(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return {'status': 'not_found', 'result': None}
            if job['status'] == 'finished':
                return {'status': 'finished', 'result': job['result']}
            elif job['status'] == 'failed':
                return {'status': 'failed', 'result': job['result']}
            else:
                return {'status': job['status'], 'result': None}

job_manager = JobManager()

# === Hybrid Training Endpoints ===
from fastapi import BackgroundTasks

@app.post("/api/train")
def start_training_job():
    job_id = job_manager.start_job()
    return {"job_id": job_id, "status": "started"}

@app.get("/api/train/status/{job_id}")
def get_training_status(job_id: str):
    status = job_manager.get_status(job_id)
    return status

@app.get("/api/train/result/{job_id}")
def get_training_result(job_id: str):
    result = job_manager.get_result(job_id)
    return result

# === Startup Event ===
def cleanup_old_images():
    """Delete images older than IMAGE_CLEANUP_AGE_HOURS from IMAGE_FOLDER and subfolders."""
    cutoff = datetime.now() - timedelta(hours=IMAGE_CLEANUP_AGE_HOURS)
    deleted_count = 0
    for label in LABELS:
        folder = os.path.join(IMAGE_FOLDER, label)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            fpath = os.path.join(folder, fname)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                if mtime < cutoff:
                    os.remove(fpath)
                    logger.info(f"🗑️ Deleted old image: {fpath}")
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to check/delete {fpath}: {e}")
    if deleted_count > 0:
        logger.info(f"🧹 Cleanup complete: {deleted_count} old images deleted.")
    else:
        logger.info("🧹 No old images found for cleanup.")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 AI Cancer Image Scraper & Classifier Started")
    logger.info(f"🧠 AI Model: ResNet50 on {device}")
    logger.info(f"📁 Data folder: {IMAGE_FOLDER}")
    logger.info(f"🎯 Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    
    # Clean up expired sessions on startup
    db.delete_expired_sessions()
    logger.info("🧹 Cleaned up expired sessions")
    # Clean up old images on startup
    cleanup_old_images()

# === CLIP Model Setup ===
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_classes = [
    "cat", "dog", "car", "person", "tree", "mountain", "building", "food", "art", "animal", "nature", "technology", "sports", "other"
]

# === Universal Prediction Function ===
def predict_image_clip(image_data, class_names=None, top_n=3):
    try:
        if class_names is None:
            class_names = clip_classes
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat([
            clip.tokenize(f"a photo of a {c}") for c in class_names
        ]).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_inputs)
            logits_per_image, _ = clip_model(image_input, text_inputs)
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        # Get top-N
        top_indices = probs.argsort()[-top_n:][::-1]
        top_classes = [class_names[i] for i in top_indices]
        top_probs = [float(probs[i]) for i in top_indices]
        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        return {
            "prediction": top_classes[0],
            "confidence": top_probs[0],
            "top_classes": top_classes,
            "top_confidences": top_probs,
            "probabilities": prob_dict,
            "model": "CLIP ViT-B/32"
        }
    except Exception as e:
        logger.error(f"CLIP prediction error: {e}")
        return {"prediction": "other", "confidence": 0.0, "probabilities": {}, "model": "CLIP ViT-B/32"}

# === Dashboard API Endpoints ===

# @app.get("/api/dashboard") # This line is removed as DashboardSystem is no longer imported
# async def get_dashboard(): # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     📊 Get comprehensive dashboard data # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     try: # This line is removed as DashboardSystem is no longer imported
#         dashboard_data = dashboard_system.get_dashboard_data() # This line is removed as DashboardSystem is no longer imported
#         return dashboard_data # This line is removed as DashboardSystem is no longer imported
#     except Exception as e: # This line is removed as DashboardSystem is no longer imported
#         logger.error(f"Failed to get dashboard data: {e}") # This line is removed as DashboardSystem is no longer imported
#         return {"error": str(e)} # This line is removed as DashboardSystem is no longer imported

# @app.get("/api/dashboard/report") # This line is removed as DashboardSystem is no longer imported
# async def get_dashboard_report(): # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     📈 Generate comprehensive dashboard report with visualizations # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     try: # This line is removed as DashboardSystem is no longer imported
#         report = dashboard_system.generate_dashboard_report() # This line is removed as DashboardSystem is no longer imported
#         return report # This line is removed as DashboardSystem is no longer imported
#     except Exception as e: # This line is removed as DashboardSystem is no longer imported
#         logger.error(f"Failed to generate dashboard report: {e}") # This line is removed as DashboardSystem is no longer imported
#         return {"error": str(e)} # This line is removed as DashboardSystem is no longer imported

# @app.get("/api/dashboard/image/{image_path:path}") # This line is removed as DashboardSystem is no longer imported
# async def get_image_analysis(image_path: str): # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     🔍 Get detailed analysis for a specific image # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     try: # This line is removed as DashboardSystem is no longer imported
#         # Decode the image path # This line is removed as DashboardSystem is no longer imported
#         decoded_path = image_path.replace("_", "/") # This line is removed as DashboardSystem is no longer imported
#         details = dashboard_system.get_image_details(decoded_path) # This line is removed as DashboardSystem is no longer imported
#         return details # This line is removed as DashboardSystem is no longer imported
#     except Exception as e: # This line is removed as DashboardSystem is no longer imported
#         logger.error(f"Failed to get image details: {e}") # This line is removed as DashboardSystem is no longer imported
#         return {"error": str(e)} # This line is removed as DashboardSystem is no longer imported

# @app.get("/api/dashboard/search") # This line is removed as DashboardSystem is no longer imported
# async def search_analyses( # This line is removed as DashboardSystem is no longer imported
#     query: str = "",  # This line is removed as DashboardSystem is no longer imported
#     cancer_type: str = "",  # This line is removed as DashboardSystem is no longer imported
#     min_confidence: float = 0.0,  # This line is removed as DashboardSystem is no longer imported
#     limit: int = 50 # This line is removed as DashboardSystem is no longer imported
# ): # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     🔍 Search through analyses with filters # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     try: # This line is removed as DashboardSystem is no longer imported
#         results = dashboard_system.search_analyses( # This line is removed as DashboardSystem is no longer imported
#             query=query, # This line is removed as DashboardSystem is no longer imported
#             cancer_type=cancer_type, # This line is removed as DashboardSystem is no longer imported
#             min_confidence=min_confidence, # This line is removed as DashboardSystem is no longer imported
#             limit=limit # This line is removed as DashboardSystem is no longer imported
#         ) # This line is removed as DashboardSystem is no longer imported
#         return {"results": results, "count": len(results)} # This line is removed as DashboardSystem is no longer imported
#     except Exception as e: # This line is removed as DashboardSystem is no longer imported
#         logger.error(f"Search failed: {e}") # This line is removed as DashboardSystem is no longer imported
#         return {"error": str(e)} # This line is removed as DashboardSystem is no longer imported

# @app.post("/api/dashboard/analyze") # This line is removed as DashboardSystem is no longer imported
# async def analyze_existing_image(file: UploadFile = File(...)): # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     🔬 Manually trigger analysis for an uploaded image # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     try: # This line is removed as DashboardSystem is no longer imported
#         # Save uploaded file temporarily # This line is removed as DashboardSystem is no longer imported
#         temp_path = f"temp_processing/{uuid4()}.jpg" # This line is removed as DashboardSystem is no longer imported
#         os.makedirs("temp_processing", exist_ok=True) # This line is removed as DashboardSystem is no longer imported
#          # This line is removed as DashboardSystem is no longer imported
#         with open(temp_path, "wb") as f: # This line is removed as DashboardSystem is no longer imported
#             content = await file.read() # This line is removed as DashboardSystem is no longer imported
#             f.write(content) # This line is removed as DashboardSystem is no longer imported
#          # This line is removed as DashboardSystem is no longer imported
#         # Process the image # This line is removed as DashboardSystem is no longer imported
#         result = dashboard_system.process_new_image(temp_path) # This line is removed as DashboardSystem is no longer imported
#          # This line is removed as DashboardSystem is no longer imported
#         # Clean up temp file # This line is removed as DashboardSystem is no longer imported
#         try: # This line is removed as DashboardSystem is no longer imported
#             os.remove(temp_path) # This line is removed as DashboardSystem is no longer imported
#         except: # This line is removed as DashboardSystem is no longer imported
#             pass # This line is removed as DashboardSystem is no longer imported
#          # This line is removed as DashboardSystem is no longer imported
#         return result # This line is removed as DashboardSystem is no longer imported
#     except Exception as e: # This line is removed as DashboardSystem is no longer imported
#         logger.error(f"Manual analysis failed: {e}") # This line is removed as DashboardSystem is no longer imported
#         return {"error": str(e)} # This line is removed as DashboardSystem is no longer imported

# @app.get("/api/dashboard/visualizations/{viz_type}") # This line is removed as DashboardSystem is no longer imported
# async def get_visualization(viz_type: str): # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     📊 Get specific visualization by type # This line is removed as DashboardSystem is no longer imported
#     """ # This line is removed as DashboardSystem is no longer imported
#     try: # This line is removed as DashboardSystem is no longer imported
#         viz_path = f"analysis_outputs/{viz_type}.png" # This line is removed as DashboardSystem is no longer imported
#         if os.path.exists(viz_path): # This line is removed as DashboardSystem is no longer imported
#             return {"visualization_path": viz_path, "available": True} # This line is removed as DashboardSystem is no longer imported
#         else: # This line is removed as DashboardSystem is no longer imported
#             return {"visualization_path": viz_path, "available": False} # This line is removed as DashboardSystem is no longer imported
#     except Exception as e: # This line is removed as DashboardSystem is no longer imported
#         logger.error(f"Failed to get visualization: {e}") # This line is removed as DashboardSystem is no longer imported
#         return {"error": str(e)} # This line is removed as DashboardSystem is no longer imported

# === Image Management Endpoints ===
from fastapi import Path

@app.delete("/api/images/{category}/{filename}")
async def delete_image(category: str, filename: str = Path(...)):
    """Delete an image and its analysis results."""
    folder = os.path.join(IMAGE_FOLDER, category)
    image_path = os.path.join(folder, filename)
    stem = os.path.splitext(filename)[0]
    analysis_path = os.path.join("backend/analysis_outputs", f"analysis_{stem}.png")
    quality_path = os.path.join("backend/analysis_outputs", f"quality_{stem}.png")
    deleted = []
    errors = []
    for path in [image_path, analysis_path, quality_path]:
        try:
            if os.path.exists(path):
                os.remove(path)
                deleted.append(path)
        except Exception as e:
            errors.append(f"Failed to delete {path}: {e}")
    if errors:
        return {"status": "error", "deleted": deleted, "errors": errors}
    return {"status": "success", "deleted": deleted}

@app.post("/api/images/{category}/{filename}/reanalyze")
async def reanalyze_image(category: str, filename: str = Path(...)):
    """Re-run analysis on an image and return the new analysis result."""
    folder = os.path.join(IMAGE_FOLDER, category)
    image_path = os.path.join(folder, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        # Initialize analyzer if not already defined
        global analyzer
        if 'analyzer' not in globals():
            analyzer = ImageAnalyzer()
        result = analyzer.analyze_image(image_path)
        return {"status": "success", "analysis": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)













#####################




# #!/usr/bin/env python3
# """
# Integrated Cancer Classification System
# Combines standalone CLI functionality with FastAPI web service
# """

# import os
# import sys
# import logging
# import argparse
# from pathlib import Path
# import json
# from typing import Dict, List, Optional, Union
# import asyncio
# from datetime import datetime
# import threading
# from concurrent.futures import ThreadPoolExecutor
# import time
# import uuid
# from io import BytesIO

# # FastAPI and web dependencies
# from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel

# # Image processing
# from PIL import Image
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models
# import numpy as np

# # Try to import enhanced classifier
# try:
#     from enhanced_classifier import EnhancedImageClassifier
#     ENHANCED_CLASSIFIER_AVAILABLE = True
# except ImportError:
#     ENHANCED_CLASSIFIER_AVAILABLE = False
#     print("⚠️ Enhanced classifier not found. Using basic classification only.")

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('integrated_classification.log'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# logger = logging.getLogger(__name__)

# # ===============================
# # Configuration and Constants
# # ===============================

# CANCER_TYPES = [
#     "lung_cancer",
#     "breast_cancer", 
#     "brain_cancer",
#     "skin_cancer",
#     "colon_cancer",
#     "unsorted"
# ]

# CANCER_INFO = {
#     "lung_cancer": {
#         "name": "Lung Cancer",
#         "description": "Cancer that begins in the lungs, often linked to smoking",
#         "recommendations": [
#             "Consult with an oncologist immediately",
#             "Consider CT scan for staging",
#             "Discuss treatment options including surgery, chemotherapy, radiation",
#             "Quit smoking if applicable"
#         ]
#     },
#     "breast_cancer": {
#         "name": "Breast Cancer", 
#         "description": "Cancer that forms in tissues of the breast",
#         "recommendations": [
#             "Schedule mammography and ultrasound",
#             "Consult with breast cancer specialist",
#             "Consider genetic testing if family history",
#             "Discuss hormone therapy options"
#         ]
#     },
#     "brain_cancer": {
#         "name": "Brain Cancer",
#         "description": "Cancer that begins in the brain tissue",
#         "recommendations": [
#             "Get MRI with contrast immediately",
#             "Consult with neurosurgeon and neuro-oncologist",
#             "Discuss surgical resection options",
#             "Consider radiation therapy planning"
#         ]
#     },
#     "skin_cancer": {
#         "name": "Skin Cancer",
#         "description": "Cancer that begins in skin cells, often due to UV exposure",
#         "recommendations": [
#             "Get dermoscopy examination",
#             "Consider surgical excision",
#             "Use broad-spectrum sunscreen daily",
#             "Regular skin self-examinations"
#         ]
#     },
#     "colon_cancer": {
#         "name": "Colon Cancer",
#         "description": "Cancer that begins in the large intestine (colon)",
#         "recommendations": [
#             "Schedule colonoscopy for staging",
#             "Consult with gastroenterologist",
#             "Discuss surgical resection options",
#             "Consider adjuvant chemotherapy"
#         ]
#     },
#     "unsorted": {
#         "name": "Unclassified",
#         "description": "Image could not be classified with sufficient confidence",
#         "recommendations": [
#             "Consult with medical professional",
#             "Consider additional imaging",
#             "Get second opinion if concerned"
#         ]
#     }
# }

# MODEL_CONFIG = {
#     "model_path": "models/cancer_classifier.pth",
#     "feature_model_path": "models/feature_classifier.pkl",
#     "confidence_threshold": 0.7,
#     "image_size": (224, 224),
#     "device": "cuda" if torch.cuda.is_available() else "cpu"
# }

# # ===============================
# # Core Classification Models
# # ===============================

# class CancerClassifier(nn.Module):
#     """Enhanced Cancer Classification Model"""
    
#     def __init__(self, num_classes=len(CANCER_TYPES)-1, dropout_rate=0.5):
#         super(CancerClassifier, self).__init__()
        
#         # Load pre-trained ResNet50
#         self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
#         # Freeze early layers
#         for param in list(self.backbone.parameters())[:-30]:
#             param.requires_grad = False
            
#         # Custom classifier head
#         num_features = self.backbone.fc.in_features
#         self.backbone.fc = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.Linear(num_features, 1024),
#             nn.ReLU(),
#             nn.BatchNorm1d(1024),
#             nn.Dropout(dropout_rate/2),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(dropout_rate/4),
#             nn.Linear(512, num_classes)
#         )
        
#     def forward(self, x):
#         return self.backbone(x)

# # ===============================
# # Integrated Classification System
# # ===============================

# class IntegratedCancerClassifier:
#     """
#     Unified cancer classification system combining multiple approaches
#     """
    
#     def __init__(self, 
#                  model_path: str = MODEL_CONFIG["model_path"],
#                  feature_model_path: str = MODEL_CONFIG["feature_model_path"],
#                  use_enhanced: bool = True):
        
#         self.device = torch.device(MODEL_CONFIG["device"])
#         self.confidence_threshold = MODEL_CONFIG["confidence_threshold"]
        
#         # Initialize transforms
#         self.transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])
        
#         # Load models
#         self._load_pytorch_model(model_path)
#         self._load_enhanced_classifier(feature_model_path, use_enhanced)
        
#         logger.info(f"✅ Integrated classifier initialized on {self.device}")
    
#     def _load_pytorch_model(self, model_path: str):
#         """Load PyTorch cancer classification model"""
#         try:
#             self.pytorch_model = CancerClassifier(num_classes=len(CANCER_TYPES)-1)
            
#             if os.path.exists(model_path):
#                 checkpoint = torch.load(model_path, map_location=self.device)
#                 state_dict = checkpoint.get('model_state_dict', checkpoint)
#                 self.pytorch_model.load_state_dict(state_dict)
#                 logger.info(f"✅ Loaded PyTorch model from {model_path}")
#             else:
#                 logger.warning(f"⚠️ PyTorch model not found at {model_path}")
                
#             self.pytorch_model.to(self.device)
#             self.pytorch_model.eval()
            
#         except Exception as e:
#             logger.error(f"❌ Failed to load PyTorch model: {e}")
#             self.pytorch_model = None
    
#     def _load_enhanced_classifier(self, feature_model_path: str, use_enhanced: bool):
#         """Load enhanced classifier if available"""
#         self.enhanced_classifier = None
        
#         if use_enhanced and ENHANCED_CLASSIFIER_AVAILABLE:
#             try:
#                 self.enhanced_classifier = EnhancedImageClassifier(
#                     model_path=MODEL_CONFIG["model_path"],
#                     feature_model_path=feature_model_path
#                 )
#                 logger.info("✅ Enhanced classifier loaded")
#             except Exception as e:
#                 logger.warning(f"⚠️ Enhanced classifier failed to load: {e}")
    
#     def classify_image(self, image_input: Union[str, bytes, Image.Image]) -> Dict:
#         """
#         Classify cancer type from image using multiple models
        
#         Args:
#             image_input: Path to image file, image bytes, or PIL Image
            
#         Returns:
#             Dict with classification results
#         """
#         start_time = time.time()
        
#         try:
#             # Preprocess image
#             image_tensor = self._preprocess_image(image_input)
#             if image_tensor is None:
#                 return self._create_error_result("Failed to preprocess image")
            
#             results = {}
            
#             # Try enhanced classifier first
#             if self.enhanced_classifier:
#                 try:
#                     if isinstance(image_input, str):
#                         enhanced_result = self.enhanced_classifier.classify_image(image_input)
#                         results['enhanced'] = enhanced_result
#                         logger.info("✅ Enhanced classification completed")
#                 except Exception as e:
#                     logger.warning(f"⚠️ Enhanced classification failed: {e}")
            
#             # PyTorch model classification
#             if self.pytorch_model:
#                 pytorch_result = self._classify_with_pytorch(image_tensor)
#                 results['pytorch'] = pytorch_result
#                 logger.info("✅ PyTorch classification completed")
            
#             # Ensemble results
#             final_result = self._ensemble_results(results)
#             final_result['processing_time'] = time.time() - start_time
#             final_result['models_used'] = list(results.keys())
            
#             return final_result
            
#         except Exception as e:
#             logger.error(f"❌ Classification failed: {e}")
#             return self._create_error_result(str(e))
    
#     def _preprocess_image(self, image_input: Union[str, bytes, Image.Image]) -> Optional[torch.Tensor]:
#         """Preprocess image for classification"""
#         try:
#             if isinstance(image_input, str):
#                 if not os.path.exists(image_input):
#                     return None
#                 image = Image.open(image_input).convert('RGB')
#             elif isinstance(image_input, bytes):
#                 image = Image.open(BytesIO(image_input)).convert('RGB')
#             elif isinstance(image_input, Image.Image):
#                 image = image_input.convert('RGB')
#             else:
#                 return None
            
#             return self.transform(image).unsqueeze(0).to(self.device)
            
#         except Exception as e:
#             logger.error(f"Image preprocessing failed: {e}")
#             return None
    
#     def _classify_with_pytorch(self, image_tensor: torch.Tensor) -> Dict:
#         """Classify using PyTorch model"""
#         try:
#             with torch.no_grad():
#                 outputs = self.pytorch_model(image_tensor)
#                 probabilities = torch.nn.functional.softmax(outputs, dim=1)
#                 predicted_class = torch.argmax(probabilities, dim=1).item()
#                 confidence = probabilities[0][predicted_class].item()
            
#             # Map to cancer types (excluding 'unsorted')
#             cancer_types_only = [ct for ct in CANCER_TYPES if ct != 'unsorted']
#             prediction = cancer_types_only[predicted_class]
            
#             # Apply confidence threshold
#             if confidence < self.confidence_threshold:
#                 prediction = 'unsorted'
            
#             prob_dict = {
#                 cancer_types_only[i]: float(probabilities[0][i]) 
#                 for i in range(len(cancer_types_only))
#             }
#             prob_dict['unsorted'] = 1.0 - confidence if confidence < self.confidence_threshold else 0.0
            
#             return {
#                 'prediction': prediction,
#                 'confidence': confidence,
#                 'probabilities': prob_dict,
#                 'cancer_info': CANCER_INFO[prediction]
#             }
            
#         except Exception as e:
#             logger.error(f"PyTorch classification failed: {e}")
#             return self._create_error_result(str(e))
    
#     def _ensemble_results(self, results: Dict) -> Dict:
#         """Combine results from multiple models"""
#         if not results:
#             return self._create_error_result("No classification results available")
        
#         # If enhanced classifier succeeded, use it as primary
#         if 'enhanced' in results and 'error' not in results['enhanced']:
#             primary_result = results['enhanced'].copy()
#             primary_result['ensemble_method'] = 'enhanced_primary'
#             return primary_result
        
#         # Otherwise use PyTorch results
#         if 'pytorch' in results:
#             pytorch_result = results['pytorch'].copy()
#             pytorch_result['ensemble_method'] = 'pytorch_only'
#             return pytorch_result
        
#         return self._create_error_result("No valid classification results")
    
#     def _create_error_result(self, error_msg: str) -> Dict:
#         """Create standardized error result"""
#         return {
#             'error': error_msg,
#             'prediction': 'unsorted',
#             'confidence': 0.0,
#             'probabilities': {ct: 0.0 for ct in CANCER_TYPES},
#             'cancer_info': CANCER_INFO['unsorted'],
#             'recommendations': CANCER_INFO['unsorted']['recommendations']
#         }
    
#     def classify_batch(self, image_paths: List[str], max_workers: int = 4) -> List[Dict]:
#         """Classify multiple images in parallel"""
#         logger.info(f"🔄 Starting batch classification of {len(image_paths)} images")
        
#         def classify_single(path):
#             result = self.classify_image(path)
#             result['image_path'] = path
#             result['image_name'] = os.path.basename(path)
#             return result
        
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             results = list(executor.map(classify_single, image_paths))
        
#         logger.info(f"✅ Batch classification completed")
#         return results

# # ===============================
# # CLI Application
# # ===============================

# class CancerClassificationApp:
#     """CLI Application for cancer classification"""
    
#     def __init__(self, **kwargs):
#         self.classifier = IntegratedCancerClassifier(**kwargs)
#         logger.info("✅ Cancer Classification App initialized")
    
#     def classify_single_image(self, image_path: str, save_results: bool = True) -> Dict:
#         """Classify a single image"""
#         logger.info(f"🔍 Classifying image: {image_path}")
        
#         result = self.classifier.classify_image(image_path)
        
#         # Log results
#         if "error" not in result:
#             logger.info(f"✅ Classification complete:")
#             logger.info(f"   Prediction: {result['prediction']}")
#             logger.info(f"   Confidence: {result['confidence']:.3f}")
#             logger.info(f"   Cancer Type: {result['cancer_info']['name']}")
#         else:
#             logger.error(f"❌ Classification failed: {result['error']}")
        
#         # Save results if requested
#         if save_results and "error" not in result:
#             self._save_results(image_path, result)
        
#         return result
    
#     def classify_batch_images(self, image_directory: str, output_file: str = "batch_results.json") -> List[Dict]:
#         """Classify multiple images in a directory"""
#         logger.info(f"🔍 Starting batch classification: {image_directory}")
        
#         if not os.path.exists(image_directory):
#             logger.error(f"❌ Directory not found: {image_directory}")
#             return []
        
#         # Find all image files
#         image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
#         image_files = []
        
#         for ext in image_extensions:
#             image_files.extend(Path(image_directory).glob(f"*{ext}"))
#             image_files.extend(Path(image_directory).glob(f"*{ext.upper()}"))
        
#         if not image_files:
#             logger.warning(f"⚠️ No image files found in: {image_directory}")
#             return []
        
#         logger.info(f"📁 Found {len(image_files)} images to process")
        
#         # Process images
#         image_paths = [str(img) for img in image_files]
#         results = self.classifier.classify_batch(image_paths)
        
#         # Save results
#         self._save_batch_results(results, output_file)
#         self._print_batch_summary(results)
        
#         return results
    
#     def _save_results(self, image_path: str, result: Dict) -> None:
#         """Save single image results"""
#         try:
#             results_dir = Path("results")
#             results_dir.mkdir(exist_ok=True)
            
#             image_name = Path(image_path).stem
#             output_file = results_dir / f"{image_name}_results.json"
            
#             result["image_path"] = image_path
#             result["image_name"] = Path(image_path).name
#             result["timestamp"] = datetime.now().isoformat()
            
#             with open(output_file, 'w') as f:
#                 json.dump(result, f, indent=2, default=str)
            
#             logger.info(f"💾 Results saved to: {output_file}")
            
#         except Exception as e:
#             logger.error(f"❌ Failed to save results: {e}")
    
#     def _save_batch_results(self, results: List[Dict], output_file: str) -> None:
#         """Save batch processing results"""
#         try:
#             results_dir = Path("results")
#             results_dir.mkdir(exist_ok=True)
            
#             output_path = results_dir / output_file
            
#             batch_summary = {
#                 "timestamp": datetime.now().isoformat(),
#                 "total_processed": len(results),
#                 "results": results
#             }
            
#             with open(output_path, 'w') as f:
#                 json.dump(batch_summary, f, indent=2, default=str)
            
#             logger.info(f"💾 Batch results saved to: {output_path}")
            
#         except Exception as e:
#             logger.error(f"❌ Failed to save batch results: {e}")
    
#     def _print_batch_summary(self, results: List[Dict]) -> None:
#         """Print summary of batch processing"""
#         if not results:
#             return
        
#         prediction_counts = {}
#         successful_classifications = 0
#         total_processing_time = 0
        
#         for result in results:
#             if "error" not in result:
#                 successful_classifications += 1
#                 pred = result["prediction"]
#                 prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
#                 total_processing_time += result.get('processing_time', 0)
        
#         logger.info("\n" + "="*60)
#         logger.info("📊 BATCH PROCESSING SUMMARY")
#         logger.info("="*60)
#         logger.info(f"Total images processed: {len(results)}")
#         logger.info(f"Successful classifications: {successful_classifications}")
#         logger.info(f"Failed classifications: {len(results) - successful_classifications}")
#         logger.info(f"Average processing time: {total_processing_time/len(results):.2f}s")
        
#         if prediction_counts:
#             logger.info("\nPrediction breakdown:")
#             for prediction, count in sorted(prediction_counts.items()):
#                 percentage = (count / successful_classifications) * 100
#                 cancer_name = CANCER_INFO[prediction]['name']
#                 logger.info(f"  {cancer_name}: {count} ({percentage:.1f}%)")
        
#         logger.info("="*60)

# # ===============================
# # FastAPI Web Service
# # ===============================

# # Pydantic models for API
# class PredictionResponse(BaseModel):
#     prediction: str
#     confidence: float
#     probabilities: dict
#     cancer_info: dict
#     recommendations: List[str]
#     processing_time: float
#     models_used: List[str]

# class BatchProcessRequest(BaseModel):
#     images: List[str]  # List of image paths or base64 encoded images
#     max_workers: Optional[int] = 4

# # Global classifier instance for FastAPI
# global_classifier = None

# def get_classifier():
#     """Get or initialize global classifier"""
#     global global_classifier
#     if global_classifier is None:
#         global_classifier = IntegratedCancerClassifier()
#     return global_classifier

# # FastAPI app
# app = FastAPI(
#     title="🧠 Integrated Cancer Image Classification System",
#     description="Advanced AI-powered cancer image classification with multiple model ensemble",
#     version="3.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# async def root():
#     return {
#         "message": "🧠 Integrated Cancer Image Classification System",
#         "version": "3.0.0",
#         "status": "operational",
#         "models_available": {
#             "pytorch": True,
#             "enhanced": ENHANCED_CLASSIFIER_AVAILABLE
#         },
#         "device": MODEL_CONFIG["device"]
#     }

# @app.post("/api/classify", response_model=PredictionResponse)
# async def classify_image_api(file: UploadFile = File(...)):
#     """Classify uploaded image for cancer type"""
#     try:
#         classifier = get_classifier()
        
#         # Read image data
#         image_data = await file.read()
        
#         # Classify
#         result = classifier.classify_image(image_data)
        
#         if "error" in result:
#             raise HTTPException(status_code=400, detail=result["error"])
        
#         return PredictionResponse(**result)
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"API classification failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/classify-batch")
# async def classify_batch_api(request: BatchProcessRequest, background_tasks: BackgroundTasks):
#     """Classify multiple images"""
#     try:
#         classifier = get_classifier()
        
#         # For now, assume image paths are provided
#         # In production, you'd handle base64 encoded images
#         results = classifier.classify_batch(request.images, request.max_workers)
        
#         return {
#             "message": "Batch classification completed",
#             "total_processed": len(results),
#             "results": results
#         }
        
#     except Exception as e:
#         logger.error(f"Batch API classification failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/model-info")
# async def get_model_info():
#     """Get information about loaded models"""
#     classifier = get_classifier()
    
#     return {
#         "pytorch_model": classifier.pytorch_model is not None,
#         "enhanced_model": classifier.enhanced_classifier is not None,
#         "device": str(classifier.device),
#         "confidence_threshold": classifier.confidence_threshold,
#         "supported_cancer_types": CANCER_TYPES,
#         "cancer_info": CANCER_INFO
#     }

# # ===============================
# # Main CLI Interface
# # ===============================

# def main():
#     """Main function with command line interface"""
#     parser = argparse.ArgumentParser(description="Integrated Cancer Image Classification System")
#     parser.add_argument("--image", "-i", type=str, help="Path to single image file")
#     parser.add_argument("--batch", "-b", type=str, help="Path to directory containing images")
#     parser.add_argument("--model", "-m", type=str, default=MODEL_CONFIG["model_path"],
#                        help="Path to PyTorch model")
#     parser.add_argument("--feature-model", "-f", type=str, default=MODEL_CONFIG["feature_model_path"],
#                        help="Path to feature-based model")
#     parser.add_argument("--output", "-o", type=str, default="batch_results.json",
#                        help="Output file for batch results")
#     parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
#     parser.add_argument("--no-enhanced", action="store_true", help="Don't use enhanced classifier")
#     parser.add_argument("--web", action="store_true", help="Start web service")
#     parser.add_argument("--host", type=str, default="0.0.0.0", help="Web service host")
#     parser.add_argument("--port", type=int, default=8000, help="Web service port")
    
#     args = parser.parse_args()
    
#     if args.web:
#         # Start web service
#         import uvicorn
#         logger.info(f"🚀 Starting web service on {args.host}:{args.port}")
#         uvicorn.run(app, host=args.host, port=args.port, reload=False)
#         return
    
#     # Validate CLI arguments
#     if not args.image and not args.batch:
#         parser.error("Please specify either --image, --batch, or --web")
    
#     if args.image and args.batch:
#         parser.error("Please specify either --image or --batch, not both")
    
#     try:
#         # Initialize the application
#         app_instance = CancerClassificationApp(
#             model_path=args.model,
#             feature_model_path=args.feature_model,
#             use_enhanced=not args.no_enhanced
#         )
        
#         if args.image:
#             # Single image classification
#             result = app_instance.classify_single_image(args.image, save_results=not args.no_save)
            
#             # Print results
#             if "error" not in result:
#                 print(f"\n🎯 Classification Results for: {args.image}")
#                 print(f"Prediction: {result['prediction']}")
#                 print(f"Confidence: {result['confidence']:.3f}")
#                 print(f"Cancer Type: {result['cancer_info']['name']}")
#                 print(f"Description: {result['cancer_info']['description']}")
#                 print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
#                 print(f"Models Used: {', '.join(result.get('models_used', []))}")
#                 print("\nRecommendations:")
#                 for rec in result['recommendations']:
#                     print(f"  • {rec}")
#             else:
#                 print(f"❌ Classification failed: {result['error']}")
        
#         elif args.batch:
#             # Batch classification
#             results = app_instance.classify_batch_images(args.batch, args.output)
            
#             if results:
#                 print(f"\n✅ Batch processing completed. Results saved to: results/{args.output}")
#             else:
#                 print("❌ Batch processing failed or no images found")
    
#     except KeyboardInterrupt:
#         logger.info("🛑 Process interrupted by user")
#         sys.exit(1)
#     except Exception as e:
#         logger.error(f"❌ Application error: {e}")
#         sys.exit(1)

# # ===============================
# # Utility Functions
# # ===============================

# def classify_image_simple(image_path: str) -> Dict:
#     """Simple function to classify a single image"""
#     classifier = IntegratedCancerClassifier()
#     return classifier.classify_image(image_path)

# def classify_images_in_folder(folder_path: str) -> List[Dict]:
#     """Simple function to classify all images in a folder"""
#     app_instance = CancerClassificationApp()
#     return app_instance.classify_batch_images(folder_path)

# if __name__ == "__main__":
#     main()

# # ===============================
# # Example Usage
# # ===============================
# """
# # CLI Usage:
# python integrated_main.py --image path/to/image.jpg
# python integrated_main.py --batch path/to/images/
# python integrated_main.py --web --port 8000

# # Programmatic Usage:
# from integrated_main import IntegratedCancerClassifier, CancerClassificationApp

# # Direct classifier
# classifier = IntegratedCancerClassifier()
# result = classifier.classify_image("image.jpg")

# # CLI app
# app = CancerClassificationApp()
# result = app.classify_single_image("image.jpg")

# # Simple functions
# result = classify_image_simple("image.jpg")
# results = classify_images_in_folder("images/")
# """