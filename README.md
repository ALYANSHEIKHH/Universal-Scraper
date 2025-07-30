# ⚙️ FastAPI AI Universal Image Classifier – Backend Overview
This backend is powered by FastAPI and includes a full pipeline for handling image classification using AI, file management, scraping, token-based auth, model training, and more.

# 📦 Installed Libraries & Modules
# FastAPI core & security
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Data modeling & validation
from pydantic import BaseModel

# JWT authentication
import jwt
from datetime import datetime, timedelta

# Security & hashing
import hashlib

# Database connection
from app.database import db

# Image processing
from PIL import Image

# Web scraping
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests

# File & OS utilities
import os
import shutil
import zipfile
from io import BytesIO
from typing import List, Optional
from pathlib import Path
import csv
import sqlite3
import threading

# Deep Learning & Classification
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import clip
import open_clip

# UUID & Utility
import uuid
from uuid import uuid4
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
import time
import logging

# Project-specific modules
from app.models.training import train_model
from app import config
from app.image_analyzer import ImageAnalyzer


## 🚀 Features

### 🔄 Automated Workflow
- **Image Scraping**: Automatically scrape images from URLs
- **AI Classification**: Multi-method classification (Deep Learning + Feature-based + Rule-based)

  ### 🎯 Classification System
- **Deep Learning**: ResNet50-based classifier
- **Feature-based**: Traditional ML with handcrafted features
- **Rule-based**: Heuristic classification based on image characteristics
- **Ensemble**: Weighted combination of all methods

