# 🧠 AI Cancer Image Scraper & Analysis Dashboard

A comprehensive AI-powered system for automated image scraping, classification, analysis, and dashboard visualization for medical images.

## 🚀 Features

### 🔄 Automated Workflow
- **Image Scraping**: Automatically scrape images from URLs
- **AI Classification**: Multi-method classification (Deep Learning + Feature-based + Rule-based)
- **Image Analysis**: Comprehensive image analysis with visualizations
- **Dashboard Generation**: Real-time dashboard with statistics and insights

### 📊 Analysis Capabilities
- **Color Analysis**: RGB/HSV analysis, dominant color extraction
- **Texture Analysis**: Local Binary Patterns, gradient analysis
- **Edge Detection**: Canny edge detection, contour analysis
- **Quality Assessment**: Sharpness, noise, contrast metrics
- **Medical Features**: Tissue coverage, feature density analysis

### 🎯 Classification System
- **Deep Learning**: ResNet50-based classifier
- **Feature-based**: Traditional ML with handcrafted features
- **Rule-based**: Heuristic classification based on image characteristics
- **Ensemble**: Weighted combination of all methods

### 📈 Dashboard Features
- **Real-time Statistics**: Live updates of analysis results
- **Interactive Visualizations**: Charts and graphs for insights
- **Search & Filter**: Advanced search capabilities
- **Image Details**: Detailed analysis for individual images
- **Recommendations**: AI-generated recommendations based on results

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Image Input   │───▶│  Classification  │───▶│   Analysis      │
│   (Scraper/     │    │   (Multi-Method) │    │   (Comprehensive)│
│    Upload)      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │◀───│   Database      │◀───│  Visualizations │
│   (Frontend)    │    │   (SQLite)      │    │   (Matplotlib)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
Cancer-Image-Scrapper/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application
│   │   ├── image_analyzer.py       # Image analysis module
│   │   ├── enhanced_classifier.py  # Multi-method classifier
│   │   ├── dashboard_system.py     # Dashboard backend
│   │   ├── models/
│   │   │   └── training.py         # Model training
│   │   └── ...
│   ├── data/                       # Image storage
│   ├── models/                     # Trained models
│   ├── analysis_outputs/           # Generated visualizations
│   ├── automation_demo.py          # Demo script
│   └── requirement.txt             # Dependencies
├── frontend/
│   └── my-app/
│       ├── pages/
│       │   ├── dashboard.tsx       # Dashboard frontend
│       │   └── gallery/
│       └── ...
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- pip
- npm

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Cancer-Image-Scrapper
```

2. **Install Python dependencies**
```bash
cd backend
pip install -r requirement.txt
```

3. **Create necessary directories**
```bash
mkdir -p data models analysis_outputs
```

### Frontend Setup

1. **Install Node.js dependencies**
```bash
cd frontend/my-app
npm install
```

2. **Start the development server**
```bash
npm run dev
```

## 🚀 Usage

### Starting the System

1. **Start the backend server**
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Start the frontend**
```bash
cd frontend/my-app
npm run dev
```

3. **Access the application**
- Frontend: http://localhost:3000
- Dashboard: http://localhost:3000/dashboard
- API Documentation: http://localhost:8000/docs

### Running the Automation Demo

```bash
cd backend
python automation_demo.py
```

This will:
1. Download sample images
2. Run the complete analysis workflow
3. Generate visualizations
4. Update the dashboard
5. Display a summary report

## 📡 API Endpoints

### Core Endpoints
- `POST /api/predict` - Single image classification
- `POST /api/scrape` - Scrape and classify images from URL
- `POST /api/upload` - Upload and classify multiple images

### Dashboard Endpoints
- `GET /api/dashboard` - Get dashboard data
- `GET /api/dashboard/report` - Generate comprehensive report
- `GET /api/dashboard/image/{path}` - Get specific image analysis
- `GET /api/dashboard/search` - Search analyses with filters
- `POST /api/dashboard/analyze` - Manual analysis trigger

### Model Endpoints
- `GET /api/model-info` - Model information and statistics
- `POST /api/train` - Start model training
- `GET /api/train/status/{job_id}` - Training status
- `GET /api/train/result/{job_id}` - Training results

## 🔧 Configuration

### Environment Variables
```bash
# Backend configuration
MAX_TOTAL_SIZE_BYTES=500000000  # 500MB limit
MAX_IMAGE_SIZE=10000000         # 10MB per image
CONFIDENCE_THRESHOLD=0.20       # Classification threshold
MAX_CONCURRENT_DOWNLOADS=5      # Concurrent download limit
```

### Model Configuration
- **Deep Learning**: ResNet50 with custom head
- **Feature-based**: Random Forest with 50+ features
- **Rule-based**: Heuristic rules for image characteristics

## 📊 Dashboard Features

### Overview Tab
- System statistics
- Cancer type distribution
- Recent activity summary

### Recent Analyses Tab
- Latest processed images
- Classification results
- Confidence scores
- Recommendations

### Visualizations Tab
- Cancer type distribution charts
- Confidence trend analysis
- Quality metrics visualization
- Color analysis charts

### Statistics Tab
- Historical data
- Performance metrics
- System health indicators

## 🔍 Analysis Details

### Image Analysis Components
1. **Basic Statistics**: Dimensions, aspect ratio, memory usage
2. **Color Analysis**: RGB/HSV means, color diversity, dominant colors
3. **Texture Analysis**: LBP, gradient magnitude, entropy
4. **Edge Analysis**: Canny edges, contour detection, edge density
5. **Quality Assessment**: Sharpness, noise, contrast, brightness
6. **Medical Features**: Tissue coverage, feature density

### Classification Methods
1. **Deep Learning (60% weight)**: ResNet50-based classification
2. **Feature-based (30% weight)**: Traditional ML with handcrafted features
3. **Rule-based (10% weight)**: Heuristic classification

### Visualization Types
- **Main Analysis**: RGB histograms, dominant colors, edge detection
- **Quality Metrics**: Sharpness, noise, contrast, brightness bars
- **Dashboard Charts**: Cancer distribution, confidence trends, statistics

## 🎯 Use Cases

### Medical Research
- Automated image classification for research datasets
- Quality assessment of medical images
- Statistical analysis of image characteristics

### Clinical Support
- Preliminary image screening
- Quality control for medical imaging
- Educational tool for medical professionals

### Data Analysis
- Large-scale image dataset analysis
- Pattern recognition in medical images
- Automated reporting and visualization

## 🔒 Security & Privacy

- **Image Processing**: All processing done locally
- **Data Storage**: SQLite database with local storage
- **No External APIs**: Self-contained analysis system
- **Configurable Limits**: Size and rate limiting

## 🚧 Development

### Adding New Analysis Methods
1. Extend `ImageAnalyzer` class
2. Add new analysis methods
3. Update visualization generation
4. Test with sample images

### Adding New Classification Methods
1. Extend `EnhancedImageClassifier` class
2. Implement new classification logic
3. Update ensemble weights
4. Validate with test data

### Customizing Visualizations
1. Modify `_generate_visualizations` methods
2. Add new chart types
3. Update dashboard frontend
4. Test visualization rendering

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at http://localhost:8000/docs
- Review the demo script for usage examples

## 🔄 Changelog

### Version 2.0.0
- ✨ Added comprehensive image analysis
- ✨ Implemented multi-method classification
- ✨ Created interactive dashboard
- ✨ Added automated workflow
- ✨ Enhanced visualization system

### Version 1.0.0
- 🎉 Initial release with basic scraping and classification

---

**Built with 💙 for AI innovation · UniversalAI © 2025** 