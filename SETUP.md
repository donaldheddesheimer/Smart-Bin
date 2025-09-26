# 🛠️ Smart Bin Setup Guide

This guide will help you set up the Smart Bin waste classification system on your local machine or server.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: 2GB free space for dependencies and models
- **GPU**: Optional but recommended (CUDA-compatible for faster inference)

### Required Software
- Git
- Python pip
- Virtual environment tool (venv, conda, or virtualenv)

## 🚀 Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/smart-bin.git
cd smart-bin
```

### 2. Create Virtual Environment
```bash
# Using venv (recommended)
python -m venv smart_bin_env
source smart_bin_env/bin/activate  # On Windows: smart_bin_env\Scripts\activate

# Or using conda
conda create -n smart_bin_env python=3.8
conda activate smart_bin_env
```

### 3. Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# For development (includes testing and linting tools)
pip install -r requirements.txt
pip install pytest pytest-cov black flake8
```

### 4. Download or Train Model

#### Option A: Use Pre-trained Model
1. Download the trained model from [releases page](https://github.com/yourusername/smart-bin/releases)
2. Place it in the `models/` directory
3. Rename to `best_model.h5`

#### Option B: Train Your Own Model
```bash
# Prepare your dataset in the following structure:
# data/
# ├── train/
# │   ├── cardboard/
# │   ├── glass/
# │   ├── metal/
# │   ├── paper/
# │   ├── plastic/
# │   └── trash/
# ├── val/
# └── test/

# Run training
python src/training.py
```

### 5. Verify Installation
```bash
# Test the CLI interface
python classify_waste.py --help

# Test with a sample image (if you have one)
python classify_waste.py --image path/to/test/image.jpg --model models/best_model.h5
```

## 📊 Dataset Setup

### Kaggle Dataset (Recommended)
1. Install Kaggle CLI: `pip install kaggle`
2. Set up Kaggle API credentials
3. Download dataset:
```bash
kaggle datasets download -d asdasdasasdas/garbage-classification
unzip garbage-classification.zip -d data/
```

### Custom Dataset
Organize your images in this structure:
```
data/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
├── val/
└── test/
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:
```bash
# Model paths
MODEL_PATH=models/best_model.h5
DATA_PATH=data/

# API configuration
FLASK_ENV=development
API_PORT=5000

# Logging
LOG_LEVEL=INFO
```

### Model Configuration
Update model parameters in `src/model_architecture.py`:
```python
# Image input size
IMAGE_SIZE = (224, 224, 3)

# Number of classes
NUM_CLASSES = 6

# Class names (in order)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
```

## 🧪 Testing Installation

### Run Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test
python -m pytest tests/test_inference.py -v
```

### Manual Testing
```bash
# Test CLI with sample data
python classify_waste.py --image tests/sample_images/plastic_bottle.jpg --model models/best_model.h5 --verbose

# Test API server
python api/app.py
# Navigate to http://localhost:5000 in browser
```

## 📱 API Server Setup

### Local Development
```bash
# Start Flask development server
python api/app.py

# Server will start on http://localhost:5000
# Open browser and navigate to the URL to use the web interface
```

### Production Deployment
```bash
# Using Gunicorn (recommended for production)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api.app:app

# Using Docker
docker build -t smart-bin .
docker run -p 5000:5000 smart-bin
```

## 🐳 Docker Setup

### Build and Run with Docker
```bash
# Create Dockerfile in project root
cat > Dockerfile << EOF
FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p models temp_uploads

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "api/app.py"]
EOF

# Build and run
docker build -t smart-bin .
docker run -p 5000:5000 -v $(pwd)/models:/app/models smart-bin
```

## 🔍 Troubleshooting

### Common Issues

#### Model Loading Error
```bash
Error: Model not found at models/best_model.h5
```
**Solution**: Ensure you have downloaded or trained a model file and placed it in the `models/` directory.

#### Import Error
```bash
ModuleNotFoundError: No module named 'tensorflow'
```
**Solution**: Activate your virtual environment and reinstall dependencies:
```bash
source smart_bin_env/bin/activate
pip install -r requirements.txt
```

#### Memory Error
```bash
ResourceExhaustedError: OOM when allocating tensor
```
**Solution**: Reduce batch size or close other applications to free memory.

#### CUDA Error (if using GPU)
```bash
Could not load dynamic library 'libcudart.so.11.0'
```
**Solution**: Install appropriate CUDA drivers or use CPU-only TensorFlow:
```bash
pip uninstall tensorflow
pip install tensorflow-cpu
```

### Performance Issues

#### Slow Inference
- Use GPU if available
- Reduce image size in preprocessing
- Use TensorFlow Lite for mobile deployment

#### High Memory Usage
- Use model quantization
- Implement batch processing for multiple images
- Clear TensorFlow session after inference

## 📊 Data Preprocessing

### Image Requirements
- **Format**: PNG, JPG, JPEG, GIF, BMP
- **Size**: Any size (will be resized to 224x224)
- **Quality**: Higher quality images generally work better
- **Background**: Clean backgrounds improve accuracy

### Data Augmentation Settings
Modify in `src/data_preprocessing.py`:
```python
# Training augmentation parameters
ROTATION_RANGE = 30        # Degrees
ZOOM_RANGE = 0.3          # Zoom factor
SHIFT_RANGE = 0.2         # Width/height shift
HORIZONTAL_FLIP = True    # Enable horizontal flipping
VERTICAL_FLIP = True      # Enable vertical flipping
```

## 🔧 Advanced Configuration

### Model Training Parameters
Edit `src/training.py`:
```python
# Training configuration
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
```

### Inference Settings
Modify `src/inference.py`:
```python
# Confidence thresholds
HIGH_CONFIDENCE = 0.8
MEDIUM_CONFIDENCE = 0.5
LOW_CONFIDENCE = 0.3

# Top-K predictions
TOP_K_PREDICTIONS = 3
```

## 📈 Performance Monitoring

### Logging Setup
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_bin.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics Collection
Track model performance:
```python
# In your inference code
import time

start_time = time.time()
result = classifier.predict(image)
inference_time = time.time() - start_time

# Log metrics
logger.info(f"Inference time: {inference_time:.3f}s")
logger.info(f"Predicted class: {result['predicted_class']}")
logger.info(f"Confidence: {result['confidence']:.3f}")
```

## 🚀 Deployment Checklist

### Pre-deployment
- [ ] All tests passing
- [ ] Model file available and tested
- [ ] Dependencies installed correctly
- [ ] Configuration files set up
- [ ] Logging configured
- [ ] Error handling implemented

### Production
- [ ] Use production WSGI server (Gunicorn)
- [ ] Set up reverse proxy (Nginx)
- [ ] Configure SSL/HTTPS
- [ ] Set up monitoring and alerts
- [ ] Implement rate limiting
- [ ] Set up backup strategy

### Security
- [ ] Update dependencies regularly
- [ ] Implement input validation
- [ ] Set up CORS properly
- [ ] Use environment variables for secrets
- [ ] Enable security headers
- [ ] Implement authentication if needed

## 💡 Tips for Best Results

### Image Quality
- Use well-lit, clear images
- Avoid cluttered backgrounds
- Center the waste item in the frame
- Ensure the item takes up most of the image

### Model Performance
- Retrain with domain-specific data if needed
- Use ensemble methods for critical applications
- Monitor confidence scores and flag low-confidence predictions
- Implement feedback loop for continuous improvement

### Deployment
- Use containerization for consistent environments
- Implement health checks
- Set up proper logging and monitoring
- Use load balancing for high traffic
- Cache frequently accessed models

## 📞 Getting Help

If you encounter issues not covered in this guide:

1. **Check the logs** for detailed error messages
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Your system specifications
   - Python/TensorFlow versions
   - Complete error traceback
   - Steps to reproduce

## 🔄 Updates and Maintenance

### Keeping Up to Date
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Update model (if new version available)
# Download new model and replace in models/ directory

# Run tests after updates
python -m pytest tests/
```

### Model Retraining
Consider retraining when:
- New waste categories emerge
- Performance degrades over time
- Domain-specific requirements change
- More training data becomes available