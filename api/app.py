"""
Flask API for Smart Bin waste classification.
Production-ready endpoint for real-time waste classification.
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import numpy as np
from PIL import Image
import io
import base64
import logging
from werkzeug.utils import secure_filename

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import WasteClassifier

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize classifier (load model once at startup)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.h5')
classifier = None

def init_classifier():
    """Initialize the waste classifier."""
    global classifier
    try:
        if os.path.exists(MODEL_PATH):
            classifier = WasteClassifier(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.error(f"Model not found at {MODEL_PATH}")
            # For demo purposes, we'll continue without a model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload interface."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Bin - AI Waste Classification</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .result { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .recyclable { color: #28a745; }
            .non-recyclable { color: #dc3545; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <h1>🗂️ Smart Bin: AI Waste Classification</h1>
        <p>Upload an image of waste to get classification and disposal recommendations.</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <input type="file" id="fileInput" name="file" accept="image/*" required>
                <p>Choose an image file (PNG, JPG, JPEG, GIF, BMP)</p>
            </div>
            <button type="submit">Classify Waste</button>
        </form>
        
        <div id="result"></div>
        
        <script>
            document.getElementById('uploadForm').onsubmit = function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a file');
                    return;
                }
                
                formData.append('file', file);
                
                // Show loading message
                document.getElementById('result').innerHTML = '<p>Classifying... Please wait.</p>';
                
                fetch('/api/classify', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const result = data.result;
                        const recyclableClass = result.is_recyclable ? 'recyclable' : 'non-recyclable';
                        const recyclableIcon = result.is_recyclable ? '♻️' : '🗑️';
                        
                        document.getElementById('result').innerHTML = `
                            <div class="result">
                                <h3>${recyclableIcon} Classification Results</h3>
                                <p><strong>Predicted Class:</strong> ${result.predicted_class.toUpperCase()}</p>
                                <p><strong>Confidence:</strong> ${result.confidence_percent.toFixed(1)}%</p>
                                <p><strong>Bin Type:</strong> <span class="${recyclableClass}">${result.bin_type}</span></p>
                                <p><strong>Recommendation:</strong> ${result.message}</p>
                                <p><strong>Disposal Confidence:</strong> ${result.disposal_confidence}</p>
                                
                                <h4>Top Predictions:</h4>
                                <ul>
                                    ${result.top_predictions.map(pred => 
                                        `<li>${pred.class}: ${pred.confidence_percent.toFixed(1)}%</li>`
                                    ).join('')}
                                </ul>
                            </div>
                        `;
                    } else {
                        document.getElementById('result').innerHTML = `
                            <div class="result">
                                <h3>Error</h3>
                                <p>${data.error}</p>
                            </div>
                        `;
                    }
                })
                .catch(error => {
                    document.getElementById('result').innerHTML = `
                        <div class="result">
                            <h3>Error</h3>
                            <p>Failed to classify image: ${error.message}</p>
                        </div>
                    `;
                });
            };
        </script>
    </body>
    </html>
    '''

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """API endpoint for image classification."""
    try:
        # Check if model is loaded
        if classifier is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please ensure the model file exists.'
            }), 500
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, GIF, BMP'
            }), 400
        
        # Read and process image
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid image file: {str(e)}'
            }), 400
        
        # Classify image
        try:
            result = classifier.classify_with_recommendation(image)
            
            return jsonify({
                'success': True,
                'result': result,
                'filename': secure_filename(file.filename)
            })
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Classification failed: {str(e)}'
            }), 500
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'version': '2.0.0'
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get available waste classes."""
    if classifier is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'success': True,
        'classes': classifier.class_names,
        'recyclable_classes': ['cardboard', 'glass', 'metal', 'paper', 'plastic'],
        'non_recyclable_classes': ['trash']
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def server_error(e):
    """Handle server errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# Cleanup function for temporary files
def cleanup_temp_files():
    """Remove temporary uploaded files."""
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp files: {str(e)}")

if __name__ == '__main__':
    # Initialize classifier
    init_classifier()
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_ENV') == 'development'
    )