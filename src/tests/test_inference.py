"""
Comprehensive test suite for the Smart Bin inference module.
Tests cover model loading, image preprocessing, predictions, and error handling.
"""
import pytest
import numpy as np
import os
import tempfile
from PIL import Image
import json
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import WasteClassifier, BatchProcessor

class TestWasteClassifier:
    """Test cases for the WasteClassifier class."""
    
    @pytest.fixture
    def mock_model_path(self):
        """Create a temporary model file path."""
        return "mock_model.h5"
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple RGB image
        image = Image.new('RGB', (224, 224), color='red')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            image.save(f.name)
            return f.name
    
    @pytest.fixture
    def mock_classifier(self, mock_model_path):
        """Create a classifier with mocked model loading."""
        with patch('inference.load_model') as mock_load:
            # Mock the loaded model
            mock_model = Mock()
            mock_model.input_shape = (None, 224, 224, 3)
            mock_model.predict.return_value = np.array([[0.1, 0.2, 0.15, 0.25, 0.2, 0.1]])
            mock_load.return_value = mock_model
            
            classifier = WasteClassifier(mock_model_path)
            return classifier
    
    def test_init_with_valid_model(self, mock_model_path):
        """Test classifier initialization with valid model path."""
        with patch('inference.load_model') as mock_load:
            mock_model = Mock()
            mock_model.input_shape = (None, 224, 224, 3)
            mock_load.return_value = mock_model
            
            classifier = WasteClassifier(mock_model_path)
            
            assert classifier.model_path == mock_model_path
            assert classifier.model is not None
            assert classifier.input_shape == (224, 224, 3)
            assert len(classifier.class_names) == 6
            mock_load.assert_called_once()
    
    def test_init_with_custom_class_names(self, mock_model_path):
        """Test classifier initialization with custom class names."""
        with patch('inference.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            custom_classes = ['class1', 'class2', 'class3']
            classifier = WasteClassifier(mock_model_path, custom_classes)
            
            assert classifier.class_names == custom_classes
    
    def test_init_with_invalid_model_path(self):
        """Test classifier initialization with invalid model path."""
        with patch('inference.load_model') as mock_load:
            mock_load.side_effect = Exception("Model not found")
            
            with pytest.raises(RuntimeError, match="Failed to load model"):
                WasteClassifier("invalid_path.h5")
    
    def test_preprocess_image_from_path(self, mock_classifier, sample_image):
        """Test image preprocessing from file path."""
        processed = mock_classifier.preprocess_image(sample_image)
        
        assert processed.shape == (1, 224, 224, 3)
        assert processed.dtype == np.float32
        # Check if values are normalized (MobileNet preprocessing)
        assert np.min(processed) >= -1.0
        assert np.max(processed) <= 1.0
    
    def test_preprocess_image_from_array(self, mock_classifier):
        """Test image preprocessing from numpy array."""
        # Create a sample image array
        image_array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        processed = mock_classifier.preprocess_image(image_array)
        
        assert processed.shape == (1, 224, 224, 3)
        assert processed.dtype == np.float32
    
    def test_preprocess_image_with_invalid_path(self, mock_classifier):
        """Test image preprocessing with invalid file path."""
        with pytest.raises(FileNotFoundError):
            mock_classifier.preprocess_image("nonexistent_image.jpg")
    
    def test_preprocess_image_with_invalid_format(self, mock_classifier):
        """Test image preprocessing with invalid image format."""
        # Create a text file instead of image
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is not an image")
            
            with pytest.raises(RuntimeError):
                mock_classifier.preprocess_image(f.name)
    
    def test_predict_single_image(self, mock_classifier, sample_image):
        """Test single image prediction."""
        result = mock_classifier.predict(sample_image)
        
        # Check result structure
        assert 'predicted_class' in result
        assert 'confidence' in result
        assert 'confidence_percent' in result
        assert 'top_predictions' in result
        assert 'all_probabilities' in result
        
        # Check data types and ranges
        assert isinstance(result['predicted_class'], str)
        assert 0.0 <= result['confidence'] <= 1.0
        assert 0.0 <= result['confidence_percent'] <= 100.0
        assert len(result['top_predictions']) <= 3
        assert len(result['all_probabilities']) == 6
    
    def test_predict_with_custom_top_k(self, mock_classifier, sample_image):
        """Test prediction with custom top-k value."""
        result = mock_classifier.predict(sample_image, top_k=5)
        
        assert len(result['top_predictions']) <= 5
        
        # Check if predictions are sorted by confidence
        confidences = [pred['confidence'] for pred in result['top_predictions']]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_predict_batch(self, mock_classifier, sample_image):
        """Test batch prediction functionality."""
        # Create multiple sample images
        image_paths = [sample_image, sample_image, sample_image]
        
        results = mock_classifier.predict_batch(image_paths)
        
        assert len(results) == 3
        for result in results:
            assert 'predicted_class' in result
            assert 'confidence' in result
    
    def test_predict_batch_with_errors(self, mock_classifier, sample_image):
        """Test batch prediction with some invalid inputs."""
        image_paths = [sample_image, "invalid_path.jpg", sample_image]
        
        results = mock_classifier.predict_batch(image_paths)
        
        assert len(results) == 3
        assert 'error' not in results[0]  # Valid image
        assert 'error' in results[1]      # Invalid path
        assert 'error' not in results[2]  # Valid image
    
    def test_get_recyclable_status(self, mock_classifier):
        """Test recyclability determination."""
        # Test recyclable classes
        recyclable_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
        for class_name in recyclable_classes:
            result = mock_classifier.get_recyclable_status(class_name)
            assert result['is_recyclable'] is True
            assert result['bin_type'] == 'Recycling Bin'
            assert 'recycling' in result['message'].lower()
        
        # Test non-recyclable class
        result = mock_classifier.get_recyclable_status('trash')
        assert result['is_recyclable'] is False
        assert result['bin_type'] == 'Trash Bin'
        assert 'trash' in result['message'].lower()
    
    def test_classify_with_recommendation(self, mock_classifier, sample_image):
        """Test complete classification with recommendation."""
        result = mock_classifier.classify_with_recommendation(sample_image)
        
        # Check all expected fields are present
        expected_fields = [
            'predicted_class', 'confidence', 'confidence_percent',
            'top_predictions', 'all_probabilities', 'is_recyclable',
            'bin_type', 'message', 'disposal_confidence'
        ]
        
        for field in expected_fields:
            assert field in result
        
        # Check disposal confidence levels
        assert result['disposal_confidence'] in ['High', 'Medium', 'Low']
    
    def test_model_not_loaded_error(self, mock_model_path):
        """Test error handling when model is not loaded."""
        classifier = WasteClassifier.__new__(WasteClassifier)
        classifier.model = None
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            classifier.predict("dummy_path.jpg")

class TestBatchProcessor:
    """Test cases for the BatchProcessor class."""
    
    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier for testing."""
        classifier = Mock()
        classifier.classify_with_recommendation.return_value = {
            'predicted_class': 'plastic',
            'confidence': 0.85,
            'confidence_percent': 85.0,
            'is_recyclable': True,
            'bin_type': 'Recycling Bin',
            'message': 'This item (plastic) should go in the recycling bin.',
            'disposal_confidence': 'High'
        }
        return classifier
    
    @pytest.fixture
    def batch_processor(self, mock_classifier):
        """Create a batch processor with mock classifier."""
        return BatchProcessor(mock_classifier)
    
    @pytest.fixture
    def sample_directory(self):
        """Create a temporary directory with sample images."""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        
        # Create sample image files
        for i in range(3):
            image = Image.new('RGB', (100, 100), color='red')
            image_path = os.path.join(temp_dir, f'test_image_{i}.jpg')
            image.save(image_path)
        
        # Create a non-image file (should be ignored)
        with open(os.path.join(temp_dir, 'readme.txt'), 'w') as f:
            f.write("This is not an image")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_process_directory_success(self, batch_processor, sample_directory):
        """Test successful directory processing."""
        results = batch_processor.process_directory(sample_directory)
        
        # Should process 3 image files, ignore 1 text file
        assert len(results) == 3
        
        for result in results:
            assert 'filename' in result
            assert 'filepath' in result
            assert 'predicted_class' in result
            assert result['predicted_class'] == 'plastic'
    
    def test_process_directory_with_output_file(self, batch_processor, sample_directory):
        """Test directory processing with output file."""
        output_file = os.path.join(sample_directory, 'results.json')
        
        results = batch_processor.process_directory(sample_directory, output_file)
        
        # Check that output file was created
        assert os.path.exists(output_file)
        
        # Check that results were saved correctly
        with open(output_file, 'r') as f:
            saved_results = json.load(f)
        
        assert len(saved_results) == 3
        assert saved_results == results
    
    def test_process_directory_nonexistent(self, batch_processor):
        """Test processing non-existent directory."""
        with pytest.raises(FileNotFoundError):
            batch_processor.process_directory("/nonexistent/directory")
    
    def test_process_directory_empty(self, batch_processor):
        """Test processing empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = batch_processor.process_directory(temp_dir)
            assert len(results) == 0
    
    def test_process_directory_with_errors(self, sample_directory):
        """Test directory processing with classification errors."""
        # Create classifier that raises errors
        mock_classifier = Mock()
        mock_classifier.classify_with_recommendation.side_effect = Exception("Classification error")
        
        processor = BatchProcessor(mock_classifier)
        results = processor.process_directory(sample_directory)
        
        assert len(results) == 3
        for result in results:
            assert 'error' in result
            assert result['predicted_class'] is None
    
    def test_generate_summary_report(self, batch_processor):
        """Test summary report generation."""
        # Create sample results
        results = [
            {
                'predicted_class': 'plastic',
                'confidence': 0.85,
                'is_recyclable': True
            },
            {
                'predicted_class': 'paper',
                'confidence': 0.92,
                'is_recyclable': True
            },
            {
                'predicted_class': 'trash',
                'confidence': 0.78,
                'is_recyclable': False
            },
            {
                'error': 'Classification failed',
                'predicted_class': None
            }
        ]
        
        summary = batch_processor.generate_summary_report(results)
        
        # Check summary structure
        assert summary['total_images'] == 4
        assert summary['successfully_processed'] == 3
        assert summary['errors'] == 1
        assert summary['recyclable_items'] == 2
        assert summary['non_recyclable_items'] == 1
        assert summary['recycling_rate'] == pytest.approx(66.67, rel=1e-2)
        assert summary['high_confidence_predictions'] == 2  # confidence > 0.8
        assert summary['high_confidence_rate'] == pytest.approx(66.67, rel=1e-2)
        
        # Check class distribution
        expected_distribution = {'plastic': 1, 'paper': 1, 'trash': 1}
        assert summary['class_distribution'] == expected_distribution
    
    def test_generate_summary_report_empty(self, batch_processor):
        """Test summary report generation with empty results."""
        summary = batch_processor.generate_summary_report([])
        
        assert summary == {}

class TestIntegration:
    """Integration tests for the complete inference pipeline."""
    
    def test_end_to_end_classification(self):
        """Test complete end-to-end classification workflow."""
        # This would typically use a real model file
        # For testing, we'll mock the model loading and prediction
        
        with patch('inference.load_model') as mock_load:
            # Mock model
            mock_model = Mock()
            mock_model.input_shape = (None, 224, 224, 3)
            mock_model.predict.return_value = np.array([[0.1, 0.2, 0.15, 0.25, 0.2, 0.1]])
            mock_load.return_value = mock_model
            
            # Create sample image
            image = Image.new('RGB', (224, 224), color='green')
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                image.save(f.name)
                image_path = f.name
            
            try:
                # Initialize classifier
                classifier = WasteClassifier("mock_model.h5")
                
                # Classify image
                result = classifier.classify_with_recommendation(image_path)
                
                # Verify complete workflow
                assert result['predicted_class'] in classifier.class_names
                assert 0 <= result['confidence'] <= 1
                assert isinstance(result['is_recyclable'], bool)
                assert result['bin_type'] in ['Recycling Bin', 'Trash Bin']
                assert len(result['message']) > 0
                assert result['disposal_confidence'] in ['High', 'Medium', 'Low']
                
            finally:
                # Cleanup
                os.unlink(image_path)

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--cov=../src/inference', '--cov-report=html'])