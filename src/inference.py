"""
Inference utilities for Smart Bin waste classification.
"""
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.applications.mobilenet_v2 as mobilenetv2

class WasteClassifier:
    """Main classifier for waste categorization."""
    
    def __init__(self, model_path, class_names=None):
        """
        Initialize the waste classifier.
        
        Args:
            model_path (str): Path to the trained model
            class_names (list): List of class names in order
        """
        self.model_path = model_path
        self.model = None
        self.input_shape = (224, 224, 3)
        
        # Default class names based on the notebook
        self.class_names = class_names or [
            'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'
        ]
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            print(f"Loading model from {self.model_path}...")
            self.model = load_model(self.model_path)
            print("Model loaded successfully!")
            
            # Get input shape from model
            if self.model.input_shape:
                self.input_shape = self.model.input_shape[1:]
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def preprocess_image(self, image_path_or_array):
        """
        Preprocess image for prediction.
        
        Args:
            image_path_or_array: Path to image file or numpy array
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            if isinstance(image_path_or_array, str):
                if not os.path.exists(image_path_or_array):
                    raise FileNotFoundError(f"Image not found: {image_path_or_array}")
                image = Image.open(image_path_or_array).convert('RGB')
            elif isinstance(image_path_or_array, np.ndarray):
                image = Image.fromarray(image_path_or_array.astype('uint8')).convert('RGB')
            else:
                image = image_path_or_array
            
            # Resize image
            image = image.resize(self.input_shape[:2])
            
            # Convert to array
            img_array = img_to_array(image)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Apply MobileNet preprocessing if needed
            # Note: This assumes the model was trained with MobileNet preprocessing
            # You might need to adjust this based on your specific model
            img_array = mobilenetv2.preprocess_input(img_array)
            
            return img_array
            
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess image: {str(e)}")
    
    def predict(self, image_path_or_array, top_k=3):
        """
        Predict waste category for a single image.
        
        Args:
            image_path_or_array: Path to image file or numpy array
            top_k (int): Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path_or_array)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get probabilities for all classes
        probabilities = predictions[0]
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_predictions = []
        
        for idx in top_indices:
            top_predictions.append({
                'class': self.class_names[idx],
                'confidence': float(probabilities[idx]),
                'confidence_percent': float(probabilities[idx] * 100)
            })
        
        # Primary prediction
        primary_prediction = top_predictions[0]
        
        return {
            'predicted_class': primary_prediction['class'],
            'confidence': primary_prediction['confidence'],
            'confidence_percent': primary_prediction['confidence_percent'],
            'top_predictions': top_predictions,
            'all_probabilities': {
                self.class_names[i]: float(probabilities[i]) 
                for i in range(len(self.class_names))
            }
        }
    
    def predict_batch(self, image_paths_or_arrays):
        """
        Predict waste categories for multiple images.
        
        Args:
            image_paths_or_arrays: List of image paths or arrays
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in image_paths_or_arrays:
            try:
                result = self.predict(image)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'predicted_class': None,
                    'confidence': 0.0
                })
        
        return results
    
    def get_recyclable_status(self, predicted_class):
        """
        Determine if the predicted class is recyclable.
        
        Args:
            predicted_class (str): Predicted waste class
            
        Returns:
            Dictionary with recyclability information
        """
        recyclable_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
        non_recyclable_classes = ['trash']
        
        is_recyclable = predicted_class.lower() in recyclable_classes
        
        return {
            'is_recyclable': is_recyclable,
            'bin_type': 'Recycling Bin' if is_recyclable else 'Trash Bin',
            'message': f"This item ({predicted_class}) should go in the {'recycling' if is_recyclable else 'trash'} bin."
        }
    
    def classify_with_recommendation(self, image_path_or_array):
        """
        Classify waste and provide disposal recommendation.
        
        Args:
            image_path_or_array: Path to image file or numpy array
            
        Returns:
            Complete classification with disposal recommendation
        """
        # Get prediction
        prediction = self.predict(image_path_or_array)
        
        # Get recyclability status
        recyclability = self.get_recyclable_status(prediction['predicted_class'])
        
        # Combine results
        result = {
            **prediction,
            **recyclability,
            'disposal_confidence': 'High' if prediction['confidence'] > 0.8 else 'Medium' if prediction['confidence'] > 0.5 else 'Low'
        }
        
        return result

class BatchProcessor:
    """Process multiple images for waste classification."""
    
    def __init__(self, classifier):
        self.classifier = classifier
    
    def process_directory(self, directory_path, output_file=None):
        """
        Process all images in a directory.
        
        Args:
            directory_path (str): Path to directory containing images
            output_file (str): Optional path to save results as JSON
            
        Returns:
            List of results for all images
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(directory_path, filename))
        
        if not image_files:
            print("No image files found in directory")
            return []
        
        print(f"Processing {len(image_files)} images...")
        
        results = []
        for i, image_path in enumerate(image_files):
            try:
                result = self.classifier.classify_with_recommendation(image_path)
                result['filename'] = os.path.basename(image_path)
                result['filepath'] = image_path
                results.append(result)
                
                print(f"Processed {i+1}/{len(image_files)}: {result['filename']} -> {result['predicted_class']} ({result['confidence_percent']:.1f}%)")
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'filename': os.path.basename(image_path),
                    'filepath': image_path,
                    'error': str(e),
                    'predicted_class': None
                })
        
        # Save results if output file specified
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results
    
    def generate_summary_report(self, results):
        """Generate summary statistics from results."""
        if not results:
            return {}
        
        # Filter successful predictions
        successful_results = [r for r in results if 'error' not in r]
        
        # Count by class
        class_counts = {}
        recyclable_count = 0
        high_confidence_count = 0
        
        for result in successful_results:
            predicted_class = result['predicted_class']
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            
            if result.get('is_recyclable', False):
                recyclable_count += 1
            
            if result['confidence'] > 0.8:
                high_confidence_count += 1
        
        total_processed = len(successful_results)
        error_count = len(results) - total_processed
        
        summary = {
            'total_images': len(results),
            'successfully_processed': total_processed,
            'errors': error_count,
            'class_distribution': class_counts,
            'recyclable_items': recyclable_count,
            'non_recyclable_items': total_processed - recyclable_count,
            'recycling_rate': (recyclable_count / total_processed * 100) if total_processed > 0 else 0,
            'high_confidence_predictions': high_confidence_count,
            'high_confidence_rate': (high_confidence_count / total_processed * 100) if total_processed > 0 else 0
        }
        
        return summary

def main():
    """Example usage of the inference system."""
    # Initialize classifier
    model_path = "models/best_model.h5"  # Update this path
    
    try:
        classifier = WasteClassifier(model_path)
        
        # Single image prediction
        image_path = "path/to/test/image.jpg"  # Update this path
        
        if os.path.exists(image_path):
            result = classifier.classify_with_recommendation(image_path)
            
            print("Prediction Results:")
            print(f"  Class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence_percent']:.1f}%")
            print(f"  Recyclable: {result['is_recyclable']}")
            print(f"  Recommendation: {result['message']}")
            
            print(f"\nTop 3 Predictions:")
            for pred in result['top_predictions']:
                print(f"  {pred['class']}: {pred['confidence_percent']:.1f}%")
        
        # Batch processing example
        batch_processor = BatchProcessor(classifier)
        directory_path = "path/to/test/images/"  # Update this path
        
        if os.path.exists(directory_path):
            results = batch_processor.process_directory(directory_path)
            summary = batch_processor.generate_summary_report(results)
            
            print(f"\nBatch Processing Summary:")
            print(f"  Total images: {summary['total_images']}")
            print(f"  Successfully processed: {summary['successfully_processed']}")
            print(f"  Recycling rate: {summary['recycling_rate']:.1f}%")
            print(f"  High confidence rate: {summary['high_confidence_rate']:.1f}%")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()