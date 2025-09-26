#!/usr/bin/env python3
"""
Command-line interface for Smart Bin waste classification.
"""
import argparse
import os
import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference import WasteClassifier, BatchProcessor

def print_prediction_result(result):
    """Pretty print prediction result."""
    print("\n" + "="*50)
    print("WASTE CLASSIFICATION RESULT")
    print("="*50)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    # Main prediction
    recyclable_icon = "♻️" if result.get('is_recyclable', False) else "🗑️"
    confidence_bar = "█" * int(result['confidence_percent'] / 10)
    
    print(f"{recyclable_icon} Class: {result['predicted_class'].upper()}")
    print(f"📊 Confidence: {result['confidence_percent']:.1f}% {confidence_bar}")
    print(f"🏷️  Bin Type: {result.get('bin_type', 'Unknown')}")
    print(f"💡 Recommendation: {result.get('message', 'N/A')}")
    
    # Top predictions
    if 'top_predictions' in result:
        print(f"\n📈 Top 3 Predictions:")
        for i, pred in enumerate(result['top_predictions'][:3], 1):
            print(f"  {i}. {pred['class']}: {pred['confidence_percent']:.1f}%")
    
    # Confidence assessment
    confidence_level = result.get('disposal_confidence', 'Unknown')
    confidence_colors = {
        'High': '🟢',
        'Medium': '🟡', 
        'Low': '🔴'
    }
    icon = confidence_colors.get(confidence_level, '⚪')
    print(f"\n{icon} Disposal Confidence: {confidence_level}")
    
    print("="*50)

def print_summary_report(summary):
    """Pretty print batch processing summary."""
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY REPORT")
    print("="*60)
    
    # Basic stats
    print(f"📊 Total Images Processed: {summary['total_images']}")
    print(f"✅ Successfully Processed: {summary['successfully_processed']}")
    if summary.get('errors', 0) > 0:
        print(f"Errors: {summary['errors']}")
    
    # Recycling stats
    if summary['successfully_processed'] > 0:
        print(f"\n♻️  RECYCLING ANALYSIS:")
        print(f"   Recyclable items: {summary['recyclable_items']}")
        print(f"   Non-recyclable items: {summary['non_recyclable_items']}")
        print(f"   Recycling rate: {summary['recycling_rate']:.1f}%")
        
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"   High confidence predictions: {summary['high_confidence_predictions']}")
        print(f"   High confidence rate: {summary['high_confidence_rate']:.1f}%")
    
    # Class distribution
    if summary.get('class_distribution'):
        print(f"\nCLASS DISTRIBUTION:")
        for class_name, count in sorted(summary['class_distribution'].items()):
            percentage = (count / summary['successfully_processed']) * 100
            bar = "▓" * int(percentage / 5)  # Scale to 20 char max
            print(f"   {class_name:>10}: {count:>3} ({percentage:>5.1f}%) {bar}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Smart Bin: AI-powered waste classification system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a single image
  python classify_waste.py --image path/to/waste.jpg --model models/best_model.h5
  
  # Process all images in a directory
  python classify_waste.py --directory path/to/images/ --model models/best_model.h5
  
  # Save batch results to file
  python classify_waste.py --directory images/ --model model.h5 --output results.json
  
  # Show detailed predictions for single image
  python classify_waste.py --image test.jpg --model model.h5 --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model file (.h5)'
    )
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image', '-i',
        type=str,
        help='Path to single image file'
    )
    input_group.add_argument(
        '--directory', '-d',
        type=str,
        help='Path to directory containing images'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for batch processing results (JSON format)'
    )
    
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
        help='Class names in model order (default: cardboard glass metal paper plastic trash)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top predictions to show (default: 3)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for warnings (default: 0.5)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output including all class probabilities'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output (overrides verbose)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if args.image and not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
        
    if args.directory and not os.path.exists(args.directory):
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)
    
    try:
        # Initialize classifier
        if not args.quiet:
            print("🤖 Initializing Smart Bin Classifier...")
            print(f"📁 Loading model: {args.model}")
        
        classifier = WasteClassifier(args.model, args.classes)
        
        if not args.quiet:
            print("Model loaded successfully!")
        
        # Single image classification
        if args.image:
            if not args.quiet:
                print(f"🔍 Analyzing image: {args.image}")
            
            result = classifier.classify_with_recommendation(args.image)
            
            if args.quiet:
                print(f"{result['predicted_class']},{result['confidence_percent']:.1f}")
            else:
                print_prediction_result(result)
                
                # Show verbose output
                if args.verbose and 'all_probabilities' in result:
                    print(f"\n📋 All Class Probabilities:")
                    for class_name, prob in result['all_probabilities'].items():
                        print(f"   {class_name:>10}: {prob*100:>6.2f}%")
                
                # Warning for low confidence
                if result['confidence'] < args.threshold:
                    print(f"\n⚠️  Warning: Low confidence prediction ({result['confidence_percent']:.1f}%)")
                    print("   Consider manual verification or retaking the image.")
        
        # Directory batch processing
        elif args.directory:
            if not args.quiet:
                print(f"📂 Processing directory: {args.directory}")
            
            batch_processor = BatchProcessor(classifier)
            results = batch_processor.process_directory(args.directory, args.output)
            
            if args.quiet:
                # Just show count of each class
                successful_results = [r for r in results if 'error' not in r]
                class_counts = {}
                for result in successful_results:
                    class_name = result['predicted_class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                for class_name, count in sorted(class_counts.items()):
                    print(f"{class_name}: {count}")
            else:
                # Generate and display summary
                summary = batch_processor.generate_summary_report(results)
                print_summary_report(summary)
                
                # Show individual results if verbose and not too many
                if args.verbose and len(results) <= 10:
                    print(f"\n📋 Individual Results:")
                    for result in results:
                        filename = result.get('filename', 'Unknown')
                        if 'error' in result:
                            print(f"   {filename}: Error - {result['error']}")
                        else:
                            icon = "♻️" if result.get('is_recyclable') else "🗑️"
                            print(f"   {icon} {filename}: {result['predicted_class']} ({result['confidence_percent']:.1f}%)")
                
                if args.output:
                    print(f"\n💾 Detailed results saved to: {args.output}")
    
    except Exception as e:
        print(f" Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()