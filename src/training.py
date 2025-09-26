"""
Training pipeline for Smart Bin waste classification models.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from data_preprocessing import DataPreprocessor
from model_architecture import ModelArchitecture, ModelCallbacks

class ModelTrainer:
    """Handles model training, evaluation, and visualization."""
    
    def __init__(self, base_data_path, model_save_path='models/', log_path='logs/'):
        self.base_data_path = base_data_path
        self.model_save_path = model_save_path
        self.log_path = log_path
        
        # Create directories if they don't exist
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        
        # Initialize components
        self.preprocessor = DataPreprocessor(base_data_path)
        self.model_builder = ModelArchitecture()
        
    def prepare_data(self, augment_training=True, test_split=0.1, val_split=0.1):
        """Prepare and split data for training."""
        print("Preparing data...")
        
        # Create dataframe
        df = self.preprocessor.create_dataframe()
        print(f"Total samples loaded: {len(df)}")
        
        # Validate data integrity
        issues = self.preprocessor.validate_data_integrity(df)
        if issues:
            print(f"Data issues found: {issues}")
        
        # Show class distribution
        class_dist = self.preprocessor.get_class_distribution(df)
        print("Class distribution:")
        for class_name, count in class_dist.items():
            print(f"  {class_name}: {count}")
        
        # Split data
        train_df, val_df, test_df = self.preprocessor.split_data(
            df, test_size=(test_split + val_split), val_size=val_split/(test_split + val_split)
        )
        
        print(f"\nData split:")
        print(f"  Training: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples") 
        print(f"  Test: {len(test_df)} samples")
        
        # Create generators
        generators = self.preprocessor.create_data_generators(
            train_df, val_df, test_df, augment=augment_training
        )
        
        self.train_generator = generators['train']
        self.val_generator = generators['validation']
        self.test_generator = generators['test']
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        return generators
    
    def train_model(self, model_type='mobilenet', epochs=20, learning_rate=0.001, 
                   early_stopping=True, save_best=True):
        """Train a model with specified configuration."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_{timestamp}"
        
        print(f"\nTraining {model_type} model...")
        
        # Create model based on type
        if model_type == 'basic_cnn':
            model = self.model_builder.create_basic_cnn()
        elif model_type == 'improved_cnn':
            model = self.model_builder.create_improved_cnn()
        elif model_type == 'mobilenet':
            model = self.model_builder.create_mobilenet_model()
        elif model_type == 'fine_tuned_mobilenet':
            model = self.model_builder.create_fine_tuned_mobilenet()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile model
        model = self.model_builder.compile_model(model, learning_rate=learning_rate)
        
        # Show model summary
        print("\nModel Architecture:")
        self.model_builder.get_model_summary(model)
        
        # Create callbacks
        callbacks = []
        
        if early_stopping:
            callbacks.append(ModelCallbacks.create_early_stopping(patience=5))
        
        if save_best:
            checkpoint_path = os.path.join(self.model_save_path, f"{model_name}_best.h5")
            callbacks.append(ModelCallbacks.create_model_checkpoint(checkpoint_path))
        
        callbacks.append(ModelCallbacks.create_reduce_lr())
        
        # Add TensorBoard logging
        log_dir = os.path.join(self.log_path, model_name)
        callbacks.append(ModelCallbacks.create_tensorboard_callback(log_dir))
        
        # Calculate steps
        steps_per_epoch = len(self.train_generator)
        validation_steps = len(self.val_generator)
        
        print(f"\nTraining configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Validation steps: {validation_steps}")
        print(f"  Learning rate: {learning_rate}")
        
        # Train model
        history = model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(self.model_save_path, f"{model_name}_final.h5")
        model.save(final_model_path)
        
        # Save training history
        history_path = os.path.join(self.log_path, f"{model_name}_history.json")
        self.save_training_history(history.history, history_path)
        
        # Save training configuration
        config = {
            'model_type': model_type,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': self.preprocessor.batch_size,
            'image_size': self.preprocessor.image_size,
            'train_samples': len(self.train_df),
            'val_samples': len(self.val_df),
            'test_samples': len(self.test_df),
            'timestamp': timestamp
        }
        config_path = os.path.join(self.log_path, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nModel saved: {final_model_path}")
        print(f"History saved: {history_path}")
        print(f"Config saved: {config_path}")
        
        return model, history.history, model_name
    
    def evaluate_model(self, model, model_name=None):
        """Evaluate model performance on test set."""
        print("\nEvaluating model on test set...")
        
        # Evaluate on test set
        test_loss, test_acc, test_top3_acc = model.evaluate(
            self.test_generator, 
            steps=len(self.test_generator),
            verbose=1
        )
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Top-3 Accuracy: {test_top3_acc:.4f}")
        
        # Generate predictions
        print("Generating predictions...")
        predictions = model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true classes
        true_classes = self.test_generator.classes
        class_labels = list(self.test_generator.class_indices.keys())
        
        # Classification report
        report = classification_report(
            true_classes, predicted_classes, 
            target_names=class_labels, 
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=class_labels))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Save evaluation results
        if model_name:
            eval_results = {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_acc),
                'test_top3_accuracy': float(test_top3_acc),
                'classification_report': report,
                'confusion_matrix': cm.tolist()
            }
            
            eval_path = os.path.join(self.log_path, f"{model_name}_evaluation.json")
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"Evaluation results saved: {eval_path}")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_top3_accuracy': test_top3_acc,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes,
            'class_labels': class_labels,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def plot_training_history(self, history, model_name=None, save_plot=True):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot loss
        ax1.plot(history['loss'], label='Training Loss', color='blue')
        ax1.plot(history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['categorical_accuracy'], label='Training Accuracy', color='blue')
        ax2.plot(history['val_categorical_accuracy'], label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_plot and model_name:
            plot_path = os.path.join(self.log_path, f"{model_name}_training_history.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved: {plot_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_labels, model_name=None, save_plot=True):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_plot and model_name:
            plot_path = os.path.join(self.log_path, f"{model_name}_confusion_matrix.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved: {plot_path}")
        
        plt.show()
    
    def save_training_history(self, history, filepath):
        """Save training history to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, values in history.items():
            if isinstance(values, np.ndarray):
                serializable_history[key] = values.tolist()
            elif isinstance(values, list):
                serializable_history[key] = [float(v) for v in values]
            else:
                serializable_history[key] = values
        
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)

def main():
    """Example training pipeline."""
    # Configuration
    DATA_PATH = "/path/to/your/dataset"
    
    # Initialize trainer
    trainer = ModelTrainer(DATA_PATH)
    
    # Prepare data
    generators = trainer.prepare_data(augment_training=True)
    
    # Train MobileNet model
    model, history, model_name = trainer.train_model(
        model_type='mobilenet',
        epochs=10,
        learning_rate=0.001
    )
    
    # Plot training history
    trainer.plot_training_history(history, model_name)
    
    # Evaluate model
    eval_results = trainer.evaluate_model(model, model_name)
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(
        eval_results['confusion_matrix'],
        eval_results['class_labels'],
        model_name
    )
    
    print(f"\nTraining completed! Final test accuracy: {eval_results['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()