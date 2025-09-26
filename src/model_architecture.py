"""
Model architecture definitions for Smart Bin waste classification.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten, Dropout, 
    GlobalAveragePooling2D, Input, Lambda
)
import tensorflow.keras.applications.mobilenet_v2 as mobilenetv2

class ModelArchitecture:
    """Contains different model architectures for waste classification."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=6):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def create_basic_cnn(self):
        """Create a basic CNN model from scratch."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_improved_cnn(self):
        """Create an improved CNN with more layers and regularization."""
        model = Sequential([
            # First Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second Block
            Conv2D(64, (3, 3), activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third Block
            Conv2D(128, (3, 3), activation='relu'),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fourth Block
            Conv2D(256, (3, 3), activation='relu'),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Classifier
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_mobilenet_model(self, weights_path=None, trainable_base=False):
        """Create MobileNetV2-based transfer learning model."""
        # Load pre-trained MobileNetV2
        if weights_path:
            base_model = mobilenetv2.MobileNetV2(
                include_top=False, 
                input_shape=self.input_shape,
                weights=None
            )
            base_model.load_weights(weights_path)
        else:
            base_model = mobilenetv2.MobileNetV2(
                include_top=False, 
                input_shape=self.input_shape,
                weights='imagenet'
            )
        
        # Freeze base model layers
        base_model.trainable = trainable_base
        
        # Create the model
        model = Sequential([
            Input(shape=self.input_shape),
            
            # Preprocessing layer
            Lambda(mobilenetv2.preprocess_input),
            
            # Pre-trained base
            base_model,
            
            # Custom classifier
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_fine_tuned_mobilenet(self, weights_path=None, fine_tune_from=100):
        """Create a fine-tuned MobileNetV2 model."""
        # Load base model
        if weights_path:
            base_model = mobilenetv2.MobileNetV2(
                include_top=False,
                input_shape=self.input_shape,
                weights=None
            )
            base_model.load_weights(weights_path)
        else:
            base_model = mobilenetv2.MobileNetV2(
                include_top=False,
                input_shape=self.input_shape,
                weights='imagenet'
            )
        
        # Freeze layers up to fine_tune_from
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False
        
        # Build model
        model = Sequential([
            Input(shape=self.input_shape),
            Lambda(mobilenetv2.preprocess_input),
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model, learning_rate=0.001, loss='categorical_crossentropy'):
        """Compile model with specified parameters."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['categorical_accuracy', 'top_3_categorical_accuracy']
        )
        
        return model
    
    def get_model_summary(self, model):
        """Get detailed model summary."""
        model.summary()
        
        # Count trainable parameters
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        
        print(f"\nTrainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"Total parameters: {trainable_params + non_trainable_params:,}")
        
        return {
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'total_params': trainable_params + non_trainable_params
        }

class ModelCallbacks:
    """Utility class for creating training callbacks."""
    
    @staticmethod
    def create_early_stopping(patience=5, monitor='val_categorical_accuracy', 
                            min_delta=0.001, restore_best_weights=True):
        """Create early stopping callback."""
        return tf.keras.callbacks.EarlyStopping(
            patience=patience,
            verbose=1,
            monitor=monitor,
            mode='max',
            min_delta=min_delta,
            restore_best_weights=restore_best_weights
        )
    
    @staticmethod
    def create_model_checkpoint(filepath, monitor='val_categorical_accuracy', 
                              save_best_only=True):
        """Create model checkpoint callback."""
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
    
    @staticmethod
    def create_reduce_lr(factor=0.2, patience=3, monitor='val_categorical_accuracy',
                        min_lr=1e-7):
        """Create learning rate reduction callback."""
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=1,
            mode='max'
        )
    
    @staticmethod
    def create_tensorboard_callback(log_dir):
        """Create TensorBoard callback."""
        return tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )

def main():
    """Example usage of model architectures."""
    # Initialize architecture builder
    model_builder = ModelArchitecture(num_classes=6)
    
    # Create different models
    print("Creating Basic CNN...")
    basic_cnn = model_builder.create_basic_cnn()
    basic_cnn = model_builder.compile_model(basic_cnn)
    model_builder.get_model_summary(basic_cnn)
    
    print("\n" + "="*50)
    print("Creating MobileNet Transfer Learning Model...")
    mobilenet_model = model_builder.create_mobilenet_model()
    mobilenet_model = model_builder.compile_model(mobilenet_model)
    model_builder.get_model_summary(mobilenet_model)
    
    # Create callbacks
    callbacks = [
        ModelCallbacks.create_early_stopping(),
        ModelCallbacks.create_model_checkpoint('best_model.h5'),
        ModelCallbacks.create_reduce_lr()
    ]
    
    print(f"\nCallbacks created: {len(callbacks)}")

if __name__ == "__main__":
    main()