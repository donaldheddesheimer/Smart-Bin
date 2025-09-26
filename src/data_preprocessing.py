"""
Data preprocessing utilities for Smart Bin waste classification.
"""
import os
import re
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.applications.mobilenet_v2 as mobilenetv2

class DataPreprocessor:
    """Handles data loading, preprocessing, and augmentation for waste classification."""
    
    def __init__(self, base_path, image_size=(224, 224), batch_size=32):
        self.base_path = base_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.categories = {
            0: 'paper', 
            1: 'cardboard', 
            2: 'plastic', 
            3: 'metal', 
            4: 'trash', 
            5: 'glass'
        }
    
    def add_class_name_prefix(self, df, col_name):
        """Add class name prefix to filename."""
        df[col_name] = df[col_name].apply(
            lambda x: x[:re.search(r"\d", x).start()] + '/' + x
        )
        return df
    
    def create_dataframe(self):
        """Create DataFrame with filenames and categories."""
        filenames_list = []
        categories_list = []
        
        for category in self.categories:
            category_path = os.path.join(self.base_path, self.categories[category])
            if os.path.exists(category_path):
                filenames = os.listdir(category_path)
                filenames_list.extend(filenames)
                categories_list.extend([category] * len(filenames))
        
        df = pd.DataFrame({
            'filename': filenames_list,
            'category': categories_list
        })
        
        # Add class name prefix to filename
        df = self.add_class_name_prefix(df, 'filename')
        
        # Convert categories to names
        df['category'] = df['category'].replace(self.categories)
        
        # Shuffle the dataframe
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.5):
        """Split data into train, validation, and test sets."""
        # First split: train and temp
        train_df, temp_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['category']
        )
        
        # Second split: validation and test from temp
        val_df, test_df = train_test_split(
            temp_df, test_size=val_size, random_state=42, stratify=temp_df['category']
        )
        
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True)
        )
    
    def create_data_generators(self, train_df, val_df, test_df=None, augment=True):
        """Create data generators for training, validation, and testing."""
        
        # Training data generator with augmentation
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                shear_range=0.1,
                zoom_range=0.3,
                horizontal_flip=True,
                vertical_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # Validation and test generators (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            self.base_path,
            x_col='filename',
            y_col='category',
            target_size=self.image_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            val_df,
            self.base_path,
            x_col='filename',
            y_col='category',
            target_size=self.image_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False
        )
        
        generators = {
            'train': train_generator,
            'validation': val_generator
        }
        
        if test_df is not None:
            test_generator = val_datagen.flow_from_dataframe(
                test_df,
                self.base_path,
                x_col='filename',
                y_col='category',
                target_size=self.image_size,
                class_mode='categorical',
                batch_size=1,
                shuffle=False
            )
            generators['test'] = test_generator
        
        return generators
    
    def mobilenet_preprocessing(self, img):
        """Apply MobileNetV2 preprocessing."""
        return mobilenetv2.preprocess_input(img)
    
    def get_class_distribution(self, df):
        """Get class distribution statistics."""
        return df['category'].value_counts().to_dict()
    
    def validate_data_integrity(self, df):
        """Validate data integrity and report issues."""
        issues = []
        
        # Check for missing files
        missing_files = []
        for _, row in df.iterrows():
            file_path = os.path.join(self.base_path, row['filename'])
            if not os.path.exists(file_path):
                missing_files.append(row['filename'])
        
        if missing_files:
            issues.append(f"Missing files: {len(missing_files)}")
        
        # Check class balance
        class_counts = self.get_class_distribution(df)
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        
        if max_count / min_count > 5:  # More than 5x difference
            issues.append("Significant class imbalance detected")
        
        return issues

def main():
    """Example usage of DataPreprocessor."""
    # Initialize preprocessor
    base_path = "/path/to/your/dataset"
    preprocessor = DataPreprocessor(base_path)
    
    # Create dataframe
    df = preprocessor.create_dataframe()
    print(f"Total samples: {len(df)}")
    print("\nClass distribution:")
    print(preprocessor.get_class_distribution(df))
    
    # Validate data
    issues = preprocessor.validate_data_integrity(df)
    if issues:
        print(f"\nData issues found: {issues}")
    
    # Split data
    train_df, val_df, test_df = preprocessor.split_data(df)
    print(f"\nData split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create generators
    generators = preprocessor.create_data_generators(train_df, val_df, test_df)
    print("\nData generators created successfully!")

if __name__ == "__main__":
    main()