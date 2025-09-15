import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
    
    def preprocess_image(self, img_path):
        """Apply preprocessing: resize, normalize, histogram equalization"""
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to 3 channels for transfer learning
        img = np.stack([img, img, img], axis=-1)
        
        return img
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create data generators with augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            color_mode='rgb'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            color_mode='rgb'
        )
        
        return train_generator, val_generator
    
    def visualize_samples(self, data_dir):
        """Visualize sample images from dataset"""
        normal_dir = os.path.join(data_dir, 'NORMAL')
        pneumonia_dir = os.path.join(data_dir, 'PNEUMONIA')
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        
        # Normal samples
        normal_files = os.listdir(normal_dir)[:4]
        for i, file in enumerate(normal_files):
            img = cv2.imread(os.path.join(normal_dir, file), cv2.IMREAD_GRAYSCALE)
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'Normal - {file}')
            axes[0, i].axis('off')
        
        # Pneumonia samples
        pneumonia_files = os.listdir(pneumonia_dir)[:4]
        for i, file in enumerate(pneumonia_files):
            img = cv2.imread(os.path.join(pneumonia_dir, file), cv2.IMREAD_GRAYSCALE)
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(f'Pneumonia - {file}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()