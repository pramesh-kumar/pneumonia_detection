import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from data_preprocessing import DataPreprocessor
from models import PneumoniaModels

class ModelTrainer:
    def __init__(self, data_dir, model_save_dir='../models'):
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.preprocessor = DataPreprocessor()
        self.model_builder = PneumoniaModels()
        
        os.makedirs(model_save_dir, exist_ok=True)
    
    def setup_data(self, batch_size=32):
        """Setup training and validation data"""
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        
        self.train_generator, self.val_generator = self.preprocessor.create_data_generators(
            train_dir, val_dir, batch_size
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Classes: {self.train_generator.class_indices}")
    
    def get_callbacks(self, model_name):
        """Setup training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_save_dir, f'{model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks
    
    def train_model(self, model, model_name, epochs=50):
        """Train the model"""
        callbacks = self.get_callbacks(model_name)
        
        history = model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, model_name):
        """Evaluate model performance"""
        # Predictions
        val_predictions = model.predict(self.val_generator)
        val_predictions_binary = (val_predictions > 0.5).astype(int)
        
        # True labels
        val_labels = self.val_generator.classes
        
        # Classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(val_labels, val_predictions_binary, 
                                  target_names=['Normal', 'Pneumonia']))
        
        # Confusion matrix
        cm = confusion_matrix(val_labels, val_predictions_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Pneumonia'],
                   yticklabels=['Normal', 'Pneumonia'])
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return val_predictions, val_predictions_binary
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title(f'{model_name} - Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title(f'{model_name} - Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title(f'{model_name} - Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title(f'{model_name} - Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self):
        """Train and compare all models"""
        models_to_train = [
            ('Custom_CNN', self.model_builder.create_custom_cnn()),
            ('ResNet50', self.model_builder.create_resnet_model()),
            ('DenseNet121', self.model_builder.create_densenet_model())
        ]
        
        results = {}
        
        for model_name, model in models_to_train:
            print(f"\n{'='*50}")
            print(f"Training {model_name}")
            print(f"{'='*50}")
            
            # Train model
            history = self.train_model(model, model_name, epochs=30)
            
            # Plot training history
            self.plot_training_history(history, model_name)
            
            # Evaluate model
            predictions, binary_predictions = self.evaluate_model(model, model_name)
            
            # Store results
            results[model_name] = {
                'model': model,
                'history': history,
                'predictions': predictions
            }
        
        return results

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer('../data/chest_xray')
    trainer.setup_data()
    results = trainer.compare_models()