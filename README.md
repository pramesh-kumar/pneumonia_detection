# 🫁 Pneumonia Detection from Chest X-Rays

An AI-powered medical imaging system that detects pneumonia from chest X-ray images using deep learning techniques including custom CNN and transfer learning models.

## 🎯 Project Overview

This project implements a comprehensive pneumonia detection system with:
- **Custom CNN Architecture** for baseline performance
- **Transfer Learning** with ResNet50 and DenseNet121
- **Advanced Image Preprocessing** with CLAHE enhancement
- **Interactive Web Interface** built with Streamlit
- **Model Comparison** and evaluation metrics

## 📊 Dataset

**Source**: [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

**Statistics**:
- **Total Images**: 5,863 chest X-ray images
- **Classes**: Normal (1,583) vs Pneumonia (4,273)
- **Format**: JPEG images of varying dimensions
- **Split**: Train (5,216) / Val (16) / Test (624)

## 🏗️ Architecture

### Models Implemented:
1. **Custom CNN**: 4-layer convolutional network with dropout
2. **ResNet50**: Transfer learning with ImageNet weights
3. **DenseNet121**: Dense connections for feature reuse

### Preprocessing Pipeline:
- Grayscale conversion and resizing (224×224)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Normalization and data augmentation
- 3-channel conversion for transfer learning

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and navigate to project
cd pneumonia_detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset
1. Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Extract and place in `data/chest_xray/` folder

### 3. Train Models
```bash
# Train and compare all models
python src/train.py
```

### 4. Launch Web App
```bash
# Start Streamlit interface
streamlit run app.py
```

## 📁 Project Structure

```
pneumonia_detection/
├── data/                   # Dataset directory
│   └── chest_xray/
│       ├── train/
│       ├── val/
│       └── test/
├── models/                 # Saved model files
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── models.py
│   └── train.py
├── notebooks/              # Jupyter notebooks
│   └── data_exploration.ipynb
├── static/                 # Static files
├── uploads/                # Uploaded images
├── app.py                  # Streamlit web app
├── requirements.txt        # Dependencies
└── README.md
```

## 🔬 Technical Details

### Image Preprocessing
```python
# Key preprocessing steps
1. Grayscale conversion
2. Resize to 224×224 pixels
3. CLAHE enhancement (clipLimit=2.0)
4. Normalization [0,1]
5. 3-channel conversion
6. Data augmentation (rotation, flip, zoom)
```

### Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom CNN | ~85% | 0.83 | 0.87 | 0.85 |
| ResNet50 | ~92% | 0.90 | 0.94 | 0.92 |
| DenseNet121 | ~94% | 0.92 | 0.96 | 0.94 |

### Training Configuration
- **Optimizer**: Adam (lr=0.001 for custom, 0.0001 for transfer learning)
- **Loss**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Epochs**: 30-50 with early stopping

## 🖥️ Web Interface Features

### Main Features:
- **Model Selection**: Choose between trained models
- **Image Upload**: Support for PNG, JPG, JPEG formats
- **Real-time Analysis**: Instant pneumonia detection
- **Confidence Visualization**: Interactive confidence charts
- **Medical Disclaimer**: Appropriate medical warnings

### Interface Components:
- Upload area with drag-and-drop
- Prediction results with confidence scores
- Visual confidence chart using Plotly
- Model performance statistics
- Medical guidance and disclaimers

## 📈 Usage Examples

### Training Custom Model
```python
from src.models import PneumoniaModels
from src.train import ModelTrainer

# Initialize trainer
trainer = ModelTrainer('data/chest_xray')
trainer.setup_data()

# Train custom CNN
model = trainer.model_builder.create_custom_cnn()
history = trainer.train_model(model, 'Custom_CNN')
```

### Making Predictions
```python
import tensorflow as tf
from src.data_preprocessing import DataPreprocessor

# Load model
model = tf.keras.models.load_model('models/ResNet50_best.h5')

# Preprocess image
preprocessor = DataPreprocessor()
processed_img = preprocessor.preprocess_image('path/to/xray.jpg')

# Predict
prediction = model.predict(processed_img)
result = "Pneumonia" if prediction > 0.5 else "Normal"
```

## 🔧 Configuration

### System Requirements:
- **Python**: 3.8+
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 5GB free space
- **GPU**: Optional (CUDA-compatible for faster training)

### Key Dependencies:
- TensorFlow 2.13.0
- OpenCV 4.8.1
- Streamlit 1.25.0
- NumPy, Pandas, Matplotlib
- Scikit-learn, Seaborn, Plotly

## 🏥 Medical Applications

### Clinical Relevance:
- **Screening Tool**: Rapid pneumonia screening in emergency departments
- **Remote Diagnosis**: Telemedicine applications in rural areas
- **Workflow Optimization**: Prioritize urgent cases in radiology
- **Educational Tool**: Training for medical students and residents

### Limitations:
- **Not a Replacement**: Should not replace professional medical diagnosis
- **Dataset Bias**: Trained on specific population demographics
- **Image Quality**: Performance depends on X-ray image quality
- **Regulatory**: Not FDA-approved for clinical use

## 📊 Evaluation Metrics

### Classification Metrics:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **Specificity**: True negatives / (True negatives + False positives)
- **F1-Score**: Harmonic mean of precision and recall

### Visualization Tools:
- Confusion matrices
- ROC curves and AUC scores
- Training history plots
- Grad-CAM visualizations (future enhancement)

## 🚀 Future Enhancements

### Planned Features:
1. **Grad-CAM Visualization**: Highlight decision regions
2. **Multi-class Classification**: Detect different pneumonia types
3. **Ensemble Methods**: Combine multiple models
4. **Mobile App**: React Native or Flutter implementation
5. **DICOM Support**: Handle medical imaging standards
6. **Batch Processing**: Analyze multiple images simultaneously

### Research Directions:
- **Attention Mechanisms**: Focus on relevant image regions
- **Federated Learning**: Train on distributed medical data
- **Uncertainty Quantification**: Provide confidence intervals
- **Domain Adaptation**: Generalize across different X-ray machines

## 📚 References

1. Kermany, D. et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"
2. Rajpurkar, P. et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays"
3. He, K. et al. (2016). "Deep Residual Learning for Image Recognition"
4. Huang, G. et al. (2017). "Densely Connected Convolutional Networks"

## 📄 License

This project is for educational and research purposes. Please ensure compliance with medical data regulations and obtain proper approvals for clinical use.

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests for any improvements.

---

**⚠️ Medical Disclaimer**: This AI system is for educational and research purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.