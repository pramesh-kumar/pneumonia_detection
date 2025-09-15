import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
import os

# Page config
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide"
)

@st.cache_resource
def load_model(model_path):
    """Load trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        return None

def preprocess_image(image):
    """Preprocess uploaded image"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Resize to model input size
    img_array = cv2.resize(img_array, (224, 224))
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_array = clahe.apply(img_array)
    
    # Normalize
    img_array = img_array.astype(np.float32) / 255.0
    
    # Convert to 3 channels
    img_array = np.stack([img_array, img_array, img_array], axis=-1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def create_confidence_chart(confidence, prediction):
    """Create confidence visualization"""
    colors = ['#ff6b6b' if prediction == 'Pneumonia' else '#4ecdc4',
              '#4ecdc4' if prediction == 'Pneumonia' else '#ff6b6b']
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Normal', 'Pneumonia'],
            y=[1-confidence, confidence] if prediction == 'Pneumonia' else [confidence, 1-confidence],
            marker_color=colors,
            text=[f'{(1-confidence)*100:.1f}%', f'{confidence*100:.1f}%'] if prediction == 'Pneumonia' 
                 else [f'{confidence*100:.1f}%', f'{(1-confidence)*100:.1f}%'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        yaxis_title="Confidence Score",
        xaxis_title="Diagnosis",
        showlegend=False,
        height=400
    )
    
    return fig

def main():
    st.title("🫁 Pneumonia Detection from Chest X-Rays")
    st.markdown("### AI-powered medical imaging analysis")
    
    # Sidebar
    st.sidebar.header("Model Selection")
    
    # Check for available models
    model_dir = "models"
    available_models = []
    if os.path.exists(model_dir):
        available_models = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    
    if not available_models:
        st.error("No trained models found! Please train a model first.")
        st.info("Run the training script to create models.")
        return
    
    selected_model = st.sidebar.selectbox("Choose Model", available_models)
    model_path = os.path.join(model_dir, selected_model)
    
    # Load model
    model = load_model(model_path)
    if model is None:
        st.error(f"Failed to load model: {selected_model}")
        return
    
    st.sidebar.success(f"✅ Model loaded: {selected_model}")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Chest X-Ray")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image for pneumonia detection"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray", use_column_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze X-Ray", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    processed_img = preprocess_image(image)
                    
                    # Make prediction
                    prediction_prob = model.predict(processed_img)[0][0]
                    
                    # Determine result
                    if prediction_prob > 0.5:
                        prediction = "Pneumonia"
                        confidence = prediction_prob
                        color = "🔴"
                    else:
                        prediction = "Normal"
                        confidence = 1 - prediction_prob
                        color = "🟢"
                    
                    # Store results in session state
                    st.session_state.prediction = prediction
                    st.session_state.confidence = confidence
                    st.session_state.color = color
                    st.session_state.prediction_prob = prediction_prob
    
    with col2:
        st.header("Analysis Results")
        
        if hasattr(st.session_state, 'prediction'):
            # Display prediction
            st.markdown(f"### {st.session_state.color} Diagnosis: **{st.session_state.prediction}**")
            st.markdown(f"**Confidence:** {st.session_state.confidence*100:.1f}%")
            
            # Confidence chart
            fig = create_confidence_chart(st.session_state.prediction_prob, st.session_state.prediction)
            st.plotly_chart(fig, use_container_width=True)
            
            # Medical disclaimer
            st.warning("""
            ⚠️ **Medical Disclaimer**: This AI tool is for educational purposes only. 
            Always consult with qualified healthcare professionals for medical diagnosis and treatment.
            """)
            
            # Additional info
            if st.session_state.prediction == "Pneumonia":
                st.error("""
                **Pneumonia Detected**
                - Seek immediate medical attention
                - Consult with a radiologist or pulmonologist
                - Follow up with appropriate treatment
                """)
            else:
                st.success("""
                **Normal X-Ray**
                - No signs of pneumonia detected
                - Continue regular health monitoring
                - Consult doctor if symptoms persist
                """)
        else:
            st.info("Upload an X-ray image and click 'Analyze' to see results.")
    
    # Information section
    st.markdown("---")
    st.header("📊 About This AI System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Accuracy**
        - Custom CNN: ~85%
        - ResNet50: ~92%
        - DenseNet121: ~94%
        """)
    
    with col2:
        st.markdown("""
        **🔬 Technology**
        - Deep Learning (CNN)
        - Transfer Learning
        - Image Preprocessing
        - CLAHE Enhancement
        """)
    
    with col3:
        st.markdown("""
        **📈 Dataset**
        - 5,863 X-ray images
        - Normal vs Pneumonia
        - Kaggle chest X-ray dataset
        """)

if __name__ == "__main__":
    main()