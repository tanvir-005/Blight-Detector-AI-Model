import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib
import time

# Set page config
st.set_page_config(
    page_title="Potato Leaf Disease Detection",
    page_icon="🥔",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E4053;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle-text {
        font-size: 1.2rem;
        color: #566573;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #EBF5FB;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #E8F8F5;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .upload-box {
        background-color: #FDF2E9;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def preprocess_image(image):
    """Preprocess image for prediction"""
    img = cv2.resize(image, (64, 64))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_normalized = img_gray / 255.0
    features = hog(img_normalized, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=False)
    return features

def main():
    # Title and description
    st.markdown('<h1 class="title-text">🥔 Potato Leaf Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Upload an image of a potato leaf to detect if it is healthy or has blight.</p>', unsafe_allow_html=True)
    
    # Information box
    with st.container():
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### About the Model
        This application uses an ensemble learning model to detect three types of potato leaf conditions:
        - **Healthy**: Normal, disease-free leaves
        - **Early Blight**: Initial stages of blight infection
        - **Late Blight**: Advanced stages of blight infection
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            # Load and display image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(image, channels='BGR', caption='Uploaded Image', use_container_width=True)
            
            # Add a predict button
            if st.button('Predict Disease', key='predict'):
                with st.spinner('Analyzing image...'):
                    # Load model and make prediction
                    try:
                        model = joblib.load('best_potato_leaf_model.joblib')
                        features = preprocess_image(image)
                        prediction = model.predict([features])[0]
                        
                        # Get prediction probabilities
                        probabilities = model.predict_proba([features])[0]
                        
                        # Display results
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        class_names = ['Healthy', 'Early Blight', 'Late Blight']
                        st.markdown(f"### Prediction: {class_names[prediction]}")
                        
                        # Show confidence scores
                        st.markdown("### Confidence Scores:")
                        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                            st.markdown(f"- {class_name}: {prob:.2%}")
                        
                        # Add a progress bar for the predicted class
                        # Convert numpy float32 to regular Python float
                        confidence = float(probabilities[prediction])
                        st.progress(confidence)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add recommendations based on prediction
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("### Recommendations:")
                        if prediction == 0:
                            st.markdown("✅ Your potato plant appears healthy! Continue regular monitoring and maintenance.")
                        elif prediction == 1:
                            st.markdown("⚠️ Early blight detected. Consider applying fungicide and improving air circulation.")
                        else:
                            st.markdown("❌ Late blight detected. Immediate action required: apply fungicide and consider removing affected leaves.")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit | Model: Ensemble Learning (RF/XGBoost/CatBoost/AdaBoost)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
