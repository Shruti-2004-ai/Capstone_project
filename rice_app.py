import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ====== CRITICAL DEPLOYMENT FIXES ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== MODEL CONFIG ======
MODEL_PATH = os.path.join(BASE_DIR, "trained_rice_model.h5")  # Absolute path
CLASS_NAMES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# ====== CORE FUNCTIONS ======
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        # Warm-up the model
        model.predict(np.zeros((1, 224, 224, 3)))
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

def process_image(image):
    img = image.resize((224, 224))
    arr = np.array(img)
    if arr.ndim == 2:  # Convert grayscale to RGB
        arr = np.stack((arr,)*3, axis=-1)
    return arr[np.newaxis, ...].astype(np.float32) / 255.0

# ====== STREAMLIT UI ======
st.set_page_config(
    page_title="Rice Classifier Pro",
    page_icon="🍚",
    layout="centered"
)

# Load model
model = load_model()

st.title("🍚 Rice Variety Classifier")
uploaded = st.file_uploader("Upload rice image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        if st.button("Classify"):
            with st.spinner("Analyzing..."):
                try:
                    processed = process_image(img)
                    preds = model.predict(processed, verbose=0)[0]
                    results = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
                    top_class = max(results, key=results.get)
                    
                    st.success(f"**Prediction:** {top_class}")
                    st.metric("Confidence", f"{results[top_class]*100:.1f}%")
                    
                    # Show all probabilities
                    st.subheader("Probabilities:")
                    for name, prob in results.items():
                        st.progress(int(prob * 100), text=f"{name}: {prob*100:.1f}%")
                        
                except Exception as e:
                    st.error(f"Classification failed: {str(e)}")

# ====== DEBUGGING (REMOVE AFTER TESTING) ======
st.sidebar.markdown("### Debug Info")
st.sidebar.write("Current directory:", os.listdir(BASE_DIR))
st.sidebar.write("Model path:", MODEL_PATH)
