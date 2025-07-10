import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain
import plotly.express as px

# ====== INITIAL SETUP ======
st.set_page_config(
    page_title="Rice Classifier Pro",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== CUSTOM STYLING ======
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")  # Create this file for custom styles

# ====== ANIMATIONS ======
def success_animation():
    rain(
        emoji="🎉",
        font_size=30,
        falling_speed=5,
        animation_length=1,
    )

# ====== TENSORFLOW IMPORT ======
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    tf.get_logger().setLevel('ERROR')
except ImportError:
    st.error("TensorFlow not found! Please install with: pip install tensorflow-cpu==2.10.0")
    st.stop()

# ====== APP HEADER ======
colored_header(
    label="🍚 Rice Variety Classifier Pro",
    description="Upload an image to identify rice varieties with AI",
    color_name="blue-70",
)

# ====== MODEL LOADING ======
MODEL_PATHS = [
    "venv/converted_model.keras",
    "venv/fixed_model.keras",
    "venv/trained_rice_model.h5"
]
CLASS_NAMES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
CLASS_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

@st.cache_resource
def load_model_safely():
    with st.spinner("🔍 Loading AI model..."):
        for model_path in MODEL_PATHS:
            if os.path.exists(model_path):
                try:
                    model = load_model(model_path)
                    model.predict(np.zeros((1, 224, 224, 3)))  # Warm-up
                    st.balloons()
                    return model
                except Exception as e:
                    st.warning(f"⚠️ Failed to load {os.path.basename(model_path)}")
        st.error("❌ No working model found!")
        return None

model = load_model_safely()

# ====== IMAGE PROCESSING ======
def process_image(image):
    img = image.resize((224, 224))
    arr = np.array(img)
    if arr.ndim == 2:  # Grayscale
        arr = np.stack((arr,)*3, axis=-1)
    return arr[np.newaxis, ...].astype(np.float32) / 255.0

# ====== PREDICTION ======
def classify(image):
    if not model: return None
    with st.spinner("🧠 Analyzing rice grains..."):
        try:
            processed = process_image(image)
            preds = model.predict(processed, verbose=0)[0]
            return {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        except Exception as e:
            st.error(f"❌ Classification error: {str(e)}")
            return None

# ====== MAIN UI ======
with st.container():
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload Rice Image")
        uploaded = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg","png","jpeg"],
            label_visibility="collapsed"
        )
        
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Your Uploaded Image", use_column_width=True)

    with col2:
        if uploaded:
            pred = classify(img)
            
            if pred:
                top = max(pred, key=pred.get)
                conf = pred[top]
                
                # Animated success
                if conf > 0.7:
                    success_animation()
                
                # Prediction card
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>🔍 Analysis Results</h3>
                    <p class="prediction-text">Variety: <span style="color:{CLASS_COLORS[CLASS_NAMES.index(top)]}">{top}</span></p>
                    <p class="confidence-text">Confidence: {conf*100:.1f}%</p>
                    <div class="progress-bar">
                        <div class="progress" style="width:{conf*100}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Store results
                st.session_state.predictions = st.session_state.get('predictions', []) + [{
                    'file': uploaded.name,
                    'prediction': top,
                    'confidence': f"{conf*100:.1f}%",
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }]
                
                # Interactive feedback
                with st.expander("💬 Provide Feedback", expanded=True):
                    feedback = st.radio(
                        "Was this prediction correct?",
                        ["Select", "✅ Correct", "❌ Incorrect"],
                        horizontal=True
                    )
                    if feedback != "Select":
                        st.session_state.feedback = st.session_state.get('feedback', []) + [{
                            'file': uploaded.name,
                            'prediction': top,
                            'correct': feedback == "✅ Correct",
                            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }]
                        st.toast("Feedback saved!", icon="👍")

# ====== VISUALIZATION ======
if uploaded and pred:
    st.markdown("---")
    st.subheader("📊 Confidence Breakdown")
    
    # Interactive Plotly chart
    fig = px.bar(
        x=list(pred.values()),
        y=list(pred.keys()),
        orientation='h',
        color=list(pred.keys()),
        color_discrete_sequence=CLASS_COLORS,
        labels={'x': 'Confidence', 'y': 'Variety'},
        height=400
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ====== HISTORY DASHBOARD ======
st.markdown("---")
tab1, tab2 = st.tabs(["📜 Prediction History", "📈 Performance Analytics"])

with tab1:
    if st.session_state.get('predictions'):
        df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(
            df.sort_values('time', ascending=False),
            column_config={
                "time": "Timestamp",
                "file": "Image",
                "prediction": "Prediction",
                "confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    help="Prediction confidence level",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full History",
            data=csv,
            file_name="rice_classification_history.csv",
            mime="text/csv"
        )

with tab2:
    if st.session_state.get('feedback'):
        fb_df = pd.DataFrame(st.session_state.feedback)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        accuracy = fb_df['correct'].mean()
        
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy*100:.1f}%")
        with col2:
            st.metric("📊 Total Samples", len(fb_df))
        with col3:
            st.metric("🔄 Last Feedback", fb_df.iloc[-1]['time'])
        
        # Visualization
        st.plotly_chart(
            px.pie(
                fb_df, 
                names='prediction', 
                title='Prediction Distribution',
                color='prediction',
                color_discrete_sequence=CLASS_COLORS
            ),
            use_container_width=True
        )

# ====== SYSTEM INFO ======
with st.expander("⚙️ Technical Details"):
    st.code(f"""
    System Information:
    Python: {os.sys.version.split()[0]}
    TensorFlow: {tf.__version__}
    Available Models: {[os.path.basename(p) for p in MODEL_PATHS if os.path.exists(p)]}
    """)

# ====== SIDEBAR ======
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Rice+Classifier", width=150)
    st.markdown("### 🧠 About This App")
    st.markdown("""
    This AI-powered app classifies rice varieties using deep learning.
    Upload clear images of rice grains for best results.
    """)
    
    st.markdown("### 📌 Quick Tips")
    st.markdown("""
    - Use well-lit, high-quality images
    - Show grains spread out, not clumped
    - Avoid shadows and reflections
    """)
    
    if st.button("🎁 Surprise Me"):
        success_animation()
        st.balloons()