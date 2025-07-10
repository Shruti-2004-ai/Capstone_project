import os
import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

# ====== Configuration ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fixed_model.keras")
CLASS_NAMES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
PREDICTION_LOG = os.path.join(BASE_DIR, "predictions.csv")
FEEDBACK_LOG = os.path.join(BASE_DIR, "feedback.csv")

# Initialize CSV files with headers
for file, headers in [(PREDICTION_LOG, ['timestamp', 'filename', 'predicted_class', 'confidence', 'probabilities']),
                      (FEEDBACK_LOG, ['timestamp', 'predicted_class', 'actual_class', 'feedback'])]:
    if not os.path.exists(file):
        with open(file, 'w', newline='') as f:
            csv.writer(f).writerow(headers)

# ====== Core Functions ======
@st.cache_resource
def load_model():
    """Load and warm up the model"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        # Warm-up with correct input shape
        warm_up_data = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model.predict(warm_up_data)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

def process_image(image):
    """Standardize image preprocessing"""
    try:
        img = image.resize((224, 224))
        arr = np.array(img)
        
        # Handle various image formats
        if arr.ndim == 2:  # Grayscale
            arr = np.stack((arr,)*3, axis=-1)
        elif arr.shape[2] == 4:  # RGBA
            arr = arr[..., :3]
            
        return np.expand_dims(arr.astype(np.float32) / 255.0, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

def log_data(filepath, data):
    """Thread-safe CSV logging"""
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def plot_probabilities(probabilities):
    """Create styled probability bar plot"""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.set_style("whitegrid")
    plot = sns.barplot(
        x=probabilities,
        y=CLASS_NAMES,
        palette="viridis",
        ax=ax
    )
    plt.xlim(0, 1)
    plt.xlabel("Confidence Score")
    plt.title("Classification Probabilities", pad=20)
    
    # Add value annotations
    for p in plot.patches:
        width = p.get_width()
        plt.text(
            width + 0.02,
            p.get_y() + p.get_height()/2,
            f"{width:.2f}",
            ha="left",
            va="center"
        )
    return fig

def load_logs():
    """Load and process log data"""
    try:
        pred_df = pd.read_csv(PREDICTION_LOG, parse_dates=['timestamp'])
        feedback_df = pd.read_csv(FEEDBACK_LOG, parse_dates=['timestamp'])
        
        # Calculate accuracy from feedback
        if not feedback_df.empty:
            feedback_df['correct'] = feedback_df['predicted_class'] == feedback_df['actual_class']
            accuracy = feedback_df['correct'].mean()
        else:
            accuracy = None
            
        return pred_df, feedback_df, accuracy
    except Exception as e:
        st.error(f"Error loading logs: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), None

# ====== Streamlit UI ======
st.set_page_config(
    page_title="Rice Classifier Pro",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    model = load_model()
    
    # Custom CSS
    st.markdown("""
    <style>
        .main {padding-top: 1rem;}
        .stProgress > div > div > div {background-color: #4CAF50;}
        .st-bb {background-color: #f0f2f6;}
        .st-at {background-color: #ffffff;}
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🍚 Rice Variety Classifier")
    st.markdown("Upload an image of rice grains to identify the variety")
    
    # Main content columns
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        uploaded = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="file_uploader"
        )
        
        if uploaded:
            try:
                img = Image.open(uploaded)
                st.image(
                    img,
                    caption="Uploaded Image",
                    use_column_width=True,
                    output_format="auto"
                )
                
                # Image metadata
                with st.expander("Image Details"):
                    st.write(f"**Format:** {img.format}")
                    st.write(f"**Size:** {img.size}")
                    st.write(f"**Mode:** {img.mode}")
                    
            except UnidentifiedImageError:
                st.error("Invalid image file. Please upload a valid JPG, PNG, or JPEG file.")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    with col2:
        if uploaded and st.button("Classify", type="primary", use_container_width=True):
            with st.spinner("Analyzing rice grains..."):
                processed_img = process_image(img)
                
                if processed_img is not None:
                    try:
                        # Make prediction
                        preds = model.predict(processed_img, verbose=0)[0]
                        results = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
                        top_class = max(results, key=results.get)
                        confidence = results[top_class] * 100
                        
                        # Log prediction
                        log_data(
                            PREDICTION_LOG,
                            [
                                datetime.datetime.now().isoformat(),
                                uploaded.name,
                                top_class,
                                confidence,
                                str(preds.tolist())
                            ]
                        )
                        
                        # Display results
                        st.success(f"**Prediction:** {top_class}")
                        st.metric("Confidence", f"{confidence:.1f}%")
                        
                        # Show probability visualization
                        st.pyplot(plot_probabilities(preds))
                        
                        # Feedback system
                        with st.expander("✏️ Provide Feedback", expanded=False):
                            feedback_col1, feedback_col2 = st.columns(2)
                            
                            with feedback_col1:
                                actual_class = st.selectbox(
                                    "Actual rice variety",
                                    [""] + CLASS_NAMES,
                                    key="feedback_class"
                                )
                            
                            with feedback_col2:
                                feedback = st.text_area(
                                    "Comments (optional)",
                                    placeholder="E.g., 'The rice was mixed varieties'",
                                    key="feedback_text"
                                )
                            
                            if st.button("Submit Feedback", key="feedback_btn"):
                                if actual_class:
                                    log_data(
                                        FEEDBACK_LOG,
                                        [
                                            datetime.datetime.now().isoformat(),
                                            top_class,
                                            actual_class,
                                            feedback
                                        ]
                                    )
                                    st.toast("✅ Feedback submitted successfully!")
                                else:
                                    st.warning("Please select the actual rice variety")
                    
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

    # ====== Analytics Dashboard ======
    st.sidebar.header("📊 Analytics Dashboard")
    
    pred_df, feedback_df, accuracy = load_logs()
    
    # Prediction Statistics
    with st.sidebar.expander("📈 Prediction History", expanded=True):
        if not pred_df.empty:
            st.write(f"Total predictions: {len(pred_df)}")
            
            # Time series chart
            daily_counts = pred_df.set_index('timestamp').resample('D').size()
            st.line_chart(daily_counts)
            
            # Class distribution
            st.subheader("Class Distribution")
            class_counts = pred_df['predicted_class'].value_counts()
            st.bar_chart(class_counts)
        else:
            st.info("No prediction history yet")

    # Accuracy Metrics
    with st.sidebar.expander("🎯 Model Accuracy", expanded=True):
        if accuracy is not None:
            st.metric("Overall Accuracy", f"{accuracy*100:.1f}%")
            
            if not feedback_df.empty:
                # Confusion matrix
                st.subheader("Confusion Matrix")
                confusion = pd.crosstab(
                    feedback_df['predicted_class'],
                    feedback_df['actual_class'],
                    margins=True
                )
                st.dataframe(confusion.style.background_gradient(cmap='Blues'))
        else:
            st.info("No feedback data yet")

    # Data Export
    with st.sidebar.expander("💾 Export Data"):
        if st.button("Download Prediction Log"):
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="rice_predictions.csv",
                mime="text/csv"
            )
        
        if st.button("Download Feedback Log"):
            csv = feedback_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="rice_feedback.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
