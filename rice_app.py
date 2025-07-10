import os
import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import StringIO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

# Streamlit Cloud config
if st.secrets.get("DEPLOYED", False):
    MODEL_PATH = "fixed_model.keras"  # Use relative path in cloud
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce cloud logs
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "fixed_model.keras")

# ====== Configuration ======
CLASS_NAMES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
PREDICTION_LOG = os.path.join(os.path.dirname(MODEL_PATH), "predictions.csv")
FEEDBACK_LOG = os.path.join(os.path.dirname(MODEL_PATH), "feedback.csv")

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
        warm_up_data = np.zeros((1, 128, 128, 3), dtype=np.float32)
        model.predict(warm_up_data)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

def process_image(image):
    try:
        img = image.resize((128, 128))
        arr = np.array(img)

        if arr.ndim == 2:
            arr = np.stack((arr,) * 3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[..., :3]

        return np.expand_dims(arr.astype(np.float32) / 255.0, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

def log_data(filepath, data):
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def plotly_probabilities(probabilities):
    df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Confidence': probabilities
    })
    fig = px.bar(
        df, 
        x='Confidence', 
        y='Class', 
        orientation='h', 
        text='Confidence',
        color='Confidence',
        color_continuous_scale='viridis'
    )
    fig.update_layout(title="Classification Probabilities", xaxis_range=[0, 1])
    return fig

@st.cache_data
def load_logs():
    try:
        pred_df = pd.read_csv(PREDICTION_LOG, parse_dates=['timestamp'])
        feedback_df = pd.read_csv(FEEDBACK_LOG, parse_dates=['timestamp'])

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

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

        if uploaded and uploaded.type.startswith("image/"):
            try:
                img = Image.open(uploaded)
                st.image(img, caption="Uploaded Image", use_column_width=True)

                with st.expander("Image Details"):
                    st.write(f"**Format:** {img.format}")
                    st.write(f"**Size:** {img.size}")
                    st.write(f"**Mode:** {img.mode}")

                st.session_state["img"] = img
                st.session_state["img_name"] = uploaded.name
            except UnidentifiedImageError:
                st.error("Invalid image file. Please upload a valid JPG, PNG, or JPEG file.")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        elif uploaded:
            st.error("Please upload a valid image file (JPG, PNG, JPEG).")

    with col2:
        if st.button("Classify", type="primary", use_container_width=True):
            if "img" in st.session_state:
                with st.spinner("Analyzing rice grains..."):
                    processed_img = process_image(st.session_state["img"])
                    if processed_img is not None:
                        try:
                            preds = model.predict(processed_img, verbose=0)[0]
                            top_class = CLASS_NAMES[np.argmax(preds)]
                            confidence = max(0, min(100, float(np.max(preds)) * 100))

                            st.session_state["prediction"] = {
                                "class": top_class,
                                "confidence": confidence,
                                "probs": preds.tolist()
                            }

                            log_data(PREDICTION_LOG, [
                                datetime.datetime.now().isoformat(),
                                st.session_state.get("img_name", "unknown"),
                                top_class,
                                confidence,
                                str(preds.tolist())
                            ])
                        except Exception as e:
                            st.error(f"Prediction failed: {str(e)}")

        if "prediction" in st.session_state:
            pred = st.session_state["prediction"]
            st.success(f"**Prediction:** {pred['class']}")
            st.metric("Confidence", f"{pred['confidence']:.1f}%")
            st.plotly_chart(plotly_probabilities(pred["probs"]), use_container_width=True)
            st.info("Not quite right? Provide feedback below to help us improve!")

            with st.expander("✏️ Provide Feedback", expanded=False):
                feedback_col1, feedback_col2 = st.columns(2)

                with feedback_col1:
                    actual_class = st.selectbox(
                        "Actual rice variety",
                        [""] + CLASS_NAMES,
                        key="feedback_class"
                    )

                with feedback_col2:
                    feedback_text = st.text_area(
                        "Comments (optional)",
                        placeholder="E.g., 'The rice was mixed varieties'",
                        key="feedback_text"
                    )

                if st.button("Submit Feedback", key="feedback_btn"):
                    try:
                        if actual_class:
                            log_data(FEEDBACK_LOG, [
                                datetime.datetime.now().isoformat(),
                                pred["class"],
                                actual_class,
                                feedback_text
                            ])
                            st.toast("✅ Feedback submitted successfully!")
                            st.session_state["feedback_class"] = ""
                            st.session_state["feedback_text"] = ""
                        else:
                            st.warning("Please select the actual rice variety")
                    except Exception as e:
                        st.error(f"Feedback submission failed: {str(e)}")

    # ====== Analytics Dashboard ======
    st.sidebar.header("📊 Analytics Dashboard")

    pred_df, feedback_df, accuracy = load_logs()

    with st.sidebar.expander("📈 Prediction History", expanded=True):
        if not pred_df.empty:
            st.write(f"Total predictions: {len(pred_df)}")

            daily_counts = pred_df.set_index('timestamp').resample('D').size()
            st.line_chart(daily_counts)

            st.subheader("Class Distribution")
            class_counts = pred_df['predicted_class'].value_counts()
            st.bar_chart(class_counts)
        else:
            st.info("No prediction history yet")

    with st.sidebar.expander("🎯 Model Accuracy", expanded=True):
        if accuracy is not None:
            st.metric("Overall Accuracy", f"{accuracy*100:.1f}%")

            if not feedback_df.empty:
                st.subheader("Confusion Matrix")
                confusion = pd.crosstab(
                    feedback_df['predicted_class'],
                    feedback_df['actual_class'],
                    margins=True
                ).reindex(
                    index=CLASS_NAMES + ['All'],
                    columns=CLASS_NAMES + ['All'],
                    fill_value=0
                )

                st.dataframe(confusion.style.background_gradient(cmap='Blues'))
        else:
            st.info("No feedback data yet")

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
