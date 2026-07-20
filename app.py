"""
=====================================================================
 Brain Tumor Detection - Streamlit Web App
 ---------------------------------------------------------------
 Upload a brain-MRI image and get a tumor / no-tumor prediction.
 The app first checks that the image looks like a real MRI scan.

 Run (inside your virtual environment):
   python -m streamlit run app.py
=====================================================================
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from PIL import Image
import streamlit as st
from tensorflow import keras

from config import IMG_SIZE, MODEL_PATH, CLASS_NAMES, CONFIDENCE_THRESHOLD
from image_check import check_image_suitability

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")


# ---------------------------------------------------------------
# Load the trained model (cached so it loads only once)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
def predict(model, pil_image):
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(keras.utils.img_to_array(img), axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])   # ~1 -> tumor, ~0 -> healthy
    has_tumor = prob > 0.5
    confidence = prob if has_tumor else (1 - prob)
    label = CLASS_NAMES[1] if has_tumor else CLASS_NAMES[0]
    return has_tumor, confidence, label


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🧠 Brain Tumor Detection")
st.markdown(
    "This system uses **deep learning (MobileNetV2)** to analyze a brain-MRI "
    "image and predict whether a tumor is present."
)
st.divider()

if not os.path.exists(MODEL_PATH):
    st.error(
        f"⚠️ Model file not found: `{MODEL_PATH}`\n\n"
        "Train the model first (run `train.py`) or place the trained "
        "model file next to this app."
    )
    st.stop()

model = load_model()

uploaded = st.file_uploader("Upload a brain-MRI image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    image = Image.open(uploaded)
    is_suitable, reason = check_image_suitability(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input image", use_container_width=True)

    with col2:
        if not is_suitable:
            st.warning("⚠️ **Unsuitable image**")
            st.write(reason)
            st.info("Please upload a real brain-MRI scan (grayscale, with a dark background).")
        else:
            with st.spinner("Analyzing..."):
                has_tumor, confidence, label = predict(model, image)

            if confidence < CONFIDENCE_THRESHOLD:
                st.warning("⚠️ **Low confidence**")
                st.write(
                    f"The model is not confident enough "
                    f"(confidence: {confidence*100:.1f}%). This image may be "
                    "unlike the training data."
                )
                st.metric("Model confidence", f"{confidence*100:.1f}%")
            else:
                st.markdown("### Result")
                if has_tumor:
                    st.error("🔴 **Tumor detected**")
                else:
                    st.success("🟢 **No tumor found**")
                st.metric("Model confidence", f"{confidence*100:.1f}%")
                st.progress(float(confidence))

    st.divider()
    st.caption(
        "⚠️ Disclaimer: this is a decision-support tool and is not a substitute "
        "for a qualified physician. Final diagnosis must be made by a radiologist."
    )
else:
    st.info("👆 Upload a brain-MRI image to begin.")

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "- **Model:** MobileNetV2 (transfer learning)\n"
        "- **Dataset:** Br35H (3000 balanced images)\n"
        "- **Evaluation:** train / validation / independent test split\n"
        "- **Performance:** high accuracy with high tumor sensitivity"
    )
    st.divider()
    st.header("Input validation")
    st.caption(
        "Before analysis, the app checks that the image looks like a real "
        "brain MRI (grayscale, dark background, real structure) and that the "
        "model is confident enough. Otherwise a warning is shown."
    )
