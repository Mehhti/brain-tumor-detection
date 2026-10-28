"""
=====================================================================
 Brain Tumor Detection - Streamlit Web App
 ---------------------------------------------------------------
 Upload a brain-MRI image and get a tumor / no-tumor prediction.
 The app first checks that the image looks like a real MRI scan.
 For a "tumor" result it also shows a Grad-CAM map of the regions
 that most influenced the prediction.

 Run (inside your virtual environment):
   python -m streamlit run app.py
=====================================================================
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import io
import json

import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import tensorflow as tf
from tensorflow import keras

from config import (BASE_DIR, MODEL_PATH, METRICS_PATH, ENSEMBLE_RESULTS_PATH,
                    CONFIDENCE_THRESHOLD)
from image_check import check_image_suitability
from inference import (input_size, prepare_image, predict_tumor_prob,
                       gradcam_heatmap, overlay_heatmap, embed, ood_scores,
                       load_ood_reference)

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")


# ---------------------------------------------------------------
# Load the trained model(s) (cached so they load only once)
# ---------------------------------------------------------------
@st.cache_resource
def load_models(model_files):
    """Load the ensemble members listed by train.py (one file for a single
    model). Returns a list of models."""
    return [keras.models.load_model(os.path.join(BASE_DIR, f)) for f in model_files]


@st.cache_resource
def load_reference(file_name):
    """Training-scan features for the out-of-distribution check."""
    return load_ood_reference(os.path.join(BASE_DIR, file_name))


def load_json(path):
    """Contents of a JSON results file, or None if not available."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def matching_cv_results(metrics, cv):
    """Nested-CV results from ensemble.py if they describe the deployed setup
    (same backbone, input size, ensemble size and number of images)."""
    if not metrics or not cv:
        return None
    same = (cv.get("backbone") == metrics.get("backbone")
            and cv.get("img_size") == metrics.get("img_size")
            and cv.get("ensemble_size") == len(metrics.get("model_files", []))
            and cv.get("n_images") == metrics["n_train"] + metrics["n_val"] + metrics["n_test"])
    return cv if same else None


metrics = load_json(METRICS_PATH)            # written by train.py
cv_results = matching_cv_results(metrics, load_json(ENSEMBLE_RESULTS_PATH))
# Decision threshold chosen by train.py (tuned on the validation set if
# TUNE_THRESHOLD is on); 0.5 if no metrics file is available.
threshold = float(metrics.get("threshold", 0.5)) if metrics else 0.5
# Model file(s) written by train.py; older metrics files list none.
model_files = tuple((metrics or {}).get("model_files") or [os.path.basename(MODEL_PATH)])


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.header("About")
    backbone_names = {"mobilenet_v2": "MobileNetV2",
                      "efficientnet_v2_b0": "EfficientNetV2-B0"}
    backbone = backbone_names.get((metrics or {}).get("backbone"), "MobileNetV2")
    ensemble_note = f", ensemble of {len(model_files)}" if len(model_files) > 1 else ""
    st.write(
        f"- **Model:** {backbone} (transfer learning{ensemble_note})\n"
        f"- **Data:** {' + '.join((metrics or {}).get('sources', ['br35h']))} "
        "(duplicate scans removed before splitting)\n"
        "- **Evaluation:** nested cross-validation + independent test split"
    )
    if cv_results:
        # Preferred: every image is tested once, so far less noisy than the
        # small single test set.
        st.subheader(f"Cross-validated performance ({cv_results['n_folds']}-fold, "
                     f"{cv_results['n_images']} images)")
        # Per data source when there are several: the overall number would be
        # dominated by the largest source.
        per_source = {}
        for f in cv_results["folds"]:
            for name, m in f.get("ensemble_per_source", {}).items():
                per_source.setdefault(name, []).append(m)
        groups = (per_source if len(per_source) > 1
                  else {"all": [f["ensemble"] for f in cv_results["folds"]]})
        for name, folds in groups.items():
            v = {k: np.array([m[k] for m in folds]) for k in
                 ("accuracy", "roc_auc", "sensitivity", "specificity")}
            title = f"**{name}**\n" if name != "all" else ""
            st.write(
                f"{title}"
                f"- **Accuracy:** {v['accuracy'].mean()*100:.1f}% ± {v['accuracy'].std(ddof=1)*100:.1f}\n"
                f"- **ROC AUC:** {v['roc_auc'].mean():.3f} ± {v['roc_auc'].std(ddof=1):.3f}\n"
                f"- **Sensitivity (tumor):** {v['sensitivity'].mean()*100:.1f}% ± {v['sensitivity'].std(ddof=1)*100:.1f}\n"
                f"- **Specificity (healthy):** {v['specificity'].mean()*100:.1f}% ± {v['specificity'].std(ddof=1)*100:.1f}"
            )
    if metrics:
        if cv_results:
            st.caption(f"Single test split (n={metrics['n_test']}): accuracy "
                       f"{metrics['accuracy']*100:.1f}%, AUC {metrics['roc_auc']:.3f}. "
                       f"Decision threshold {threshold:.2f}.")
        else:
            st.subheader(f"Test-set performance (n={metrics['n_test']})")
            st.write(
                f"- **Accuracy:** {metrics['accuracy']*100:.1f}%\n"
                f"- **ROC AUC:** {metrics['roc_auc']:.3f}\n"
                f"- **Sensitivity (tumor):** {metrics['sensitivity']*100:.1f}%\n"
                f"- **Specificity (healthy):** {metrics['specificity']*100:.1f}%\n"
                f"- **Decision threshold:** {threshold:.2f}"
            )
        if metrics.get("n_duplicates_removed"):
            st.caption(f"{metrics['n_duplicates_removed']} duplicate images of "
                       f"{metrics['n_images_found']} were removed before splitting.")
    else:
        st.caption("Run `train.py` to compute test-set metrics.")
    st.divider()
    st.header("Input validation")
    st.caption(
        "Before analysis, the app checks that the image looks like a real "
        "brain MRI (grayscale, dark background, real structure), that it "
        "resembles the training scans in the model's own feature space, and "
        "that the model is confident enough. Otherwise a warning is shown."
    )


# ---------------------------------------------------------------
# Analysis of one uploaded file
# ---------------------------------------------------------------
def analyze(model, data):
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        # Decoded and resized exactly like in training.
        batch = prepare_image(data, input_size(model))
    except (UnidentifiedImageError, OSError, tf.errors.InvalidArgumentError):
        st.error("⚠️ This file could not be read as an image. "
                 "Please upload a JPG, PNG or BMP file.")
        return

    is_suitable, reason = check_image_suitability(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input image", width="stretch")

    with col2:
        if not is_suitable:
            st.warning("⚠️ **Unsuitable image**")
            st.write(reason)
            st.info("Please upload a real brain-MRI scan (grayscale, with a dark background).")
            return

        if reference is not None:
            # Feature-space check: is this image like the training scans?
            score = float(ood_scores(embed(model, batch), reference["embeddings"],
                                     reference["k"])[0])
            if score > reference["threshold"]:
                st.warning("⚠️ **Unlike the training scans**")
                st.write(
                    "This image looks different from every scan the model was "
                    "trained on (for example a CT scan, an X-ray, another body "
                    "part, or an MRI view or sequence that was not in the "
                    "training data), so its prediction would not be reliable."
                )
                st.caption(f"Distance score {score:.3f} > threshold "
                           f"{reference['threshold']:.3f}")
                return

        with st.spinner("Analyzing..."):
            prob = predict_tumor_prob(model, batch)   # ~1 -> tumor, ~0 -> healthy
        has_tumor = prob >= threshold
        confidence = prob if has_tumor else (1 - prob)
        details = (f"P(tumor) = {prob*100:.1f}% · decision threshold = "
                   f"{threshold*100:.0f}%")

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning("⚠️ **Low confidence**")
            st.write(
                f"The model is not confident enough "
                f"(confidence: {confidence*100:.1f}%). This image may be "
                "unlike the training data."
            )
            st.metric("Model confidence", f"{confidence*100:.1f}%")
            st.caption(details)
            return

        st.markdown("### Result")
        if has_tumor:
            st.error("🔴 **Tumor detected**")
        else:
            st.success("🟢 **No tumor found**")
        st.metric("Model confidence", f"{confidence*100:.1f}%")
        st.progress(float(confidence))
        st.caption(details)

    if has_tumor:
        with st.spinner("Computing Grad-CAM..."):
            heatmap = gradcam_heatmap(model, batch)
        st.image(overlay_heatmap(image, heatmap), width="stretch",
                 caption="Grad-CAM: regions that most influenced the tumor "
                         "prediction (red = strongest). The map is coarse and "
                         "shows the rough area, not the exact tumor outline.")


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🧠 Brain Tumor Detection")
st.markdown(
    f"This system uses **deep learning ({backbone})** to analyze a brain-MRI "
    "image and predict whether a tumor is present."
)
st.divider()

missing = [f for f in model_files if not os.path.exists(os.path.join(BASE_DIR, f))]
if missing:
    st.error(
        f"⚠️ Model file(s) not found: {', '.join(f'`{f}`' for f in missing)}\n\n"
        "Train the model first (run `train.py`) or place the trained "
        "model file(s) next to this app."
    )
    st.stop()

model = load_models(model_files)        # a list: one model, or an ensemble
# Only use a reference written by the same training run as the models.
reference = (load_reference(metrics["ood_reference_file"])
             if metrics and metrics.get("ood_reference_file") else None)

uploaded = st.file_uploader("Upload a brain-MRI image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    analyze(model, uploaded.getvalue())
    st.divider()
    st.caption(
        "⚠️ Disclaimer: this is a decision-support tool and is not a substitute "
        "for a qualified physician. Final diagnosis must be made by a radiologist."
    )
else:
    st.info("👆 Upload a brain-MRI image to begin.")
