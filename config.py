"""
Central configuration for the Brain Tumor Detection project.
Edit the paths and hyperparameters here in one place.
"""

import os

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
# Folder that directly contains the `yes/` and `no/` subfolders.
# Change this to match your machine.
#   - On Linux / WSL:  "/mnt/c/Users/<you>/Downloads/brain_tumor_dataset"
#   - On Windows:      r"C:\Users\<you>\Downloads\brain_tumor_dataset"
DATA_DIR = os.environ.get("DATA_DIR", "data")

# Where the trained model is saved / loaded from.
MODEL_PATH = "best_model_br35h.keras"

# Optional local ImageNet weights for MobileNetV2 (useful in restricted
# networks where the automatic download is blocked). If the file exists
# it is used; otherwise weights are downloaded automatically.
LOCAL_WEIGHTS_PATH = os.path.join(
    DATA_DIR, "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5")

# ---------------------------------------------------------------
# Image / training hyperparameters
# ---------------------------------------------------------------
IMG_SIZE    = 160     # MobileNetV2 input size
BATCH_SIZE  = 32
EPOCHS      = 20      # stage 1 (head training)
FINE_EPOCHS = 10      # stage 2 (fine-tuning)
SEED        = 42

# Class order MUST match alphabetical folder order: no=0, yes=1
CLASS_NAMES = ["no", "yes"]

# ---------------------------------------------------------------
# Input-image suitability thresholds (used by the web app)
# Loosened so that genuine MRI scans are accepted while clearly
# unrelated images (color photos, documents) are rejected.
# ---------------------------------------------------------------
GRAYNESS_THRESHOLD   = 35     # max avg difference between RGB channels
DARK_RATIO_THRESHOLD = 0.12   # min fraction of dark pixels
STD_THRESHOLD        = 18.0   # min brightness variation (MRI has structure)
CONFIDENCE_THRESHOLD = 0.55   # min model confidence to show a result
