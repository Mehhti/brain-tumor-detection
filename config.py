"""
Central configuration for the Brain Tumor Detection project.
Edit the paths and hyperparameters here in one place.
"""

import os

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
# All default paths are resolved relative to this file, so the scripts
# work no matter which directory they are launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder that directly contains the `yes/` and `no/` subfolders.
# Change this to match your machine (or set the DATA_DIR env variable).
#   - On Linux / WSL:  "/mnt/c/Users/<you>/Downloads/brain_tumor_dataset"
#   - On Windows:      r"C:\Users\<you>\Downloads\brain_tumor_dataset"
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))

# Optional second dataset: "Brain Tumor Classification (MRI)" by Sartaj
# Bhuvaji (github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet).
# Point SARTAJ_DIR at the folder that contains Training/ and Testing/.
SARTAJ_DIR = os.environ.get("SARTAJ_DIR", os.path.join(BASE_DIR, "data_sartaj"))

# Every source lists, per class, the sub-folders that hold its images.
# Sources are read in the order of USE_SOURCES; when the same scan occurs
# in several sources, the copy from the first source is kept.
_SARTAJ_TUMORS = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
DATA_SOURCES = {
    "br35h": {"dir": DATA_DIR, "classes": {"no": ["no"], "yes": ["yes"]}},
    "sartaj": {"dir": SARTAJ_DIR, "classes": {
        "no": ["Training/no_tumor", "Testing/no_tumor"],
        "yes": [f"{split}/{t}" for split in ("Training", "Testing") for t in _SARTAJ_TUMORS],
    }},
}
# Both sources, chosen by cross-validation: with balanced source x label
# weighting (train.compute_sample_weights) the combined model matched the
# Br35H-only model on Br35H (60 vs 61 errors of 413) and generalised far
# better to Sartaj (AUC 0.964 vs 0.760 for a Br35H-only model). Without
# that weighting, adding Sartaj made Br35H much worse (97 errors).
USE_SOURCES = ["br35h", "sartaj"]

# Where the trained model is saved / loaded from.
MODEL_PATH = os.path.join(BASE_DIR, "best_model_br35h.keras")

# Training outputs.
HISTORY_PLOT_PATH = os.path.join(BASE_DIR, "training_history.png")
ROC_PLOT_PATH     = os.path.join(BASE_DIR, "roc_curve.png")
METRICS_PATH      = os.path.join(BASE_DIR, "test_metrics.json")
DUPLICATES_REPORT_PATH = os.path.join(BASE_DIR, "duplicates_report.json")
CV_RESULTS_PATH   = os.path.join(BASE_DIR, "cv_results.json")
ENSEMBLE_RESULTS_PATH = os.path.join(BASE_DIR, "ensemble_cv_results.json")
OOD_REFERENCE_PATH = os.path.join(BASE_DIR, "ood_reference.npz")

# Out-of-distribution check in the app (see inference.py): an upload whose
# feature-space distance to its OOD_K-th nearest training scan exceeds the
# OOD_QUANTILE of the training scans' own (leave-one-out) distances is
# flagged as unlike the training data. With 0.99, about 1% of genuine
# scans are expected to be flagged.
OOD_K = 5
OOD_QUANTILE = 0.99

# Optional folder with locally downloaded ImageNet weights (useful in
# restricted networks where the automatic download is blocked). If the
# backbone's weight file (see backbones.py, e.g.
# mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5)
# exists there it is used; otherwise weights are downloaded automatically.
LOCAL_WEIGHTS_DIR = DATA_DIR

# ---------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------
# EfficientNetV2-B0 at 224 px was chosen by cross-validation: over 15 paired
# runs (same splits and seeds) it beat MobileNetV2 at 160 px by +1.9 accuracy
# points on average (84.4% vs 82.4%, AUC 0.925 vs 0.900, better in 9/15 runs).
# The direction is consistent but not statistically significant (p ~ 0.16).
# MobileNetV2 at 224 px gave no gain. See README "Results".
BACKBONE    = "efficientnet_v2_b0"   # see backbones.py: mobilenet_v2, efficientnet_v2_b0
IMG_SIZE    = 224     # input size (the app reads it from the saved model)
FINE_TUNE_LAYERS = 30 # stage 2 unfreezes this many layers at the top of the base
BATCH_SIZE  = 32
EPOCHS      = 20      # stage 1 (head training)
FINE_EPOCHS = 10      # stage 2 (fine-tuning)
SEED        = 42

# Stratified 3-way split of the file list: 80% train / 10% val / 10% test
VAL_FRACTION  = 0.1
TEST_FRACTION = 0.1

# Stratified k-fold cross-validation (cross_validate.py). Each fold is the
# test set once; VAL_FRACTION of all images is split off the remaining
# folds for early stopping / checkpointing.
CV_FOLDS = 5

# Number of models trained by train.py and averaged by the app (each with
# its own random validation split and seed); ensemble.py evaluates this.
# A single training run is partly luck (61-69 errors of 416 across seeds
# in nested CV, Br35H only); averaging 3 removes that luck. Final setup
# (both sources): 85.3% on Br35H and 96.1% on Sartaj in nested CV.
ENSEMBLE_SIZE = 3

# Decision threshold for "tumor". If True, it is tuned on the validation
# set (Youden's J, see train.py) and stored in test_metrics.json for the
# app; otherwise 0.5 is used. Both are always reported.
# Off by default: in 5-fold CV the tuned threshold varied from 0.51 to
# 0.65 (0.81 on the single split), because ~60 validation images are too
# few to place it reliably. It traded sensitivity for specificity without
# improving accuracy (87.0% +/- 3.1 vs 87.5% +/- 1.2 at 0.5) and made
# sensitivity unstable (86.9% +/- 9.0 vs 90.8% +/- 3.3), which is the
# wrong trade for a screening aid.
TUNE_THRESHOLD = False

# ---------------------------------------------------------------
# Duplicate / near-duplicate removal (perceptual hash, see dedup.py)
# Done BEFORE the split so copies of one scan cannot end up in both
# train and test. Calibrated on Br35H: at a distance <= 6 (of 64 bits)
# matched pairs are almost always the same scan re-saved with other
# brightness / contrast / crop; from 8 on, different scans start to match.
# ---------------------------------------------------------------
DEDUPLICATE              = True
DUPLICATE_MAX_DISTANCE   = 6      # max Hamming distance between pHashes
# Also catch left-right / top-bottom flipped and 180-degree rotated copies.
# (Matching only left-right mirrors missed 230 copy pairs in Br35H.)
DUPLICATE_MATCH_FLIPS    = True
# A hash match between images with different labels only counts as "the
# same scan" if their pixels correlate at least this much (see dedup.py).
DUPLICATE_CONFLICT_MIN_CORR = 0.95

# ---------------------------------------------------------------
# Manual label review (see label_review.json)
# Images listed there are excluded or relabelled before deduplication,
# e.g. scans labelled "no" that show a visible lesion. Decisions are
# meant to be confirmed or changed by a qualified reader.
# ---------------------------------------------------------------
LABEL_REVIEW_PATH = os.path.join(BASE_DIR, "label_review.json")

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
