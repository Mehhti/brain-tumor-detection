"""
=====================================================================
 Brain Tumor Detection - Training Script
 ---------------------------------------------------------------
 Method: Two-stage Transfer Learning with MobileNetV2
 Dataset: Br35H :: Brain Tumor Detection 2020 (3000 balanced images)
          https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

 Key practices:
   1) Duplicate / near-duplicate images (perceptual hash, see dedup.py)
      are removed BEFORE splitting, so copies of one scan cannot end up
      in both the training and the test set.
   2) Standard 3-way split: train / validation / independent test.
      The split is done once on the FILE LIST (stratified), so the
      three sets are guaranteed to be disjoint.
   3) Full-buffer shuffle for proper data mixing (training set only).
   4) Class weighting for robustness against class imbalance.
   5) Two-stage training (frozen base -> fine-tuning).
   6) The final test score is computed on the SAVED best model,
      i.e. exactly the model the web app uses.

 The building blocks below are plain functions, so cross_validate.py
 reuses exactly the same data loading, training and evaluation.

 Expected folder layout (see config.py -> DATA_DIR):
   data_dir/
       yes/    -> 1500 tumorous MRI images
       no/     -> 1500 non-tumorous MRI images
       pred/   -> (ignored)
=====================================================================
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

from config import (DATA_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, FINE_EPOCHS,
                    SEED, MODEL_PATH, LOCAL_WEIGHTS_DIR, CLASS_NAMES,
                    VAL_FRACTION, TEST_FRACTION, HISTORY_PLOT_PATH,
                    ROC_PLOT_PATH, METRICS_PATH, DEDUPLICATE,
                    DUPLICATE_MAX_DISTANCE, DUPLICATE_MATCH_FLIPS,
                    DUPLICATES_REPORT_PATH, TUNE_THRESHOLD, BACKBONE,
                    FINE_TUNE_LAYERS, ENSEMBLE_SIZE, LABEL_REVIEW_PATH,
                    DATA_SOURCES, USE_SOURCES, OOD_REFERENCE_PATH, OOD_K,
                    OOD_QUANTILE, DUPLICATE_CONFLICT_MIN_CORR)
from dedup import find_duplicates
from backbones import get_backbone

AUTOTUNE = tf.data.AUTOTUNE


def imagenet_weights():
    """Local ImageNet weights for BACKBONE if available (helps in restricted
    networks), otherwise "imagenet" to download them."""
    path = os.path.join(LOCAL_WEIGHTS_DIR,
                        get_backbone(BACKBONE)["weights_file"](IMG_SIZE))
    return path if os.path.exists(path) else "imagenet"


IMAGE_EXTENSIONS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


# ---------------------------------------------------------------
# Load the file list
# ---------------------------------------------------------------
def list_image_files(data_dir, class_names):
    """Return (paths, labels) for every image under data_dir/<class>/."""
    paths, labels = [], []
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        for root, _, files in os.walk(class_dir):
            for name in sorted(files):
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    paths.append(os.path.join(root, name))
                    labels.append(label)
    return np.array(paths), np.array(labels, dtype="int32")


def list_source_files(name):
    """Return (paths, labels) for one entry of DATA_SOURCES."""
    source = DATA_SOURCES[name]
    paths, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        for sub in source["classes"][class_name]:
            folder = os.path.join(source["dir"], sub)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"[{name}] folder not found: {folder}")
            for root, _, files in os.walk(folder):
                for f in sorted(files):
                    if f.lower().endswith(IMAGE_EXTENSIONS):
                        paths.append(os.path.join(root, f))
                        labels.append(label)
    return np.array(paths), np.array(labels, dtype="int32")


def display_path(path):
    """'<source>/<path inside the source>' with forward slashes, e.g.
    'br35h/no/no1102.jpg' - used in reports and in label_review.json."""
    for name, source in DATA_SOURCES.items():
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(source["dir"]))
        if not rel.startswith(".."):
            return f"{name}/{rel}".replace("\\", "/")
    return path.replace("\\", "/")


def strata(labels, sources):
    """Stratification key per image: source x label."""
    return np.char.add(np.asarray(sources, dtype=str), np.asarray(labels).astype(str))


def make_folds(labels, sources, n_folds):
    """Stratified k-fold, made separately within each source and merged, so
    a source's folds do not change when other sources are added.
    Returns a list of (rest_idx, test_idx)."""
    test_of = np.empty(len(labels), dtype=int)
    for name in np.unique(sources):
        idx = np.flatnonzero(sources == name)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        for k, (_, test) in enumerate(skf.split(idx, labels[idx])):
            test_of[idx[test]] = k
    return [(np.flatnonzero(test_of != k), np.flatnonzero(test_of == k))
            for k in range(n_folds)]


# ---------------------------------------------------------------
# Remove duplicate / near-duplicate images (before any split)
#   One image is kept per group of copies; groups whose copies carry
#   different labels are dropped entirely. The full list of removed
#   files is written to DUPLICATES_REPORT_PATH for manual review.
# ---------------------------------------------------------------
def remove_duplicates(all_paths, all_labels):
    """Return (paths, labels, n_removed, copies) with duplicates removed.
    copies maps each kept path to all paths of its duplicate group."""
    n_found = len(all_paths)
    print(f"\nChecking for duplicate / near-duplicate images (pHash distance "
          f"<= {DUPLICATE_MAX_DISTANCE}"
          f"{', flipped/rotated copies included' if DUPLICATE_MATCH_FLIPS else ''}) ...")
    keep, dup_groups, conflict_groups, unreadable = find_duplicates(
        all_paths, all_labels, DUPLICATE_MAX_DISTANCE, DUPLICATE_MATCH_FLIPS,
        DUPLICATE_CONFLICT_MIN_CORR)

    def rel(i):
        return display_path(all_paths[i])

    n_removed = int((~keep).sum())
    removed_per_class = {name: int(((all_labels == c) & ~keep).sum())
                         for c, name in enumerate(CLASS_NAMES)}
    n_dup_copies = sum(len(g) - 1 for g in dup_groups)
    n_conflict = sum(len(g) for g in conflict_groups)

    print(f"  {n_dup_copies} duplicate copies removed "
          f"({len(dup_groups)} groups, one image kept per group)")
    print(f"  {n_conflict} images removed from {len(conflict_groups)} groups "
          f"with conflicting labels")
    if unreadable:
        print(f"  {len(unreadable)} unreadable files skipped")
    print(f"  -> removed {n_removed} of {n_found} images "
          f"({', '.join(f'{k}={v}' for k, v in removed_per_class.items())}); "
          f"{n_found - n_removed} images remain")

    report = {
        "settings": {"max_distance": DUPLICATE_MAX_DISTANCE,
                     "match_flips": DUPLICATE_MATCH_FLIPS,
                     "conflict_min_corr": DUPLICATE_CONFLICT_MIN_CORR},
        "n_images_found": n_found,
        "n_images_kept": n_found - n_removed,
        "n_removed": n_removed,
        "removed_per_class": removed_per_class,
        "n_duplicate_copies_removed": n_dup_copies,
        "n_conflict_images_removed": n_conflict,
        "n_unreadable": len(unreadable),
        "duplicate_groups": [{"kept": rel(g[0]), "removed": [rel(i) for i in g[1:]]}
                             for g in dup_groups],
        "conflict_groups": [[rel(i) for i in g] for g in conflict_groups],
        "unreadable": [rel(i) for i in unreadable],
    }
    with open(DUPLICATES_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("  Details saved to:", DUPLICATES_REPORT_PATH)

    copies = {all_paths[g[0]]: [all_paths[i] for i in g] for g in dup_groups}
    return all_paths[keep], all_labels[keep], n_removed, copies


# ---------------------------------------------------------------
# Manual label review (LABEL_REVIEW_PATH)
#   {"images": [{"path": "br35h/no/no1102.jpg", "action": "exclude", ...},
#               {"path": "...", "action": "relabel", "label": "yes"}, ...]}
#   Paths are "<source>/<path inside the source>" (see display_path).
#   A decision applies to the image's whole duplicate group, so a re-saved
#   copy under another file name cannot slip back in with the old label.
# ---------------------------------------------------------------
def _rel_key(path):
    return display_path(path).lower()


def apply_label_review(paths, labels, copies=None):
    """Return (paths, labels, summary) after applying the review decisions."""
    if not os.path.exists(LABEL_REVIEW_PATH):
        return paths, labels, {"n_excluded": 0, "n_relabelled": 0}
    with open(LABEL_REVIEW_PATH, encoding="utf-8") as f:
        decisions = {d["path"].replace("\\", "/").lower(): d
                     for d in json.load(f)["images"]}

    copies = copies or {}
    keep = np.ones(len(paths), dtype=bool)
    labels = labels.copy()
    used, excluded, relabelled = set(), [], []
    for i, path in enumerate(paths):
        group = [_rel_key(p) for p in copies.get(path, [path])]
        decision = next((decisions[k] for k in group if k in decisions), None)
        if decision is None:
            continue
        used.update(k for k in group if k in decisions)
        if decision["action"] == "exclude":
            keep[i] = False
            excluded.append(_rel_key(path))
        elif decision["action"] == "relabel":
            labels[i] = CLASS_NAMES.index(decision["label"])
            relabelled.append(_rel_key(path))
        elif decision["action"] != "keep":
            raise ValueError(f"Unknown review action {decision['action']!r}")

    print(f"\nLabel review ({os.path.basename(LABEL_REVIEW_PATH)}): "
          f"{len(excluded)} excluded, {len(relabelled)} relabelled")
    missing = sorted(set(decisions) - used)
    if missing:
        print("  Not found among the kept images:", ", ".join(missing))
    return paths[keep], labels[keep], {"n_excluded": len(excluded),
                                       "n_relabelled": len(relabelled)}


def source_of(paths):
    """Source name of every path (see display_path)."""
    return np.array([display_path(p).split("/", 1)[0] for p in paths])


def prepare_dataset():
    """List the images of all USE_SOURCES, remove duplicates (also across
    sources) and apply the label review.
    Returns (paths, labels, sources, info) - shared by train.py,
    cross_validate.py and ensemble.py so they all use exactly the same data."""
    paths, labels = [], []
    for name in USE_SOURCES:
        p, l = list_source_files(name)
        print(f"Found {len(p)} images in [{name}] {DATA_SOURCES[name]['dir']}")
        paths.append(p)
        labels.append(l)
    paths, labels = np.concatenate(paths), np.concatenate(labels)
    info = {"sources": list(USE_SOURCES), "n_images_found": len(paths),
            "n_duplicates_removed": 0}
    copies = {}
    if DEDUPLICATE:
        paths, labels, info["n_duplicates_removed"], copies = remove_duplicates(paths, labels)
    paths, labels, review = apply_label_review(paths, labels, copies)
    info["n_label_review_excluded"] = review["n_excluded"]
    info["n_label_review_relabelled"] = review["n_relabelled"]
    sources = source_of(paths)
    info["per_source"] = {
        name: {cls: int(((sources == name) & (labels == c)).sum())
               for c, cls in enumerate(CLASS_NAMES)} for name in USE_SOURCES}
    print(f"Images used: {len(paths)} "
          + " | ".join(f"{name}: " + ", ".join(f"{k}={v}" for k, v in counts.items())
                       for name, counts in info["per_source"].items()))
    return paths, labels, sources, info


# ---------------------------------------------------------------
# Class weights (computed from training labels)
# ---------------------------------------------------------------
def compute_class_weight(train_labels):
    n0 = int((train_labels == 0).sum())
    n1 = int((train_labels == 1).sum())
    total = n0 + n1
    class_weight = {0: total / (2.0 * n0), 1: total / (2.0 * n1)}
    print(f"Counts -> no={n0}, yes={n1} | class_weight={class_weight}")
    return class_weight


def compute_sample_weights(labels, sources):
    """Per-image weights giving every source x label group the same total
    weight. With several sources, class weighting alone is not enough: if
    one source supplies most of the healthy scans and another most of the
    tumors, the model learns "this source's image style = this label".
    Joint CV of Br35H + Sartaj with plain class weights showed exactly that
    (Br35H tumors were missed far more often)."""
    keys = strata(labels, sources)
    groups, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    weights = len(keys) / (len(groups) * counts[inverse])
    print("Sample weights per source x label: " + ", ".join(
        f"{g}={len(keys) / (len(groups) * c):.2f} (n={c})" for g, c in zip(groups, counts)))
    return weights.astype("float32")


def training_datasets(train_paths, train_labels, val_paths, val_labels):
    """(train_ds, val_ds, class_weight) with the right weighting: class
    weights for a single source; balanced source x label sample weights for
    several sources, applied to validation too, so early stopping and model
    selection also treat every source x label group equally."""
    if len(np.unique(source_of(train_paths))) == 1:
        return (make_dataset(train_paths, train_labels, shuffle=True),
                make_dataset(val_paths, val_labels, shuffle=False),
                compute_class_weight(train_labels))
    train_w = compute_sample_weights(train_labels, source_of(train_paths))
    val_w = compute_sample_weights(val_labels, source_of(val_paths))
    return (make_dataset(train_paths, train_labels, shuffle=True, weights=train_w),
            make_dataset(val_paths, val_labels, shuffle=False, weights=val_w),
            None)


# ---------------------------------------------------------------
# Performance pipeline (full-buffer shuffle for real mixing)
#   Images are decoded and resized the same way as
#   keras.utils.image_dataset_from_directory (RGB, bilinear).
# ---------------------------------------------------------------
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    return img, label


@tf.autograph.experimental.do_not_convert          # plain TF ops, no conversion needed
def _load_weighted_image(path, label, weight):
    img, label = load_image(path, label)
    return img, label, weight


def make_dataset(paths, labels, shuffle, weights=None):
    """Batched (image, label) dataset, or (image, label, weight) if per-image
    weights are given (Keras then weights the loss and the metrics)."""
    labels = labels.astype("float32").reshape(-1, 1)   # binary label mode
    if weights is None:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(load_image, num_parallel_calls=AUTOTUNE).cache()
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights))
        ds = ds.map(_load_weighted_image, num_parallel_calls=AUTOTUNE).cache()
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# ---------------------------------------------------------------
# Build the transfer-learning model (with data augmentation)
# ---------------------------------------------------------------
def build_model(weights="auto"):
    """Return (model, base_model) with the base frozen, ready for stage 1.
    weights: "auto" (local file or ImageNet download), or e.g. None for a
    randomly initialised base (tests)."""
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    backbone = get_backbone(BACKBONE)
    base_model = backbone["build"](
        IMG_SIZE, imagenet_weights() if weights == "auto" else weights)
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    if backbone["preprocess"] is not None:
        x = backbone["preprocess"](x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model, base_model


# ---------------------------------------------------------------
# Two-stage training
#   Callbacks use the validation set only, never the test set.
#   The same ModelCheckpoint is shared by both stages, so the file
#   on disk always holds the best model across stage 1 AND stage 2.
#   EarlyStopping, however, is created fresh for each stage: Keras
#   does not reset its best score between fit() calls, so a shared
#   instance would judge fine-tuning against the stage-1 best and
#   stop it early before it had a chance to improve.
# ---------------------------------------------------------------
def make_early_stopping():
    return keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=6,
        restore_best_weights=True)


def train_model(train_ds, val_ds, class_weight, model_path, verbose="auto",
                show_summary=False):
    """Train both stages; the best model is saved to model_path.
    Returns (history, history_fine)."""
    model, base_model = build_model()
    if show_summary:
        model.summary()

    shared_callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_auc", mode="max", save_best_only=True),
    ]

    # Stage 1: train the classification head (frozen base)
    print("\n=== Stage 1: training the head (base frozen) ===")
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=EPOCHS, class_weight=class_weight,
                        callbacks=[make_early_stopping(), *shared_callbacks],
                        verbose=verbose)

    # Stage 2: fine-tuning (unfreeze the top FINE_TUNE_LAYERS layers)
    print("\n=== Stage 2: fine-tuning ===")
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),   # very small LR for fine-tuning
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")])

    history_fine = model.fit(train_ds, validation_data=val_ds,
                             epochs=FINE_EPOCHS, class_weight=class_weight,
                             callbacks=[make_early_stopping(), *shared_callbacks],
                             verbose=verbose)
    return history, history_fine


# ---------------------------------------------------------------
# Decision threshold
#   A probability >= threshold means "tumor". Besides the default 0.5,
#   a threshold can be tuned on the VALIDATION set (never the test set)
#   by maximizing Youden's J = sensitivity + specificity - 1, which
#   balances missed tumors against false alarms.
# ---------------------------------------------------------------
def choose_threshold(y_true, y_prob):
    """Return the threshold that maximizes Youden's J on (y_true, y_prob).
    Ties are broken towards 0.5; the threshold is placed halfway between
    neighbouring scores so it is not glued to a single image's score."""
    # Keep every distinct score as a candidate (decreasing order), so the
    # midpoint below never skips over an image.
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
    j = tpr - fpr
    candidates = np.flatnonzero(np.isclose(j, j.max()))
    midpoints = []
    for k in candidates:
        upper = min(thresholds[k], 1.0)                # thresholds[0] is +inf
        lower = thresholds[k + 1] if k + 1 < len(thresholds) else 0.0
        midpoints.append((upper + lower) / 2.0)
    midpoints = np.array(midpoints)
    return float(midpoints[np.argmin(np.abs(midpoints - 0.5))])


def compute_metrics(y_true, y_prob, threshold):
    """Classification metrics for probabilities y_prob at a threshold."""
    y_pred = (y_prob >= threshold).astype("int32")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_test": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,  # tumor recall
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,  # healthy recall
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
    }


# ---------------------------------------------------------------
# Evaluation of the SAVED best model on a held-out set
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Ensemble members
#   Member 0 uses the standard train/val split. Further members are
#   trained on the same pool (train + val) with a different random
#   stratified validation split and seed, so every member keeps almost
#   all training data and the members still differ. The test set is
#   never part of the pool.
# ---------------------------------------------------------------
def member_model_path(i):
    """best_model_br35h.keras for member 0, best_model_br35h_2.keras, ..."""
    if i == 0:
        return MODEL_PATH
    root, ext = os.path.splitext(MODEL_PATH)
    return f"{root}_{i + 1}{ext}"


def member_split(pool_paths, pool_labels, member, val_size, pool_sources):
    """Random train/val split of the pool for one member, stratified by
    source x label. Returns (train_paths, val_paths, train_labels, val_labels)."""
    return train_test_split(pool_paths, pool_labels, test_size=val_size,
                            stratify=strata(pool_labels, pool_sources),
                            random_state=SEED + 100 * member)


def build_ood_reference(model_path, ref_paths, ref_labels, test_paths, test_labels):
    """Save the training scans' feature vectors and a calibrated distance
    threshold for the app's out-of-distribution check (see inference.py).
    The threshold is the OOD_QUANTILE of the leave-one-out distances within
    the reference. Returns a summary dict, including how many genuine test
    scans the check would reject."""
    from inference import embed, ood_scores      # local: avoids a cycle at import

    model = keras.models.load_model(model_path)

    def embeddings(paths, labels):
        images = np.concatenate([x.numpy() for x, _ in make_dataset(paths, labels, shuffle=False)])
        return embed(model, images)

    reference = embeddings(ref_paths, ref_labels)
    sims = reference @ reference.T
    np.fill_diagonal(sims, -np.inf)                       # leave one out
    loo = 1.0 - np.sort(sims, axis=1)[:, -OOD_K]
    threshold = float(np.quantile(loo, OOD_QUANTILE))
    test_scores = ood_scores(embeddings(test_paths, test_labels), reference, OOD_K)
    np.savez_compressed(OOD_REFERENCE_PATH, embeddings=reference.astype("float16"),
                        threshold=threshold, k=OOD_K)
    summary = {"threshold": threshold, "k": OOD_K, "quantile": OOD_QUANTILE,
               "n_reference": int(len(reference)),
               "test_rejected": int((test_scores > threshold).sum()),
               "n_test": int(len(test_scores))}
    print(f"OOD reference: {summary['n_reference']} scans, threshold {threshold:.4f}; "
          f"would reject {summary['test_rejected']}/{summary['n_test']} genuine test scans "
          f"-> {OOD_REFERENCE_PATH}")
    return summary


def per_source_metrics(labels, probs, sources, threshold):
    """compute_metrics for each source that has both classes in the set."""
    out = {}
    for name in np.unique(sources):
        m = sources == name
        if len(np.unique(labels[m])) == 2:
            out[str(name)] = compute_metrics(labels[m], probs[m], threshold)
    return out


def evaluate_saved_model(model_path, val_paths, val_labels, test_paths, test_labels):
    """Reload model_path, tune a threshold on the validation set and score
    the test set at 0.5 and at the tuned threshold.
    Returns (test_prob, tuned_threshold, metrics_at_0.5, metrics_at_tuned)."""
    # model_path may also be a list of paths (an ensemble): probabilities
    # are then averaged over the members.
    model_paths = model_path if isinstance(model_path, (list, tuple)) else [model_path]
    val_ds = make_dataset(val_paths, val_labels, shuffle=False)
    test_ds = make_dataset(test_paths, test_labels, shuffle=False)
    val_prob, test_prob = 0.0, 0.0
    for path in model_paths:
        best_model = keras.models.load_model(path)
        # The datasets are not shuffled, so predictions line up with the labels.
        val_prob = val_prob + best_model.predict(val_ds, verbose=0).ravel()
        test_prob = test_prob + best_model.predict(test_ds, verbose=0).ravel()
    val_prob, test_prob = val_prob / len(model_paths), test_prob / len(model_paths)

    tuned = choose_threshold(val_labels, val_prob)
    return (test_prob, tuned,
            compute_metrics(test_labels, test_prob, 0.5),
            compute_metrics(test_labels, test_prob, tuned))


def main():
    # -----------------------------------------------------------
    # 1) Reproducibility
    # -----------------------------------------------------------
    keras.utils.set_random_seed(SEED)   # seeds Python, NumPy and TensorFlow

    # -----------------------------------------------------------
    # 2) Load the file list, remove duplicates and apply the label
    #    review (all before the split)
    # -----------------------------------------------------------
    all_paths, all_labels, all_sources, data_info = prepare_dataset()

    # -----------------------------------------------------------
    # 3) Split the file list once (3-way split, stratified by
    #    source x label): 80% train / 10% validation / 10% test.
    #    Splitting file paths (instead of take/skip on a shuffled
    #    tf.data pipeline) guarantees the three sets never overlap.
    # -----------------------------------------------------------
    (train_paths, temp_paths, train_labels, temp_labels,
     train_sources, temp_sources) = train_test_split(
        all_paths, all_labels, all_sources, test_size=VAL_FRACTION + TEST_FRACTION,
        stratify=strata(all_labels, all_sources), random_state=SEED)

    (val_paths, test_paths, val_labels, test_labels,
     val_sources, test_sources) = train_test_split(
        temp_paths, temp_labels, temp_sources,
        test_size=TEST_FRACTION / (VAL_FRACTION + TEST_FRACTION),
        stratify=strata(temp_labels, temp_sources), random_state=SEED)

    # Sanity check: the three sets must be disjoint.
    assert not set(train_paths) & set(val_paths)
    assert not set(train_paths) & set(test_paths)
    assert not set(val_paths) & set(test_paths)

    print("Classes:", CLASS_NAMES)
    print(f"Images -> train: {len(train_paths)}, val: {len(val_paths)}, "
          f"test: {len(test_paths)}")

    # -----------------------------------------------------------
    # 4) Train (class weights, augmentation, two stages), once per
    #    ensemble member (ENSEMBLE_SIZE = 1: a single model)
    # -----------------------------------------------------------
    print(f"Backbone: {BACKBONE} @ {IMG_SIZE}px | ensemble size: {ENSEMBLE_SIZE}")
    pool_paths = np.concatenate([train_paths, val_paths])
    pool_labels = np.concatenate([train_labels, val_labels])
    pool_sources = np.concatenate([train_sources, val_sources])
    model_paths = [member_model_path(i) for i in range(ENSEMBLE_SIZE)]
    for i, path in enumerate(model_paths):
        if i == 0:
            m_train, m_val, m_train_labels, m_val_labels = (
                train_paths, val_paths, train_labels, val_labels)
        else:
            m_train, m_val, m_train_labels, m_val_labels = member_split(
                pool_paths, pool_labels, i, len(val_paths) / len(pool_paths),
                pool_sources)
        if ENSEMBLE_SIZE > 1:
            print(f"\n--- Ensemble member {i + 1}/{ENSEMBLE_SIZE} -> {path} ---")
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED + 1000 * i)
        histories = train_model(
            *training_datasets(m_train, m_train_labels, m_val, m_val_labels),
            path, show_summary=(i == 0))
        if i == 0:
            history, history_fine = histories      # plotted below

    # -----------------------------------------------------------
    # 5) Final evaluation on the INDEPENDENT test set
    #    The best model(s) are reloaded from disk, so the reported score
    #    belongs to exactly what the web app will use. The test data was
    #    never used for training, early stopping, or model selection, so
    #    the score is fair and defensible. (With an ensemble, the optional
    #    tuned threshold uses member 0's validation set, which the other
    #    members may have trained on, so it is only indicative.)
    # -----------------------------------------------------------
    print(f"\n=== Final evaluation on the independent test set "
          f"({len(model_paths)} model(s)) ===")
    y_prob, tuned, at_default, at_tuned = evaluate_saved_model(
        model_paths, val_paths, val_labels, test_paths, test_labels)
    y_true = test_labels

    # The app uses this threshold (see TUNE_THRESHOLD in config.py).
    test_metrics = at_tuned if TUNE_THRESHOLD else at_default
    threshold = test_metrics["threshold"]
    y_pred = (y_prob >= threshold).astype("int32")

    print(f"\nDecision threshold: {threshold:.3f} "
          f"({'tuned on validation, Youden J' if TUNE_THRESHOLD else 'fixed'})")
    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("\nClassification Report (Test):")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=CLASS_NAMES, digits=4))
    print(f"ROC AUC (Test): {test_metrics['roc_auc']:.4f}")
    for name, m in (("threshold 0.5", at_default), (f"tuned {tuned:.3f}", at_tuned)):
        print(f"  {name:>15}: accuracy {m['accuracy']:.4f}, "
              f"sensitivity {m['sensitivity']:.4f}, specificity {m['specificity']:.4f}")

    metrics = {
        "model_path": os.path.basename(MODEL_PATH),
        "model_files": [os.path.basename(p) for p in model_paths],
        "backbone": BACKBONE,
        "img_size": IMG_SIZE,
        **data_info,
        "n_train": int(len(train_paths)),
        "n_val": int(len(val_paths)),
        "threshold_source": "validation (Youden's J)" if TUNE_THRESHOLD else "fixed",
        **test_metrics,
        "test_per_source": per_source_metrics(test_labels, y_prob, test_sources, threshold),
        # Reference = member 0's training scans (the app embeds with member 0).
        "ood": build_ood_reference(model_paths[0], train_paths, train_labels,
                                   test_paths, test_labels),
        "ood_reference_file": os.path.basename(OOD_REFERENCE_PATH),
        "metrics_at_0.5": at_default,
        "metrics_at_val_tuned": at_tuned,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Test metrics saved to:", METRICS_PATH)

    # -----------------------------------------------------------
    # 6) Plot the ROC curve (test set)
    # -----------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {test_metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance")
    plt.scatter([1 - test_metrics["specificity"]], [test_metrics["sensitivity"]],
                color="red", zorder=3, label=f"Threshold {threshold:.2f}")
    plt.title("ROC Curve - Independent Test Set")
    plt.xlabel("False Positive Rate (1 - specificity)")
    plt.ylabel("True Positive Rate (sensitivity)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH)

    # -----------------------------------------------------------
    # 7) Plot training history
    # -----------------------------------------------------------
    fine_start = len(history.history["accuracy"]) - 1

    plt.figure(figsize=(12, 5))
    for i, metric in enumerate(["accuracy", "auc"]):
        plt.subplot(1, 2, i + 1)
        plt.plot(history.history[metric] + history_fine.history[metric],
                 label=f"Train {metric}")
        plt.plot(history.history[f"val_{metric}"] + history_fine.history[f"val_{metric}"],
                 label=f"Val {metric}")
        plt.axvline(fine_start, color="gray", linestyle="--", label="Start Fine-Tuning")
        plt.title(f"{metric.upper()} - Br35H Dataset")
        plt.xlabel("Epoch"); plt.ylabel(metric); plt.legend()
    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    plt.show()

    print("\nDone. Best model(s) saved to:", ", ".join(model_paths))


if __name__ == "__main__":
    main()
