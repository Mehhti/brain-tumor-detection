"""
=====================================================================
 Brain Tumor Detection - Training Script
 ---------------------------------------------------------------
 Method: Two-stage Transfer Learning with MobileNetV2
 Dataset: Br35H :: Brain Tumor Detection 2020 (3000 balanced images)
          https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

 Key practices:
   1) Standard 3-way split: train / validation / independent test
      -> the final score is reported on a test set the model never saw.
   2) Full-buffer shuffle for proper data mixing.
   3) Class weighting for robustness against class imbalance.
   4) Two-stage training (frozen base -> fine-tuning).

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

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

from config import (DATA_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, FINE_EPOCHS,
                    SEED, MODEL_PATH, LOCAL_WEIGHTS_PATH, CLASS_NAMES)

# ---------------------------------------------------------------
# 1) Reproducibility
# ---------------------------------------------------------------
tf.random.set_seed(SEED)
np.random.seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE

# Use local ImageNet weights if available (helps in restricted networks),
# otherwise download them from the internet.
WEIGHTS = LOCAL_WEIGHTS_PATH if os.path.exists(LOCAL_WEIGHTS_PATH) else "imagenet"

# ---------------------------------------------------------------
# 2) Load data with a standard 3-way split
#    First 80% train / 20% temp, then split temp into val + test.
#    Result: 80% train / 10% validation / 10% independent test.
# ---------------------------------------------------------------
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2, subset="training", seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    label_mode="binary", class_names=CLASS_NAMES)

temp_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2, subset="validation", seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    label_mode="binary", class_names=CLASS_NAMES)

class_names = train_ds.class_names
print("Classes:", class_names)

# Split the 20% temp set into validation (first half) and test (second half)
temp_batches = tf.data.experimental.cardinality(temp_ds)
val_ds  = temp_ds.take(temp_batches // 2)
test_ds = temp_ds.skip(temp_batches // 2)

print(f"Batches -> train: {tf.data.experimental.cardinality(train_ds)}, "
      f"val: {tf.data.experimental.cardinality(val_ds)}, "
      f"test: {tf.data.experimental.cardinality(test_ds)}")

# ---------------------------------------------------------------
# 3) Class weights (computed from training labels)
# ---------------------------------------------------------------
train_labels = np.concatenate([y.numpy().flatten() for _, y in train_ds], axis=0)
n0 = int((train_labels == 0).sum())
n1 = int((train_labels == 1).sum())
total = n0 + n1
class_weight = {0: total / (2.0 * n0), 1: total / (2.0 * n1)}
print(f"Counts -> no={n0}, yes={n1} | class_weight={class_weight}")

# ---------------------------------------------------------------
# 4) Performance pipeline (full-buffer shuffle for real mixing)
# ---------------------------------------------------------------
train_ds = train_ds.cache().shuffle(total, seed=SEED).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ---------------------------------------------------------------
# 5) Data augmentation
# ---------------------------------------------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# ---------------------------------------------------------------
# 6) Build the transfer-learning model
# ---------------------------------------------------------------
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False, weights=WEIGHTS)
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")])
model.summary()

# ---------------------------------------------------------------
# 7) Callbacks (use validation set only, never the test set)
# ---------------------------------------------------------------
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=6,
        restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True),
]

# ---------------------------------------------------------------
# 8) Stage 1: train the classification head (frozen base)
# ---------------------------------------------------------------
print("\n=== Stage 1: training the head (base frozen) ===")
history = model.fit(train_ds, validation_data=val_ds,
                    epochs=EPOCHS, class_weight=class_weight,
                    callbacks=callbacks)

# ---------------------------------------------------------------
# 9) Stage 2: fine-tuning (unfreeze the last 30 layers)
# ---------------------------------------------------------------
print("\n=== Stage 2: fine-tuning ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),   # very small LR for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")])

history_fine = model.fit(train_ds, validation_data=val_ds,
                         epochs=FINE_EPOCHS, class_weight=class_weight,
                         callbacks=callbacks)

# ---------------------------------------------------------------
# 10) Final evaluation on the INDEPENDENT test set
#     This data was never used for training, early stopping,
#     or model selection, so the score is fair and defensible.
# ---------------------------------------------------------------
print("\n=== Final evaluation on the independent test set ===")
y_true, y_pred = [], []
for images, lbl in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend((preds > 0.5).astype("int32").flatten())
    y_true.extend(lbl.numpy().flatten().astype("int32"))

print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report (Test):")
print(classification_report(y_true, y_pred, target_names=class_names))

# ---------------------------------------------------------------
# 11) Plot training history
# ---------------------------------------------------------------
acc     = history.history["accuracy"]     + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]

plt.figure(figsize=(8, 5))
plt.plot(acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.axvline(len(history.history["accuracy"]) - 1, color="gray",
            linestyle="--", label="Start Fine-Tuning")
plt.title("Accuracy - Br35H Dataset")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

print("\nDone. Best model saved to:", MODEL_PATH)
