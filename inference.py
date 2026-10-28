"""
Inference helpers shared by the web app (and the tests).

  - input_size():          the image size a saved model expects.
  - prepare_image():       turn uploaded image bytes into the model input
                           exactly like training does (TensorFlow decoding,
                           RGB, bilinear resize). PIL decodes JPEGs slightly
                           differently, so it is not used for model input.
  - predict_tumor_prob():  P(tumor) for one image.
  - gradcam_heatmap():     Grad-CAM map showing which regions of the scan
                           pushed the model towards its prediction.
  - overlay_heatmap():     blend that map over the original image.

`models` may be a single model or a list (an ensemble); predictions and
Grad-CAM maps are then averaged over the members.

Grad-CAM uses the last convolutional feature map of the backbone, which
is small (5x5 for MobileNetV2 at 160 px, 7x7 at 224 px), so the map is
coarse: it points at the rough area that drove the decision, it does
not outline the tumor.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from matplotlib import colormaps

from backbones import backbone_of


def _as_list(models):
    return list(models) if isinstance(models, (list, tuple)) else [models]


def input_size(models):
    """Square input size of the (first) model, e.g. 160."""
    return int(_as_list(models)[0].input_shape[1])


def prepare_image(image_bytes, size):
    """Return a (1, size, size, 3) float32 batch for the model.
    Mirrors train.load_image(); raises tf.errors.InvalidArgumentError for
    data TensorFlow cannot decode."""
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size), method="bilinear")
    return img[tf.newaxis].numpy()


def predict_tumor_prob(models, batch):
    """P(tumor) for a single prepared image (mean over ensemble members)."""
    return float(np.mean([m.predict(batch, verbose=0)[0][0]
                          for m in _as_list(models)]))


def _split_model(model):
    """Return (base_model, head_layers) of the trained classifier.

    The classifier is: augmentation -> [preprocessing] -> pretrained base
    -> GlobalAveragePooling -> Dropout -> Dense. The base is the nested
    functional model (the augmentation block is a Sequential)."""
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.Model) and not isinstance(layer, keras.Sequential):
            return layer, model.layers[i + 1:]
    raise ValueError("No nested base model found in the classifier.")


def _base_inputs(base_model, batch):
    """Apply the backbone's preprocessing to a raw 0-255 batch."""
    x = tf.convert_to_tensor(batch, dtype=tf.float32)
    preprocess = backbone_of(base_model)["preprocess"]
    return preprocess(x) if preprocess is not None else x


# ---------------------------------------------------------------
# Out-of-distribution check
#   An image is compared with the training scans in the model's own
#   feature space (globally pooled backbone features of the first model,
#   L2-normalised). Its score is the cosine distance to its k-th nearest
#   training scan; a score above the calibrated threshold means "unlike
#   anything the model was trained on" (e.g. a CT scan, an X-ray, a photo),
#   so its prediction should not be trusted.
# ---------------------------------------------------------------
def embed(models, batch, chunk=32):
    """L2-normalised feature vectors, shape (n, d), for a prepared batch."""
    base_model, head = _split_model(_as_list(models)[0])
    pool = head[0]                                     # GlobalAveragePooling2D
    out = []
    for i in range(0, len(batch), chunk):
        x = _base_inputs(base_model, batch[i:i + chunk])
        out.append(pool(base_model(x, training=False)).numpy())
    feats = np.concatenate(out).astype("float32")
    return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def ood_scores(embeddings, reference, k=5):
    """Cosine distance of each embedding to its k-th nearest reference row."""
    sims = np.asarray(embeddings, "float32") @ np.asarray(reference, "float32").T
    kth = np.sort(sims, axis=1)[:, -k]
    return 1.0 - kth


def load_ood_reference(path):
    """dict(embeddings, threshold, k) saved by train.py, or None."""
    try:
        with np.load(path) as data:
            return {"embeddings": data["embeddings"],
                    "threshold": float(data["threshold"]), "k": int(data["k"])}
    except (OSError, KeyError, ValueError):
        return None


def _gradcam_single(model, batch, for_tumor, layer):
    base_model, head = _split_model(model)
    x = _base_inputs(base_model, batch)
    if layer is None:
        target_model = base_model                  # final feature map
    else:
        # Expose an intermediate (higher-resolution) feature map as well.
        target_model = keras.Model(base_model.inputs,
                                   [base_model.get_layer(layer).output, base_model.output])

    with tf.GradientTape() as tape:
        outputs = target_model(x, training=False)
        features, out = (outputs, outputs) if layer is None else outputs
        tape.watch(features)
        for head_layer in head:
            out = head_layer(out, training=False)
        score = out[:, 0] if for_tumor else 1.0 - out[:, 0]

    grads = tape.gradient(score, features)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))             # one per channel
    return tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1)).numpy()


def gradcam_heatmap(models, batch, for_tumor=True, layer=None):
    """Grad-CAM map in [0, 1].

    layer: None uses the final feature map (e.g. 7x7 at 224 px); a layer
    name of the backbone uses that layer instead. Higher-resolution layers
    (e.g. "block5e_add", 14x14) and Grad-CAM++ were tried on Br35H tumors:
    the earlier layer highlighted the skull rim more often and Grad-CAM++
    gave practically the same maps, so the final layer is the default.
    for_tumor=True explains the "tumor" score, False the "no tumor" score.
    Augmentation layers are inactive at inference and are skipped."""
    cam = np.mean([_gradcam_single(m, batch, for_tumor, layer)
                   for m in _as_list(models)], axis=0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def overlay_heatmap(pil_image, heatmap, alpha=0.4):
    """Blend a [0, 1] heatmap (jet colormap) over the image; returns PIL RGB."""
    base = pil_image.convert("RGB")
    heat = Image.fromarray(np.uint8(heatmap * 255)).resize(
        base.size, Image.Resampling.BILINEAR)
    colored = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    colored = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(base, colored, alpha)
