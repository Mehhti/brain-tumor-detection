"""
Supported pretrained backbones (ImageNet weights, Keras applications).

Each entry knows how to build the base model, how to preprocess raw
0-255 RGB input for it, how its nested model is named (so a saved
classifier can be recognised again, e.g. by Grad-CAM), and the file
name of its locally cached ImageNet weights.
"""

from tensorflow import keras


def _mobilenet_v2(size, weights):
    return keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights=weights)


def _efficientnet_v2_b0(size, weights):
    # Rescaling / normalization are built into the base model.
    return keras.applications.EfficientNetV2B0(
        input_shape=(size, size, 3), include_top=False, weights=weights,
        include_preprocessing=True)


BACKBONES = {
    "mobilenet_v2": {
        "build": _mobilenet_v2,
        "preprocess": keras.applications.mobilenet_v2.preprocess_input,
        "name_prefix": "mobilenetv2",
        "weights_file": lambda size:
            f"mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_{size}_no_top.h5",
    },
    "efficientnet_v2_b0": {
        "build": _efficientnet_v2_b0,
        "preprocess": None,
        "name_prefix": "efficientnetv2-b0",
        "weights_file": lambda size: "efficientnetv2-b0_notop.h5",
    },
}


def get_backbone(name):
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[name]


def backbone_of(base_model):
    """Return the BACKBONES entry that built this (nested) base model."""
    for spec in BACKBONES.values():
        if base_model.name.startswith(spec["name_prefix"]):
            return spec
    raise ValueError(f"Unrecognised base model {base_model.name!r}")
