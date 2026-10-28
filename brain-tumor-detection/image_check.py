"""
Input-image suitability check.

Goal: accept genuine brain-MRI scans (which vary a lot in background
darkness and brain size) while rejecting clearly unrelated inputs such
as color photos or scanned documents.

The check is intentionally lenient: it is a light guard, not a strict
classifier. Three simple, robust signals are used:

  1) Grayness    - MRI scans are (near) grayscale, so the RGB channels
                   should be close to each other.
  2) Dark pixels - MRI scans usually contain some dark background.
  3) Structure   - MRI scans have real intensity variation (not a flat
                   solid color), measured by the standard deviation.
"""

import numpy as np
from config import (IMG_SIZE, GRAYNESS_THRESHOLD,
                    DARK_RATIO_THRESHOLD, STD_THRESHOLD)


def check_image_suitability(pil_image):
    """
    Return (is_suitable: bool, reason: str).
    `reason` is "ok" when the image is accepted.
    """
    rgb = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(rgb).astype("float32")
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # 1) Grayscale check: RGB channels should be close for MRI scans.
    color_diff = (np.abs(r - g).mean()
                  + np.abs(g - b).mean()
                  + np.abs(r - b).mean()) / 3.0
    if color_diff > GRAYNESS_THRESHOLD:
        return False, "The image looks colored; brain MRI scans are grayscale."

    gray = arr.mean(axis=2) / 255.0

    # 2) Some dark background is expected in MRI scans.
    dark_ratio = float((gray < 0.15).mean())
    if dark_ratio < DARK_RATIO_THRESHOLD:
        return False, "The image lacks a dark background typical of brain MRI scans."

    # 3) MRI scans have real structure (not a flat, uniform image).
    if float(gray.std() * 255.0) < STD_THRESHOLD:
        return False, "The image is too flat/uniform to be a brain MRI scan."

    return True, "ok"
