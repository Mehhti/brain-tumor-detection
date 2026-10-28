"""Synthetic test images (no dataset needed)."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def fake_scan(seed, size=200):
    """A distinct grayscale 'scan': textured ellipse on a dark background."""
    rng = np.random.default_rng(seed)
    texture = Image.fromarray((rng.random((24, 24)) * 255).astype("uint8"))
    texture = texture.resize((size, size), Image.Resampling.BICUBIC)
    texture = texture.filter(ImageFilter.GaussianBlur(2))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((size * 0.12, size * 0.08, size * 0.88, size * 0.92), fill=255)
    return Image.composite(texture, Image.new("L", (size, size), 5), mask)
