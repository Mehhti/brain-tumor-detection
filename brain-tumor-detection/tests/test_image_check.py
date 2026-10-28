import numpy as np
from PIL import Image

from image_check import check_image_suitability
from helpers import fake_scan


def test_grayscale_scan_is_accepted():
    ok, reason = check_image_suitability(fake_scan(0))
    assert ok and reason == "ok"


def test_color_photo_is_rejected():
    arr = np.zeros((200, 300, 3), np.uint8)
    arr[..., 0] = np.linspace(0, 255, 300)[None]
    arr[..., 1] = 120
    arr[..., 2] = np.linspace(255, 0, 200)[:, None]
    ok, reason = check_image_suitability(Image.fromarray(arr))
    assert not ok and "colored" in reason


def test_bright_document_is_rejected():
    page = Image.new("L", (200, 200), 235)          # white page, no dark area
    ok, reason = check_image_suitability(page)
    assert not ok and "dark background" in reason


def test_flat_image_is_rejected():
    ok, reason = check_image_suitability(Image.new("L", (200, 200), 10))
    assert not ok and "flat" in reason
