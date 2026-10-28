"""Tests for the app's inference path, using an untrained model
(random weights, so no ImageNet download and no trained model file)."""

import numpy as np
import pytest
import tensorflow as tf

import train
from config import IMG_SIZE
from inference import (input_size, prepare_image, predict_tumor_prob,
                       gradcam_heatmap, overlay_heatmap, _split_model, embed,
                       ood_scores, load_ood_reference)
from helpers import fake_scan


def feature_map_size(model):
    """Spatial size of the backbone's last feature map (= Grad-CAM size)."""
    return tuple(_split_model(model)[0].output_shape[1:3])


@pytest.fixture(scope="module")
def model():
    model, _ = train.build_model(weights=None)    # random init, offline
    return model


@pytest.mark.parametrize("fmt, suffix", [("PNG", ".png"), ("JPEG", ".jpg"), ("BMP", ".bmp")])
def test_app_preprocessing_matches_training(tmp_path, fmt, suffix):
    path = tmp_path / f"scan{suffix}"
    fake_scan(7, size=233).convert("RGB").save(path, fmt)
    app_input = prepare_image(path.read_bytes(), IMG_SIZE)[0]
    train_input = train.load_image(tf.constant(str(path)), 0)[0].numpy()
    assert app_input.shape == (IMG_SIZE, IMG_SIZE, 3)
    np.testing.assert_array_equal(app_input, train_input)


def test_unreadable_bytes_raise():
    with pytest.raises(tf.errors.InvalidArgumentError):
        prepare_image(b"definitely not an image", IMG_SIZE)


def test_prediction_is_a_probability(model, tmp_path):
    path = tmp_path / "scan.png"
    fake_scan(1).save(path)
    p = predict_tumor_prob(model, prepare_image(path.read_bytes(), input_size(model)))
    assert 0.0 <= p <= 1.0


def test_gradcam_heatmap_and_overlay(model, tmp_path):
    img = fake_scan(2)
    path = tmp_path / "scan.png"
    img.save(path)
    batch = prepare_image(path.read_bytes(), input_size(model))
    for for_tumor in (True, False):
        heatmap = gradcam_heatmap(model, batch, for_tumor=for_tumor)
        assert heatmap.shape == feature_map_size(model)
        assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    overlay = overlay_heatmap(img, heatmap)
    assert overlay.size == img.size and overlay.mode == "RGB"


def test_ensemble_averages_members(model, tmp_path):
    other, _ = train.build_model(weights=None)
    path = tmp_path / "scan.png"
    fake_scan(3).save(path)
    batch = prepare_image(path.read_bytes(), input_size(model))
    p1, p2 = predict_tumor_prob(model, batch), predict_tumor_prob(other, batch)
    assert predict_tumor_prob([model, other], batch) == pytest.approx((p1 + p2) / 2, abs=1e-6)
    heatmap = gradcam_heatmap([model, other], batch)
    assert heatmap.shape == feature_map_size(model) and heatmap.max() <= 1.0


@pytest.mark.parametrize("backbone, size, cam_size", [
    ("mobilenet_v2", 160, (5, 5)),
    ("efficientnet_v2_b0", 224, (7, 7)),
])
def test_each_backbone_and_input_size(monkeypatch, tmp_path, backbone, size, cam_size):
    monkeypatch.setattr(train, "BACKBONE", backbone)
    monkeypatch.setattr(train, "IMG_SIZE", size)
    model, _ = train.build_model(weights=None)
    assert input_size(model) == size
    path = tmp_path / "scan.png"
    fake_scan(4).save(path)
    batch = prepare_image(path.read_bytes(), input_size(model))
    assert 0.0 <= predict_tumor_prob(model, batch) <= 1.0
    assert gradcam_heatmap(model, batch).shape == cam_size


def test_embeddings_are_unit_vectors(model, tmp_path):
    batch = np.concatenate([prepare_image(_png(tmp_path, s), input_size(model)) for s in (1, 2, 3)])
    emb = embed(model, batch, chunk=2)
    assert emb.shape[0] == 3
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-5)


def test_ood_score_is_distance_to_kth_neighbour():
    reference = np.eye(4, dtype="float32")              # four orthogonal unit vectors
    query = np.array([[1, 0, 0, 0]], dtype="float32")
    assert ood_scores(query, reference, k=1)[0] == pytest.approx(0.0)   # identical
    assert ood_scores(query, reference, k=2)[0] == pytest.approx(1.0)   # orthogonal


def test_missing_ood_reference_is_none(tmp_path):
    assert load_ood_reference(str(tmp_path / "missing.npz")) is None
    path = tmp_path / "ref.npz"
    np.savez_compressed(path, embeddings=np.eye(3, dtype="float16"), threshold=0.5, k=2)
    ref = load_ood_reference(str(path))
    assert ref["threshold"] == 0.5 and ref["k"] == 2 and ref["embeddings"].shape == (3, 3)


def _png(tmp_path, seed):
    path = tmp_path / f"scan{seed}.png"
    fake_scan(seed).save(path)
    return path.read_bytes()


def test_gradcam_on_an_intermediate_layer(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "BACKBONE", "mobilenet_v2")
    monkeypatch.setattr(train, "IMG_SIZE", 160)
    model, _ = train.build_model(weights=None)
    batch = prepare_image(_png(tmp_path, 5), input_size(model))
    heatmap = gradcam_heatmap(model, batch, layer="block_13_expand_relu")
    assert heatmap.shape == (10, 10) and 0.0 <= heatmap.min() and heatmap.max() <= 1.0
