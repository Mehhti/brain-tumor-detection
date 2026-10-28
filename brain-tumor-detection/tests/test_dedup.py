import numpy as np
import pytest
from PIL import ImageEnhance, ImageOps

from dedup import find_duplicates, phash
from helpers import fake_scan


def _write(tmp_path, img, name, **kw):
    path = tmp_path / name
    img.save(path, **kw)
    return str(path)


def _dataset(tmp_path, n_distinct=20):
    paths, labels = [], []
    for i in range(n_distinct):
        paths.append(_write(tmp_path, fake_scan(i), f"orig{i:02d}.png"))
        labels.append(i % 2)
    return paths, labels


def test_identical_images_have_identical_hash(tmp_path):
    a = _write(tmp_path, fake_scan(1), "a.png")
    b = _write(tmp_path, fake_scan(1), "b.png")
    assert phash(a) == phash(b)
    assert len(phash(a)) == 4          # original, mirror, vertical flip, 180-degree


def test_distinct_images_are_all_kept(tmp_path):
    paths, labels = _dataset(tmp_path)
    keep, dups, conflicts, unreadable = find_duplicates(paths, np.array(labels), 6)
    assert keep.all() and not dups and not conflicts and not unreadable


def test_copies_conflicts_and_broken_files(tmp_path):
    paths, labels = _dataset(tmp_path)
    copies = {
        "copy_exact.png": (fake_scan(0), 0, {}),
        "copy_resized.jpg": (fake_scan(0).resize((140, 140)), 0, {"quality": 70}),
        "copy_bright.png": (ImageEnhance.Brightness(fake_scan(1)).enhance(1.25), 1, {}),
        "copy_mirror.png": (ImageOps.mirror(fake_scan(3)), 1, {}),
        "copy_vflip.png": (ImageOps.flip(fake_scan(5)), 1, {}),
        "copy_rot180.png": (fake_scan(6).rotate(180), 0, {}),
        "copy_wrong_label.png": (fake_scan(4), 1, {}),    # orig04 has label 0
    }
    for name, (img, label, kw) in copies.items():
        paths.append(_write(tmp_path, img, name, **kw))
        labels.append(label)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    paths.append(str(broken))
    labels.append(0)

    keep, dups, conflicts, unreadable = find_duplicates(paths, np.array(labels), 6)

    removed = {p.replace("\\", "/").rsplit("/", 1)[-1] for p in np.array(paths)[~keep]}
    assert removed == set(copies) | {"orig04.png", "broken.png"}
    assert len(conflicts) == 1 and len(unreadable) == 1
    # The first image of every duplicate group (the original) is kept.
    assert all(keep[g[0]] for g in dups)


@pytest.mark.parametrize("transform", [ImageOps.mirror, ImageOps.flip,
                                       lambda img: img.rotate(180)])
def test_flip_matching_can_be_disabled(tmp_path, transform):
    paths, labels = _dataset(tmp_path)
    paths.append(_write(tmp_path, transform(fake_scan(3)), "copy_flipped.png"))
    labels.append(1)
    keep, *_ = find_duplicates(paths, np.array(labels), 6, match_flips=False)
    assert keep[-1]
    keep, *_ = find_duplicates(paths, np.array(labels), 6, match_flips=True)
    assert not keep[-1]


def test_cross_label_hash_match_needs_pixel_agreement(tmp_path, monkeypatch):
    # Two clearly different scans whose hashes are forced to be identical,
    # as happens by coincidence in larger datasets.
    import dedup
    a = _write(tmp_path, fake_scan(1), "a.png")
    b = _write(tmp_path, fake_scan(2), "b.png")
    monkeypatch.setattr(dedup, "phash", lambda path: [np.uint64(7)] * 4)
    assert dedup.pixel_similarity(a, b) < 0.95

    keep, dups, conflicts, _ = dedup.find_duplicates([a, b], np.array([0, 1]), 6)
    assert keep.all() and not conflicts            # different labels: kept apart

    keep, dups, conflicts, _ = dedup.find_duplicates([a, b], np.array([0, 0]), 6)
    assert keep.tolist() == [True, False] and dups  # same label: merged


def test_pixel_similarity_ignores_flips_and_contrast(tmp_path):
    a = _write(tmp_path, fake_scan(3), "a.png")
    b = _write(tmp_path, ImageEnhance.Contrast(ImageOps.flip(fake_scan(3))).enhance(1.5), "b.png")
    from dedup import pixel_similarity
    assert pixel_similarity(a, b) > 0.95
