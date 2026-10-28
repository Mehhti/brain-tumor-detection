import numpy as np
from sklearn.model_selection import StratifiedKFold

import train
from helpers import fake_scan


def _make_source(root, layout):
    """layout: {relative folder: number of images}"""
    seed = 0
    for folder, n in layout.items():
        (root / folder).mkdir(parents=True)
        for i in range(n):
            fake_scan(seed).save(root / folder / f"img{i}.jpg", "JPEG")
            seed += 1


def test_sources_are_listed_and_named(tmp_path, monkeypatch):
    _make_source(tmp_path / "a", {"no": 2, "yes": 3})
    _make_source(tmp_path / "b", {"Train/healthy": 1, "Train/t1": 2, "Test/t2": 1})
    monkeypatch.setattr(train, "DATA_SOURCES", {
        "a": {"dir": str(tmp_path / "a"), "classes": {"no": ["no"], "yes": ["yes"]}},
        "b": {"dir": str(tmp_path / "b"), "classes": {"no": ["Train/healthy"],
                                                     "yes": ["Train/t1", "Test/t2"]}},
    })
    paths, labels = train.list_source_files("b")
    assert sorted(labels.tolist()) == [0, 1, 1, 1]
    assert train.display_path(paths[0]).startswith("b/Train/")
    assert set(train.source_of(paths)) == {"b"}


def test_folds_of_a_source_do_not_change_when_another_is_added():
    rng = np.random.default_rng(0)
    labels_a = rng.integers(0, 2, 60)
    labels_b = rng.integers(0, 2, 90)
    alone = train.make_folds(labels_a, np.array(["a"] * 60), 5)
    labels = np.concatenate([labels_a, labels_b])
    sources = np.array(["a"] * 60 + ["b"] * 90)
    joint = train.make_folds(labels, sources, 5)
    for (_, t_alone), (_, t_joint) in zip(alone, joint):
        assert np.array_equal(np.sort(t_alone), np.sort(t_joint[t_joint < 60]))
    # every image is tested exactly once
    assert sorted(np.concatenate([t for _, t in joint]).tolist()) == list(range(150))


def test_single_source_folds_equal_plain_stratified_kfold():
    labels = np.random.default_rng(1).integers(0, 2, 80)
    folds = train.make_folds(labels, np.array(["x"] * 80), 5)
    plain = StratifiedKFold(5, shuffle=True, random_state=train.SEED).split(labels, labels)
    for (_, a), (_, b) in zip(folds, plain):
        assert np.array_equal(np.sort(a), np.sort(b))


def test_sample_weights_balance_every_source_label_group():
    labels = np.array([0] * 10 + [1] * 90 + [0] * 40 + [1] * 10)
    sources = np.array(["a"] * 100 + ["b"] * 50)
    w = train.compute_sample_weights(labels, sources)
    totals = {(s, y): w[(sources == s) & (labels == y)].sum()
              for s in ("a", "b") for y in (0, 1)}
    assert np.allclose(list(totals.values()), len(labels) / 4)
    assert np.isclose(w.mean(), 1.0)


def test_weighted_dataset_yields_weights(tmp_path):
    for i in range(3):
        fake_scan(i).save(tmp_path / f"{i}.png")
    paths = np.array([str(tmp_path / f"{i}.png") for i in range(3)])
    ds = train.make_dataset(paths, np.array([0, 1, 1]), shuffle=False,
                            weights=np.array([2.0, 0.5, 0.5], dtype="float32"))
    x, y, w = next(iter(ds))
    assert x.shape[0] == 3 and w.numpy().tolist() == [2.0, 0.5, 0.5]
