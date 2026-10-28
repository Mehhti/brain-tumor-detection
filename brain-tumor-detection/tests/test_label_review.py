import json

import numpy as np
import pytest

import train


@pytest.fixture
def review(tmp_path, monkeypatch):
    """Point train at a temporary data dir and review file; returns a writer."""
    data_dir = tmp_path / "data"
    review_path = tmp_path / "label_review.json"
    monkeypatch.setattr(train, "DATA_SOURCES", {"br35h": {"dir": str(data_dir), "classes": {}}})
    monkeypatch.setattr(train, "LABEL_REVIEW_PATH", str(review_path))

    def write(images):
        review_path.write_text(json.dumps({"images": images}))
        return data_dir
    return write


def _paths(data_dir, *names):
    return np.array([str(data_dir / n) for n in names])


def test_no_review_file_changes_nothing(review, tmp_path):
    train.LABEL_REVIEW_PATH = str(tmp_path / "missing.json")
    paths, labels = np.array(["a", "b"]), np.array([0, 1])
    out_paths, out_labels, summary = train.apply_label_review(paths, labels)
    assert list(out_paths) == ["a", "b"] and summary == {"n_excluded": 0, "n_relabelled": 0}


def test_exclude_relabel_and_keep(review):
    data_dir = review([
        {"path": "br35h/no/a.jpg", "action": "exclude"},
        {"path": "br35h/no/b.jpg", "action": "relabel", "label": "yes"},
        {"path": "br35h/yes/c.jpg", "action": "keep"},
    ])
    paths = _paths(data_dir, "no/a.jpg", "no/b.jpg", "yes/c.jpg", "yes/d.jpg")
    out_paths, out_labels, summary = train.apply_label_review(paths, np.array([0, 0, 1, 1]))
    assert [p.replace("\\", "/").rsplit("/", 2)[-2:] for p in out_paths] == [
        ["no", "b.jpg"], ["yes", "c.jpg"], ["yes", "d.jpg"]]
    assert out_labels.tolist() == [1, 1, 1]
    assert summary == {"n_excluded": 1, "n_relabelled": 1}


def test_decision_on_a_removed_copy_applies_to_the_kept_image(review):
    # no/copy.jpg was removed as a duplicate of no/kept.jpg; reviewing the copy
    # must still exclude the kept image (same scan, same doubtful label).
    data_dir = review([{"path": "BR35H/NO/copy.jpg", "action": "exclude"}])
    kept = _paths(data_dir, "no/kept.jpg", "no/other.jpg")
    copies = {kept[0]: [kept[0], str(data_dir / "no/copy.jpg")]}
    out_paths, _, summary = train.apply_label_review(kept, np.array([0, 0]), copies)
    assert len(out_paths) == 1 and out_paths[0].endswith("other.jpg")
    assert summary["n_excluded"] == 1


def test_unknown_action_is_an_error(review):
    data_dir = review([{"path": "br35h/no/a.jpg", "action": "delete"}])
    with pytest.raises(ValueError):
        train.apply_label_review(_paths(data_dir, "no/a.jpg"), np.array([0]))
