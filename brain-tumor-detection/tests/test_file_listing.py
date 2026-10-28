import numpy as np
import pytest

from train import list_image_files
from helpers import fake_scan


def test_only_class_folders_and_image_files_are_listed(tmp_path):
    for cls in ("no", "yes", "pred"):
        (tmp_path / cls).mkdir()
    fake_scan(0).save(tmp_path / "no" / "a.png")
    fake_scan(1).save(tmp_path / "no" / "b.JPG", "JPEG")
    fake_scan(2).save(tmp_path / "yes" / "c.jpeg", "JPEG")
    fake_scan(3).save(tmp_path / "pred" / "ignored.png")
    (tmp_path / "no" / "Thumbs.db").write_text("junk")

    paths, labels = list_image_files(str(tmp_path), ["no", "yes"])

    names = sorted(p.replace("\\", "/").rsplit("/", 1)[-1] for p in paths)
    assert names == ["a.png", "b.JPG", "c.jpeg"]
    assert sorted(labels.tolist()) == [0, 0, 1]


def test_missing_class_folder_is_an_error(tmp_path):
    (tmp_path / "no").mkdir()
    with pytest.raises(FileNotFoundError):
        list_image_files(str(tmp_path), ["no", "yes"])
