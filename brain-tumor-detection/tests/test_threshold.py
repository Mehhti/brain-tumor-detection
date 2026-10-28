import numpy as np
import pytest

from train import choose_threshold, compute_metrics


def _j(y, p, t):
    m = compute_metrics(y, p, t)
    return m["sensitivity"] + m["specificity"] - 1


def test_perfectly_separable_scores_give_threshold_between_classes():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    t = choose_threshold(y, p)
    assert 0.2 < t <= 0.8
    assert t == pytest.approx(0.5)        # ties are broken towards 0.5


def test_threshold_never_skips_over_an_image():
    # Regression test: roc_curve drops "intermediate" thresholds by default,
    # which once put the cut at 0.35 and misclassified the 0.6 healthy scan.
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.6, 0.7, 0.65, 0.9, 0.95])
    t = choose_threshold(y, p)
    assert _j(y, p, t) == pytest.approx(2 / 3)
    assert compute_metrics(y, p, t)["specificity"] == pytest.approx(2 / 3)


def test_tuned_threshold_reaches_best_possible_youden_j():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(4, 40))
        y = rng.integers(0, 2, n)
        if len(set(y)) < 2:
            continue
        p = np.round(rng.random(n), 2)
        best = max(_j(y, p, c) for c in np.unique(np.r_[p, 1.01]))
        assert _j(y, p, choose_threshold(y, p)) == pytest.approx(best)


def test_compute_metrics_counts():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.2, 0.7, 0.4, 0.8, 0.9])
    m = compute_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(1 / 2)
    assert m["threshold"] == 0.5
