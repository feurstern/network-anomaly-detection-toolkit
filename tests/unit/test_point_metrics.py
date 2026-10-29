import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from network_anomaly.evaluation.point import harmonic_mean, point_metrics


def test_hand_computed_confusion_matrix() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int8)
    flagged = np.array([1, 1, 0, 1, 0, 0, 0, 0], dtype=bool)
    scores = np.array([9, 8, 2, 7, 1, 1, 1, 1], dtype=float)
    metrics = point_metrics(labels, flagged, scores)
    assert (metrics.true_positives, metrics.false_positives) == (2, 1)
    assert (metrics.true_negatives, metrics.false_negatives) == (4, 1)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.false_positive_rate == pytest.approx(1 / 5)
    assert metrics.pr_auc == pytest.approx(average_precision_score(labels, scores))


def test_unscored_rows_count_as_not_flagged_and_rank_last() -> None:
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    flagged = np.array([True, True, False, False])
    scores = np.array([5.0, np.nan, 1.0, 2.0])  # the second anomaly was never scored
    metrics = point_metrics(labels, flagged, scores)
    assert metrics.true_positives == 1  # a flag on an unscored row cannot count
    assert metrics.false_negatives == 1
    assert metrics.scored == 3
    expected = average_precision_score(labels, np.array([5.0, 0.0, 1.0, 2.0]))
    assert metrics.pr_auc == pytest.approx(expected)


def test_metrics_are_undefined_rather_than_misleading() -> None:
    metrics = point_metrics(np.zeros(4, dtype=np.int8), np.zeros(4, bool), np.ones(4))
    assert metrics.pr_auc is None  # one class only
    assert metrics.precision is None  # nothing flagged
    assert metrics.recall is None  # nothing to find
    assert metrics.false_positive_rate == 0.0


def test_harmonic_mean() -> None:
    assert harmonic_mean(None, 0.5) is None
    assert harmonic_mean(0.0, 0.0) == 0.0
    assert harmonic_mean(1.0, 0.5) == pytest.approx(2 / 3)
