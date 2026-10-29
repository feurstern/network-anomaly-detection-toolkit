"""Point-level metrics: every scored time step is one prediction.

* Rows that could not be scored (warm-up, missing data) count as **not flagged**, so anomalies
  missed because a detector was not ready still count as misses.
* PR-AUC is average precision over the raw scores; unscored rows get ``min(score) - 1`` rather than
  being dropped, for the same reason. It is threshold-free; the other metrics use the deployed
  thresholds. It is undefined (``None``) when only one class is present.
* Point-adjusted metrics (one hit marks a whole event as found) are deliberately not computed:
  they reward long events and make even random scores look good (Kim et al., AAAI 2022).
"""

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score

from network_anomaly.config import StrictModel


class PointMetrics(StrictModel):
    rows: int
    scored: int
    positives: int
    flagged: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_rate: float | None
    pr_auc: float | None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def harmonic_mean(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    total = precision + recall
    return 0.0 if total == 0 else 2 * precision * recall / total


def point_metrics(
    labels: npt.NDArray[np.int8],
    flagged: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
) -> PointMetrics:
    """Confusion-matrix metrics at the deployed thresholds plus threshold-free PR-AUC."""
    actual = labels.astype(bool)
    predicted = flagged & np.isfinite(scores)
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    tn = int((~predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    pr_auc = None
    if 0 < actual.sum() < len(actual):
        finite = np.isfinite(scores)
        fill = float(np.min(scores[finite])) - 1.0 if finite.any() else 0.0
        pr_auc = float(average_precision_score(actual, np.where(finite, scores, fill)))
    return PointMetrics(
        rows=len(actual),
        scored=int(np.isfinite(scores).sum()),
        positives=int(actual.sum()),
        flagged=tp + fp,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=harmonic_mean(precision, recall),
        false_positive_rate=_ratio(fp, fp + tn),
        pr_auc=pr_auc,
    )
