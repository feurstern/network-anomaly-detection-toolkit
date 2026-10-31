"""Which metrics drove a detector's score on a row - computed in a way that fits each model.

* **Per-metric terms** (``static_threshold``, ``robust_zscore``): the score *is* the maximum of one
  term per metric (value / limit, or normalised directional robust z), so the terms explain it
  exactly. The optional autoencoder's score is the *sum* of per-metric reconstruction errors,
  which explains it exactly too (``reconstruction_error``).
* **Grouped occlusion** (``isolation_forest``, ``one_class_svm``): there is no additive
  decomposition, so each metric's features (``z_local``, ``z_season``, ``roc``) are replaced
  *together* by their typical training values (the training medians) and the row is re-scored.

  - *drop-one*: contribution = score - score with that metric made typical;
  - *keep-one*, used when drop-one saturates (several metrics are extreme together, so making one
    typical barely lowers the score): contribution = score with only that metric left as observed
    - score with every metric typical.

  Metrics are grouped because a metric's three features are strongly correlated; splitting credit
  between them would be arbitrary.

Contributions rank metrics for investigation. They are not additive, not probabilities and not
causal: a metric that moved with the anomaly can rank first without being its cause. SHAP is not
used: it would add a heavy dependency, divide credit arbitrarily among correlated engineered
features and depend on an arbitrary background sample.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from network_anomaly.detectors.base import Detector
from network_anomaly.features.builder import FeatureFrame, feature_groups
from network_anomaly.schemas import ExplanationMethod

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

SATURATION_FRACTION = 0.25


@dataclass(frozen=True)
class Attribution:
    """Per-metric contributions for selected rows, aligned with ``metrics``."""

    metrics: tuple[str, ...]
    contributions: FloatArray  # (rows x metrics); NaN where a metric does not take part
    methods: tuple[ExplanationMethod, ...]  # one per row


def occlusion(
    detector: Detector,
    features: FloatArray,
    typical: FloatArray,
    saturation: float = SATURATION_FRACTION,
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Grouped occlusion for ``features`` (rows x detector.feature_schema).

    Returns (contributions rows x metrics, whether keep-one was used per row). Every variant of
    every row is scored in one batch.
    """
    n_rows = len(features)
    groups = feature_groups(detector.feature_schema, detector.metrics)
    metrics = [metric for metric, positions in groups.items() if positions]
    variants = [features]
    for metric in metrics:
        dropped = features.copy()
        dropped[:, groups[metric]] = typical[groups[metric]]
        variants.append(dropped)
    for metric in metrics:
        kept = np.tile(typical, (n_rows, 1))
        kept[:, groups[metric]] = features[:, groups[metric]]
        variants.append(kept)
    variants.append(typical[np.newaxis, :])
    scores = detector.score_features(np.vstack(variants))
    observed = scores[:n_rows]
    count = len(metrics)
    without = scores[n_rows : (count + 1) * n_rows].reshape(count, n_rows).T
    only = scores[(count + 1) * n_rows : (2 * count + 1) * n_rows].reshape(count, n_rows).T
    baseline = scores[-1]
    drop_one = observed[:, np.newaxis] - without
    keep_one = only - baseline
    excess = observed - baseline
    best_drop = drop_one.max(axis=1) if count else np.zeros(n_rows)
    saturated = (excess > 0) & (best_drop < saturation * excess)
    chosen = np.where(saturated[:, np.newaxis], keep_one, drop_one)
    contributions = np.full((n_rows, len(detector.metrics)), np.nan)
    for index, metric in enumerate(detector.metrics):
        if metric in metrics:
            contributions[:, index] = chosen[:, metrics.index(metric)]
    return contributions, saturated


def attribute(
    detector: Detector,
    ff: FeatureFrame,
    positions: IntArray,
    feature_medians: Mapping[str, float],
) -> Attribution:
    """Contributions of each metric to the scores of ``ff`` rows at ``positions``."""
    terms = detector.metric_terms(ff, positions)
    if terms is not None:
        return Attribution(detector.metrics, terms, (detector.term_method,) * len(positions))
    typical = np.array([feature_medians.get(c, 0.0) for c in detector.feature_schema])
    contributions, saturated = occlusion(detector, detector.matrix(ff, positions), typical)
    methods: tuple[ExplanationMethod, ...] = tuple(
        "occlusion_keep_one" if keep else "occlusion_drop_one" for keep in saturated
    )
    return Attribution(detector.metrics, contributions, methods)
