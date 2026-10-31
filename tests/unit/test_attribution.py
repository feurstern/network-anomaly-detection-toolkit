"""Attribution: exact per-metric terms for statistical detectors, grouped occlusion otherwise."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from network_anomaly.config import StrictModel
from network_anomaly.detectors import create_detector, default_config
from network_anomaly.detectors.base import BoolArray, Detector, FloatArray
from network_anomaly.explanations.attribution import attribute, occlusion
from network_anomaly.features.builder import FeatureBuilder, FeatureFrame, feature_groups
from network_anomaly.ingestion.validation import MetricsFrame

FIVE_MINUTES = pd.Timedelta("5min")
TRAIN_END = pd.Timestamp("2026-01-07T00:00:00Z")
SPIKE = pd.Timestamp("2026-01-08T12:00:00Z")


class _MaxAbs(Detector):
    """Toy model: score = largest |feature|. Removing one of two equal extremes changes nothing."""

    name = "max_abs"

    def __init__(self) -> None:
        super().__init__(StrictModel())
        self.metrics = ("a", "b", "c")
        self.feature_schema = (
            "a__z_local",
            "a__roc",
            "b__z_local",
            "b__roc",
            "c__z_local",
            "c__roc",
        )
        self._fitted = True

    def score_features(self, features: FloatArray) -> FloatArray:
        result: FloatArray = np.abs(features).max(axis=1)
        return result

    def _fit(self, ff: FeatureFrame, rows: BoolArray) -> None: ...

    def _score(self, ff: FeatureFrame, rows: BoolArray) -> FloatArray:
        return self.score_features(self.matrix(ff, rows))

    def _state(self, directory: Path) -> None: ...

    def _restore(self, directory: Path) -> None: ...


TYPICAL = np.zeros(6)


def test_features_are_grouped_by_metric() -> None:
    groups = feature_groups(("a__z_local", "b__roc", "a__roc", "calendar__weekend"), ("a", "b"))
    assert groups == {"a": [0, 2], "b": [1]}


def test_drop_one_ranks_the_metric_whose_removal_lowers_the_score_most() -> None:
    row = np.array([[8.0, 2.0, 1.0, 0.0, 0.5, 0.0]])
    contributions, keep_one = occlusion(_MaxAbs(), row, TYPICAL)
    assert not keep_one[0]
    np.testing.assert_allclose(contributions[0], [7.0, 0.0, 0.0])


def test_saturated_drop_one_falls_back_to_keep_one() -> None:
    row = np.array([[8.0, 0.0, 8.0, 0.0, 1.0, 0.0]])  # a and b are both extreme
    contributions, keep_one = occlusion(_MaxAbs(), row, TYPICAL)
    assert keep_one[0]
    np.testing.assert_allclose(contributions[0], [8.0, 8.0, 1.0])


def _frames(metrics: MetricsFrame) -> tuple[FeatureFrame, FeatureFrame, FeatureBuilder]:
    builder = FeatureBuilder().fit(metrics.until(TRAIN_END, FIVE_MINUTES), FIVE_MINUTES)
    ff = builder.transform(metrics)
    return ff.select((ff.timestamps < TRAIN_END).to_numpy(), "train"), ff, builder


@pytest.fixture(scope="module")
def spiked(
    make_metrics: Callable[..., MetricsFrame],
) -> tuple[FeatureFrame, FeatureFrame, FeatureBuilder, np.ndarray]:
    metrics = make_metrics(seed=4, days=4, gap_fraction=0.0)
    data = metrics.data.copy()
    target = (data["host_id"] == "h1") & (data["timestamp"].dt.round(FIVE_MINUTES) == SPIKE)
    data.loc[target, "cpu_usage_percent"] = 99.0
    data.loc[target, "http_5xx_rate"] = 0.4
    train, ff, builder = _frames(MetricsFrame(data, "host_id", metrics.metrics))
    position = np.flatnonzero((ff.entities == "h1") & (ff.timestamps == SPIKE).to_numpy())
    return train, ff, builder, position


@pytest.mark.parametrize("name", ["isolation_forest", "one_class_svm"])
def test_occlusion_on_a_fitted_model_ranks_the_spiked_metrics_first(
    name: str, spiked: tuple[FeatureFrame, FeatureFrame, FeatureBuilder, np.ndarray]
) -> None:
    train, ff, builder, position = spiked
    detector = create_detector(default_config(name)).fit(train)
    result = attribute(detector, ff, position, builder.state.feature_medians)
    assert result.methods[0] in ("occlusion_drop_one", "occlusion_keep_one")
    ranked = [result.metrics[i] for i in np.argsort(-result.contributions[0])]
    assert set(ranked[:2]) == {"cpu_usage_percent", "http_5xx_rate"}


def test_robust_zscore_terms_reproduce_the_score_exactly(
    spiked: tuple[FeatureFrame, FeatureFrame, FeatureBuilder, np.ndarray],
) -> None:
    train, ff, builder, _ = spiked
    detector = create_detector(default_config("robust_zscore")).fit(train)
    positions = np.flatnonzero(detector.rows(ff))[:500]
    result = attribute(detector, ff, positions, builder.state.feature_medians)
    assert set(result.methods) == {"metric_terms"}
    np.testing.assert_allclose(result.contributions.max(axis=1), detector.score(ff)[positions])


def test_static_terms_are_value_over_limit_and_nan_without_a_limit(
    spiked: tuple[FeatureFrame, FeatureFrame, FeatureBuilder, np.ndarray],
) -> None:
    train, ff, builder, position = spiked
    detector = create_detector(default_config("static_threshold")).fit(train)
    terms = attribute(detector, ff, position, builder.state.feature_medians).contributions[0]
    by_metric = dict(zip(detector.metrics, terms, strict=True))
    assert np.isnan(by_metric["requests_per_second"])  # no default limit
    assert by_metric["cpu_usage_percent"] == pytest.approx(99.0 / 90.0)
    assert by_metric["http_5xx_rate"] == pytest.approx(0.4 / 0.05)
    assert detector.input_metrics == ("cpu_usage_percent", "http_5xx_rate")
