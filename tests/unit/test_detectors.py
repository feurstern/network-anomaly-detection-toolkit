"""Contract tests that every detector must pass."""

import importlib.util
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from network_anomaly.detectors import (
    DETECTOR_TYPES,
    Detector,
    create_detector,
    default_config,
    load_detector,
)
from network_anomaly.detectors.statistical import (
    RobustZScoreConfig,
    RobustZScoreDetector,
    StaticThresholdConfig,
)
from network_anomaly.errors import (
    ArtifactError,
    ConfigError,
    DataValidationError,
    LeakageError,
    NotFittedError,
)
from network_anomaly.features.builder import FeatureBuilder, FeatureFrame
from network_anomaly.ingestion.validation import MetricsFrame
from network_anomaly.schemas import TIMESTAMP_COLUMN

FIVE_MINUTES = pd.Timedelta("5min")
TRAIN_END = pd.Timestamp("2026-01-07T00:00:00Z")
HAS_TORCH = importlib.util.find_spec("torch") is not None
DETECTORS = [
    pytest.param(name, marks=pytest.mark.skipif(not HAS_TORCH, reason="needs the torch extra"))
    if name == "autoencoder"
    else name
    for name in sorted(DETECTOR_TYPES)
]


def _frames(metrics: MetricsFrame) -> tuple[FeatureFrame, FeatureFrame]:
    """(training frame, whole-timeline frame) for ``metrics``."""
    builder = FeatureBuilder().fit(metrics.until(TRAIN_END, FIVE_MINUTES), FIVE_MINUTES)
    ff = builder.transform(metrics)
    return ff.select((ff.timestamps < TRAIN_END).to_numpy(), "train"), ff


@pytest.fixture(scope="module")
def frames(make_metrics: Callable[..., MetricsFrame]) -> tuple[FeatureFrame, FeatureFrame]:
    return _frames(make_metrics(seed=3, days=4))


def _fitted(name: str, train: FeatureFrame) -> Detector:
    return create_detector(default_config(name)).fit(train)


@pytest.mark.parametrize("name", DETECTORS)
def test_scores_one_value_per_row_and_nan_where_unscorable(
    name: str, frames: tuple[FeatureFrame, FeatureFrame]
) -> None:
    train, ff = frames
    detector = _fitted(name, train)
    scores = detector.score(ff)
    assert scores.shape == (len(ff),)
    assert scores.dtype == np.float64
    scorable = detector.rows(ff)
    assert np.isnan(scores[~scorable]).all()
    assert np.isfinite(scores[scorable]).mean() > 0.99


@pytest.mark.parametrize("name", DETECTORS)
def test_higher_scores_mean_more_anomalous(
    name: str, make_metrics: Callable[..., MetricsFrame]
) -> None:
    metrics = make_metrics(seed=4, days=4, gap_fraction=0.0)
    data = metrics.data.copy()
    when = pd.Timestamp("2026-01-08T12:00:00Z")
    target = (data["host_id"] == "h1") & (data[TIMESTAMP_COLUMN].dt.round(FIVE_MINUTES) == when)
    data.loc[target, "cpu_usage_percent"] = 99.0
    data.loc[target, "http_5xx_rate"] = 0.4
    data.loc[target, "requests_per_second"] *= 8
    train, ff = _frames(MetricsFrame(data, "host_id", metrics.metrics))
    scores = _fitted(name, train).score(ff)
    row = ((ff.keys["host_id"] == "h1") & (ff.timestamps == when)).to_numpy()
    others = np.isfinite(scores) & ~row
    assert scores[row].item() > np.quantile(scores[others], 0.999)


@pytest.mark.parametrize("name", DETECTORS)
def test_fitting_is_deterministic_and_survives_save_and_load(
    name: str, frames: tuple[FeatureFrame, FeatureFrame], tmp_path: Path
) -> None:
    train, ff = frames
    first = _fitted(name, train)
    second = _fitted(name, train)
    np.testing.assert_array_equal(first.score(ff), second.score(ff))
    first.save(tmp_path / name)
    restored = load_detector(tmp_path / name)
    assert type(restored) is type(first)
    np.testing.assert_array_equal(restored.score(ff), first.score(ff))


@pytest.mark.parametrize("name", DETECTORS)
def test_fit_refuses_anything_but_training_data(
    name: str, frames: tuple[FeatureFrame, FeatureFrame]
) -> None:
    _, ff = frames
    for role in ("all", "validation", "test"):
        with pytest.raises(LeakageError):
            create_detector(default_config(name)).fit(ff.select(np.ones(len(ff), bool), role))


@pytest.mark.parametrize("name", DETECTORS)
def test_scoring_before_fitting_is_an_error(
    name: str, frames: tuple[FeatureFrame, FeatureFrame], tmp_path: Path
) -> None:
    _, ff = frames
    detector = create_detector(default_config(name))
    with pytest.raises(NotFittedError):
        detector.score(ff)
    with pytest.raises(NotFittedError):
        detector.save(tmp_path)


def test_robust_zscore_needs_enough_training_rows(
    frames: tuple[FeatureFrame, FeatureFrame],
) -> None:
    train, _ = frames
    tiny = train.select(np.arange(len(train)) < 20, "train")
    with pytest.raises(DataValidationError, match="at least 50 scorable training rows"):
        RobustZScoreDetector().fit(tiny)


def test_robust_zscore_respects_metric_directions(
    frames: tuple[FeatureFrame, FeatureFrame],
) -> None:
    train, ff = frames
    detector = RobustZScoreDetector().fit(train)
    rows = ff.scorable
    deviations = detector.directional(ff, rows)
    cpu = list(detector.metrics).index("cpu_usage_percent")
    rps = list(detector.metrics).index("requests_per_second")
    z = ff.X[["cpu_usage_percent__z_local", "requests_per_second__z_local"]].to_numpy()[rows]
    np.testing.assert_array_equal(deviations[:, cpu], np.maximum(z[:, 0], 0))  # only increases
    np.testing.assert_array_equal(deviations[:, rps], np.abs(z[:, 1]))  # both directions
    overridden = RobustZScoreDetector(RobustZScoreConfig(directions={"cpu_usage_percent": "down"}))
    overridden.fit(train)
    np.testing.assert_array_equal(overridden.directional(ff, rows)[:, cpu], np.maximum(-z[:, 0], 0))
    assert all(value >= 1.0 for value in detector.normalisers.values())


def test_static_threshold_scores_value_over_limit(
    frames: tuple[FeatureFrame, FeatureFrame],
) -> None:
    train, ff = frames
    config = StaticThresholdConfig(limits={"cpu_usage_percent": 50.0, "memory_usage_percent": 80.0})
    detector = create_detector(config).fit(train)
    assert detector.limits == {"cpu_usage_percent": 50.0}  # absent metrics are skipped
    scores = detector.score(ff)
    values = ff.context["cpu_usage_percent__value"].to_numpy()
    observed = ff.observed
    np.testing.assert_allclose(scores[observed], values[observed] / 50.0)
    no_match = StaticThresholdConfig(limits={"memory_usage_percent": 80.0})
    with pytest.raises(ConfigError, match="none of the static limits apply"):
        create_detector(no_match).fit(train)


def test_static_threshold_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="limits must be positive"):
        StaticThresholdConfig(limits={"cpu_usage_percent": 0.0})


def test_loading_checks_the_manifest(
    frames: tuple[FeatureFrame, FeatureFrame], tmp_path: Path
) -> None:
    train, _ = frames
    _fitted("robust_zscore", train).save(tmp_path / "model")
    manifest = tmp_path / "model" / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"version": "1"', '"version": "99"'))
    with pytest.raises(ArtifactError, match="not robust_zscore v1"):
        load_detector(tmp_path / "model")
    with pytest.raises(ArtifactError, match="cannot read"):
        load_detector(tmp_path / "missing")


def test_scoring_with_different_metrics_is_rejected(
    frames: tuple[FeatureFrame, FeatureFrame],
) -> None:
    train, ff = frames
    detector = _fitted("robust_zscore", train)
    reduced = FeatureFrame(
        ff.X.drop(columns=["cpu_usage_percent__z_local"]),
        ff.keys,
        ff.context,
        ff.available,
        ff.quality,
        ff.entity_column,
        ff.metrics,
    )
    with pytest.raises(DataValidationError, match="fitted on different features"):
        detector.score(reduced)


def test_unknown_detector_name_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="unknown detector"):
        default_config("magic")


def test_isolation_forest_contamination_maps_to_a_training_quantile() -> None:
    from network_anomaly.pipelines import DetectionConfig

    config = DetectionConfig.model_validate(
        {"detector": {"type": "isolation_forest", "contamination": 0.02}}
    )
    threshold = config.resolved_threshold()
    assert threshold.strategy == "train_quantile"
    assert threshold.quantile == pytest.approx(0.98)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match=r"either detector\.contamination or threshold"):
        DetectionConfig.model_validate(
            {
                "detector": {"type": "isolation_forest", "contamination": 0.02},
                "threshold": {"strategy": "fixed", "value": 0.6},
            }
        )


def test_one_class_svm_trains_on_a_capped_host_stratified_sample(
    frames: tuple[FeatureFrame, FeatureFrame],
) -> None:
    from network_anomaly.detectors.sklearn_models import (
        OneClassSVMConfig,
        OneClassSVMDetector,
        stratified_sample,
    )

    train, _ = frames
    detector = OneClassSVMDetector(OneClassSVMConfig(max_train_samples=300)).fit(train)
    assert detector.scaler is not None
    assert detector.scaler.n_features_in_ == len(detector.feature_schema)
    entities = np.array(["a"] * 900 + ["b"] * 100, dtype=object)
    sample = stratified_sample(entities, 100, np.random.default_rng(0))
    assert len(sample) == 100
    assert (entities[sample] == "a").sum() == 90


def test_model_detectors_use_the_configured_feature_kinds(
    frames: tuple[FeatureFrame, FeatureFrame],
) -> None:
    from network_anomaly.detectors.sklearn_models import (
        IsolationForestConfig,
        IsolationForestDetector,
    )

    train, _ = frames
    detector = IsolationForestDetector(IsolationForestConfig(feature_kinds=("z_local",))).fit(train)
    assert detector.feature_schema == tuple(f"{m}__z_local" for m in train.metrics)
