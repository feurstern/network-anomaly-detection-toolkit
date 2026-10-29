import math

import numpy as np
import pytest

from network_anomaly.errors import ConfigError, DataValidationError, LeakageError
from network_anomaly.scoring.thresholds import (
    MIN_EXCEEDANCES,
    FixedThresholdConfig,
    HostThresholds,
    QuantileThresholdConfig,
    TargetFprConfig,
    fit_thresholds,
)


def _hosts(*counts: tuple[str, int]) -> np.ndarray:
    return np.array([name for name, count in counts for _ in range(count)], dtype=object)


def test_fixed_threshold_flags_scores_at_or_above_the_value() -> None:
    thresholds = fit_thresholds(
        FixedThresholdConfig(value=1.0), np.array([]), _hosts(), role="train"
    )
    scores = np.array([0.5, 1.0, 2.0, np.nan])
    flags = thresholds.apply(scores, _hosts(("a", 4)))
    assert flags.tolist() == [False, True, True, False]  # NaN is never flagged
    assert thresholds.fitted_on is None


def test_train_quantile_uses_training_scores() -> None:
    scores = np.arange(1000, dtype=float)
    config = QuantileThresholdConfig(strategy="train_quantile", quantile=0.99, per_host=False)
    thresholds = fit_thresholds(config, scores, _hosts(("a", 1000)), role="train")
    assert thresholds.global_threshold == pytest.approx(np.quantile(scores, 0.99))
    assert thresholds.n_samples == 1000


@pytest.mark.parametrize(
    ("strategy", "role"),
    [("train_quantile", "validation"), ("clean_quantile", "train")],
)
def test_each_strategy_is_fitted_on_its_own_segment(strategy: str, role: str) -> None:
    config = QuantileThresholdConfig(strategy=strategy)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="are fitted on"):
        fit_thresholds(config, np.ones(10), _hosts(("a", 10)), role=role)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "config",
    [
        FixedThresholdConfig(value=1.0),
        QuantileThresholdConfig(),
        TargetFprConfig(),
    ],
)
def test_thresholds_are_never_fitted_on_test_data(config: object) -> None:
    with pytest.raises(LeakageError):
        fit_thresholds(config, np.ones(10), _hosts(("a", 10)), role="test")  # type: ignore[arg-type]


def test_per_host_thresholds_need_enough_samples_and_fall_back_to_global() -> None:
    quantile = 0.99
    minimum = math.ceil(MIN_EXCEEDANCES / (1 - quantile))
    rng = np.random.default_rng(0)
    hosts = _hosts(("busy", minimum), ("quiet", minimum - 1))
    scores = np.concatenate([rng.normal(10, 1, minimum), rng.normal(0, 1, minimum - 1)])
    thresholds = fit_thresholds(
        QuantileThresholdConfig(quantile=quantile), scores, hosts, role="validation"
    )
    assert set(thresholds.per_host) == {"busy"}
    assert thresholds.hosts_using_global == ("quiet",)
    lookup = _hosts(("busy", 1), ("quiet", 1), ("never-seen", 1))
    values = thresholds.thresholds_for(lookup)
    assert values[0] == pytest.approx(np.quantile(scores[:minimum], quantile))
    assert values[1] == values[2] == pytest.approx(thresholds.global_threshold)
    assert thresholds.scope_for(lookup).tolist() == ["host", "global", "global"]


def test_target_fpr_uses_only_rows_labelled_normal() -> None:
    rng = np.random.default_rng(1)
    normal = rng.normal(0, 1, 5000)
    anomalous = rng.normal(8, 1, 500)
    scores = np.concatenate([normal, anomalous])
    labels = np.concatenate([np.zeros(5000), np.ones(500)]).astype(np.int8)
    config = TargetFprConfig(target_fpr=0.01, per_host=False)
    thresholds = fit_thresholds(
        config, scores, _hosts(("a", 5500)), role="validation", labels=labels
    )
    false_positive_rate = (normal >= thresholds.global_threshold).mean()
    assert false_positive_rate == pytest.approx(0.01, abs=0.002)
    with pytest.raises(ConfigError, match="needs anomaly labels"):
        fit_thresholds(config, scores, _hosts(("a", 5500)), role="validation")


def test_no_scored_rows_is_an_actionable_error() -> None:
    with pytest.raises(DataValidationError, match="no scored rows"):
        fit_thresholds(
            QuantileThresholdConfig(), np.full(5, np.nan), _hosts(("a", 5)), role="validation"
        )


def test_thresholds_round_trip_through_json() -> None:
    thresholds = fit_thresholds(
        QuantileThresholdConfig(quantile=0.9),
        np.arange(500, dtype=float),
        _hosts(("a", 250), ("b", 250)),
        role="validation",
    )
    restored = HostThresholds.model_validate_json(thresholds.model_dump_json())
    assert restored == thresholds
