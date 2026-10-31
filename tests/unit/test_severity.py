"""Severity is a rule table on metric deviations; it never looks at detector scores."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from network_anomaly.features.builder import FeatureBuilder
from network_anomaly.ingestion.validation import MetricsFrame
from network_anomaly.scoring.severity import (
    SeverityRules,
    classify_severity,
    directional_deviations,
)

RULES = SeverityRules()
METRICS = ("requests_per_second", "http_5xx_rate", "cpu_usage_percent")  # 5xx is critical


def _deviations(*rows: tuple[float, float, float]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def test_a_critical_metric_far_off_for_long_enough_is_critical() -> None:
    severity, reason = classify_severity(
        _deviations((0, 9, 0), (0, 8, 0), (0, 12, 0)), METRICS, RULES
    )
    assert severity == "critical"
    assert "http_5xx_rate" in reason
    assert "3 points" in reason


def test_a_short_critical_excursion_is_only_high() -> None:
    assert classify_severity(_deviations((0, 9, 0), (0, 8, 0)), METRICS, RULES)[0] == "high"


def test_a_non_critical_metric_can_reach_high_but_not_critical() -> None:
    rows = _deviations((20, 0, 0), (20, 0, 0), (20, 0, 0))
    assert classify_severity(rows, METRICS, RULES)[0] == "high"


def test_several_moderate_deviations_at_once_are_high() -> None:
    severity, reason = classify_severity(_deviations((4.5, 4.2, 5.0)), METRICS, RULES)
    assert severity == "high"
    assert "3 metrics" in reason


def test_medium_and_low() -> None:
    assert classify_severity(_deviations((5.0, 0, 0)), METRICS, RULES)[0] == "medium"
    assert classify_severity(_deviations((3.9, 3.9, 0)), METRICS, RULES)[0] == "low"
    assert classify_severity(np.zeros((0, 3)), METRICS, RULES)[0] == "low"


def test_rules_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="medium_z <= high_z <= critical_z"):
        SeverityRules(medium_z=7.0, high_z=6.0)


def test_deviations_count_only_the_meaningful_direction(
    make_metrics: Callable[..., MetricsFrame],
) -> None:
    metrics = make_metrics(seed=2, days=3, gap_fraction=0.0)
    data = metrics.data.copy()
    when = pd.Timestamp("2026-01-07T12:00:00Z")
    row = (data["host_id"] == "h0") & (data["timestamp"].dt.round("5min") == when)
    data.loc[row, "cpu_usage_percent"] = 0.0  # a drop: CPU only counts increases
    data.loc[row, "requests_per_second"] *= 0.1  # a drop: traffic counts both ways
    frame = MetricsFrame(data, "host_id", metrics.metrics)
    five_minutes = pd.Timedelta("5min")
    ff = FeatureBuilder().fit(frame, five_minutes).transform(frame)
    position = np.flatnonzero((ff.entities == "h0") & (ff.timestamps == when).to_numpy())
    deviations = directional_deviations(ff, position)[0]
    by_metric = dict(zip(ff.metrics, deviations, strict=True))
    assert by_metric["cpu_usage_percent"] == 0.0
    assert by_metric["requests_per_second"] > 10
    assert (deviations >= 0).all()
