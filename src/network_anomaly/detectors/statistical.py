"""Baseline detectors that engineers can reason about without a model.

* :class:`StaticThresholdDetector` - fixed per-metric limits, i.e. classic alert rules.
* :class:`RobustZScoreDetector` - deviation from each host's own rolling median, measured in robust
  noise units (rolling median absolute deviation of recent residuals).
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, field_validator

from network_anomaly.config import StrictModel
from network_anomaly.detectors.base import BoolArray, Detector, FloatArray
from network_anomaly.errors import ArtifactError, ConfigError
from network_anomaly.features.builder import FeatureFrame
from network_anomaly.schemas import Direction, metric_spec

STATE_FILE = "state.json"

DEFAULT_LIMITS: dict[str, float] = {
    "cpu_usage_percent": 90.0,
    "memory_usage_percent": 90.0,
    "http_5xx_rate": 0.05,
    "connection_error_rate": 0.05,
    "response_time_p95_ms": 1000.0,
}


class StaticThresholdConfig(StrictModel):
    type: Literal["static_threshold"] = "static_threshold"
    limits: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_LIMITS),
        description="Upper limit per metric in raw units; a row scores value / limit.",
    )

    @field_validator("limits")
    @classmethod
    def _positive(cls, limits: dict[str, float]) -> dict[str, float]:
        if not limits:
            raise ValueError("at least one limit is required")
        bad = [name for name, limit in limits.items() if not limit > 0]
        if bad:
            raise ValueError(f"limits must be positive: {bad}")
        return limits


class StaticThresholdDetector(Detector):
    """Score = largest ``value / limit`` over metrics with a limit; 1.0 means a limit is reached.

    It learns nothing from data and needs no baseline, so it scores from the first observation.
    Its natural operating point is the fixed threshold 1.0.
    """

    name = "static_threshold"
    requires_baseline = False
    min_training_rows = 1
    config: StaticThresholdConfig

    def __init__(self, config: StaticThresholdConfig | None = None) -> None:
        super().__init__(config or StaticThresholdConfig())
        self.limits: dict[str, float] = {}

    def _fit(self, ff: FeatureFrame, rows: BoolArray) -> None:
        self.limits = {m: v for m, v in self.config.limits.items() if m in ff.metrics}
        if not self.limits:
            raise ConfigError(
                "none of the static limits apply to the metrics in this data",
                hint=f"Limits are set for {sorted(self.config.limits)}.",
            )

    def _score(self, ff: FeatureFrame, rows: BoolArray) -> FloatArray:
        ratios = np.column_stack(
            [
                ff.context[f"{metric}__value"].to_numpy(dtype=np.float64)[rows] / limit
                for metric, limit in self.limits.items()
            ]
        )
        # fmax ignores missing metrics; a row with every limited metric missing stays NaN.
        return np.fmax.reduce(ratios, axis=1)

    def _state(self, directory: Path) -> None:
        (directory / STATE_FILE).write_text(json.dumps({"limits": self.limits}, indent=2) + "\n")

    def _restore(self, directory: Path) -> None:
        try:
            self.limits = json.loads((directory / STATE_FILE).read_text())["limits"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read detector state in {directory}: {exc}") from exc


class RobustZScoreConfig(StrictModel):
    type: Literal["robust_zscore"] = "robust_zscore"
    normalisation_quantile: float = Field(
        default=0.99,
        gt=0.5,
        lt=1.0,
        description="Each metric's directional |z| is divided by this training quantile so that "
        "naturally spiky metrics do not dominate the maximum.",
    )
    directions: dict[str, Direction] = Field(
        default_factory=dict,
        description="Override which deviations count per metric (default: the metric registry).",
    )


class RobustZScoreDetector(Detector):
    """Score = max over metrics of directional robust z, normalised per metric.

    The robust z of a metric is its deviation from the host's rolling median in units of the
    rolling MAD of recent residuals (computed causally by the feature builder). Direction follows
    domain meaning: errors, latency, CPU and memory only count increases; traffic counts both ways.
    ``fit`` learns one number per metric: the training quantile of its directional z, used to put
    metrics on a common footing. Hosts never seen in training are scored like any other host.
    """

    name = "robust_zscore"
    default_feature_kinds = ("z_local",)
    config: RobustZScoreConfig

    def __init__(self, config: RobustZScoreConfig | None = None) -> None:
        super().__init__(config or RobustZScoreConfig())
        self.normalisers: dict[str, float] = {}

    def directional(self, ff: FeatureFrame, rows: BoolArray) -> FloatArray:
        """(rows x metrics) deviations that count, clipped at zero for the ignored direction."""
        z = self.matrix(ff, rows)
        columns = []
        for index, metric in enumerate(self.metrics):
            direction = self.config.directions.get(metric, metric_spec(metric).direction)
            values = z[:, index]
            if direction == "up":
                values = np.maximum(values, 0.0)
            elif direction == "down":
                values = np.maximum(-values, 0.0)
            else:
                values = np.abs(values)
            columns.append(values)
        return np.column_stack(columns)

    def _fit(self, ff: FeatureFrame, rows: BoolArray) -> None:
        deviations = self.directional(ff, rows)
        available = ff.available[list(self.metrics)].to_numpy(dtype=bool)[rows]
        quantile = self.config.normalisation_quantile
        self.normalisers = {}
        for index, metric in enumerate(self.metrics):
            values = deviations[available[:, index], index]
            learned = float(np.quantile(values, quantile)) if values.size else 1.0
            self.normalisers[metric] = max(learned, 1.0)

    def _score(self, ff: FeatureFrame, rows: BoolArray) -> FloatArray:
        normalisers = np.array([self.normalisers[m] for m in self.metrics])
        result: FloatArray = (self.directional(ff, rows) / normalisers).max(axis=1)
        return result

    def _state(self, directory: Path) -> None:
        state = {"normalisers": self.normalisers}
        (directory / STATE_FILE).write_text(json.dumps(state, indent=2) + "\n")

    def _restore(self, directory: Path) -> None:
        try:
            self.normalisers = json.loads((directory / STATE_FILE).read_text())["normalisers"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read detector state in {directory}: {exc}") from exc
