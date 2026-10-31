"""Alert severity from metric deviations - never from detector scores.

Scores are unitless and detector-specific, so they cannot say how serious a situation is. Severity
is a rule table on how far metrics moved from their causal baselines, in robust noise units, in
the direction that matters operationally (see the metric registry):

==========  ======================================================================================
severity    rule (first match wins)
==========  ======================================================================================
critical    a *critical* metric (a user-facing symptom: 5xx rate, p95 latency, connection errors)
            deviates by >= ``critical_z`` on at least ``critical_points`` points
high        any metric deviates by >= ``high_z``, or at least ``breadth_metrics`` metrics deviate
            by >= ``medium_z`` at the same point
medium      any metric deviates by >= ``medium_z``
low         otherwise: the detector flagged a combination of smaller deviations
==========  ======================================================================================

A metric's deviation on a row is the larger of its recent (``z_local``) and same-time
(``z_season``) deviation, counted only in its meaningful direction. Severity is a triage aid, not a
probability that an incident is happening.
"""

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from network_anomaly.config import StrictModel
from network_anomaly.features.builder import FeatureFrame
from network_anomaly.schemas import AlertSeverity, metric_spec

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

SEVERITY_ORDER: tuple[AlertSeverity, ...] = ("critical", "high", "medium", "low")


class SeverityRules(StrictModel):
    critical_z: float = Field(default=8.0, gt=0)
    critical_points: int = Field(default=3, ge=1)
    high_z: float = Field(default=6.0, gt=0)
    medium_z: float = Field(default=4.0, gt=0)
    breadth_metrics: int = Field(default=3, ge=2)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.medium_z <= self.high_z <= self.critical_z:
            raise ValueError("severity rules need medium_z <= high_z <= critical_z")
        return self


def directional_deviations(ff: FeatureFrame, positions: IntArray) -> FloatArray:
    """(rows x metrics) deviations in each metric's meaningful direction, >= 0.

    Unavailable features are neutral (0) in the feature frame, so they never raise severity.
    """
    columns = []
    for metric in ff.metrics:
        direction = metric_spec(metric).direction
        z = ff.X[[f"{metric}__z_local", f"{metric}__z_season"]].to_numpy(np.float64)[positions]
        if direction == "up":
            z = np.maximum(z, 0.0)
        elif direction == "down":
            z = np.maximum(-z, 0.0)
        else:
            z = np.abs(z)
        columns.append(z.max(axis=1))
    if not columns:
        return np.zeros((len(positions), 0))
    return np.column_stack(columns)


def classify_severity(
    deviations: FloatArray, metrics: Sequence[str], rules: SeverityRules
) -> tuple[AlertSeverity, str]:
    """Apply the rule table to the deviations of an alert's rows; returns (severity, reason)."""
    if deviations.size == 0:
        return "low", "no metric deviation could be computed"
    for index, metric in enumerate(metrics):
        if not metric_spec(metric).critical:
            continue
        points = int((deviations[:, index] >= rules.critical_z).sum())
        if points >= rules.critical_points:
            return (
                "critical",
                f"{metric} (user-facing) was at least {rules.critical_z:g} noise units from its "
                f"baseline on {points} points",
            )
    peaks = deviations.max(axis=0)
    largest = int(np.argmax(peaks))
    if peaks[largest] >= rules.high_z:
        return (
            "high",
            f"{metrics[largest]} reached {peaks[largest]:.1f} noise units from its baseline "
            f"(>= {rules.high_z:g})",
        )
    wide = deviations >= rules.medium_z
    busiest = int(np.argmax(wide.sum(axis=1)))
    if int(wide[busiest].sum()) >= rules.breadth_metrics:
        names = [m for m, hit in zip(metrics, wide[busiest], strict=True) if hit]
        return (
            "high",
            f"{len(names)} metrics were at least {rules.medium_z:g} noise units from their "
            f"baselines at the same time ({', '.join(names)})",
        )
    if peaks[largest] >= rules.medium_z:
        return (
            "medium",
            f"{metrics[largest]} reached {peaks[largest]:.1f} noise units from its baseline "
            f"(>= {rules.medium_z:g})",
        )
    return (
        "low",
        f"largest deviation {metrics[largest]} at {peaks[largest]:.1f} noise units "
        f"(< {rules.medium_z:g}); the detector flagged a combination of smaller changes",
    )
