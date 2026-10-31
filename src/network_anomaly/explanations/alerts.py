"""From flagged points to explainable alerts.

A single flagged point is a weak signal: one noisy sample or a scrape glitch can produce it.
Alerts therefore group flagged points per entity, tolerate short interruptions, and only fire once
enough points have been flagged, like an alert rule with a ``for:`` duration in production:

* flagged points of one entity separated by at most ``merge_gap`` unflagged or missing intervals
  belong to the same alert;
* an alert needs ``min_points`` flagged points. It *fires* at its ``min_points``-th point
  (``fired_at``, when an operator would be notified) and is *active* from ``fired_at`` until
  ``end``, one interval after its last flagged point.

:func:`assemble_alerts` then explains each alert at its peak (the row with the highest score):
which metrics contributed (see :mod:`~network_anomaly.explanations.attribution`), observed values
against both causal baselines, a severity from deviation rules (never from the score), a one-line
summary that describes co-occurring deviations without asserting a cause, suggested checks and the
data-quality problems that limit how far the alert can be trusted.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field

from network_anomaly.config import StrictModel
from network_anomaly.detectors.base import Detector
from network_anomaly.explanations.attribution import attribute
from network_anomaly.explanations.playbook import suggested_checks
from network_anomaly.features.builder import FeatureFrame
from network_anomaly.schemas import (
    Alert,
    DataQualityWarning,
    MetricContribution,
    metric_spec,
)
from network_anomaly.scoring.severity import (
    SeverityRules,
    classify_severity,
    directional_deviations,
)
from network_anomaly.scoring.thresholds import HostThresholds, ScoreReference

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

# A metric is mentioned as co-occurring only if it deviated at least this much (noise units).
CO_OCCURRENCE_Z = 3.0


class AlertingConfig(StrictModel):
    merge_gap: int = Field(
        default=2, ge=0, description="Unflagged or missing intervals allowed inside one alert."
    )
    min_points: int = Field(
        default=2, ge=1, description="Flagged points required before an alert fires."
    )
    top_contributors: int = Field(default=3, ge=1, le=20, description="Metrics listed per alert.")
    severity: SeverityRules = Field(default_factory=SeverityRules)


@dataclass(frozen=True)
class AlertInterval:
    """One alert on one entity. Times are grid timestamps (UTC); ``end`` is exclusive."""

    entity: str
    start: pd.Timestamp
    fired_at: pd.Timestamp
    end: pd.Timestamp
    peak_at: pd.Timestamp
    peak_score: float
    flagged_points: int


def build_alert_intervals(
    entities: npt.NDArray[np.object_],
    times: pd.Series,
    flagged: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    frequency: pd.Timedelta,
    config: AlertingConfig,
) -> list[AlertInterval]:
    """Group flagged rows into alerts, ordered by firing time then entity."""
    rows = pd.DataFrame(
        {
            "entity": entities,
            "time": times.to_numpy(),
            "score": scores,
        }
    ).loc[flagged]
    rows["time"] = pd.to_datetime(rows["time"], utc=True)
    rows = rows.sort_values(["entity", "time"], kind="stable")
    largest_step = (config.merge_gap + 1) * frequency
    alerts: list[AlertInterval] = []
    for entity, group in rows.groupby("entity", sort=True):
        block = (group["time"].diff() > largest_step).cumsum()
        for _, points in group.groupby(block, sort=True):
            if len(points) < config.min_points:
                continue
            peak = points.iloc[int(np.argmax(points["score"].to_numpy()))]
            alerts.append(
                AlertInterval(
                    entity=str(entity),
                    start=points["time"].iloc[0],
                    fired_at=points["time"].iloc[config.min_points - 1],
                    end=points["time"].iloc[-1] + frequency,
                    peak_at=peak["time"],
                    peak_score=float(peak["score"]),
                    flagged_points=len(points),
                )
            )
    return sorted(alerts, key=lambda alert: (alert.fired_at, alert.entity))


# --------------------------------------------------------------------------------------------------
# Wording
# --------------------------------------------------------------------------------------------------


def _bytes(value: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if abs(value) < 1000:
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1000
    return f"{value:,.1f} TB"


def format_value(value: float | None, unit: str) -> str:
    """A metric value in its unit, readable at a glance (fractions are shown as percentages)."""
    if value is None or not math.isfinite(value):
        return "n/a"
    if unit == "fraction":
        return f"{value:.2%}"
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "ms":
        return f"{value:,.0f} ms"
    if unit == "bytes/s":
        return f"{_bytes(value)}/s"
    if unit == "requests/s":
        return f"{value:,.1f} req/s"
    if unit == "connections":
        return f"{value:,.0f} connections"
    return f"{value:.4g}" if unit == "unknown" else f"{value:.4g} {unit}"


def _directional(z: float | None, metric: str) -> float:
    """Size of a deviation in the metric's meaningful direction (0 for the ignored direction)."""
    if z is None:
        return 0.0
    direction = metric_spec(metric).direction
    if direction == "up":
        return max(z, 0.0)
    if direction == "down":
        return max(-z, 0.0)
    return abs(z)


def _reference(item: MetricContribution) -> tuple[float | None, float | None, str]:
    """The baseline that shows the meaningful deviation best.

    During a long incident the recent baseline absorbs the anomaly (a 5xx surge an hour old looks
    like a *drop* against the last hour), so the same-time baseline is used whenever it shows the
    larger deviation in the metric's meaningful direction.
    """
    if item.seasonal_baseline is not None and _directional(
        item.seasonal_z, item.metric
    ) > _directional(item.robust_z, item.metric):
        return item.seasonal_baseline, item.seasonal_z, "typical at this time of day"
    return item.baseline, item.robust_z, "typical recently"


def _change(observed: float, reference: float, unit: str) -> str | None:
    if unit in ("fraction", "%"):
        points = (observed - reference) * (100.0 if unit == "fraction" else 1.0)
        return f"{points:+.2f} pp" if unit == "fraction" else f"{points:+.1f} pp"
    if reference == 0:
        return None
    return f"{100.0 * (observed - reference) / reference:+,.0f}%"


def describe_metric(item: MetricContribution) -> str:
    """'<metric> <observed> vs <typical> typical ... (<change>, z <z>)'."""
    observed = format_value(item.observed, item.unit)
    reference, z, label = _reference(item)
    if item.observed is None or reference is None:
        return f"{item.metric} {observed} (no baseline yet)"
    details = [_change(item.observed, reference, item.unit)]
    if z is not None:
        details.append(f"z {z:+.1f}")
    shown = ", ".join(d for d in details if d)
    return f"{item.metric} {observed} vs {format_value(reference, item.unit)} {label}" + (
        f" ({shown})" if shown else ""
    )


def _join(parts: Sequence[str]) -> str:
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def summarise(
    entity: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    contributions: Sequence[MetricContribution],
) -> str:
    """One sentence: what deviated, against which baseline, when, and what moved with it."""
    same_day = start.date() == end.date()
    when = (
        f"from {start:%Y-%m-%d %H:%M} to {end:%H:%M} UTC"
        if same_day
        else (f"from {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC")
    )
    if not contributions:
        return f"{entity}: anomalous combination of metrics {when}."
    lead, *others = contributions
    text = f"{entity}: {describe_metric(lead)} {when}"
    companions = []
    for item in others:
        reference, z, _ = _reference(item)
        if _directional(z, item.metric) >= CO_OCCURRENCE_Z:
            companions.append(
                f"{item.metric} ({format_value(item.observed, item.unit)} vs "
                f"{format_value(reference, item.unit)})"
            )
    if companions:
        text += f"; co-occurred with {_join(companions)}"
    return text + "."


# --------------------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------------------


class _RowLocator:
    """Positions of feature-frame rows by (entity, grid time), via per-entity sorted times."""

    def __init__(self, ff: FeatureFrame) -> None:
        nanos = ff.timestamps.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        times = nanos.view(np.int64)
        codes, uniques = pd.factorize(pd.Series(ff.entities, dtype=object))
        self._blocks: dict[str, tuple[IntArray, npt.NDArray[np.int64]]] = {}
        for code, entity in enumerate(uniques):
            positions = np.flatnonzero(codes == code)
            ordered = positions[np.argsort(times[positions], kind="stable")]
            self._blocks[str(entity)] = (ordered, times[ordered])

    def span(self, entity: str, start: pd.Timestamp, end: pd.Timestamp) -> IntArray:
        positions, times = self._blocks[entity]
        lo, hi = np.searchsorted(times, [start.value, end.value], side="left")
        return positions[lo:hi]

    def position(self, entity: str, time: pd.Timestamp) -> int:
        positions, times = self._blocks[entity]
        index = int(np.searchsorted(times, time.value))
        if index >= len(times) or times[index] != time.value:
            raise KeyError(f"no row for {entity} at {time}")
        return int(positions[index])


def _number(value: float, digits: int) -> float | None:
    return round(float(value), digits) if math.isfinite(value) else None


def _contribution(
    metric: str, value: float, row: Mapping[str, float], available: bool
) -> MetricContribution:
    spec = metric_spec(metric)
    observed = _number(row[f"{metric}__value"], 6)
    baseline = _number(row[f"{metric}__baseline"], 6)
    seasonal = _number(row[f"{metric}__seasonal_baseline"], 6)
    percent = None
    if observed is not None and baseline:
        percent = round(100.0 * (observed - baseline) / baseline, 2)
    return MetricContribution(
        metric=metric,
        family=spec.family,
        unit=spec.unit,
        observed=observed,
        baseline=baseline,
        seasonal_baseline=seasonal,
        deviation_percent=percent,
        robust_z=_number(row[f"{metric}__z_local"], 3) if available else None,
        seasonal_z=_number(row[f"{metric}__z_season"], 3)
        if available and seasonal is not None
        else None,
        contribution=round(float(value), 6),
    )


@dataclass(frozen=True)
class _RowStatus:
    """Per-row data-quality flags of a feature frame as NumPy arrays."""

    metrics: tuple[str, ...]
    observed: npt.NDArray[np.bool_]
    available: npt.NDArray[np.bool_]  # rows x metrics
    season_ready: npt.NDArray[np.bool_]
    entity_seen: npt.NDArray[np.bool_]
    masked_cells: npt.NDArray[np.int64]

    @classmethod
    def of(cls, ff: FeatureFrame) -> "_RowStatus":
        quality = ff.quality
        return cls(
            metrics=ff.metrics,
            observed=ff.observed,
            available=ff.available[list(ff.metrics)].to_numpy(dtype=bool),
            season_ready=quality["season_ready"].to_numpy(dtype=bool),
            entity_seen=quality["entity_seen"].to_numpy(dtype=bool),
            masked_cells=quality["masked_cells"].to_numpy(dtype=np.int64),
        )

    def is_available(self, peak: int, metric: str) -> bool:
        return bool(self.available[peak, self.metrics.index(metric)])


def _quality_warnings(
    status: _RowStatus,
    rows: IntArray,
    peak: int,
    peak_row: Mapping[str, float],
    detector: Detector,
    segment: str,
    fitted_on: str | None,
) -> list[DataQualityWarning]:
    warnings: list[DataQualityWarning] = []
    gaps = int((~status.observed[rows]).sum())
    if gaps:
        warnings.append(
            DataQualityWarning(
                code="telemetry_gap",
                message=f"{gaps} of {len(rows)} intervals in the alert have no data",
            )
        )
    masked = int(status.masked_cells[rows].sum())
    if masked:
        warnings.append(
            DataQualityWarning(
                code="masked_cells",
                message=f"{masked} invalid value(s) in the alert were masked during validation",
            )
        )
    missing = [
        metric
        for metric in detector.input_metrics
        if not math.isfinite(peak_row[f"{metric}__value"])
        or (detector.requires_baseline and not status.is_available(peak, metric))
    ]
    if missing:
        warnings.append(
            DataQualityWarning(
                code="missing_metrics",
                message=f"{len(missing)} metric(s) had no value or no baseline at the peak and "
                "were treated as typical",
                metrics=tuple(missing),
            )
        )
    if "z_season" in detector.feature_kinds and not status.season_ready[peak]:
        warnings.append(
            DataQualityWarning(
                code="seasonal_baseline_unavailable",
                message="no same-time baseline from previous days yet (cold start); seasonal "
                "deviations were treated as typical",
            )
        )
    if not status.entity_seen[peak]:
        warnings.append(
            DataQualityWarning(
                code="unseen_entity",
                message="this entity was not in the training data: pooled noise floors and the "
                "global threshold apply",
            )
        )
    if segment == "train" or segment == fitted_on:
        fitted = "detector" if segment == "train" else "threshold"
        warnings.append(
            DataQualityWarning(
                code="in_sample",
                message=f"the alert lies in the {segment} period used to fit the {fitted}; "
                "treat it as in-sample",
            )
        )
    return warnings


def assemble_alerts(
    ff: FeatureFrame,
    scores: FloatArray,
    flagged: npt.NDArray[np.bool_],
    segments: npt.NDArray[np.object_],
    *,
    detector: Detector,
    thresholds: HostThresholds,
    feature_medians: Mapping[str, float],
    frequency: pd.Timedelta,
    config: AlertingConfig,
    reference: ScoreReference | None = None,
) -> list[Alert]:
    """Group flagged rows into alerts and explain each one at its peak."""
    intervals = build_alert_intervals(
        ff.entities, ff.timestamps, flagged, scores, frequency, config
    )
    if not intervals:
        return []
    locator = _RowLocator(ff)
    status = _RowStatus.of(ff)
    peaks = np.array([locator.position(a.entity, a.peak_at) for a in intervals], dtype=np.intp)
    attribution = attribute(detector, ff, peaks, feature_medians)
    peak_values = pd.concat([ff.context.iloc[peaks], ff.X.iloc[peaks]], axis=1).to_dict("records")
    entities = np.array([a.entity for a in intervals], dtype=object)
    limits = thresholds.thresholds_for(entities)
    scopes = thresholds.scope_for(entities)
    alerts = []
    for index, interval in enumerate(intervals):
        peak = int(peaks[index])
        rows = locator.span(interval.entity, interval.start, interval.end)
        observed_rows = rows[status.observed[rows]]
        severity, reason = classify_severity(
            directional_deviations(ff, observed_rows), ff.metrics, config.severity
        )
        values = attribution.contributions[index]
        order = np.argsort(-np.nan_to_num(values, nan=-np.inf), kind="stable")
        ranked = [int(i) for i in order if np.isfinite(values[i])]
        row = {str(k): float(v) for k, v in peak_values[index].items()}
        contributions = tuple(
            _contribution(ff.metrics[i], values[i], row, bool(status.available[peak, i]))
            for i in ranked[: config.top_contributors]
        )
        segment = str(segments[locator.position(interval.entity, interval.fired_at)])
        alerts.append(
            Alert(
                alert_id=f"{interval.entity}@{interval.start:%Y%m%dT%H%M%SZ}",
                entity_column=ff.entity_column,
                entity=interval.entity,
                start=interval.start.to_pydatetime(),
                fired_at=interval.fired_at.to_pydatetime(),
                end=interval.end.to_pydatetime(),
                peak_at=interval.peak_at.to_pydatetime(),
                duration_minutes=(interval.end - interval.start).total_seconds() / 60.0,
                flagged_points=interval.flagged_points,
                segment=segment,
                detector=detector.name,
                detector_version=detector.version,
                score=round(interval.peak_score, 6),
                threshold=round(float(limits[index]), 6),
                threshold_scope="host" if scopes[index] == "host" else "global",
                threshold_strategy=thresholds.strategy,
                reference_exceedance=None
                if reference is None
                else round(reference.exceedance(interval.peak_score), 6),
                severity=severity,
                severity_reason=reason,
                explanation_method=attribution.methods[index],
                contributions=contributions,
                metric_ranking=tuple(ff.metrics[i] for i in ranked),
                peak_deviations={
                    metric: _number(row[f"{metric}__z_local"], 3)
                    if status.available[peak, j]
                    else None
                    for j, metric in enumerate(ff.metrics)
                },
                summary=summarise(interval.entity, interval.start, interval.end, contributions),
                suggested_checks=suggested_checks([c.metric for c in contributions]),
                data_quality=tuple(
                    _quality_warnings(
                        status, rows, peak, row, detector, segment, thresholds.fitted_on
                    )
                ),
            )
        )
    return alerts
