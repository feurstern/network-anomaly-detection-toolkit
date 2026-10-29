"""Evaluate detections against ground truth - on the test segment unless asked otherwise.

Evaluation works on one table with a row per observed input row (``entity``, grid ``time``,
``segment``, ``scored``, ``score``, ``flagged``, ``is_anomaly``). The table is built either from a
detections CSV plus the labelled input file (CLI) or from an in-memory detection result
(benchmarks), so both paths compute identical metrics.

Evaluating training or validation rows measures in-sample behaviour (the detector or its threshold
was fitted on them); it is refused unless explicitly allowed and is then labelled as in-sample.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pydantic import AwareDatetime

from network_anomaly.config import StrictModel
from network_anomaly.errors import ArtifactError, ConfigError, DataValidationError, LeakageError
from network_anomaly.evaluation.events import (
    EventMatchingConfig,
    EventMetrics,
    event_window,
    match_events,
)
from network_anomaly.evaluation.point import PointMetrics, point_metrics
from network_anomaly.explanations.alerts import AlertingConfig, build_alert_intervals
from network_anomaly.ingestion.events import read_events, sibling_path
from network_anomaly.ingestion.validation import LabelFrame, load_metrics
from network_anomaly.pipelines import DetectionResult, align_labels
from network_anomaly.schemas import TIMESTAMP_COLUMN, Event

logger = logging.getLogger(__name__)

SEGMENT_ORDER = ("train", "validation", "test")


class EvaluationReport(StrictModel):
    detector: str
    segments: tuple[str, ...]
    in_sample: bool
    window_start: AwareDatetime
    window_end: AwareDatetime
    frequency_seconds: float
    alerting: AlertingConfig
    matching: EventMatchingConfig
    point: PointMetrics
    events: EventMetrics

    def summary(self) -> str:
        p, e = self.point, self.events

        def num(value: float | None, digits: int = 3) -> str:
            return "n/a" if value is None else f"{value:.{digits}f}"

        scope = " (IN-SAMPLE: fitted on these rows)" if self.in_sample else ""
        anomalous = p.positives / p.rows if p.rows else 0.0
        lines = [
            f"Evaluation of {self.detector} on {', '.join(self.segments)}{scope}: "
            f"{self.window_start.isoformat()} -> {self.window_end.isoformat()}",
            f"Point level: {p.scored:,} of {p.rows:,} rows scored, "
            f"{anomalous:.1%} labelled anomalous",
            f"  precision {num(p.precision)}  recall {num(p.recall)}  F1 {num(p.f1)}  "
            f"FPR {num(p.false_positive_rate, 4)}  PR-AUC {num(p.pr_auc)}",
            f"Event level (alert after {self.alerting.min_points} flagged points, gaps <= "
            f"{self.alerting.merge_gap}; window -{self.matching.early_tolerance}/"
            f"+{self.matching.late_tolerance} intervals):",
            f"  events {e.events}, detected {e.detected} (recall {num(e.recall)}); "
            f"alerts {e.alerts}, false {e.false_alerts} (precision {num(e.precision)}); "
            f"F1 {num(e.f1)}",
            f"  delay median {num(e.median_delay_minutes, 1)} min, p90 "
            f"{num(e.p90_delay_minutes, 1)} min; false alerts {num(e.false_alerts_per_day, 2)}"
            f"/day ({num(e.false_alerts_per_host_day, 3)} per host-day)",
        ]
        if e.excluded_events:
            lines.append(f"  {e.excluded_events} event(s) started earlier and were not evaluated")
        if e.by_scenario:
            groups = ", ".join(f"{k} {v.detected}/{v.events}" for k, v in e.by_scenario.items())
            lines.append(f"  detected by scenario: {groups}")
        return "\n".join(lines)


def events_from_labels(table: pd.DataFrame, frequency: pd.Timedelta) -> list[Event]:
    """Reconstruct events from point labels: by ``anomaly_event_id`` when present, else as
    contiguous runs of anomalous intervals per entity."""
    positive = table[table["is_anomaly"] == 1].sort_values(["entity", "time"])
    if positive.empty:
        return []
    if "anomaly_event_id" in positive and positive["anomaly_event_id"].notna().all():
        keys = positive["anomaly_event_id"]
    else:
        new_run = (positive["entity"] != positive["entity"].shift()) | (
            positive["time"].diff() > frequency
        )
        keys = new_run.cumsum().astype(str)
    events = []
    for key, rows in positive.groupby(keys, sort=False):
        first = rows.iloc[0]
        scenario = first.get("anomaly_scenario")
        events.append(
            Event(
                event_id=str(key) if "anomaly_event_id" in rows else f"run-{key}",
                host_id=str(first["entity"]),
                scenario=str(scenario) if pd.notna(scenario) else "unknown",
                start=rows["time"].min().to_pydatetime(),
                end=(rows["time"].max() + frequency).to_pydatetime(),
            )
        )
    return events


def _segment_window(
    split: dict[str, str], segments: Sequence[str]
) -> tuple[pd.Timestamp, pd.Timestamp]:
    bounds = {
        "train": ("start", "train_end"),
        "validation": ("train_end", "validation_end"),
        "test": ("validation_end", "end"),
    }
    starts = [pd.Timestamp(split[bounds[s][0]]) for s in segments]
    ends = [pd.Timestamp(split[bounds[s][1]]) for s in segments]
    return min(starts), max(ends)


def evaluate_table(
    table: pd.DataFrame,
    events: Sequence[Event],
    *,
    detector: str,
    split: dict[str, str],
    frequency: pd.Timedelta,
    segments: Sequence[str] = ("test",),
    allow_in_sample: bool = False,
    alerting: AlertingConfig | None = None,
    matching: EventMatchingConfig | None = None,
) -> EvaluationReport:
    """Point and event metrics for the rows of ``segments``."""
    unknown = [s for s in segments if s not in SEGMENT_ORDER]
    if unknown or not segments:
        raise ConfigError(f"unknown segment(s) {unknown}; choose from {', '.join(SEGMENT_ORDER)}")
    in_sample = any(s != "test" for s in segments)
    if in_sample and not allow_in_sample:
        raise LeakageError(
            "train and validation rows were used for fitting or calibration; evaluating them "
            "measures in-sample behaviour",
            hint="Evaluate the test segment, or pass --allow-in-sample to do it knowingly.",
        )
    alerting = alerting or AlertingConfig()
    matching = matching or EventMatchingConfig()
    ordered = tuple(s for s in SEGMENT_ORDER if s in segments)
    window_start, window_end = _segment_window(split, ordered)

    entities = table["entity"].to_numpy(dtype=object)
    times = table["time"]
    excluded = [e for e in events if pd.Timestamp(e.start) < window_start < pd.Timestamp(e.end)]
    excluded_rows = np.zeros(len(table), dtype=bool)
    for event in excluded:
        excluded_rows |= (
            (entities == event.host_id)
            & (times >= pd.Timestamp(event.start)).to_numpy()
            & (times < pd.Timestamp(event.end)).to_numpy()
        )
    selected = table["segment"].isin(ordered).to_numpy() & ~excluded_rows
    scores = table["score"].to_numpy(dtype=np.float64)
    flagged = table["flagged"].to_numpy(dtype=bool)
    point = point_metrics(
        table["is_anomaly"].to_numpy(dtype=np.int8)[selected], flagged[selected], scores[selected]
    )

    # Alerts are built over the whole timeline: one that fired earlier may still be active.
    alerts = build_alert_intervals(entities, times, flagged, scores, frequency, alerting)
    in_any_window = np.zeros(len(table), dtype=bool)
    for event in events:
        start, end = event_window(event, frequency, matching)
        in_any_window |= (
            (entities == event.host_id) & (times >= start).to_numpy() & (times < end).to_numpy()
        )
    normal_scored = selected & table["scored"].to_numpy(dtype=bool) & ~in_any_window
    normal_host_days = float(normal_scored.sum()) * frequency / pd.Timedelta(days=1)
    event_metrics, _ = match_events(
        events,
        alerts,
        window_start=window_start,
        window_end=window_end,
        frequency=frequency,
        config=matching,
        normal_host_days=normal_host_days,
    )
    return EvaluationReport(
        detector=detector,
        segments=ordered,
        in_sample=in_sample,
        window_start=window_start.to_pydatetime(),
        window_end=window_end.to_pydatetime(),
        frequency_seconds=frequency.total_seconds(),
        alerting=alerting,
        matching=matching,
        point=point,
        events=event_metrics,
    )


def table_from_result(result: DetectionResult, labels: LabelFrame | None) -> pd.DataFrame:
    """Evaluation table for an in-memory detection result (observed rows only)."""
    ff = result.features
    aligned = align_labels(ff, labels)
    if aligned is None:
        raise DataValidationError("evaluation needs an is_anomaly label column")
    table = pd.DataFrame(
        {
            "entity": ff.entities,
            "time": ff.timestamps.to_numpy(),
            "segment": result.segments,
            "scored": np.isfinite(result.scores),
            "score": result.scores,
            "flagged": result.flagged,
            "is_anomaly": aligned,
        }
    )
    return table.loc[ff.observed].reset_index(drop=True)


_DETECTION_COLUMNS = ("timestamp", "segment", "status", "score", "is_flagged")


def read_detections(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a detections CSV and its ``.meta.json`` sidecar written by ``detect``."""
    meta_path = sibling_path(path, "meta.json")
    try:
        meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"cannot read {meta_path}: {exc}",
            hint="Keep the .meta.json written by `detect` next to the detections CSV.",
        ) from exc
    entity = meta.get("entity_column", "host_id")
    try:
        frame = pd.read_csv(path, dtype={entity: str, "segment": str, "status": str})
    except (OSError, pd.errors.ParserError) as exc:
        raise ArtifactError(f"cannot read detections {path}: {exc}") from exc
    missing = [c for c in (*_DETECTION_COLUMNS, entity) if c not in frame.columns]
    if missing:
        raise DataValidationError(f"{path}: not a detections file; missing {', '.join(missing)}")
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True, format="ISO8601")
    frame[TIMESTAMP_COLUMN] = frame[TIMESTAMP_COLUMN].dt.as_unit("ns")
    return frame, meta


def evaluate_files(
    data_path: Path,
    predictions_path: Path,
    *,
    events_path: Path | None = None,
    segments: Sequence[str] = ("test",),
    allow_in_sample: bool = False,
    alerting: AlertingConfig | None = None,
    matching: EventMatchingConfig | None = None,
) -> EvaluationReport:
    """Evaluate a detections CSV against the labelled input it was computed from."""
    detections, meta = read_detections(predictions_path)
    entity = meta.get("entity_column", "host_id")
    frequency = pd.Timedelta(seconds=float(meta["features"]["frequency_seconds"]))
    validated = load_metrics(data_path, entity_column=entity, frequency=frequency)
    if validated.labels is None or "is_anomaly" not in validated.labels.data:
        raise DataValidationError(
            f"{data_path} has no is_anomaly column; evaluation needs labelled data"
        )
    labels = validated.labels.data.rename(columns={entity: "entity", TIMESTAMP_COLUMN: "source"})
    detections = detections.rename(columns={entity: "entity", TIMESTAMP_COLUMN: "source"})
    joined = detections.merge(labels, on=["entity", "source"], how="left", validate="one_to_one")
    unmatched = int(joined["is_anomaly"].isna().sum())
    if unmatched or len(joined) != len(labels):
        raise DataValidationError(
            f"{predictions_path} does not match {data_path}: {unmatched} detection row(s) have no "
            "labelled input row",
            hint="Evaluate detections against the same file that was passed to `detect`.",
        )
    table = pd.DataFrame(
        {
            "entity": joined["entity"].astype(str),
            "time": joined["source"].dt.round(to_offset(frequency)),
            "segment": joined["segment"],
            "scored": joined["status"].eq("scored"),
            "score": joined["score"].astype(np.float64),
            "flagged": joined["is_flagged"].astype(bool),
            "is_anomaly": joined["is_anomaly"].astype(np.int8),
        }
    )
    for column in ("anomaly_event_id", "anomaly_scenario"):
        if column in joined:
            table[column] = joined[column]
    if events_path is None and sibling_path(data_path, "events.csv").is_file():
        events_path = sibling_path(data_path, "events.csv")
    if events_path is not None:
        events = read_events(events_path, entity_column=entity)
        logger.info("using events file", extra={"path": str(events_path), "events": len(events)})
    else:
        events = events_from_labels(table, frequency)
        logger.info("reconstructed events from labels", extra={"events": len(events)})
    saved_alerting = meta.get("alerting")
    return evaluate_table(
        table,
        events,
        detector=str(meta["detector"]["name"]),
        split=meta["split"],
        frequency=frequency,
        segments=segments,
        allow_in_sample=allow_in_sample,
        alerting=alerting
        or (AlertingConfig.model_validate(saved_alerting) if saved_alerting else None),
        matching=matching,
    )
