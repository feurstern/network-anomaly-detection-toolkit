"""Event-level metrics: does an operator get a timely alert for each incident, and how many
alerts are noise?

Matching rules (same entity only):

* **Event window** - ``[start - early_tolerance, end + late_tolerance)`` in sampling intervals. The
  late tolerance accepts detectors that react a little after the labelled end; the early tolerance
  absorbs clock jitter.
* **Detection** - an event is detected if any alert is *active* (``fired_at`` to ``end``) inside
  its window. One alert covering several events detects all of them (``merged_detections``);
  several alerts on one event detect it once (``fragmentation`` = matched alerts per detected
  event). An alert that fired before the window but is still active at its start also counts.
* **False alert** - an alert that fires inside the evaluation window and touches no event window.
  Alerts touching an event that started *before* the window (excluded below) are neither true nor
  false.
* **Event precision** = detected / (detected + false alerts): fragmented alerts cannot inflate it.
* **Detection delay** = first time an alert is active in the window minus the event start, never
  negative. For gradual events it is also reported from ``detectable_from`` (when the injected
  effect first exceeded two noise standard deviations).
* **Excluded events** - events that started before the evaluation window belong to the earlier
  segment; they are not evaluated here.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import Field

from network_anomaly.config import StrictModel
from network_anomaly.evaluation.point import harmonic_mean
from network_anomaly.explanations.alerts import AlertInterval
from network_anomaly.schemas import Event


class EventMatchingConfig(StrictModel):
    early_tolerance: int = Field(default=1, ge=0, description="Intervals before an event start.")
    late_tolerance: int = Field(default=3, ge=0, description="Intervals after an event end.")


class GroupRecall(StrictModel):
    events: int
    detected: int
    recall: float | None


class EventMetrics(StrictModel):
    events: int
    detected: int
    recall: float | None
    alerts: int
    false_alerts: int
    precision: float | None
    f1: float | None
    median_delay_minutes: float | None
    p90_delay_minutes: float | None
    median_delay_from_detectable_minutes: float | None
    false_alerts_per_day: float | None
    false_alerts_per_host_day: float | None
    fragmentation: float | None
    merged_detections: int
    excluded_events: int
    by_scenario: dict[str, GroupRecall]
    by_severity: dict[str, GroupRecall]


@dataclass(frozen=True)
class EventMatch:
    """How one evaluated event was (or was not) detected."""

    event: Event
    alerts: tuple[AlertInterval, ...]
    delay_minutes: float | None
    delay_from_detectable_minutes: float | None


def event_window(
    event: Event, frequency: pd.Timedelta, config: EventMatchingConfig
) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(event.start) - config.early_tolerance * frequency,
        pd.Timestamp(event.end) + config.late_tolerance * frequency,
    )


def _active_in(alert: AlertInterval, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return alert.fired_at < end and alert.end > start


def _minutes(delta: pd.Timedelta) -> float:
    return max(delta.total_seconds(), 0.0) / 60.0


def _group_recall(matches: Sequence[EventMatch], key: str) -> dict[str, GroupRecall]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for match in matches:
        group = str(getattr(match.event, key) or "unknown")
        totals[group][0] += 1
        totals[group][1] += bool(match.alerts)
    return {
        group: GroupRecall(events=total, detected=hits, recall=hits / total)
        for group, (total, hits) in sorted(totals.items())
    }


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def match_events(
    events: Sequence[Event],
    alerts: Sequence[AlertInterval],
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    frequency: pd.Timedelta,
    config: EventMatchingConfig,
    normal_host_days: float,
) -> tuple[EventMetrics, list[EventMatch]]:
    """Match alerts to the events that start inside ``[window_start, window_end)``."""
    evaluated = [e for e in events if window_start <= pd.Timestamp(e.start) < window_end]
    excluded = [
        e
        for e in events
        if pd.Timestamp(e.start) < window_start and pd.Timestamp(e.end) > window_start
    ]
    by_entity: dict[str, list[AlertInterval]] = defaultdict(list)
    for alert in alerts:
        by_entity[alert.entity].append(alert)

    matches: list[EventMatch] = []
    alert_hits: dict[int, int] = defaultdict(int)
    for event in evaluated:
        start, end = event_window(event, frequency, config)
        hits = tuple(a for a in by_entity[event.host_id] if _active_in(a, start, end))
        for alert in hits:
            alert_hits[id(alert)] += 1
        delay = detectable_delay = None
        if hits:
            first_active = min(max(alert.fired_at, start) for alert in hits)
            delay = _minutes(first_active - pd.Timestamp(event.start))
            if event.detectable_from is not None:
                detectable_delay = _minutes(first_active - pd.Timestamp(event.detectable_from))
        matches.append(EventMatch(event, hits, delay, detectable_delay))

    windows = [(e.host_id, *event_window(e, frequency, config)) for e in (*evaluated, *excluded)]
    in_window = [a for a in alerts if window_start <= a.fired_at < window_end]
    false_alerts = [
        a
        for a in in_window
        if not any(a.entity == host and _active_in(a, s, e) for host, s, e in windows)
    ]
    detected = [m for m in matches if m.alerts]
    recall = len(detected) / len(evaluated) if evaluated else None
    flagged_events = len(detected) + len(false_alerts)
    precision = len(detected) / flagged_events if flagged_events else None
    delays = [m.delay_minutes for m in detected if m.delay_minutes is not None]
    detectable = [
        m.delay_from_detectable_minutes
        for m in detected
        if m.delay_from_detectable_minutes is not None
    ]
    calendar_days = (window_end - window_start) / pd.Timedelta(days=1)
    metrics = EventMetrics(
        events=len(evaluated),
        detected=len(detected),
        recall=recall,
        alerts=len(in_window),
        false_alerts=len(false_alerts),
        precision=precision,
        f1=harmonic_mean(precision, recall),
        median_delay_minutes=_quantile(delays, 0.5),
        p90_delay_minutes=_quantile(delays, 0.9),
        median_delay_from_detectable_minutes=_quantile(detectable, 0.5),
        false_alerts_per_day=len(false_alerts) / calendar_days if calendar_days > 0 else None,
        false_alerts_per_host_day=len(false_alerts) / normal_host_days
        if normal_host_days > 0
        else None,
        fragmentation=sum(len(m.alerts) for m in detected) / len(detected) if detected else None,
        merged_detections=sum(1 for count in alert_hits.values() if count > 1),
        excluded_events=len(excluded),
        by_scenario=_group_recall(matches, "scenario"),
        by_severity=_group_recall(matches, "severity"),
    )
    return metrics, matches
