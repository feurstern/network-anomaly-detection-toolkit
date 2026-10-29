"""Hand-built scenarios for the event-matching rules documented in evaluation/events.py."""

import pandas as pd
import pytest

from network_anomaly.evaluation.events import (
    EventMatch,
    EventMatchingConfig,
    EventMetrics,
    match_events,
)
from network_anomaly.explanations.alerts import AlertInterval
from network_anomaly.schemas import Event

STEP = pd.Timedelta("5min")
T0 = pd.Timestamp("2026-01-05T00:00:00Z")
WINDOW = (T0, T0 + pd.Timedelta(days=2))
CONFIG = EventMatchingConfig(early_tolerance=1, late_tolerance=3)


def at(steps: int) -> pd.Timestamp:
    return T0 + steps * STEP


def event(name: str, start: int, end: int, *, host: str = "web-1", **extra: object) -> Event:
    return Event.model_validate(
        {
            "event_id": name,
            "host_id": host,
            "scenario": extra.pop("scenario", "traffic_spike"),
            "start": at(start).to_pydatetime(),
            "end": at(end).to_pydatetime(),
            **extra,
        }
    )


def alert(fired: int, end: int, *, host: str = "web-1", start: int | None = None) -> AlertInterval:
    return AlertInterval(
        entity=host,
        start=at(fired - 1 if start is None else start),
        fired_at=at(fired),
        end=at(end),
        peak_at=at(fired),
        peak_score=1.0,
        flagged_points=2,
    )


def run(
    events: list[Event], alerts: list[AlertInterval], host_days: float = 2.0
) -> tuple[EventMetrics, list[EventMatch]]:
    return match_events(
        events,
        alerts,
        window_start=WINDOW[0],
        window_end=WINDOW[1],
        frequency=STEP,
        config=CONFIG,
        normal_host_days=host_days,
    )


def test_an_alert_inside_the_event_detects_it_with_its_delay() -> None:
    metrics, matches = run([event("e1", 100, 110)], [alert(102, 106)])
    assert (metrics.detected, metrics.recall, metrics.precision) == (1, 1.0, 1.0)
    assert metrics.median_delay_minutes == pytest.approx(10.0)
    assert matches[0].delay_minutes == pytest.approx(10.0)


def test_tolerances_extend_the_matching_window() -> None:
    # The window ends 3 intervals after the event (exclusive), i.e. step 113.
    late = run([event("e1", 100, 110)], [alert(112, 114)])[0]
    too_late = run([event("e1", 100, 110)], [alert(113, 115)])[0]
    early = run([event("e1", 100, 110)], [alert(98, 100)])[0]  # active at step 99 = start - 1
    assert late.detected == 1
    assert too_late.detected == 0
    assert too_late.false_alerts == 1
    assert early.detected == 1
    assert early.median_delay_minutes == 0.0  # delays are never negative


def test_one_alert_spanning_two_events_detects_both() -> None:
    metrics, _ = run([event("e1", 100, 104), event("e2", 106, 110)], [alert(101, 109)])
    assert metrics.detected == 2
    assert metrics.merged_detections == 1
    assert metrics.false_alerts == 0


def test_several_alerts_on_one_event_count_once() -> None:
    metrics, _ = run([event("e1", 100, 130)], [alert(101, 105), alert(115, 120)])
    assert metrics.detected == 1
    assert metrics.alerts == 2
    assert metrics.fragmentation == 2.0
    assert metrics.precision == 1.0  # fragments are neither extra true nor false positives


def test_false_alerts_and_their_rates() -> None:
    metrics, _ = run([event("e1", 100, 110)], [alert(300, 302), alert(400, 402)], host_days=4.0)
    assert metrics.false_alerts == 2
    assert metrics.precision == 0.0
    assert metrics.false_alerts_per_day == pytest.approx(1.0)  # 2 alerts in a 2-day window
    assert metrics.false_alerts_per_host_day == pytest.approx(0.5)


def test_alerts_on_another_host_do_not_match() -> None:
    metrics, _ = run([event("e1", 100, 110)], [alert(102, 106, host="web-2")])
    assert metrics.detected == 0
    assert metrics.false_alerts == 1


def test_events_started_before_the_window_are_excluded_but_not_penalised() -> None:
    before = event("old", -10, 20)  # starts before the window, ends inside it
    metrics, _ = run([before, event("e1", 100, 110)], [alert(5, 10), alert(102, 104)])
    assert metrics.events == 1
    assert metrics.excluded_events == 1
    assert metrics.false_alerts == 0  # the alert on the excluded event is neither true nor false


def test_alert_fired_before_the_window_can_still_detect() -> None:
    metrics, _ = run([event("e1", 2, 10)], [alert(-3, 6)])
    assert metrics.detected == 1
    assert metrics.alerts == 0  # it fired outside the window, so it is not counted as an alert here


def test_delay_from_detectable_and_breakdowns() -> None:
    gradual = event(
        "ramp",
        100,
        200,
        scenario="traffic_ramp",
        severity="low",
        detectable_from=at(150).to_pydatetime(),
    )
    missed = event("spike", 300, 310, severity="high")
    metrics, _ = run([gradual, missed], [alert(160, 170)])
    assert metrics.median_delay_minutes == pytest.approx(300.0)
    assert metrics.median_delay_from_detectable_minutes == pytest.approx(50.0)
    assert metrics.by_scenario["traffic_ramp"].recall == 1.0
    assert metrics.by_scenario["traffic_spike"].recall == 0.0
    assert metrics.by_severity["high"].detected == 0


def test_no_events_and_no_alerts() -> None:
    metrics, matches = run([], [])
    assert matches == []
    assert (metrics.recall, metrics.precision, metrics.f1) == (None, None, None)
    assert metrics.false_alerts_per_host_day == 0.0
