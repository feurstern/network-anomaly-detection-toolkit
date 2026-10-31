"""HitRate@P% of alert explanations against the metrics an event actually affected."""

import pandas as pd
import pytest

from network_anomaly.evaluation.events import EventMatch
from network_anomaly.evaluation.explanations import explanation_hit_rates
from network_anomaly.explanations.alerts import AlertInterval
from network_anomaly.schemas import Alert, Event

T0 = pd.Timestamp("2026-01-05T00:00:00Z")
STEP = pd.Timedelta("5min")


def _match(affected: tuple[str, ...], start: int = 0) -> EventMatch:
    event = Event(
        event_id=f"e{start}",
        host_id="web-1",
        scenario="test",
        start=(T0 + start * STEP).to_pydatetime(),
        end=(T0 + (start + 4) * STEP).to_pydatetime(),
        affected_metrics=affected,
    )
    interval = AlertInterval(
        entity="web-1",
        start=T0 + start * STEP,
        fired_at=T0 + (start + 1) * STEP,
        end=T0 + (start + 3) * STEP,
        peak_at=T0 + (start + 1) * STEP,
        peak_score=5.0,
        flagged_points=2,
    )
    return EventMatch(event, (interval,), 5.0, None)


def _alert(start: int, ranking: tuple[str, ...]) -> Alert:
    moment = (T0 + start * STEP).to_pydatetime()
    return Alert.model_validate(
        {
            "alert_id": f"a{start}",
            "entity_column": "host_id",
            "entity": "web-1",
            "start": moment,
            "fired_at": moment,
            "end": moment,
            "peak_at": moment,
            "duration_minutes": 15.0,
            "flagged_points": 2,
            "segment": "test",
            "detector": "test",
            "detector_version": "1",
            "score": 5.0,
            "threshold": 1.0,
            "threshold_scope": "global",
            "threshold_strategy": "fixed",
            "reference_exceedance": None,
            "severity": "low",
            "severity_reason": "",
            "explanation_method": "metric_terms",
            "contributions": (),
            "metric_ranking": ranking,
            "peak_deviations": {},
            "summary": "",
            "suggested_checks": (),
            "data_quality": (),
        }
    )


def test_hit_rates_follow_the_omnianomaly_definition() -> None:
    # Two affected metrics: @100% looks at the top 2 ranked, @150% at the top 3.
    matches = [_match(("cpu", "latency"), start=0), _match(("errors",), start=10)]
    alerts = [
        _alert(0, ("cpu", "traffic", "latency", "errors")),
        _alert(10, ("traffic", "errors")),
    ]
    metrics = explanation_hit_rates(matches, alerts)
    assert metrics.events == 2
    assert metrics.hit_rate_at_100 == pytest.approx((0.5 + 0.0) / 2)
    assert metrics.hit_rate_at_150 == pytest.approx((1.0 + 0.0) / 2)
    assert metrics.top_metric_hit_rate == pytest.approx(0.5)


def test_events_without_ground_truth_or_explanation_are_skipped() -> None:
    metrics = explanation_hit_rates([_match(()), _match(("cpu",), start=10)], [])
    assert metrics.events == 0
    assert metrics.hit_rate_at_100 is None
