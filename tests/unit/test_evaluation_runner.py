"""Evaluation orchestration: label-derived events, in-sample guard, straddling events."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from network_anomaly.errors import ConfigError, LeakageError
from network_anomaly.evaluation.runner import EvaluationReport, evaluate_table, events_from_labels
from network_anomaly.schemas import Event

STEP = pd.Timedelta("5min")
T0 = pd.Timestamp("2026-01-05T00:00:00Z")
SPLIT = {
    "start": T0.isoformat(),
    "train_end": (T0 + 100 * STEP).isoformat(),
    "validation_end": (T0 + 200 * STEP).isoformat(),
    "end": (T0 + 300 * STEP).isoformat(),
}


def _table(anomalous: list[range], flagged: list[range]) -> pd.DataFrame:
    steps = np.arange(300)
    labels = np.zeros(300, dtype=np.int8)
    flags = np.zeros(300, dtype=bool)
    for span in anomalous:
        labels[list(span)] = 1
    for span in flagged:
        flags[list(span)] = True
    return pd.DataFrame(
        {
            "entity": "web-1",
            "time": [T0 + int(s) * STEP for s in steps],
            "segment": np.where(steps < 100, "train", np.where(steps < 200, "validation", "test")),
            "scored": True,
            "score": np.where(flags, 5.0, 1.0),
            "flagged": flags,
            "is_anomaly": labels,
        }
    )


def test_events_are_reconstructed_from_contiguous_label_runs() -> None:
    events = events_from_labels(_table([range(10, 15), range(40, 42)], []), STEP)
    assert [(e.start, e.end) for e in events] == [
        ((T0 + 10 * STEP).to_pydatetime(), (T0 + 15 * STEP).to_pydatetime()),
        ((T0 + 40 * STEP).to_pydatetime(), (T0 + 42 * STEP).to_pydatetime()),
    ]


def test_events_are_reconstructed_from_event_ids_when_present() -> None:
    table = _table([range(10, 15)], [])
    table["anomaly_event_id"] = np.where(table["is_anomaly"] == 1, "web-1-ev001", None)
    table["anomaly_scenario"] = np.where(table["is_anomaly"] == 1, "cpu_saturation", None)
    (event,) = events_from_labels(table, STEP)
    assert (event.event_id, event.scenario) == ("web-1-ev001", "cpu_saturation")


def _evaluate(table: pd.DataFrame, events: list[Event], **kwargs: Any) -> EvaluationReport:
    return evaluate_table(table, events, detector="test", split=SPLIT, frequency=STEP, **kwargs)


def test_only_the_test_segment_is_evaluated_by_default() -> None:
    table = _table([range(50, 55), range(250, 255)], [range(50, 55), range(250, 255)])
    report = _evaluate(table, events_from_labels(table, STEP))
    assert report.segments == ("test",)
    assert report.events.events == 1
    assert report.point.rows == 100
    assert report.events.recall == 1.0


def test_in_sample_evaluation_must_be_requested_explicitly() -> None:
    table = _table([range(50, 55)], [range(50, 55)])
    with pytest.raises(LeakageError, match="in-sample"):
        _evaluate(table, [], segments=("train",))
    report = _evaluate(
        table, events_from_labels(table, STEP), segments=("train",), allow_in_sample=True
    )
    assert report.in_sample
    assert "IN-SAMPLE" in report.summary()
    with pytest.raises(ConfigError, match="unknown segment"):
        _evaluate(table, [], segments=("holdout",))


def test_rows_of_events_straddling_the_window_start_are_excluded() -> None:
    table = _table([range(195, 210)], [])  # event starts in validation, continues into test
    report = _evaluate(table, events_from_labels(table, STEP))
    assert report.events.events == 0
    assert report.events.excluded_events == 1
    assert report.point.positives == 0  # its test-period rows are not counted as misses
    assert report.point.rows == 100 - 10
