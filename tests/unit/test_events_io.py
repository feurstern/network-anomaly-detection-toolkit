from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from network_anomaly.errors import DataValidationError
from network_anomaly.ingestion.events import read_events, sibling_path, write_events
from network_anomaly.schemas import Event


def _event(**overrides: object) -> Event:
    fields: dict[str, object] = {
        "event_id": "web-01-ev001",
        "host_id": "web-01",
        "scenario": "traffic_spike",
        "severity": "high",
        "start": datetime(2026, 1, 5, 10, tzinfo=UTC),
        "end": datetime(2026, 1, 5, 11, tzinfo=UTC),
        "detectable_from": datetime(2026, 1, 5, 10, 5, tzinfo=UTC),
        "affected_metrics": ("requests_per_second", "cpu_usage_percent"),
        "peak_effect_sigma": 12.5,
        "params": {"magnitude": 3.2, "direction": "tx"},
    }
    return Event.model_validate(fields | overrides)


def test_events_round_trip(tmp_path: Path) -> None:
    events = [_event(), _event(event_id="web-02-ev001", host_id="web-02", detectable_from=None)]
    path = tmp_path / "demo.events.csv"
    write_events(events, path)
    assert read_events(path) == events


def test_minimal_user_supplied_events_file(tmp_path: Path) -> None:
    path = tmp_path / "incidents.csv"
    path.write_text(
        "service,start,end\ncheckout,2026-01-05T10:00:00Z,2026-01-05T10:30:00+00:00\n",
        encoding="utf-8",
    )
    (event,) = read_events(path, entity_column="service")
    assert event.host_id == "checkout"
    assert event.scenario == "unknown"
    assert event.event_id == "event-2"
    assert event.end - event.start == timedelta(minutes=30)


def test_invalid_event_rows_are_reported_with_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "host_id,start,end\n"
        "web-01,2026-01-05T10:00:00Z,2026-01-05T09:00:00Z\n"
        "web-01,2026-01-05T10:00:00,2026-01-05T11:00:00Z\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError) as info:
        read_events(path)
    message = str(info.value)
    assert "2 invalid event row(s)" in message
    assert "line 2: Value error, event event-2: end must be after start" in message
    assert "line 3: timestamp '2026-01-05T10:00:00' has no UTC offset" in message


def test_events_file_needs_entity_start_and_end(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("host_id,start\nweb-01,2026-01-05T10:00:00Z\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="missing required column"):
        read_events(path)


def test_sibling_path() -> None:
    assert sibling_path(Path("data/demo.csv"), "events.csv") == Path("data/demo.events.csv")
