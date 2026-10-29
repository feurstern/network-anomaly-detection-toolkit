"""Event-level ground truth files: ``<name>.events.csv`` and ``<name>.dq_events.csv``.

An events file lists anomaly intervals per entity. Only ``host_id`` (or the configured entity
column), ``start`` and ``end`` are required; ``end`` is exclusive. Timestamps must be ISO 8601
with a UTC offset. List fields are ``;``-separated and ``params`` is a JSON object.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from network_anomaly.errors import DataValidationError
from network_anomaly.ingestion.readers import MISSING_TOKENS, read_csv_strings
from network_anomaly.schemas import DEFAULT_ENTITY_COLUMN, DataQualityEvent, Event

EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "host_id",
    "scenario",
    "severity",
    "start",
    "end",
    "detectable_from",
    "affected_metrics",
    "peak_effect_sigma",
    "params",
)
DATA_QUALITY_COLUMNS: tuple[str, ...] = (
    "dq_event_id",
    "host_id",
    "kind",
    "start",
    "end",
    "metrics",
    "detail",
)


def sibling_path(path: Path, suffix: str) -> Path:
    """``data/demo.csv`` + ``"events.csv"`` -> ``data/demo.events.csv``."""
    stem = path.name.removesuffix(".csv")
    return path.with_name(f"{stem}.{suffix}")


def format_timestamp(value: datetime) -> str:
    """ISO 8601 in UTC with a ``Z`` suffix and second precision."""
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_events(events: Sequence[Event], path: Path) -> None:
    rows = [
        {
            "event_id": event.event_id,
            "host_id": event.host_id,
            "scenario": event.scenario,
            "severity": event.severity or "",
            "start": format_timestamp(event.start),
            "end": format_timestamp(event.end),
            "detectable_from": format_timestamp(event.detectable_from)
            if event.detectable_from
            else "",
            "affected_metrics": ";".join(event.affected_metrics),
            "peak_effect_sigma": ""
            if event.peak_effect_sigma is None
            else f"{event.peak_effect_sigma:.2f}",
            "params": json.dumps(event.params, sort_keys=True),
        }
        for event in events
    ]
    _write(pd.DataFrame(rows, columns=list(EVENT_COLUMNS)), path)


def write_data_quality_events(events: Sequence[DataQualityEvent], path: Path) -> None:
    rows = [
        {
            "dq_event_id": event.dq_event_id,
            "host_id": event.host_id,
            "kind": event.kind,
            "start": format_timestamp(event.start),
            "end": format_timestamp(event.end),
            "metrics": ";".join(event.metrics),
            "detail": event.detail,
        }
        for event in events
    ]
    _write(pd.DataFrame(rows, columns=list(DATA_QUALITY_COLUMNS)), path)


def _optional(value: str) -> str | None:
    return None if value in MISSING_TOKENS else value


def _split(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(";") if part) if _optional(value) else ()


def read_events(path: Path, *, entity_column: str = DEFAULT_ENTITY_COLUMN) -> list[Event]:
    """Read an events file, raising :class:`DataValidationError` with line numbers on problems."""
    raw = read_csv_strings(path)
    missing = [c for c in (entity_column, "start", "end") if c not in raw.frame.columns]
    if missing:
        raise DataValidationError(
            f"{path}: events file is missing required column(s): {', '.join(missing)}",
            hint=f"An events file needs at least '{entity_column}', 'start' and 'end'.",
        )
    events: list[Event] = []
    problems: list[str] = []
    records = [
        {str(key): str(value) for key, value in row.items()}
        for row in raw.frame.to_dict(orient="records")
    ]
    for line, record in zip(raw.line_numbers.tolist(), records, strict=True):
        try:
            params = json.loads(record["params"]) if _optional(record.get("params", "")) else {}
            peak = _optional(record.get("peak_effect_sigma", ""))
            events.append(
                Event(
                    event_id=_optional(record.get("event_id", "")) or f"event-{line}",
                    host_id=record[entity_column],
                    scenario=_optional(record.get("scenario", "")) or "unknown",
                    severity=_optional(record.get("severity", "")),  # type: ignore[arg-type]
                    start=_parse_aware(record["start"]),
                    end=_parse_aware(record["end"]),
                    detectable_from=_parse_aware(record["detectable_from"])
                    if _optional(record.get("detectable_from", ""))
                    else None,
                    affected_metrics=_split(record.get("affected_metrics", "")),
                    peak_effect_sigma=float(peak) if peak else None,
                    params=params,
                )
            )
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            reason = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
            problems.append(f"line {line}: {reason}")
    if problems:
        shown = "\n  ".join(problems[:5])
        raise DataValidationError(
            f"{path}: {len(problems)} invalid event row(s):\n  {shown}",
            hint="Timestamps need ISO 8601 with a UTC offset and end must be after start.",
        )
    return events


def _parse_aware(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError(f"timestamp {text!r} has no UTC offset")
    return value.astimezone(UTC)
