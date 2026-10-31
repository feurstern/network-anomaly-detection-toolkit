"""Machine-readable outputs: the detections CSV, its JSON provenance sidecar and the alerts JSONL.

The CSV has one row per *observed* input row (gap rows are internal), keyed by the original
timestamp and entity so it can be joined back to the input and its labels:

``timestamp, <entity>, segment, status, score, threshold, threshold_scope, is_flagged, <metrics>``

* ``segment`` - train / validation / test; train and validation rows are in-sample.
* ``status`` - ``scored``, ``warmup`` (baseline not ready yet) or ``no_data``.
* ``score`` - unitless, higher = more anomalous, not a probability; empty when not scored.

Alerts are written as JSON Lines, one :class:`~network_anomaly.schemas.Alert` per line, ordered by
firing time.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from network_anomaly import __version__
from network_anomaly.errors import ArtifactError
from network_anomaly.ingestion.events import sibling_path
from network_anomaly.pipelines import DetectionResult
from network_anomaly.preprocessing import SOURCE_TIMESTAMP
from network_anomaly.schemas import TIMESTAMP_COLUMN, Alert

SCORE_DECIMALS = 6


def format_timestamps(values: pd.Series) -> pd.Series:
    """ISO 8601 UTC strings; sub-second digits only when the data has them. NaT stays missing."""
    utc = values.dt.tz_convert("UTC")
    present = utc.dropna()
    fractional = bool(((present.dt.microsecond != 0) | (present.dt.nanosecond != 0)).any())
    return utc.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ" if fractional else "%Y-%m-%dT%H:%M:%SZ")


def detection_table(result: DetectionResult) -> pd.DataFrame:
    ff = result.features
    observed = ff.observed
    scores = result.scores
    thresholds = result.bundle.thresholds
    entities = ff.entities
    requires_baseline = result.bundle.detector.requires_baseline
    ready = ff.quality["local_ready"].to_numpy(dtype=bool)
    status = np.where(
        np.isfinite(scores), "scored", np.where(requires_baseline & ~ready, "warmup", "no_data")
    )
    table = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: ff.keys[SOURCE_TIMESTAMP],
            ff.entity_column: entities,
            "segment": result.segments,
            "status": status,
            "score": np.round(scores, SCORE_DECIMALS),
            "threshold": np.round(thresholds.thresholds_for(entities), SCORE_DECIMALS),
            "threshold_scope": thresholds.scope_for(entities),
            "is_flagged": result.flagged.astype(np.int8),
        }
    )
    for metric in ff.metrics:
        table[metric] = ff.context[f"{metric}__value"].to_numpy()
    table = table.loc[observed].reset_index(drop=True)
    table[TIMESTAMP_COLUMN] = format_timestamps(table[TIMESTAMP_COLUMN])
    return table


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detection_metadata(result: DetectionResult, input_path: Path | None) -> dict[str, Any]:
    bundle = result.bundle
    features = bundle.features
    return {
        "toolkit_version": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": None
        if input_path is None
        else {"path": str(input_path), "sha256": file_sha256(input_path)},
        "detector": {
            "name": bundle.detector.name,
            "version": bundle.detector.version,
            "config": bundle.detector.config.model_dump(mode="json"),
            "score_direction": "higher = more anomalous (unitless, not a probability)",
        },
        "features": {
            "config": features.config.model_dump(mode="json"),
            "frequency_seconds": features.frequency_seconds,
            "training_start": features.training_start.isoformat(),
            "training_end": features.training_end.isoformat(),
            "training_entities": list(features.training_entities),
        },
        "entity_column": result.features.entity_column,
        "split": bundle.split.as_dict(),
        "thresholds": bundle.thresholds.model_dump(mode="json"),
        "alerting": bundle.config.alerting.model_dump(mode="json"),
        "segments": result.segment_counts(),
    }


def write_detections(
    result: DetectionResult, path: Path, input_path: Path | None = None
) -> tuple[Path, Path]:
    """Write the detections CSV and ``<name>.meta.json``; returns both paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    detection_table(result).to_csv(path, index=False, lineterminator="\n", na_rep="")
    meta_path = sibling_path(path, "meta.json")
    meta = detection_metadata(result, input_path)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path, meta_path


def write_alerts(alerts: list[Alert], path: Path) -> Path:
    """Write alerts as JSON Lines (an empty file when there are none)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for alert in alerts:
            handle.write(alert.model_dump_json() + "\n")
    return path


def read_alerts(path: Path) -> list[Alert]:
    """Read alerts written by :func:`write_alerts`."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArtifactError(f"cannot read alerts {path}: {exc}") from exc
    alerts = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            alerts.append(Alert.model_validate_json(line))
        except ValidationError as exc:
            raise ArtifactError(f"{path}:{number}: not an alert record: {exc}") from exc
    return alerts
