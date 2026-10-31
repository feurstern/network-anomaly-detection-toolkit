"""Convert Prometheus range-query results into the metrics table.

The toolkit never connects to Prometheus; it reads JSON you exported with the HTTP API, one query
per canonical metric, for example::

    curl -sG http://prometheus:9090/api/v1/query_range \\
      --data-urlencode 'query=sum by (instance) (rate(http_requests_total[5m]))' \\
      -d start=2026-01-05T00:00:00Z -d end=2026-02-09T00:00:00Z -d step=300 > rps.json

    network-anomaly convert prometheus --metric requests_per_second=rps.json \\
      --metric cpu_usage_percent=cpu.json --output data/processed/metrics.csv

Each series becomes one entity, named by a label (``instance`` by default). Queries must return
at most one series per entity value (aggregate ``by (instance)``) and use the units of the
canonical schema (e.g. percent for CPU, fractions for error rates). Prometheus ``NaN`` samples
become missing values; infinite samples are rejected rather than converted silently.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from network_anomaly.errors import ArtifactError, DataValidationError
from network_anomaly.schemas import DEFAULT_ENTITY_COLUMN, TIMESTAMP_COLUMN


def read_range_query(path: Path, metric: str, entity_label: str = "instance") -> pd.DataFrame:
    """One metric from a ``query_range`` response: columns timestamp, entity, ``metric``."""
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read Prometheus response {path}: {exc}") from exc
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    if payload.get("status") != "success" or data.get("resultType") != "matrix":
        raise DataValidationError(
            f"{path}: not a successful range-query (matrix) response",
            hint="Export with /api/v1/query_range, not /api/v1/query.",
        )
    frames = []
    seen: set[str] = set()
    for series in data.get("result", []):
        labels = series.get("metric", {})
        if entity_label not in labels:
            raise DataValidationError(
                f"{path}: a series has no {entity_label!r} label (labels: {sorted(labels)})",
                hint="Aggregate the query by that label or choose another --entity-label.",
            )
        entity = str(labels[entity_label])
        if entity in seen:
            raise DataValidationError(
                f"{path}: several series share {entity_label}={entity!r}",
                hint=f"Aggregate the query, e.g. sum by ({entity_label}) (...).",
            )
        seen.add(entity)
        values = series.get("values", [])
        try:  # Prometheus sends samples as strings; "NaN" is a legal missing value
            samples = np.array([float(v) for _, v in values], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"{path}: {entity} has a malformed sample: {exc}") from exc
        if np.isinf(samples).any():
            raise DataValidationError(f"{path}: {entity} has infinite samples")
        frames.append(
            pd.DataFrame(
                {
                    TIMESTAMP_COLUMN: pd.to_datetime([t for t, _ in values], unit="s", utc=True),
                    DEFAULT_ENTITY_COLUMN: entity,
                    metric: samples,
                }
            )
        )
    if not frames:
        raise DataValidationError(f"{path}: the query returned no series")
    return pd.concat(frames, ignore_index=True)


def prometheus_to_metrics(
    queries: Mapping[str, Path], entity_label: str = "instance"
) -> pd.DataFrame:
    """Join several exported queries (metric name -> file) into one wide metrics table."""
    table: pd.DataFrame | None = None
    for metric, path in queries.items():
        frame = read_range_query(path, metric, entity_label)
        keys = [TIMESTAMP_COLUMN, DEFAULT_ENTITY_COLUMN]
        table = frame if table is None else table.merge(frame, on=keys, how="outer")
    if table is None:
        raise DataValidationError("give at least one --metric NAME=FILE")
    return table.sort_values([DEFAULT_ENTITY_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)
