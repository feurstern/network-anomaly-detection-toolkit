"""Prometheus range-query exports become the metrics table, without silent conversions."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from network_anomaly.errors import DataValidationError
from network_anomaly.ingestion.prometheus import prometheus_to_metrics, read_range_query

T0 = 1767571200  # 2026-01-05T00:00:00Z


def _response(series: list[dict[str, Any]], result_type: str = "matrix") -> dict[str, Any]:
    return {"status": "success", "data": {"resultType": result_type, "result": series}}


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _series(instance: str, values: list[str]) -> dict[str, Any]:
    return {
        "metric": {"instance": instance, "job": "web"},
        "values": [[T0 + 300 * i, value] for i, value in enumerate(values)],
    }


def test_queries_are_joined_per_entity_and_time(tmp_path: Path) -> None:
    rps = _write(
        tmp_path / "rps.json", _response([_series("a", ["10", "NaN"]), _series("b", ["5"])])
    )
    cpu = _write(tmp_path / "cpu.json", _response([_series("a", ["40.5", "41"])]))
    table = prometheus_to_metrics({"requests_per_second": rps, "cpu_usage_percent": cpu})
    assert list(table.columns) == [
        "timestamp",
        "host_id",
        "requests_per_second",
        "cpu_usage_percent",
    ]
    assert len(table) == 3
    assert np.isnan(table.loc[1, "requests_per_second"])  # Prometheus NaN stays missing
    assert np.isnan(table.loc[2, "cpu_usage_percent"])  # host b had no CPU series
    assert str(table["timestamp"].dt.tz) == "UTC"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_response([_series("a", ["1"])], result_type="vector"), "range-query"),
        (_response([{"metric": {"job": "web"}, "values": [[T0, "1"]]}]), "no 'instance' label"),
        (_response([_series("a", ["1"]), _series("a", ["2"])]), "several series"),
        (_response([_series("a", ["1", "+Inf"])]), "infinite"),
        (_response([_series("a", ["1", "lots"])]), "malformed sample"),
        (_response([]), "no series"),
    ],
)
def test_bad_exports_are_rejected(tmp_path: Path, payload: dict[str, Any], message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        read_range_query(_write(tmp_path / "q.json", payload), "requests_per_second")
