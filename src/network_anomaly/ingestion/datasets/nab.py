"""Numenta Anomaly Benchmark (NAB): labelled real server metrics, one metric per series.

Source: https://github.com/numenta/NAB, pinned to the MIT-licensed commit in :mod:`.pins` (older
releases were AGPL-3.0). Used here: the 17 ``realAWSCloudwatch`` series and two ``realKnownCause``
server series, with the label windows of ``labels/combined_windows.json``.

How the adapter maps them, and what it has to assume:

* Each series becomes its own file; ``host_id`` is the series name.
* Timestamps carry no time zone; they are taken as UTC.
* Units are inferred from the AWS CloudWatch metric names. ``NetworkIn``, ``DiskWriteBytes`` and
  ELB ``RequestCount`` are totals per 5-minute period and are divided by 300 to give per-second
  rates. Disk *write* bytes go into ``disk_io_bytes_per_second`` (reads are not included), and
  the EC2 request latency series (an average) into ``response_time_p50_ms``.
* Three series repeat 11 timestamps (a daylight-saving bug in the source); the first row of each
  repeated timestamp is kept and the number dropped is reported.
* NAB windows are deliberately wide (about 10 % of a series) and centred on each labelled point, so
  point-level precision is not meaningful; use event-level metrics. Every row inside a window is
  labelled ``is_anomaly = 1`` and each window is one event.
"""

import json
from pathlib import Path

import pandas as pd

from network_anomaly.errors import ArtifactError
from network_anomaly.ingestion.datasets.common import Prepared, write_table
from network_anomaly.ingestion.datasets.download import RemoteFile
from network_anomaly.ingestion.datasets.pins import NAB_COMMIT, NAB_FILES
from network_anomaly.ingestion.events import write_events
from network_anomaly.schemas import DEFAULT_ENTITY_COLUMN, TIMESTAMP_COLUMN, Event

BASE_URL = f"https://raw.githubusercontent.com/numenta/NAB/{NAB_COMMIT}/"
WINDOWS = "labels/combined_windows.json"
STEP = pd.Timedelta("5min")

# Series name prefix -> (canonical metric, divisor turning the value into the metric's unit).
METRIC_MAP: dict[str, tuple[str, float]] = {
    "ec2_cpu_utilization": ("cpu_usage_percent", 1.0),
    "rds_cpu_utilization": ("cpu_usage_percent", 1.0),
    "cpu_utilization_asg_misconfiguration": ("cpu_usage_percent", 1.0),
    "ec2_disk_write_bytes": ("disk_io_bytes_per_second", 300.0),
    "ec2_network_in": ("network_rx_bytes_per_second", 300.0),
    "iio_us-east-1_i-a2eb1cd9_NetworkIn": ("network_rx_bytes_per_second", 300.0),
    "elb_request_count": ("requests_per_second", 300.0),
    "ec2_request_latency_system_failure": ("response_time_p50_ms", 1.0),
}
GENERIC_METRIC = "value"  # series whose unit is unknown (grok_asg_anomaly)


def remote_files() -> list[RemoteFile]:
    return [
        RemoteFile(url=BASE_URL + path, target=path, size=size, sha256=sha256)
        for path, (size, sha256) in sorted(NAB_FILES.items())
    ]


def _metric(stem: str) -> tuple[str, float]:
    for prefix, mapping in METRIC_MAP.items():
        if stem.startswith(prefix):
            return mapping
    return GENERIC_METRIC, 1.0


def prepare(raw: Path, output: Path) -> Prepared:
    """Convert downloaded NAB series into canonical metrics files plus event files."""
    windows_path = raw / WINDOWS
    if not windows_path.is_file():
        raise ArtifactError(
            f"{windows_path} not found", hint="Run `network-anomaly datasets download nab` first."
        )
    windows: dict[str, list[list[str]]] = json.loads(windows_path.read_text(encoding="utf-8"))
    prepared = Prepared("nab")
    for path in sorted(p for p in NAB_FILES if p.startswith("data/") and p.endswith(".csv")):
        source = raw / path
        stem = source.stem
        series = pd.read_csv(source, dtype={"timestamp": str, "value": float})
        series[TIMESTAMP_COLUMN] = pd.to_datetime(series["timestamp"]).dt.tz_localize("UTC")
        repeated = int(series[TIMESTAMP_COLUMN].duplicated().sum())
        if repeated:
            series = series.drop_duplicates(TIMESTAMP_COLUMN, keep="first")
            prepared.notes.append(f"{stem}: dropped {repeated} repeated timestamp(s), kept first")
        metric, divisor = _metric(stem)
        table = pd.DataFrame(
            {
                TIMESTAMP_COLUMN: series[TIMESTAMP_COLUMN],
                DEFAULT_ENTITY_COLUMN: stem,
                metric: series["value"] / divisor,
                "is_anomaly": 0,
                "anomaly_event_id": "",
            }
        ).reset_index(drop=True)
        events = []
        key = path.removeprefix("data/")
        for index, (start_text, end_text) in enumerate(windows.get(key, []), start=1):
            start = pd.Timestamp(start_text, tz="UTC")
            end = pd.Timestamp(end_text, tz="UTC") + STEP  # windows include their last point
            event_id = f"{stem}-w{index}"
            inside = (table[TIMESTAMP_COLUMN] >= start) & (table[TIMESTAMP_COLUMN] < end)
            table.loc[inside, "is_anomaly"] = 1
            table.loc[inside, "anomaly_event_id"] = event_id
            events.append(
                Event(
                    event_id=event_id,
                    host_id=stem,
                    scenario="nab_window",
                    start=start.to_pydatetime(),
                    end=end.to_pydatetime(),
                    affected_metrics=(metric,),
                )
            )
        target = write_table(table.sort_values(TIMESTAMP_COLUMN), output / f"{stem}.csv")
        write_events(events, output / f"{stem}.events.csv")
        prepared.files.append(target)
    prepared.notes.append(
        "units of the CloudWatch series are inferred from their names (see `docs/datasets.md`)"
    )
    prepared.notes.append("NAB windows are wide: compare detectors on event-level metrics")
    example = output / "ec2_cpu_utilization_5f5533.csv"
    prepared.commands = [
        f"network-anomaly detect --input {example} --detector robust_zscore "
        f"--output reports/nab/detections.csv",
        f"network-anomaly evaluate --input {example} --predictions reports/nab/detections.csv",
    ]
    return prepared
