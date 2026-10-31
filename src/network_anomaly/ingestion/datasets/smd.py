"""Server Machine Dataset (SMD): 28 machines x 38 unnamed metrics, with interpretation labels.

Source: https://github.com/NetManAIOps/OmniAnomaly (MIT), pinned to a commit in :mod:`.pins`. Each
machine has an unlabelled training half, a test half with 0/1 labels per row, and
``interpretation_label`` lines (``start-end:dims``) naming the dimensions engineers attributed
each anomaly to - the only verified ground truth here for scoring explanations.

How the adapter maps it, and what it has to assume:

* SMD has no timestamps. Rows are placed on a 1-minute grid (as described by the dataset authors)
  starting at an arbitrary 2026-01-01T00:00Z, training half first, then the test half. Calendar
  features are meaningless for SMD and are not used by default.
* The 38 dimensions are already scaled to about [0, 1] and are unnamed; they become generic
  metrics ``dim_01`` .. ``dim_38`` (validation warns that they are not canonical).
* The training half is unlabelled. Its rows carry ``is_anomaly = 0`` only because the schema needs
  a value; the suggested split keeps them out of evaluation (train and validation only).
* Events are contiguous runs of labelled test rows; their ``affected_metrics`` are the dimensions
  of every interpretation segment overlapping the run. Interpretation spans are 0-based and
  end-exclusive; a few spans are shorter than their label run (a known inconsistency).
"""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from network_anomaly.errors import ArtifactError, ConfigError
from network_anomaly.ingestion.datasets.common import Prepared, write_table
from network_anomaly.ingestion.datasets.download import RemoteFile
from network_anomaly.ingestion.datasets.pins import SMD_COMMIT, SMD_FILES
from network_anomaly.ingestion.events import format_timestamp, write_events
from network_anomaly.schemas import DEFAULT_ENTITY_COLUMN, TIMESTAMP_COLUMN, Event

BASE_URL = f"https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/{SMD_COMMIT}/"
ROOT = "ServerMachineDataset"
PARTS = ("train", "test", "test_label", "interpretation_label")
START = datetime(2026, 1, 1, tzinfo=UTC)
STEP = pd.Timedelta("1min")
DEFAULT_MACHINES = ("machine-1-1",)
VALIDATION_SHARE = 0.2  # last share of the training half used to calibrate thresholds


def machines() -> tuple[str, ...]:
    """Every machine in the pinned release."""
    return tuple(sorted(Path(path).stem for path in SMD_FILES if path.startswith(f"{ROOT}/train/")))


def resolve_machines(selection: str | None) -> tuple[str, ...]:
    """``None`` -> the default machine; ``all`` -> every machine; else a comma-separated list."""
    if selection is None:
        return DEFAULT_MACHINES
    available = machines()
    if selection == "all":
        return available
    chosen = tuple(name.strip() for name in selection.split(",") if name.strip())
    unknown = [name for name in chosen if name not in available]
    if unknown or not chosen:
        raise ConfigError(
            f"unknown SMD machine(s): {', '.join(unknown) or '(none given)'}",
            hint=f"Choose from {available[0]} .. {available[-1]}, or 'all'.",
        )
    return chosen


def remote_files(selection: str | None = None) -> list[RemoteFile]:
    files = [f"{ROOT}/LICENSE"]
    files += [f"{ROOT}/{part}/{name}.txt" for name in resolve_machines(selection) for part in PARTS]
    return [
        RemoteFile(
            url=BASE_URL + path,
            target=path,
            size=SMD_FILES[path][0],
            sha256=SMD_FILES[path][1],
            git_sha1=SMD_FILES[path][2],
        )
        for path in files
    ]


def _runs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) row ranges of consecutive 1s."""
    padded = np.concatenate([[0], labels.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _interpretations(path: Path) -> list[tuple[int, int, tuple[int, ...]]]:
    spans = []
    for line in path.read_text(encoding="utf-8").split():
        span, _, dims = line.partition(":")
        start, _, end = span.partition("-")
        spans.append((int(start), int(end), tuple(int(d) for d in dims.split(",") if d)))
    return spans


def prepare(raw: Path, output: Path, selection: str | None = None) -> Prepared:
    """Convert downloaded SMD machines into canonical metrics files plus event files."""
    prepared = Prepared("smd")
    for name in resolve_machines(selection):
        paths = {part: raw / ROOT / part / f"{name}.txt" for part in PARTS}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise ArtifactError(
                f"SMD files for {name} are missing: {missing[0]}",
                hint=f"Run `network-anomaly datasets download smd --machines {name}` first.",
            )
        train = np.loadtxt(paths["train"], delimiter=",", dtype=np.float64, ndmin=2)
        test = np.loadtxt(paths["test"], delimiter=",", dtype=np.float64, ndmin=2)
        labels = np.loadtxt(paths["test_label"], dtype=np.int8, ndmin=1)
        if len(labels) != len(test) or train.shape[1] != test.shape[1]:
            raise ArtifactError(f"{name}: train, test and label files do not line up")
        dims = [f"dim_{index:02d}" for index in range(1, train.shape[1] + 1)]
        times = pd.date_range(START, periods=len(train) + len(test), freq=STEP)
        table = pd.DataFrame(np.vstack([train, test]), columns=dims)
        table.insert(0, DEFAULT_ENTITY_COLUMN, name)
        table.insert(0, TIMESTAMP_COLUMN, pd.Series(times))
        table["is_anomaly"] = np.concatenate([np.zeros(len(train), np.int8), labels])
        table["anomaly_event_id"] = ""
        spans = _interpretations(paths["interpretation_label"])
        events = []
        offset = len(train)
        for number, (start, end) in enumerate(_runs(labels), start=1):
            event_id = f"{name}-e{number:03d}"
            affected = sorted({d for a, b, ds in spans if a < end and b > start for d in ds})
            table.loc[offset + start : offset + end - 1, "anomaly_event_id"] = event_id
            events.append(
                Event(
                    event_id=event_id,
                    host_id=name,
                    scenario="smd_incident",
                    start=times[offset + start].to_pydatetime(),
                    end=(times[offset + start] + (end - start) * STEP).to_pydatetime(),
                    affected_metrics=tuple(dims[d - 1] for d in affected if 0 < d <= len(dims)),
                )
            )
        target = write_table(table, output / f"{name}.csv")
        write_events(events, output / f"{name}.events.csv")
        prepared.files.append(target)
        train_until = times[int(len(train) * (1 - VALIDATION_SHARE))]
        test_start = times[offset]
        prepared.commands.append(
            f"network-anomaly detect --input {target} --detector isolation_forest --freq 1min "
            f"--train-until {format_timestamp(train_until.to_pydatetime())} "
            f"--validation-until {format_timestamp(test_start.to_pydatetime())} "
            f"--output reports/smd/{name}.csv"
        )
    prepared.notes += [
        "SMD has no timestamps: rows were placed on a 1-minute grid from 2026-01-01T00:00Z",
        "the training half is unlabelled; use the printed split so it is never evaluated",
        "events list the interpretation-label dimensions, so `evaluate --output-dir` reports "
        "explanation hit rates",
    ]
    return prepared
