"""Run a benchmark: identical datasets, splits and rules for every detector.

1. **Datasets** - one synthetic dataset per (preset, seed) plus its clean twin (the same data
   without injected anomalies), generated in parallel and written to a temporary directory.
2. **Accuracy** - one task per (dataset, detector), each in a fresh worker process limited to one
   BLAS/OpenMP thread: fit features and the detector on the training segment, score the dataset
   and its clean twin, fit every operating point without touching test rows, then evaluate the
   test segment. The clean twin's test segment shows pure false-alert behaviour.
3. **Timing** - one task for feature building and one per detector, run *sequentially*, each in a
   fresh process on the first dataset: warm-up runs, then the median of repeated runs timed with
   ``perf_counter_ns``. Peak memory is read on a separate final run.

Quality metrics are deterministic for a configuration and environment; timings are not. A task
that fails is recorded with its traceback and does not stop the others.
"""

import gc
import json
import logging
import multiprocessing
import os
import statistics
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from threadpoolctl import threadpool_limits

from network_anomaly.benchmarks.config import BenchmarkConfig, OperatingPoint
from network_anomaly.benchmarks.environment import (
    current_rss_bytes,
    peak_rss_bytes,
    reset_peak_rss,
)
from network_anomaly.benchmarks.report import write_outputs
from network_anomaly.config import parse_frequency
from network_anomaly.detectors import Detector, create_detector
from network_anomaly.errors import DataValidationError
from network_anomaly.evaluation.explanations import explanation_hit_rates
from network_anomaly.evaluation.runner import assess_table, evaluation_table
from network_anomaly.explanations.alerts import AlertInterval, assemble_alerts
from network_anomaly.features.builder import FeatureBuilder, FeatureFrame
from network_anomaly.ingestion.events import read_events
from network_anomaly.ingestion.validation import load_metrics
from network_anomaly.pipelines import (
    FittedScores,
    align_labels,
    calibrate,
    fit_and_score,
    fit_clean_thresholds,
    split_for,
)
from network_anomaly.schemas import Event
from network_anomaly.scoring.thresholds import (
    FixedThresholdConfig,
    HostThresholds,
    QuantileThresholdConfig,
    TargetFprConfig,
)
from network_anomaly.synthetic.generator import generate_dataset, write_dataset
from network_anomaly.synthetic.scenarios import Preset

logger = logging.getLogger(__name__)

Record = dict[str, Any]
Progress = Callable[[str], None]
FloatArray = npt.NDArray[np.float64]

# A false alert on the clean twin counts as "after a baseline shift" within this long of one.
SHIFT_WINDOW = pd.Timedelta(hours=6)
# Short, local data-quality problems. Missing cells and timestamp jitter are spread over a host's
# whole timeline, so "an alert during them" would say nothing.
LOCAL_DATA_QUALITY = frozenset({"telemetry_gap", "metric_dropout", "noise_burst", "glitch"})


def _init_worker() -> None:
    """Runs once in every fresh worker: one BLAS/OpenMP thread, quiet library logging."""
    threadpool_limits(limits=1)
    logging.getLogger("network_anomaly").setLevel(logging.ERROR)


# --------------------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------------------


def prepare_dataset(config_json: str, preset: str, seed: int, directory: str) -> Record:
    """Generate one dataset with its clean twin and describe it (worker function)."""
    config = BenchmarkConfig.model_validate_json(config_json)
    dataset = generate_dataset(config.generator_config(Preset(preset), seed))
    paths = write_dataset(
        dataset, Path(directory) / f"{preset}-seed{seed}.csv", with_clean_twin=True
    )
    files = json.loads(paths["metadata"].read_text(encoding="utf-8"))["files"]
    meta = dataset.metadata
    return {
        "preset": preset,
        "seed": seed,
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": files["data"]["sha256"],
        "clean_sha256": files["clean"]["sha256"],
        "rows": meta["rows"],
        "anomalous_rows": meta["anomalous_rows"],
        "events": len(dataset.events),
        "events_by_scenario": meta["events_by_scenario"],
        "baseline_shifts": {
            host["host_id"]: [shift["at"] for shift in host["baseline_shifts"]]
            for host in meta["hosts"]
        },
    }


@dataclass(frozen=True)
class _Scored:
    features: FeatureFrame
    scores: FloatArray
    segments: npt.NDArray[np.object_]


Windows = dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]


@dataclass(frozen=True)
class _Context:
    """Ground truth and known *normal* disturbances of one dataset."""

    events: list[Event]
    data_quality: Windows  # local data-quality problems per host
    after_shifts: Windows  # SHIFT_WINDOW after each legitimate baseline shift per host

    @classmethod
    def load(cls, dataset: Record) -> "_Context":
        paths = dataset["paths"]
        quality = pd.read_csv(paths["data_quality_events"], dtype=str, keep_default_na=False)
        windows: Windows = {}
        for row in quality.to_dict("records"):
            if row["kind"] in LOCAL_DATA_QUALITY:
                windows.setdefault(str(row["host_id"]), []).append(
                    (pd.Timestamp(str(row["start"])), pd.Timestamp(str(row["end"])))
                )
        shifts = {
            host: [(pd.Timestamp(at), pd.Timestamp(at) + SHIFT_WINDOW) for at in ats]
            for host, ats in dataset["baseline_shifts"].items()
        }
        return cls(read_events(Path(paths["events"])), windows, shifts)

    @staticmethod
    def overlaps(windows: Windows, alert: AlertInterval) -> bool:
        return any(
            alert.fired_at < end and alert.end > start
            for start, end in windows.get(alert.entity, [])
        )

    @staticmethod
    def time_share(windows: Windows, entities: npt.NDArray[np.object_], times: pd.Series) -> float:
        """Share of the given (entity, time) rows that fall inside a window of their entity."""
        if len(times) == 0:
            return 0.0
        covered = np.zeros(len(times), dtype=bool)
        values = times.reset_index(drop=True)
        for host, spans in windows.items():
            rows = entities == host
            for start, end in spans:
                covered |= rows & ((values >= start) & (values < end)).to_numpy(dtype=bool)
        return float(covered.mean())


# --------------------------------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------------------------------


def _operating_point(
    point: OperatingPoint,
    config: BenchmarkConfig,
    fitted: FittedScores,
    labels: npt.NDArray[np.int8],
    clean: _Scored,
) -> HostThresholds:
    """Fit one operating point. None of them looks at the test segment."""
    settings = config.thresholds
    ff, scores, segments, split = fitted.features, fitted.scores, fitted.segments, fitted.split
    if point == "fixed":
        return calibrate(FixedThresholdConfig(value=1.0), scores, ff, segments, split, labels)
    if point == "target_fpr":
        target = TargetFprConfig(target_fpr=settings.target_fpr, per_host=settings.per_host)
        return calibrate(target, scores, ff, segments, split, labels)
    quantile = QuantileThresholdConfig(
        strategy="train_quantile" if point == "train_quantile" else "clean_quantile",
        quantile=settings.quantile,
        per_host=settings.per_host,
    )
    if point == "clean_quantile":
        return fit_clean_thresholds(
            quantile, clean.features, clean.scores, clean.segments, split, source="clean twin"
        )
    return calibrate(quantile, scores, ff, segments, split, labels)


def _thresholds_summary(thresholds: HostThresholds) -> Record:
    return {
        "strategy": thresholds.strategy,
        "fitted_on": thresholds.fitted_on,
        "source": thresholds.source,
        "global": thresholds.global_threshold,
        "per_host": len(thresholds.per_host),
        "hosts_using_global": len(thresholds.hosts_using_global),
        "samples": thresholds.n_samples,
    }


def _evaluate_point(
    config: BenchmarkConfig,
    fitted: FittedScores,
    labels: npt.NDArray[np.int8],
    clean: _Scored,
    thresholds: HostThresholds,
    context: _Context,
    frequency: pd.Timedelta,
    detector: str,
) -> Record:
    split = fitted.split.as_dict()
    common: dict[str, Any] = {
        "detector": detector,
        "split": split,
        "frequency": frequency,
        "alerting": config.alerting,
        "matching": config.matching,
    }
    flagged = thresholds.apply(fitted.scores, fitted.features.entities)
    table = evaluation_table(fitted.features, fitted.scores, flagged, fitted.segments, labels)
    outcome = assess_table(table, context.events, **common)
    report = outcome.report
    explained = assemble_alerts(
        fitted.features,
        fitted.scores,
        flagged & (fitted.segments == "test"),
        fitted.segments,
        detector=fitted.detector,
        thresholds=thresholds,
        feature_medians=fitted.builder.state.feature_medians,
        frequency=frequency,
        config=config.alerting,
    )
    clean_flagged = thresholds.apply(clean.scores, clean.features.entities)
    clean_table = evaluation_table(
        clean.features,
        clean.scores,
        clean_flagged,
        clean.segments,
        np.zeros(len(clean.features), dtype=np.int8),
    )
    clean_outcome = assess_table(clean_table, [], **common)
    start, end = fitted.split.validation_end, fitted.split.end
    clean_alerts = [a for a in clean_outcome.alerts if start <= a.fired_at < end]
    return {
        "thresholds": _thresholds_summary(thresholds),
        "test": {
            "point": report.point.model_dump(mode="json"),
            "events": report.events.model_dump(mode="json"),
            "explanations": explanation_hit_rates(outcome.matches, explained).model_dump(
                mode="json"
            ),
        },
        "clean_test": {
            "alerts": len(clean_alerts),
            "false_alerts_per_host_day": clean_outcome.report.events.false_alerts_per_host_day,
            "false_positive_rate": clean_outcome.report.point.false_positive_rate,
            "alerts_in_data_quality_windows": sum(
                context.overlaps(context.data_quality, alert) for alert in clean_alerts
            ),
            "alerts_after_baseline_shifts": sum(
                context.overlaps(context.after_shifts, alert) for alert in clean_alerts
            ),
        },
    }


def accuracy_task(config_json: str, dataset: Record, detector_index: int) -> Record:
    """Fit, score and evaluate one detector on one dataset at every operating point."""
    config = BenchmarkConfig.model_validate_json(config_json)
    detector = config.detectors[detector_index]
    record: Record = {
        "kind": "accuracy",
        "preset": dataset["preset"],
        "seed": dataset["seed"],
        "detector": detector.type,
        "dataset_sha256": dataset["sha256"],
        "status": "ok",
    }
    started = time.perf_counter()
    try:
        frequency = parse_frequency(config.freq)
        paths = dataset["paths"]
        validated = load_metrics(Path(paths["data"]), frequency=frequency)
        clean_metrics = load_metrics(Path(paths["clean"]), frequency=frequency).metrics
        context = _Context.load(dataset)
        fitted = fit_and_score(validated.metrics, config.detection_config(detector), frequency)
        clean = _Scored(*fitted.score_other(clean_metrics))
        labels = align_labels(fitted.features, validated.labels)
        if labels is None:
            raise DataValidationError(f"{paths['data']} has no is_anomaly labels")
        test = (clean.segments == "test") & clean.features.observed
        entities, times = clean.features.entities[test], clean.features.timestamps[test]
        record["clean_test_time_share"] = {
            "data_quality": context.time_share(context.data_quality, entities, times),
            "after_baseline_shifts": context.time_share(context.after_shifts, entities, times),
        }
        record["operating_points"] = {
            point: _evaluate_point(
                config,
                fitted,
                labels,
                clean,
                _operating_point(point, config, fitted, labels, clean),
                context,
                frequency,
                detector.type,
            )
            for point in config.operating_points_for(detector.type)
        }
    except Exception:  # recorded so that one failing task does not abort the benchmark
        record["status"] = "error"
        record["error"] = traceback.format_exc()
    record["wall_seconds"] = round(time.perf_counter() - started, 3)
    return record


# --------------------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------------------


def _repeat(action: Callable[[], object], warmup: int, repeats: int) -> list[float]:
    """Seconds per run after ``warmup`` untimed runs; garbage is collected before each run."""
    for _ in range(warmup):
        action()
    seconds = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter_ns()
        action()
        seconds.append((time.perf_counter_ns() - start) / 1e9)
    return seconds


def _peak_memory(action: Callable[[], object]) -> Record:
    gc.collect()
    baseline = current_rss_bytes()
    was_reset = reset_peak_rss()
    action()
    peak, method = peak_rss_bytes()
    if method == "VmHWM" and not was_reset:
        method = "VmHWM (not reset: includes earlier work in the process)"
    return {"peak_rss_bytes": peak, "baseline_rss_bytes": baseline, "method": method}


def _phase(seconds: list[float], rows: int) -> Record:
    median = statistics.median(seconds)
    return {
        "seconds": [round(s, 6) for s in seconds],
        "median_seconds": round(median, 6),
        "rows": rows,
        "microseconds_per_row": round(median / rows * 1e6, 3) if rows else None,
        "rows_per_second": round(rows / median, 1) if median > 0 else None,
    }


def timing_task(config_json: str, dataset: Record, detector_index: int | None) -> Record:
    """Time feature building (``detector_index`` None) or one detector's fit and score."""
    config = BenchmarkConfig.model_validate_json(config_json)
    settings = config.timing
    name = "features" if detector_index is None else config.detectors[detector_index].type
    record: Record = {
        "kind": "timing",
        "preset": dataset["preset"],
        "seed": dataset["seed"],
        "detector": name,
        "dataset_sha256": dataset["sha256"],
        "warmup": settings.warmup,
        "repeats": settings.repeats,
        "status": "ok",
    }
    try:
        frequency = parse_frequency(config.freq)
        metrics = load_metrics(Path(dataset["paths"]["data"]), frequency=frequency).metrics
        split = split_for(metrics, config.split, frequency)
        train = metrics.until(split.train_end, frequency)

        def build() -> FeatureFrame:
            return FeatureBuilder(config.features).fit(train, frequency).transform(metrics)

        if detector_index is None:
            seconds = _repeat(build, settings.warmup, settings.repeats)
            record["phases"] = {"features": _phase(seconds, len(metrics.data))}
            record["memory"] = _peak_memory(build)
        else:
            detector_config = config.detectors[detector_index]
            ff = build()
            train_ff = ff.select(split.segment(ff.timestamps) == "train", "train")
            fitted: list[Detector] = []

            def fit() -> None:
                fitted[:] = [create_detector(detector_config).fit(train_ff)]

            fit_seconds = _repeat(fit, settings.warmup, settings.repeats)
            detector = fitted[0]
            score_seconds = _repeat(lambda: detector.score(ff), settings.warmup, settings.repeats)
            record["phases"] = {
                "fit": _phase(fit_seconds, int(detector.rows(train_ff).sum())),
                "score": _phase(score_seconds, int(detector.rows(ff).sum())),
            }
            record["memory"] = _peak_memory(
                lambda: create_detector(detector_config).fit(train_ff).score(ff)
            )
    except Exception:  # recorded so that one failing task does not abort the benchmark
        record["status"] = "error"
        record["error"] = traceback.format_exc()
    return record


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkRun:
    directory: Path
    datasets: list[Record]
    records: list[Record]

    @property
    def failures(self) -> list[Record]:
        return [record for record in self.records if record["status"] != "ok"]


def _pool(workers: int) -> ProcessPoolExecutor:
    # A fresh process per task (no state leaks between detectors), started with "spawn".
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
        initializer=_init_worker,
    )


@contextmanager
def _dataset_directory(run_directory: Path, keep: bool) -> Iterator[Path]:
    if keep:
        path = run_directory / "datasets"
        path.mkdir()
        yield path
    else:
        with tempfile.TemporaryDirectory(prefix="network-anomaly-benchmark-") as temporary:
            yield Path(temporary)


def _run_directory(output: Path, started: datetime, fingerprint: str) -> Path:
    base = output / f"{started:%Y%m%dT%H%M%SZ}-{fingerprint}"
    candidate, suffix = base, 1
    while candidate.exists():
        suffix += 1
        candidate = base.with_name(f"{base.name}-{suffix}")
    candidate.mkdir(parents=True)
    return candidate


def _describe(record: Record, headline: str) -> str:
    label = f"{record['detector']} {record['preset']} seed {record['seed']}"
    if record["status"] != "ok":
        last = record.get("error", "").strip().splitlines()[-1:] or ["unknown error"]
        return f"{label}: FAILED ({last[0]})"
    if record["kind"] == "timing":
        phases = ", ".join(
            f"{name} {phase['median_seconds']:.2f}s" for name, phase in record["phases"].items()
        )
        return f"timing {record['detector']}: {phases}"
    events = record["operating_points"][headline]["test"]["events"]
    recall = events["recall"]
    false = events["false_alerts_per_host_day"]
    return (
        f"{label}: event recall {'n/a' if recall is None else f'{recall:.2f}'}, "
        f"false alerts/host-day {'n/a' if false is None else f'{false:.3f}'} ({headline})"
    )


def _order(record: Record) -> tuple[str, str, int, str]:
    return (record["kind"], record["preset"], int(record["seed"]), record["detector"])


def run_benchmark(
    config: BenchmarkConfig,
    output: Path,
    *,
    workers: int | None = None,
    skip_timing: bool = False,
    keep_datasets: bool = False,
    progress: Progress | None = None,
) -> BenchmarkRun:
    """Run every task of ``config`` and write all outputs into a new run directory."""
    report_progress = progress or (lambda _message: None)
    started = datetime.now(UTC)
    directory = _run_directory(output, started, config.fingerprint())
    config_json = config.model_dump_json()
    pairs = [(preset.value, seed) for preset in config.presets for seed in config.seeds]
    jobs = workers or os.cpu_count() or 1
    headline = config.thresholds.headline
    records: list[Record] = []
    with _dataset_directory(directory, keep_datasets) as data_directory:
        with _pool(min(jobs, len(pairs))) as pool:
            futures = [
                pool.submit(prepare_dataset, config_json, preset, seed, str(data_directory))
                for preset, seed in pairs
            ]
            datasets = [future.result() for future in futures]
        report_progress(f"generated {len(datasets)} dataset(s) with clean twins")

        tasks = [(dataset, index) for dataset in datasets for index in range(len(config.detectors))]
        with _pool(min(jobs, len(tasks))) as pool:
            pending = [pool.submit(accuracy_task, config_json, d, i) for d, i in tasks]
            for done, future in enumerate(as_completed(pending), start=1):
                record = future.result()
                records.append(record)
                report_progress(f"[{done}/{len(tasks)}] {_describe(record, headline)}")

        if config.timing.enabled and not skip_timing:
            for index in (None, *range(len(config.detectors))):
                with _pool(1) as pool:
                    record = pool.submit(timing_task, config_json, datasets[0], index).result()
                records.append(record)
                report_progress(_describe(record, headline))
    records.sort(key=_order)
    write_outputs(
        directory,
        config,
        datasets=datasets,
        records=records,
        started=started,
        finished=datetime.now(UTC),
    )
    return BenchmarkRun(directory, datasets, records)
