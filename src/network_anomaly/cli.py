"""Command-line interface: ``network-anomaly``.

The CLI only parses options, reports progress and maps failures to exit codes (see
:class:`network_anomaly.errors.ExitCode`). All behaviour lives in library modules so it can be
reused from Python and tested without a terminal.
"""

import json
import logging
import math
import os
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

from network_anomaly import __version__
from network_anomaly.benchmarks.config import BenchmarkConfig
from network_anomaly.benchmarks.report import flatten, summarise
from network_anomaly.benchmarks.runner import BenchmarkRun, run_benchmark
from network_anomaly.config import load_yaml_config, parse_config, parse_frequency
from network_anomaly.errors import ConfigError, ExitCode, ToolkitError
from network_anomaly.evaluation.events import EventMatchingConfig
from network_anomaly.evaluation.runner import evaluate_inputs, load_evaluation_inputs
from network_anomaly.explanations.alerts import AlertingConfig
from network_anomaly.flows import FlowDetectionConfig, FlowModel, aggregate_flows, evaluate_flows
from network_anomaly.ingestion.datasets import (
    describe_datasets,
    download_dataset,
    prepare_dataset,
)
from network_anomaly.ingestion.events import sibling_path
from network_anomaly.ingestion.prometheus import prometheus_to_metrics
from network_anomaly.ingestion.validation import load_flows, load_metrics, validate_file
from network_anomaly.logging_setup import configure_logging
from network_anomaly.pipelines import (
    DetectionConfig,
    DetectionResult,
    ModelBundle,
    apply_bundle,
    run_detection,
)
from network_anomaly.preprocessing import infer_frequency
from network_anomaly.reporting.export import (
    format_timestamps,
    read_alerts,
    write_alerts,
    write_detections,
)
from network_anomaly.reporting.markdown import write_evaluation_report
from network_anomaly.schemas import (
    DEFAULT_ENTITY_COLUMN,
    Alert,
    describe_flow_schema,
    describe_metric_schema,
)
from network_anomaly.scoring.severity import SEVERITY_ORDER
from network_anomaly.synthetic.generator import (
    GeneratorConfig,
    dataset_paths,
    generate_dataset,
    write_dataset,
)
from network_anomaly.synthetic.scenarios import Preset, ScenarioName

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="network-anomaly",
    help="Detect, explain and evaluate anomalies in infrastructure metrics and network flows. "
    "Anomalies are signals for investigation, not proof of an attack.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LogFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


class SchemaName(StrEnum):
    METRICS = "metrics"
    FLOWS = "flows"


class OnInvalidChoice(StrEnum):
    ERROR = "error"
    MASK = "mask"


class DetectorName(StrEnum):
    STATIC_THRESHOLD = "static_threshold"
    ROBUST_ZSCORE = "robust_zscore"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    AUTOENCODER = "autoencoder"


class ThresholdStrategy(StrEnum):
    FIXED = "fixed"
    TRAIN_QUANTILE = "train_quantile"
    CLEAN_QUANTILE = "clean_quantile"
    TARGET_FPR = "target_fpr"


@contextmanager
def cli_errors() -> Iterator[None]:
    """Report expected failures concisely and exit with their documented code."""
    try:
        yield
    except ToolkitError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=int(exc.exit_code)) from exc
    except BrokenPipeError:
        # The reader went away (e.g. `| head`): stop quietly, as other command-line tools do.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        raise typer.Exit(code=int(ExitCode.INTERNAL)) from None
    except (typer.Exit, typer.Abort):
        raise
    except Exception as exc:
        logger.debug("unexpected failure", exc_info=True)
        typer.echo(
            f"internal error: {type(exc).__name__}: {exc}\n"
            "Re-run with --log-level DEBUG for a traceback and please report the issue.",
            err=True,
        )
        raise typer.Exit(code=int(ExitCode.INTERNAL)) from exc


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"network-anomaly {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            envvar="NETWORK_ANOMALY_LOG_LEVEL",
            case_sensitive=False,
            help="Minimum level of log messages written to stderr.",
        ),
    ] = LogLevel.INFO,
    log_format: Annotated[
        LogFormat,
        typer.Option(
            "--log-format",
            envvar="NETWORK_ANOMALY_LOG_FORMAT",
            case_sensitive=False,
            help="'text' for terminals, 'json' for one JSON object per line.",
        ),
    ] = LogFormat.TEXT,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Only log warnings and errors.")
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Detect, explain and evaluate anomalies in infrastructure metrics and network flows."""
    configure_logging("WARNING" if quiet else log_level.value, log_format.value)


@app.command()
def schema(
    name: Annotated[
        SchemaName, typer.Argument(help="Which input schema to describe.")
    ] = SchemaName.METRICS,
) -> None:
    """Print a documented input schema: columns, units and valid ranges."""
    with cli_errors():
        text = describe_metric_schema() if name is SchemaName.METRICS else describe_flow_schema()
        typer.echo(text)


@app.command()
def generate(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            dir_okay=False,
            help="Destination CSV; events, data-quality events and metadata are written beside it.",
        ),
    ],
    seed: Annotated[int | None, typer.Option(min=0, help="Random seed [default: 42].")] = None,
    preset: Annotated[
        Preset | None, typer.Option(help="Difficulty preset [default: moderate].")
    ] = None,
    scenario: Annotated[
        list[ScenarioName] | None,
        typer.Option(
            "--scenario",
            "-s",
            help="Anomaly scenario to inject; repeat for several [default: all].",
        ),
    ] = None,
    no_anomalies: Annotated[
        bool, typer.Option("--no-anomalies", help="Generate normal-only data (no scenarios).")
    ] = False,
    hosts: Annotated[int | None, typer.Option(min=1, help="Number of hosts [default: 6].")] = None,
    days: Annotated[
        float | None, typer.Option(min=0.1, help="Length of the dataset in days [default: 35].")
    ] = None,
    freq: Annotated[
        str | None, typer.Option(help="Sampling interval, e.g. 1min or 5min [default: 5min].")
    ] = None,
    data_quality: Annotated[
        bool | None,
        typer.Option(
            "--data-quality/--no-data-quality",
            help="Inject the preset's data-quality problems [default: on].",
        ),
    ] = None,
    n_datasets: Annotated[
        int, typer.Option(min=1, max=100, help="Independent datasets to generate from the seed.")
    ] = 1,
    with_clean_twin: Annotated[
        bool,
        typer.Option(help="Also write <name>.clean.csv: the same data without anomalies."),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="YAML generator settings; command-line options override them.",
        ),
    ] = None,
) -> None:
    """Generate deterministic synthetic metrics with ground-truth anomaly labels."""
    with cli_errors():
        settings: dict[str, Any] = (
            load_yaml_config(config, GeneratorConfig).model_dump() if config else {}
        )
        overrides = {
            "seed": seed,
            "preset": preset,
            "hosts": hosts,
            "days": days,
            "freq": freq,
            "data_quality": data_quality,
            "scenarios": tuple(scenario) if scenario else None,
        }
        settings |= {key: value for key, value in overrides.items() if value is not None}
        if no_anomalies:
            settings["scenarios"] = ()
        generator_config = parse_config(settings, GeneratorConfig, source="generate options")
        for index, path in enumerate(dataset_paths(output, n_datasets)):
            dataset = generate_dataset(generator_config, dataset_index=index)
            written = write_dataset(dataset, path, with_clean_twin=with_clean_twin)
            meta = dataset.metadata
            share = meta["anomalous_rows"] / max(meta["rows"], 1)
            by_scenario = ", ".join(f"{k}={v}" for k, v in meta["events_by_scenario"].items())
            typer.echo(
                f"Wrote {path}\n"
                f"  seed {generator_config.seed} (dataset {index}) | preset "
                f"{generator_config.preset} | {generator_config.hosts} hosts | "
                f"{generator_config.days:g} days at {generator_config.freq} | "
                f"{meta['rows']:,} rows ({share:.1%} anomalous)\n"
                f"  {len(dataset.events)} anomaly events{': ' + by_scenario if by_scenario else ''}"
                f"\n  {len(dataset.data_quality_events)} data-quality events (never labelled as "
                "anomalies)\n"
                f"  also wrote: {', '.join(p.name for k, p in written.items() if k != 'data')}"
            )


@app.command()
def validate(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input", "-i", exists=True, dir_okay=False, readable=True, help="CSV file to check."
        ),
    ],
    schema_name: Annotated[
        SchemaName, typer.Option("--schema", help="Schema to validate against.")
    ] = SchemaName.METRICS,
    entity_column: Annotated[
        str, typer.Option(help="Column identifying the host or service (metrics schema).")
    ] = DEFAULT_ENTITY_COLUMN,
    freq: Annotated[
        str | None,
        typer.Option(help="Expected sampling interval, e.g. 5min; inferred when omitted."),
    ] = None,
    assume_utc: Annotated[
        bool, typer.Option(help="Interpret timestamps without a UTC offset as UTC.")
    ] = False,
    on_invalid: Annotated[
        OnInvalidChoice,
        typer.Option(help="'error' rejects invalid cells; 'mask' blanks them and reports each."),
    ] = OnInvalidChoice.ERROR,
    report: Annotated[
        Path | None, typer.Option(dir_okay=False, help="Write the full report as JSON.")
    ] = None,
) -> None:
    """Check a CSV file against the documented schema. Exits with code 3 if it is invalid."""
    with cli_errors():
        result = validate_file(
            input_path,
            schema_name.value,
            entity_column=entity_column,
            frequency=parse_frequency(freq) if freq else None,
            assume_utc=assume_utc,
            on_invalid=on_invalid.value,
        )
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                json.dumps(result.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
            )
        typer.echo(result.summary())
        if report is not None:
            typer.echo(f"Full report written to {report}")
        if not result.is_valid:
            raise typer.Exit(code=int(ExitCode.DATA_VALIDATION))


def _detection_config(
    config: Path | None,
    detector: DetectorName | None,
    split: dict[str, Any],
    threshold_strategy: ThresholdStrategy | None,
    threshold: dict[str, Any],
) -> DetectionConfig:
    """Merge a YAML config with command-line overrides (options win)."""
    settings: dict[str, Any] = (
        load_yaml_config(config, DetectionConfig).model_dump(exclude_unset=True) if config else {}
    )
    if detector is not None:
        configured = settings.get("detector", {}).get("type")
        if configured is not None and configured != detector.value:
            raise ConfigError(
                f"--detector {detector.value} conflicts with detector '{configured}' in {config}"
            )
        settings.setdefault("detector", {"type": detector.value})
    if "detector" not in settings:
        raise ConfigError("choose a detector with --detector or a --config file")
    if split:
        settings["split"] = {**settings.get("split", {}), **split}
    if threshold_strategy is not None:
        settings["threshold"] = {"strategy": threshold_strategy.value, **threshold}
    elif threshold:
        if settings.get("threshold") is None:
            raise ConfigError("threshold options need --threshold-strategy or a config file")
        settings["threshold"] = {**settings["threshold"], **threshold}
    return parse_config(settings, DetectionConfig, source="detect options")


def _describe(result: DetectionResult) -> str:
    bundle = result.bundle
    thresholds = bundle.thresholds
    lines = [
        f"Detector {bundle.detector.name} v{bundle.detector.version}; "
        f"training ends {bundle.split.train_end.isoformat()}, "
        f"validation ends {bundle.split.validation_end.isoformat()}",
        f"Threshold {thresholds.strategy}: global {thresholds.global_threshold:.4g}"
        + (f", per-host for {len(thresholds.per_host)} host(s)" if thresholds.per_host else "")
        + (" (fitted on the known-clean file)" if thresholds.source == "known_clean" else ""),
    ]
    for segment, counts in result.segment_counts().items():
        lines.append(
            f"  {segment:<10} rows {counts['rows']:>8,}  scored {counts['scored']:>8,}  "
            f"flagged {counts['flagged']:>6,}"
        )
    return "\n".join(lines)


def _describe_alerts(alerts: list[Alert], shown: int = 3) -> str:
    if not alerts:
        return "Alerts: none"
    segments = Counter(a.segment for a in alerts)
    severities = Counter(a.severity for a in alerts)
    lines = [
        f"Alerts: {len(alerts)} ("
        + ", ".join(f"{segments[s]} {s}" for s in ("train", "validation", "test") if segments[s])
        + "); "
        + ", ".join(f"{severities[s]} {s}" for s in SEVERITY_ORDER if severities[s])
    ]
    test = [a for a in alerts if a.segment == "test"]
    top = sorted(test, key=lambda a: (SEVERITY_ORDER.index(a.severity), -a.score))[:shown]
    if top:
        lines.append("Most severe alerts in the test period:")
        lines += [f"  [{a.severity}] {a.summary}" for a in top]
    return "\n".join(lines)


@app.command()
def detect(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", exists=True, dir_okay=False, help="Metrics CSV to score."),
    ],
    detector: Annotated[
        DetectorName | None, typer.Option("--detector", "-d", help="Detector to fit.")
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="YAML detection config; options override it."
        ),
    ] = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", dir_okay=False, help="Detections CSV to write.")
    ] = Path("reports/detections.csv"),
    entity_column: Annotated[
        str, typer.Option(help="Column identifying the host or service.")
    ] = DEFAULT_ENTITY_COLUMN,
    freq: Annotated[
        str | None, typer.Option(help="Sampling interval, e.g. 5min; inferred when omitted.")
    ] = None,
    assume_utc: Annotated[
        bool, typer.Option(help="Interpret timestamps without a UTC offset as UTC.")
    ] = False,
    train_fraction: Annotated[
        float | None, typer.Option(min=0.01, max=0.99, help="Share of time used for training.")
    ] = None,
    validation_fraction: Annotated[
        float | None, typer.Option(min=0.0, max=0.98, help="Share of time used for calibration.")
    ] = None,
    train_until: Annotated[
        str | None, typer.Option(help="End of training (ISO 8601 with offset); overrides fraction.")
    ] = None,
    validation_until: Annotated[
        str | None, typer.Option(help="End of the calibration period (ISO 8601 with offset).")
    ] = None,
    threshold_strategy: Annotated[
        ThresholdStrategy | None, typer.Option(help="How the decision threshold is chosen.")
    ] = None,
    quantile: Annotated[
        float | None, typer.Option(help="Quantile for train_quantile / clean_quantile.")
    ] = None,
    threshold_value: Annotated[
        float | None, typer.Option(help="Threshold for the fixed strategy.")
    ] = None,
    target_fpr: Annotated[
        float | None, typer.Option(help="False-positive rate for target_fpr.")
    ] = None,
    clean_validation: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Known-clean copy of the validation period (e.g. <name>.clean.csv): "
            "clean_quantile thresholds are fitted on its rows in the validation window.",
        ),
    ] = None,
    save_model: Annotated[
        Path | None, typer.Option(file_okay=False, help="Save the fitted model to this directory.")
    ] = None,
    model: Annotated[
        Path | None,
        typer.Option(
            exists=True, file_okay=False, help="Score with a saved model instead of fitting."
        ),
    ] = None,
    alerts: Annotated[
        Path | None,
        typer.Option(
            "--alerts",
            dir_okay=False,
            help="Explained alerts as JSON Lines [default: <output>.alerts.jsonl].",
        ),
    ] = None,
) -> None:
    """Score every row with a detector; write detections and explained alerts.

    The detector and features are fitted on the training period only; thresholds are calibrated on
    the validation period (or as the chosen strategy requires) and never on the test period. Each
    alert lists the metrics that drove it, their observed and typical values, a severity from
    deviation rules, suggested checks and data-quality warnings.
    """
    with cli_errors():
        split = {
            key: value
            for key, value in {
                "train_fraction": train_fraction,
                "validation_fraction": validation_fraction,
                "train_until": train_until,
                "validation_until": validation_until,
            }.items()
            if value is not None
        }
        threshold = {
            key: value
            for key, value in {
                "quantile": quantile,
                "value": threshold_value,
                "target_fpr": target_fpr,
            }.items()
            if value is not None
        }
        if model is not None:
            fitting = (detector, config, split, threshold_strategy, threshold, save_model)
            if any(fitting) or clean_validation:
                raise ConfigError("--model cannot be combined with fitting options")
            bundle = ModelBundle.load(model)
            validated = load_metrics(
                input_path,
                entity_column=entity_column,
                frequency=bundle.features.frequency,
                assume_utc=assume_utc,
            )
            result = apply_bundle(validated, bundle)
        else:
            detection_config = _detection_config(
                config, detector, split, threshold_strategy, threshold
            )
            frequency = parse_frequency(freq) if freq else None
            validated = load_metrics(
                input_path, entity_column=entity_column, frequency=frequency, assume_utc=assume_utc
            )
            frequency = frequency or infer_frequency(validated.metrics)
            clean = (
                load_metrics(
                    clean_validation,
                    entity_column=entity_column,
                    frequency=frequency,
                    assume_utc=assume_utc,
                )
                if clean_validation
                else None
            )
            result = run_detection(validated, detection_config, frequency, clean=clean)
            if save_model is not None:
                result.bundle.save(save_model)
        csv_path, meta_path = write_detections(result, output, input_path)
        explained = result.alerts()
        alerts_path = write_alerts(explained, alerts or sibling_path(output, "alerts.jsonl"))
        typer.echo(_describe(result))
        typer.echo(_describe_alerts(explained))
        typer.echo(f"Wrote {csv_path}, {meta_path.name} and {alerts_path}")
        if save_model is not None:
            typer.echo(f"Saved model to {save_model}")


class SegmentName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@app.command()
def evaluate(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input", "-i", exists=True, dir_okay=False, help="Labelled CSV given to detect."
        ),
    ],
    predictions: Annotated[
        Path,
        typer.Option(
            "--predictions", "-p", exists=True, dir_okay=False, help="CSV written by detect."
        ),
    ],
    events: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Ground-truth events [default: <input>.events.csv if present, else derived "
            "from the labels].",
        ),
    ] = None,
    segment: Annotated[
        list[SegmentName] | None,
        typer.Option("--segment", "-s", help="Segment(s) to evaluate [default: test]."),
    ] = None,
    allow_in_sample: Annotated[
        bool, typer.Option(help="Allow evaluating train/validation rows (in-sample).")
    ] = False,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
            help="Write metrics.json, report.md and figures/*.png into this directory.",
        ),
    ] = None,
    alerts: Annotated[
        Path | None,
        typer.Option(
            "--alerts",
            exists=True,
            dir_okay=False,
            help="Alerts JSONL from detect, for the report [default: "
            "<predictions>.alerts.jsonl if present].",
        ),
    ] = None,
    plot_entity: Annotated[
        str | None,
        typer.Option(help="Entity for the time-series figure [default: most labelled events]."),
    ] = None,
    merge_gap: Annotated[
        int | None, typer.Option(min=0, help="Override: gap intervals allowed inside an alert.")
    ] = None,
    min_points: Annotated[
        int | None, typer.Option(min=1, help="Override: flagged points before an alert fires.")
    ] = None,
    early_tolerance: Annotated[
        int, typer.Option(min=0, help="Intervals before an event that still count as a match.")
    ] = 1,
    late_tolerance: Annotated[
        int, typer.Option(min=0, help="Intervals after an event that still count as a match.")
    ] = 3,
) -> None:
    """Compute point- and event-level metrics for detections against labelled data."""
    with cli_errors():
        alerting = None
        if merge_gap is not None or min_points is not None:
            defaults = AlertingConfig()
            alerting = AlertingConfig(
                merge_gap=defaults.merge_gap if merge_gap is None else merge_gap,
                min_points=defaults.min_points if min_points is None else min_points,
            )
        inputs = load_evaluation_inputs(input_path, predictions, events_path=events)
        outcome = evaluate_inputs(
            inputs,
            segments=tuple(s.value for s in segment) if segment else ("test",),
            allow_in_sample=allow_in_sample,
            alerting=alerting,
            matching=EventMatchingConfig(
                early_tolerance=early_tolerance, late_tolerance=late_tolerance
            ),
        )
        written = None
        if output_dir is not None:
            default_alerts = sibling_path(predictions, "alerts.jsonl")
            alerts_path = alerts or (default_alerts if default_alerts.is_file() else None)
            written = write_evaluation_report(
                output_dir,
                inputs,
                outcome,
                input_path=input_path,
                predictions_path=predictions,
                alerts=read_alerts(alerts_path) if alerts_path else None,
                plot_entity=plot_entity,
            )
        typer.echo(outcome.report.summary())
        if written is not None:
            figures = sorted(path.name for path in written.values() if path.suffix == ".png")
            typer.echo(
                f"Wrote {written['report']}, {written['metrics'].name} and "
                f"{len(figures)} figure(s): {', '.join(figures)}"
            )


def _describe_benchmark(run: BenchmarkRun, config: BenchmarkConfig) -> str:
    headline = config.thresholds.headline
    summary = summarise(flatten(run.records))
    summary = summary[summary["operating_point"] == headline]
    columns = {
        "event_recall": "recall",
        "event_precision": "precision",
        "false_alerts_per_host_day": "false/host-day",
        "clean_false_alerts_per_host_day": "clean false/host-day",
    }
    lines = []
    for preset in config.presets:
        lines.append(f"{preset.value} preset, {headline} operating point (mean ± std over seeds):")
        lines.append(f"  {'detector':<18}" + "".join(f"{label:>24}" for label in columns.values()))
        for detector in config.detectors:
            cells = []
            for metric in columns:
                row = summary[
                    (summary["preset"] == preset.value)
                    & (summary["detector"] == detector.type)
                    & (summary["metric"] == metric)
                ]
                if row.empty:
                    cells.append("n/a")
                    continue
                mean, std = float(row["mean"].iloc[0]), float(row["std"].iloc[0])
                cells.append(f"{mean:.3f}" + (f" ± {std:.3f}" if math.isfinite(std) else ""))
            lines.append(f"  {detector.type:<18}" + "".join(f"{cell:>24}" for cell in cells))
    return "\n".join(lines)


@app.command()
def benchmark(
    config: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="YAML benchmark configuration [default: built-in defaults, as in "
            "configs/benchmark.yaml].",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Directory for run folders."),
    ] = Path("reports/benchmark"),
    workers: Annotated[
        int | None, typer.Option(min=1, help="Parallel worker processes [default: CPU count].")
    ] = None,
    skip_timing: Annotated[
        bool, typer.Option(help="Skip runtime and memory measurements (quality metrics only).")
    ] = False,
    keep_datasets: Annotated[
        bool, typer.Option(help="Keep the generated datasets inside the run folder.")
    ] = False,
) -> None:
    """Compare detectors on identical synthetic datasets; write a reproducible report.

    Every detector is fitted on the same training segments and evaluated on the same test
    segments at several operating points (threshold strategies). Results are averaged over seeds;
    synthetic results do not demonstrate production effectiveness.
    """
    with cli_errors():
        settings = load_yaml_config(config, BenchmarkConfig) if config else BenchmarkConfig()
        datasets = len(settings.presets) * len(settings.seeds)
        typer.echo(
            f"Benchmark '{settings.name}': {datasets} dataset(s) x {len(settings.detectors)} "
            f"detector(s) = {datasets * len(settings.detectors)} accuracy task(s)"
            + ("" if skip_timing or not settings.timing.enabled else ", then timing runs")
        )
        run = run_benchmark(
            settings,
            output,
            workers=workers,
            skip_timing=skip_timing,
            keep_datasets=keep_datasets,
            progress=typer.echo,
        )
        typer.echo(_describe_benchmark(run, settings))
        typer.echo(f"Wrote {run.directory}/ (report.md, summary.csv, tasks.jsonl, manifest.json)")
        if run.failures:
            typer.echo(f"error: {len(run.failures)} task(s) failed; see report.md", err=True)
            raise typer.Exit(code=int(ExitCode.INTERNAL))


# --------------------------------------------------------------------------------------------------
# Public datasets
# --------------------------------------------------------------------------------------------------

datasets_app = typer.Typer(
    help="Public datasets: list them, download pinned files (verified), convert them.",
    no_args_is_help=True,
)
app.add_typer(datasets_app, name="datasets")


class DatasetName(StrEnum):
    NAB = "nab"
    SMD = "smd"
    CICIDS2017_IMPROVED = "cicids2017-improved"
    CIDDS_001 = "cidds-001"


class IpPolicyChoice(StrEnum):
    HASH = "hash"
    DROP = "drop"
    KEEP = "keep"


class CiddsPart(StrEnum):
    EXTERNAL = "external"
    INTERNAL = "internal"


DataDir = Annotated[
    Path, typer.Option(file_okay=False, help="Where downloads are kept (one folder per dataset).")
]
Machines = Annotated[
    str | None,
    typer.Option(help="SMD machines: comma-separated names or 'all' [default: machine-1-1]."),
]


@datasets_app.command("list")
def datasets_list() -> None:
    """Show the supported datasets, their licences, sizes and caveats."""
    with cli_errors():
        typer.echo(describe_datasets())


@datasets_app.command("download")
def datasets_download(
    name: Annotated[DatasetName, typer.Argument(help="Dataset to download.")],
    data_dir: DataDir = Path("data/raw"),
    machines: Machines = None,
    accept_terms: Annotated[
        bool, typer.Option(help="Accept the dataset's terms of use (shown when required).")
    ] = False,
) -> None:
    """Download a dataset's pinned files and verify every checksum (the only network use)."""
    with cli_errors():
        paths = download_dataset(
            name.value,
            data_dir,
            accept_terms=accept_terms,
            machines=machines,
            progress=typer.echo,
        )
        typer.echo(f"{len(paths)} file(s) verified in {data_dir / name.value}")


@datasets_app.command("prepare")
def datasets_prepare(
    name: Annotated[DatasetName, typer.Argument(help="Dataset to convert.")],
    data_dir: DataDir = Path("data/raw"),
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Where converted files are written.")
    ] = Path("data/processed"),
    machines: Machines = None,
    ip_policy: Annotated[
        IpPolicyChoice,
        typer.Option(help="Flow datasets: hash (keyed pseudonyms), drop or keep addresses."),
    ] = IpPolicyChoice.HASH,
    days: Annotated[
        str | None,
        typer.Option(help="CICIDS2017: comma-separated days, e.g. monday,tuesday [default: all]."),
    ] = None,
    part: Annotated[
        CiddsPart, typer.Option(help="CIDDS-001: external server or the 4 GB OpenStack part.")
    ] = CiddsPart.EXTERNAL,
    attempted_as_anomalous: Annotated[
        bool, typer.Option(help="CICIDS2017: count '... - Attempted' flows as anomalies.")
    ] = False,
) -> None:
    """Convert a downloaded dataset into the toolkit's canonical schema."""
    with cli_errors():
        prepared = prepare_dataset(
            name.value,
            data_dir,
            output_dir,
            machines=machines,
            ip_policy=ip_policy.value,
            days=tuple(d.strip() for d in days.split(",")) if days else None,
            part=part.value,
            attempted_as_anomalous=attempted_as_anomalous,
        )
        typer.echo(prepared.summary())


# --------------------------------------------------------------------------------------------------
# Network flows
# --------------------------------------------------------------------------------------------------

flows_app = typer.Typer(
    help="Network flows: aggregate them into the metrics table, or score individual flows.",
    no_args_is_help=True,
)
app.add_typer(flows_app, name="flows")


@flows_app.command("aggregate")
def flows_aggregate(
    input_paths: Annotated[
        list[Path],
        typer.Option(
            "--input",
            "-i",
            exists=True,
            dir_okay=False,
            help="Flow records (CSV); repeat to combine several files, e.g. one per day.",
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", dir_okay=False, help="Metrics CSV to write.")
    ],
    window: Annotated[str, typer.Option(help="Window length, e.g. 1min or 5min.")] = "5min",
    entity: Annotated[
        str | None,
        typer.Option(help="Flow field to aggregate by, e.g. dst_addr [default: all flows]."),
    ] = None,
    top: Annotated[
        int | None,
        typer.Option(min=1, help="Keep the busiest N entities; merge the rest into 'other'."),
    ] = None,
    assume_utc: Annotated[
        bool, typer.Option(help="Interpret timestamps without a UTC offset as UTC.")
    ] = False,
) -> None:
    """Aggregate flow records per time window into metrics every detector understands."""
    with cli_errors():
        parts = [load_flows(path, assume_utc=assume_utc).flows.data for path in input_paths]
        flows = pd.concat(parts, ignore_index=True).sort_values("timestamp", kind="stable")
        table = aggregate_flows(
            flows, window=parse_frequency(window), entity=entity, top_entities=top
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        written = table.assign(timestamp=format_timestamps(table["timestamp"]))
        written.to_csv(output, index=False, lineterminator="\n", na_rep="")
        metrics = [c for c in table.columns if c not in ("timestamp", "host_id", "is_anomaly")]
        labelled = int(table["is_anomaly"].sum()) if "is_anomaly" in table else None
        typer.echo(
            f"Wrote {output}: {len(table):,} windows of {window} for "
            f"{table['host_id'].nunique()} entit(y/ies) from {len(flows):,} flows\n"
            f"  metrics: {', '.join(metrics)}"
            + (f"\n  windows with labelled anomalous flows: {labelled:,}" if labelled else "")
        )


@flows_app.command("detect")
def flows_detect(
    train: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Flows from a period assumed normal."),
    ],
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", exists=True, dir_okay=False, help="Flows to score."),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", dir_okay=False, help="Per-flow scores CSV.")
    ] = Path("reports/flow_scores.csv"),
    target_fpr: Annotated[
        float,
        typer.Option(min=0.0001, max=0.4, help="Share of training flows above the threshold."),
    ] = 0.01,
    assume_utc: Annotated[
        bool, typer.Option(help="Interpret timestamps without a UTC offset as UTC.")
    ] = False,
) -> None:
    """Score individual flows with an Isolation Forest fitted on a normal period."""
    with cli_errors():
        train_flows = load_flows(train, assume_utc=assume_utc).flows.data
        flows = load_flows(input_path, assume_utc=assume_utc).flows.data
        if flows["timestamp"].min() < train_flows["timestamp"].max():
            logger.warning("the scored flows start before the training flows end")
        model = FlowModel.fit(train_flows, FlowDetectionConfig(target_fpr=target_fpr))
        scores = model.score(flows)
        flagged = scores >= model.threshold
        kept = [c for c in ("src_port", "dst_port", "protocol", "label") if c in flows]
        table = flows[["timestamp", *kept]].reset_index(drop=True)
        table["timestamp"] = format_timestamps(table["timestamp"])
        table["score"] = np.round(scores, 6)
        table["is_flagged"] = flagged.astype(np.int8)
        if "is_anomaly" in flows:
            table["is_anomaly"] = flows["is_anomaly"].to_numpy(dtype=np.int8)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False, lineterminator="\n", na_rep="")
        summary: dict[str, Any] = {
            "train": {"path": str(train), "flows_used": model.training_flows},
            "threshold": model.threshold,
            "target_fpr_on_training_flows": target_fpr,
            "score_direction": "higher = more unusual (unitless, not a probability)",
        }
        typer.echo(
            f"Scored {len(flows):,} flows against {model.training_flows:,} training flows; "
            f"{int(flagged.sum()):,} flagged (threshold {model.threshold:.4g})"
        )
        if "is_anomaly" in flows:
            result = evaluate_flows(
                flows["is_anomaly"].to_numpy(dtype=np.int8),
                scores,
                flagged,
                flows.get("label"),
            )
            summary["evaluation"] = result.model_dump(mode="json")

            def num(value: float | None) -> str:
                return "n/a" if value is None else f"{value:.3f}"

            typer.echo(
                f"  labelled anomalous {result.anomalous:,}: precision {num(result.precision)}, "
                f"recall {num(result.recall)}, FPR {num(result.false_positive_rate)}, "
                f"PR-AUC {num(result.pr_auc)}, recall at 1% FPR "
                f"{num(result.recall_at_1pct_fpr)}"
            )
            for label, recall in sorted(result.recall_by_label.items()):
                typer.echo(f"    {label}: recall {recall:.3f}")
        meta = sibling_path(output, "meta.json")
        meta.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"Wrote {output} and {meta.name}")


# --------------------------------------------------------------------------------------------------
# Format conversion
# --------------------------------------------------------------------------------------------------

convert_app = typer.Typer(
    help="Convert monitoring exports into the metrics schema (files only; no connections).",
    no_args_is_help=True,
)
app.add_typer(convert_app, name="convert")


@convert_app.command("prometheus")
def convert_prometheus(
    metric: Annotated[
        list[str],
        typer.Option(
            "--metric",
            "-m",
            help="CANONICAL_NAME=FILE: a /api/v1/query_range response for one metric; repeat.",
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", dir_okay=False, help="Metrics CSV to write.")
    ],
    entity_label: Annotated[
        str, typer.Option(help="Series label that names the host or service.")
    ] = "instance",
) -> None:
    """Join exported Prometheus range queries into one metrics CSV."""
    with cli_errors():
        queries: dict[str, Path] = {}
        for item in metric:
            name, sep, file = item.partition("=")
            if not sep or not name or not file:
                raise ConfigError(f"--metric expects NAME=FILE, got {item!r}")
            queries[name.strip()] = Path(file.strip())
        table = prometheus_to_metrics(queries, entity_label)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.assign(timestamp=format_timestamps(table["timestamp"])).to_csv(
            output, index=False, lineterminator="\n", na_rep=""
        )
        typer.echo(
            f"Wrote {output}: {len(table):,} rows, {table['host_id'].nunique()} entit(y/ies), "
            f"metrics {', '.join(queries)}; check it with `network-anomaly validate`"
        )
