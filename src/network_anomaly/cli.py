"""Command-line interface: ``network-anomaly``.

The CLI only parses options, reports progress and maps failures to exit codes (see
:class:`network_anomaly.errors.ExitCode`). All behaviour lives in library modules so it can be
reused from Python and tested without a terminal.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from network_anomaly import __version__
from network_anomaly.config import load_yaml_config, parse_config, parse_frequency
from network_anomaly.errors import ConfigError, ExitCode, ToolkitError
from network_anomaly.evaluation.events import EventMatchingConfig
from network_anomaly.evaluation.runner import evaluate_files
from network_anomaly.explanations.alerts import AlertingConfig
from network_anomaly.ingestion.validation import load_metrics, validate_file
from network_anomaly.logging_setup import configure_logging
from network_anomaly.pipelines import (
    DetectionConfig,
    DetectionResult,
    ModelBundle,
    apply_bundle,
    run_detection,
)
from network_anomaly.preprocessing import infer_frequency
from network_anomaly.reporting.export import write_detections
from network_anomaly.schemas import (
    DEFAULT_ENTITY_COLUMN,
    describe_flow_schema,
    describe_metric_schema,
)
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
        typer.echo(result.summary())
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                json.dumps(result.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
            )
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
        + (f", per-host for {len(thresholds.per_host)} host(s)" if thresholds.per_host else ""),
    ]
    for segment, counts in result.segment_counts().items():
        lines.append(
            f"  {segment:<10} rows {counts['rows']:>8,}  scored {counts['scored']:>8,}  "
            f"flagged {counts['flagged']:>6,}"
        )
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
    save_model: Annotated[
        Path | None, typer.Option(file_okay=False, help="Save the fitted model to this directory.")
    ] = None,
    model: Annotated[
        Path | None,
        typer.Option(
            exists=True, file_okay=False, help="Score with a saved model instead of fitting."
        ),
    ] = None,
) -> None:
    """Score every row with a detector and write detections (score, threshold, flag, segment).

    The detector and features are fitted on the training period only; thresholds are calibrated on
    the validation period (or as the chosen strategy requires) and never on the test period.
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
            if detector or config or split or threshold_strategy or threshold or save_model:
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
            result = run_detection(validated, detection_config, frequency)
            if save_model is not None:
                result.bundle.save(save_model)
        csv_path, meta_path = write_detections(result, output, input_path)
        typer.echo(_describe(result))
        typer.echo(f"Wrote {csv_path} and {meta_path.name}")
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
        Path | None, typer.Option(file_okay=False, help="Write metrics.json into this directory.")
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
        report = evaluate_files(
            input_path,
            predictions,
            events_path=events,
            segments=tuple(s.value for s in segment) if segment else ("test",),
            allow_in_sample=allow_in_sample,
            alerting=alerting,
            matching=EventMatchingConfig(
                early_tolerance=early_tolerance, late_tolerance=late_tolerance
            ),
        )
        typer.echo(report.summary())
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "metrics.json"
            path.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")
            typer.echo(f"Wrote {path}")
