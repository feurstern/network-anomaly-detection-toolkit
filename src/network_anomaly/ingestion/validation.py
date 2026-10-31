"""Validate raw tables against the documented metrics and flow schemas.

Checks are vectorised over whole columns. Each problem becomes a :class:`ValidationIssue` with a
stable code, the affected column, example values, file line numbers and a fix hint. Nothing is
coerced silently: every cell is parsed exactly, counted as missing (see ``MISSING_TOKENS``) or
reported. With ``on_invalid="mask"`` invalid metric cells become NaN *and* are recorded in a mask
that travels with the data so later stages can warn about them.
"""

import difflib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.tseries.frequencies import to_offset

from network_anomaly.config import interval_text, snap_interval
from network_anomaly.errors import DataValidationError
from network_anomaly.ingestion.readers import MISSING_TOKENS, RawTable, read_csv_strings
from network_anomaly.schemas import (
    DEFAULT_ENTITY_COLUMN,
    FLOW_FIELD_REGISTRY,
    FLOW_VOLUME_FIELDS,
    LABEL_COLUMNS,
    METRIC_REGISTRY,
    TIMESTAMP_COLUMN,
    EntityStats,
    IssueSeverity,
    SchemaName,
    ValidationIssue,
    ValidationReport,
    metric_spec,
)

logger = logging.getLogger(__name__)

OnInvalid = Literal["error", "mask"]

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]

_MAX_LINES = 5
_MAX_EXAMPLES = 3
_LINE = "__line__"
_MASK_PREFIX = "__mask__"
_OFFSET_SUFFIX = r"(?:[zZ]|[+-]\d{2}(?::?\d{2})?)$"
_IRREGULAR_FRACTION = 0.25
_MIXED_FREQUENCY_TOLERANCE = 0.10
_TIMESTAMP_HINT = (
    "Use ISO 8601 with a UTC offset, e.g. 2026-01-05T14:30:00Z or 2026-01-05T16:30:00+02:00."
)


class InvalidDataError(DataValidationError):
    """An input file failed validation; ``report`` holds every finding."""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__(report.summary())
        self.report = report


@dataclass(frozen=True)
class MetricsFrame:
    """Validated wide metrics: UTC timestamps, entity ids and float64 metrics. Never labels.

    Rows are sorted by (entity, timestamp) with a fresh RangeIndex. ``invalid_mask`` has one boolean
    column per metric (same row order) marking cells that failed validation and were masked to NaN.
    Instances are treated as read-only.
    """

    data: pd.DataFrame
    entity_column: str
    metrics: tuple[str, ...]
    invalid_mask: pd.DataFrame | None = None
    source: str = "<memory>"

    def __post_init__(self) -> None:
        expected = [TIMESTAMP_COLUMN, self.entity_column, *self.metrics]
        if list(self.data.columns) != expected:
            raise ValueError(
                f"MetricsFrame columns must be {expected}, got {list(self.data.columns)}"
            )
        if str(self.data[TIMESTAMP_COLUMN].dtype) != "datetime64[ns, UTC]":
            raise ValueError("MetricsFrame timestamps must have dtype datetime64[ns, UTC]")
        leaked = sorted(set(self.metrics) & set(LABEL_COLUMNS))
        if leaked:
            raise ValueError(f"label columns can never be metrics: {leaked}")
        for metric in self.metrics:
            if self.data[metric].dtype != np.float64:
                raise ValueError(f"metric {metric!r} must be float64")

    @property
    def entities(self) -> tuple[str, ...]:
        return tuple(str(entity) for entity in pd.unique(self.data[self.entity_column]))

    def until(self, end: pd.Timestamp, frequency: pd.Timedelta) -> "MetricsFrame":
        """Rows whose grid interval starts before ``end`` (e.g. the training period)."""
        cells = self.data[TIMESTAMP_COLUMN].dt.round(to_offset(frequency))
        keep = (cells < end).to_numpy(dtype=bool)
        mask = self.invalid_mask
        return MetricsFrame(
            data=self.data.loc[keep].reset_index(drop=True),
            entity_column=self.entity_column,
            metrics=self.metrics,
            invalid_mask=None if mask is None else mask.loc[keep].reset_index(drop=True),
            source=self.source,
        )


@dataclass(frozen=True)
class LabelFrame:
    """Ground-truth columns, row-aligned with the :class:`MetricsFrame` they were split from."""

    data: pd.DataFrame
    entity_column: str

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(
            c for c in self.data.columns if c not in (TIMESTAMP_COLUMN, self.entity_column)
        )


@dataclass(frozen=True)
class ValidatedMetrics:
    metrics: MetricsFrame
    labels: LabelFrame | None
    report: ValidationReport


@dataclass(frozen=True)
class FlowFrame:
    """Validated flow records sorted by timestamp. Sensitive columns are kept as supplied."""

    data: pd.DataFrame
    source: str = "<memory>"


@dataclass(frozen=True)
class ValidatedFlows:
    flows: FlowFrame
    report: ValidationReport


class _IssueLog:
    """Collects issues while validation runs."""

    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def add(
        self,
        code: str,
        severity: IssueSeverity,
        message: str,
        *,
        column: str | None = None,
        lines: npt.NDArray[np.int64] | None = None,
        values: Iterable[object] | None = None,
        n_affected: int | None = None,
        hint: str | None = None,
    ) -> None:
        shown_lines: tuple[int, ...] = ()
        count = n_affected or 0
        if lines is not None:
            count = n_affected if n_affected is not None else len(lines)
            shown_lines = tuple(int(line) for line in lines[:_MAX_LINES])
        examples: tuple[str, ...] = ()
        if values is not None:
            unique: dict[str, None] = {}
            for value in values:
                unique.setdefault(str(value), None)
                if len(unique) == _MAX_EXAMPLES:
                    break
            examples = tuple(unique)
        self.issues.append(
            ValidationIssue(
                code=code,
                severity=severity,
                message=message,
                column=column,
                n_affected=count,
                lines=shown_lines,
                examples=examples,
                hint=hint,
            )
        )


def _is_missing(text: pd.Series) -> BoolArray:
    return text.isin(MISSING_TOKENS).to_numpy(dtype=bool)


def _parse_numbers(text: pd.Series) -> tuple[FloatArray, BoolArray, BoolArray]:
    """Return ``(values, missing, malformed)``; values are NaN wherever missing or malformed."""
    missing = _is_missing(text)
    candidates = text.to_numpy(dtype=object).copy()
    candidates[missing] = None
    values = np.asarray(pd.to_numeric(candidates, errors="coerce"), dtype=np.float64)
    malformed = ~missing & np.isnan(values)
    return values, missing, malformed


def _close_match(name: str, candidates: Iterable[str]) -> str | None:
    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.75)
    return matches[0] if matches else None


def _validate_timestamps(
    text: pd.Series, lines: npt.NDArray[np.int64], log: _IssueLog, *, assume_utc: bool
) -> pd.Series:
    missing = _is_missing(text)
    candidates = text.to_numpy(dtype=object).copy()
    candidates[missing] = None
    parsed = pd.to_datetime(pd.Series(candidates), utc=True, format="ISO8601", errors="coerce")
    bad = ~missing & parsed.isna().to_numpy(dtype=bool)
    if missing.any():
        log.add(
            "missing_timestamp",
            "error",
            "rows without a timestamp",
            column=TIMESTAMP_COLUMN,
            lines=lines[missing],
            hint=_TIMESTAMP_HINT,
        )
    if bad.any():
        log.add(
            "bad_timestamp",
            "error",
            "timestamps that are not valid ISO 8601",
            column=TIMESTAMP_COLUMN,
            lines=lines[bad],
            values=text[bad],
            hint=_TIMESTAMP_HINT,
        )
    has_offset = text.str.contains(_OFFSET_SUFFIX, regex=True).to_numpy(dtype=bool)
    naive = ~missing & ~bad & ~has_offset
    if naive.any():
        if assume_utc:
            log.add(
                "naive_timestamp",
                "warning",
                "timestamps without a UTC offset were interpreted as UTC (--assume-utc)",
                column=TIMESTAMP_COLUMN,
                lines=lines[naive],
            )
        else:
            log.add(
                "naive_timestamp",
                "error",
                "timestamps have no UTC offset, so their time zone is ambiguous",
                column=TIMESTAMP_COLUMN,
                lines=lines[naive],
                values=text[naive],
                hint="Append 'Z' or an offset such as +02:00, or pass --assume-utc if the "
                "values are already UTC.",
            )
    return parsed.dt.as_unit("ns")


def _validate_label_columns(
    frame: pd.DataFrame,
    label_columns: Sequence[str],
    lines: npt.NDArray[np.int64],
    log: _IssueLog,
) -> dict[str, npt.NDArray[np.int8] | pd.Series]:
    parsed: dict[str, npt.NDArray[np.int8] | pd.Series] = {}
    for column in label_columns:
        text = frame[column]
        if column != "is_anomaly":
            parsed[column] = text.mask(pd.Series(_is_missing(text), index=text.index))
            continue
        values, missing, malformed = _parse_numbers(text)
        invalid = malformed | (~missing & ~np.isin(values, (0.0, 1.0)))
        if missing.any():
            log.add(
                "missing_label",
                "error",
                "'is_anomaly' is empty; labelled data needs 0 or 1 on every row",
                column=column,
                lines=lines[missing],
            )
        if invalid.any():
            log.add(
                "invalid_label",
                "error",
                "'is_anomaly' must be 0 or 1",
                column=column,
                lines=lines[invalid],
                values=text[invalid],
            )
        parsed[column] = np.nan_to_num(values, nan=0.0).astype(np.int8)
    return parsed


def _validate_metric_column(
    column: str,
    text: pd.Series,
    lines: npt.NDArray[np.int64],
    log: _IssueLog,
    on_invalid: OnInvalid,
) -> tuple[FloatArray, BoolArray | None] | None:
    """Parse one metric column. Returns ``None`` when the column is not usable as a metric."""
    values, missing, malformed = _parse_numbers(text)
    present = ~missing
    if not present.any():
        log.add(
            "empty_column", "warning", f"'{column}' has no values and is ignored", column=column
        )
        return None
    canonical = column in METRIC_REGISTRY
    if not canonical and malformed[present].all():
        log.add(
            "ignored_column",
            "warning",
            f"'{column}' is not numeric and is ignored",
            column=column,
            values=text[present],
            hint="Only numeric columns are metrics; identifiers and text never become features.",
        )
        return None
    spec = metric_spec(column)
    if not canonical:
        suggestion = _close_match(column, METRIC_REGISTRY)
        log.add(
            "unknown_metric",
            "warning",
            f"'{column}' is not a canonical metric; it is used as a generic metric",
            column=column,
            hint=f"Did you mean '{suggestion}'?"
            if suggestion
            else "Generic metrics have no unit or range checks.",
        )

    severity: IssueSeverity = "warning" if on_invalid == "mask" else "error"
    action = " (masked to missing)" if on_invalid == "mask" else ""
    invalid = np.zeros(len(values), dtype=bool)
    if malformed.any():
        log.add(
            "non_numeric",
            severity,
            f"non-numeric values in '{column}'{action}",
            column=column,
            lines=lines[malformed],
            values=text[malformed],
            hint="Use plain numbers with '.' as the decimal separator and no units or "
            "thousands separators.",
        )
        invalid |= malformed
    non_finite = np.isinf(values)
    if non_finite.any():
        log.add(
            "non_finite",
            severity,
            f"infinite values in '{column}'{action}",
            column=column,
            lines=lines[non_finite],
            values=text[non_finite],
        )
        invalid |= non_finite
    finite = np.isfinite(values)
    out_of_range = np.zeros(len(values), dtype=bool)
    if spec.min_value is not None:
        out_of_range |= finite & (values < spec.min_value)
    if spec.max_value is not None:
        out_of_range |= finite & (values > spec.max_value)
    if out_of_range.any():
        offending = values[out_of_range]
        hint = f"Valid range for {column} is {spec.range_text()} ({spec.unit})."
        if spec.unit == "fraction" and bool(np.all((offending > 1) & (offending <= 100))):
            hint = (
                "Rates are fractions between 0 and 1; these look like percentages: divide by 100."
            )
        log.add(
            "out_of_range",
            severity,
            f"values outside {spec.range_text()} in '{column}'{action}",
            column=column,
            lines=lines[out_of_range],
            values=text[out_of_range],
            hint=hint,
        )
        invalid |= out_of_range
    usable = finite & ~invalid
    if spec.unit == "%" and usable.any() and 0 < float(np.max(values[usable])) <= 1.0:
        log.add(
            "percent_scale_suspect",
            "warning",
            f"every value of '{column}' is <= 1, but percentages are expected in [0, 100]",
            column=column,
            hint="If the values are fractions between 0 and 1, multiply them by 100.",
        )
    if not invalid.any() or on_invalid == "error":
        return values, None
    masked = values.copy()
    masked[invalid] = np.nan
    return masked, invalid


def _infer_frequency(data: pd.DataFrame, entity_column: str, log: _IssueLog) -> pd.Timedelta | None:
    """Median spacing between consecutive observations across entities, snapped to a common
    interval so that clock jitter cannot produce an odd grid such as 301 s."""
    deltas = data.groupby(entity_column, sort=False)[TIMESTAMP_COLUMN].diff().dt.total_seconds()
    per_entity = deltas.groupby(data[entity_column], sort=False).median().dropna()
    per_entity = per_entity[per_entity > 0]
    if per_entity.empty:
        return None
    frequency = snap_interval(float(per_entity.median()))
    seconds = frequency.total_seconds()
    if seconds <= 0:
        return None
    disagreeing = per_entity[(per_entity - seconds).abs() > _MIXED_FREQUENCY_TOLERANCE * seconds]
    if len(disagreeing):
        log.add(
            "mixed_frequencies",
            "warning",
            f"{len(disagreeing)} entities are sampled at a different interval than the typical "
            f"{seconds:g}s",
            values=[str(entity) for entity in disagreeing.index],
            n_affected=len(disagreeing),
            hint="Resample them to a common interval or pass --freq explicitly.",
        )
    return frequency


def _check_grid(
    data: pd.DataFrame, entity_column: str, frequency: pd.Timedelta, log: _IssueLog
) -> pd.Series:
    """Report off-grid timestamps and grid-cell collisions; return the grid-rounded timestamps."""
    rounded = data[TIMESTAMP_COLUMN].dt.round(to_offset(frequency))
    interval = interval_text(frequency)
    offset = (data[TIMESTAMP_COLUMN] - rounded).dt.total_seconds()
    # Jitter is measured around each entity's typical offset: a series sampled regularly at
    # :02, :07, ... is regular, just not aligned with the clock.
    phase = offset.groupby(data[entity_column]).transform("median")
    limit = _IRREGULAR_FRACTION * frequency.total_seconds()
    irregular = ((offset - phase).abs() > limit).to_numpy()
    lines = data[_LINE].to_numpy(dtype=np.int64)
    shifted = (phase.abs() > limit).to_numpy()
    if shifted.any():
        log.add(
            "grid_phase_offset",
            "info",
            f"timestamps are regular but offset from the clock-aligned {interval} grid by up to "
            f"{phase.abs().max():.0f}s; they are placed on the nearest grid point",
            column=TIMESTAMP_COLUMN,
            n_affected=int(shifted.sum()),
        )
    if irregular.any():
        log.add(
            "irregular_timestamps",
            "warning",
            f"timestamps more than {_IRREGULAR_FRACTION:.0%} of the {interval} interval away "
            "from the regular grid",
            column=TIMESTAMP_COLUMN,
            lines=lines[irregular],
            hint="Check exporter clocks or pass the intended interval with --freq.",
        )
    collisions = pd.DataFrame({"entity": data[entity_column], "cell": rounded}).duplicated(
        keep=False
    )
    if collisions.any():
        log.add(
            "grid_collision",
            "error",
            f"several observations of the same entity fall into one {interval} interval",
            column=TIMESTAMP_COLUMN,
            lines=lines[collisions.to_numpy(dtype=bool)],
            hint="Aggregate them to one row per interval, or pass a shorter interval with --freq.",
        )
    return rounded


def _entity_stats(
    data: pd.DataFrame,
    entity_column: str,
    metrics: Sequence[str],
    rounded: pd.Series | None,
    frequency: pd.Timedelta | None,
) -> list[EntityStats]:
    stats: list[EntityStats] = []
    for entity, group in data.groupby(entity_column, sort=True):
        missing_intervals = 0
        if rounded is not None and frequency is not None:
            cells = rounded.loc[group.index]
            expected = int((cells.max() - cells.min()) / frequency) + 1
            missing_intervals = expected - int(cells.nunique())
        stats.append(
            EntityStats(
                entity=str(entity),
                rows=len(group),
                start=group[TIMESTAMP_COLUMN].min().to_pydatetime(),
                end=group[TIMESTAMP_COLUMN].max().to_pydatetime(),
                missing_intervals=missing_intervals,
                missing_cells=int(group[list(metrics)].isna().to_numpy().sum()),
            )
        )
    return stats


def _quality_checks(data: pd.DataFrame, metrics: Sequence[str], log: _IssueLog) -> None:
    lines = data[_LINE].to_numpy(dtype=np.int64)
    missing_counts = {m: int(data[m].isna().sum()) for m in metrics if data[m].isna().any()}
    if missing_counts:
        detail = ", ".join(f"{m}={count}" for m, count in missing_counts.items())
        log.add(
            "missing_cells",
            "info",
            f"missing metric values: {detail}",
            n_affected=sum(missing_counts.values()),
        )
    for metric in metrics:
        values = data[metric].to_numpy()
        finite = values[np.isfinite(values)]
        if len(finite) > 1 and float(finite.min()) == float(finite.max()):
            log.add(
                "constant_metric",
                "warning",
                f"'{metric}' never changes, so it carries no information about anomalies",
                column=metric,
            )
    if {"response_time_p50_ms", "response_time_p95_ms"} <= set(metrics):
        inverted = (data["response_time_p95_ms"] < data["response_time_p50_ms"]).to_numpy()
        if inverted.any():
            log.add(
                "latency_percentiles_inverted",
                "warning",
                "response_time_p95_ms is below response_time_p50_ms",
                column="response_time_p95_ms",
                lines=lines[inverted],
                hint="Check that the two percentile columns are not swapped.",
            )


def validate_metrics_table(
    raw: RawTable,
    *,
    entity_column: str = DEFAULT_ENTITY_COLUMN,
    frequency: pd.Timedelta | None = None,
    assume_utc: bool = False,
    on_invalid: OnInvalid = "error",
) -> tuple[ValidatedMetrics | None, ValidationReport]:
    """Validate a raw table against the metrics schema.

    Returns the validated frames (``None`` if there are errors) and the full report.
    """
    log = _IssueLog()
    frame = raw.frame
    lines = raw.line_numbers

    def build_report(rows: int, **fields: object) -> ValidationReport:
        return ValidationReport.model_validate(
            {
                "source": raw.source,
                "schema_name": "metrics",
                "rows": rows,
                "columns": raw.columns,
                "issues": tuple(log.issues),
                **fields,
            }
        )

    for required in (TIMESTAMP_COLUMN, entity_column):
        if required not in frame.columns:
            suggestion = _close_match(required, raw.columns)
            hint = (
                "Name the column that identifies hosts or services with --entity-column."
                if required == entity_column
                else "The time column must be called 'timestamp'."
            )
            if suggestion:
                hint += f" Did you mean '{suggestion}'?"
            log.add(
                "missing_column", "error", f"required column '{required}' is missing", hint=hint
            )
    if log.has_errors:
        return None, build_report(len(frame))

    timestamps = _validate_timestamps(frame[TIMESTAMP_COLUMN], lines, log, assume_utc=assume_utc)
    entity = frame[entity_column]
    entity_missing = _is_missing(entity)
    if entity_missing.any():
        log.add(
            "missing_entity",
            "error",
            f"rows without a value in '{entity_column}'",
            column=entity_column,
            lines=lines[entity_missing],
        )
    label_columns = [column for column in LABEL_COLUMNS if column in frame.columns]
    labels = _validate_label_columns(frame, label_columns, lines, log)

    reserved = {TIMESTAMP_COLUMN, entity_column, *LABEL_COLUMNS}
    metric_values: dict[str, FloatArray] = {}
    masks: dict[str, BoolArray] = {}
    for column in raw.columns:
        if column in reserved:
            continue
        parsed = _validate_metric_column(column, frame[column], lines, log, on_invalid)
        if parsed is None:
            continue
        metric_values[column], mask = parsed
        if mask is not None:
            masks[column] = mask
    if not metric_values:
        log.add(
            "no_metrics",
            "error",
            "no usable numeric metric columns",
            hint="See `network-anomaly schema metrics` for the supported columns.",
        )
    if log.has_errors:
        return None, build_report(len(frame))

    work = pd.DataFrame({TIMESTAMP_COLUMN: timestamps, entity_column: entity.to_numpy()})
    for name, values in metric_values.items():
        work[name] = values
    for name, label in labels.items():
        work[name] = label
    for name, mask in masks.items():
        work[_MASK_PREFIX + name] = mask
    work[_LINE] = lines
    work = work.sort_values([entity_column, TIMESTAMP_COLUMN, _LINE], kind="stable")
    work = work.reset_index(drop=True)

    value_columns = [TIMESTAMP_COLUMN, entity_column, *metric_values, *labels]
    exact = work.duplicated(subset=value_columns, keep="first").to_numpy()
    if exact.any():
        log.add(
            "duplicate_rows",
            "warning",
            f"{int(exact.sum())} exact duplicate row(s) were removed",
            lines=work.loc[exact, _LINE].to_numpy(dtype=np.int64),
        )
        work = work.loc[~exact].reset_index(drop=True)
    conflicting = work.duplicated(subset=[entity_column, TIMESTAMP_COLUMN], keep=False).to_numpy()
    if conflicting.any():
        log.add(
            "conflicting_duplicates",
            "error",
            "rows share an entity and timestamp but have different values",
            column=TIMESTAMP_COLUMN,
            lines=work.loc[conflicting, _LINE].to_numpy(dtype=np.int64),
            hint="Each (entity, timestamp) pair must appear once; aggregate or remove extra rows.",
        )
        return None, build_report(len(work))

    resolved = frequency or _infer_frequency(work, entity_column, log)
    rounded = None
    if resolved is None:
        log.add(
            "frequency_unknown",
            "info",
            "sampling interval could not be inferred (every entity has a single row)",
        )
    else:
        rounded = _check_grid(work, entity_column, resolved, log)
    metrics = tuple(metric_values)
    stats = _entity_stats(work, entity_column, metrics, rounded, resolved)
    missing_intervals = sum(s.missing_intervals for s in stats)
    if missing_intervals:
        log.add(
            "missing_intervals",
            "info",
            f"{missing_intervals} expected interval(s) have no row (telemetry gaps)",
            n_affected=missing_intervals,
        )
    _quality_checks(work, metrics, log)
    report = build_report(
        len(work),
        metrics=metrics,
        label_columns=tuple(label_columns),
        frequency_seconds=resolved.total_seconds() if resolved is not None else None,
        entities=tuple(stats),
    )
    if log.has_errors:
        return None, report

    invalid_mask = None
    if masks:
        invalid_mask = pd.DataFrame(
            {
                m: work[_MASK_PREFIX + m].to_numpy(dtype=bool)
                if m in masks
                else np.zeros(len(work), bool)
                for m in metrics
            }
        )
    metrics_frame = MetricsFrame(
        data=work[[TIMESTAMP_COLUMN, entity_column, *metrics]].copy(),
        entity_column=entity_column,
        metrics=metrics,
        invalid_mask=invalid_mask,
        source=raw.source,
    )
    label_frame = None
    if label_columns:
        label_frame = LabelFrame(
            data=work[[TIMESTAMP_COLUMN, entity_column, *label_columns]].copy(),
            entity_column=entity_column,
        )
    return ValidatedMetrics(metrics=metrics_frame, labels=label_frame, report=report), report


_FLOW_NUMERIC_KINDS = frozenset({"port", "count", "duration"})


def validate_flows_table(
    raw: RawTable, *, assume_utc: bool = False, on_invalid: OnInvalid = "error"
) -> tuple[ValidatedFlows | None, ValidationReport]:
    """Validate a raw table against the network-flow schema."""
    log = _IssueLog()
    frame = raw.frame
    lines = raw.line_numbers
    sensitive: list[str] = []
    label_columns: list[str] = []

    def build_report(rows: int) -> ValidationReport:
        return ValidationReport(
            source=raw.source,
            schema_name="flows",
            rows=rows,
            columns=raw.columns,
            label_columns=tuple(label_columns),
            sensitive_columns=tuple(sensitive),
            issues=tuple(log.issues),
        )

    if TIMESTAMP_COLUMN not in frame.columns:
        log.add("missing_column", "error", "required column 'timestamp' (flow start) is missing")
        return None, build_report(len(frame))
    if not any(column in frame.columns for column in FLOW_VOLUME_FIELDS):
        log.add(
            "missing_volume_field",
            "error",
            "flow records need at least one volume column",
            hint=f"Provide one of: {', '.join(FLOW_VOLUME_FIELDS)}.",
        )

    parsed: dict[str, object] = {
        TIMESTAMP_COLUMN: _validate_timestamps(
            frame[TIMESTAMP_COLUMN], lines, log, assume_utc=assume_utc
        )
    }
    severity: IssueSeverity = "warning" if on_invalid == "mask" else "error"
    for column in raw.columns:
        if column == TIMESTAMP_COLUMN:
            continue
        spec = FLOW_FIELD_REGISTRY.get(column)
        if spec is None:
            suggestion = _close_match(column, FLOW_FIELD_REGISTRY)
            log.add(
                "unknown_column",
                "warning",
                f"'{column}' is not part of the flow schema and is ignored",
                column=column,
                hint=f"Did you mean '{suggestion}'?" if suggestion else None,
            )
            continue
        text = frame[column]
        if spec.sensitive:
            sensitive.append(column)
        if spec.kind == "label":
            label_columns.append(column)
        if spec.kind not in _FLOW_NUMERIC_KINDS and column != "is_anomaly":
            parsed[column] = text.mask(pd.Series(_is_missing(text), index=text.index))
            continue
        values, _, malformed = _parse_numbers(text)
        finite = np.isfinite(values)
        invalid = malformed | np.isinf(values)
        if spec.min_value is not None:
            invalid |= finite & (values < spec.min_value)
        if spec.max_value is not None:
            invalid |= finite & (values > spec.max_value)
        if spec.kind == "port" or column == "is_anomaly":
            invalid |= finite & (values != np.round(values))
        if invalid.any():
            column_severity: IssueSeverity = "error" if column == "is_anomaly" else severity
            log.add(
                "invalid_value",
                column_severity,
                f"invalid values in '{column}' (expected {spec.kind}"
                + (f", range [{spec.min_value:g}, {spec.max_value:g}]" if spec.max_value else "")
                + ")",
                column=column,
                lines=lines[invalid],
                values=text[invalid],
            )
            if on_invalid == "mask":
                values = values.copy()
                values[invalid] = np.nan
        parsed[column] = values
    if sensitive:
        log.add(
            "sensitive_columns",
            "warning",
            f"columns that can identify people or systems: {', '.join(sensitive)}",
            hint="Pseudonymise or drop them before sharing files; they are never model features.",
        )
    if log.has_errors:
        return None, build_report(len(frame))

    data = pd.DataFrame(parsed)
    data[_LINE] = lines
    data = data.sort_values([TIMESTAMP_COLUMN, _LINE], kind="stable").reset_index(drop=True)
    duplicates = data.drop(columns=_LINE).duplicated(keep="first").to_numpy()
    if duplicates.any():
        log.add(
            "duplicate_flows",
            "info",
            f"{int(duplicates.sum())} flow record(s) are exact copies of an earlier record",
            lines=data.loc[duplicates, _LINE].to_numpy(dtype=np.int64),
        )
    report = build_report(len(data))
    flows = FlowFrame(data=data.drop(columns=_LINE), source=raw.source)
    return ValidatedFlows(flows=flows, report=report), report


def validate_file(
    path: Path,
    schema_name: SchemaName = "metrics",
    *,
    entity_column: str = DEFAULT_ENTITY_COLUMN,
    frequency: pd.Timedelta | None = None,
    assume_utc: bool = False,
    on_invalid: OnInvalid = "error",
) -> ValidationReport:
    """Validate a CSV file and return the report (structural read errors still raise)."""
    raw = read_csv_strings(path)
    if schema_name == "flows":
        return validate_flows_table(raw, assume_utc=assume_utc, on_invalid=on_invalid)[1]
    return validate_metrics_table(
        raw,
        entity_column=entity_column,
        frequency=frequency,
        assume_utc=assume_utc,
        on_invalid=on_invalid,
    )[1]


def load_metrics(
    path: Path,
    *,
    entity_column: str = DEFAULT_ENTITY_COLUMN,
    frequency: pd.Timedelta | None = None,
    assume_utc: bool = False,
    on_invalid: OnInvalid = "error",
) -> ValidatedMetrics:
    """Read and validate a metrics CSV, raising :class:`InvalidDataError` if it has errors."""
    validated, report = validate_metrics_table(
        read_csv_strings(path),
        entity_column=entity_column,
        frequency=frequency,
        assume_utc=assume_utc,
        on_invalid=on_invalid,
    )
    if validated is None:
        raise InvalidDataError(report)
    if report.warnings:
        logger.warning(
            "input has validation warnings",
            extra={"path": str(path), "warnings": len(report.warnings)},
        )
    return validated


def load_flows(
    path: Path, *, assume_utc: bool = False, on_invalid: OnInvalid = "error"
) -> ValidatedFlows:
    """Read and validate a flows CSV, raising :class:`InvalidDataError` if it has errors."""
    validated, report = validate_flows_table(
        read_csv_strings(path), assume_utc=assume_utc, on_invalid=on_invalid
    )
    if validated is None:
        raise InvalidDataError(report)
    return validated
