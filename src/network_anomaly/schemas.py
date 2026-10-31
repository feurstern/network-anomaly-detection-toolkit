"""Documented input schemas: canonical infrastructure metrics and network-flow fields.

The metric registry is the single source of truth for each metric's unit, valid range,
transformation space and operational direction. Validation, feature engineering, explanations and
the ``schema`` CLI command all read it, so documentation and behaviour cannot drift apart.

Metrics data is *wide*: one row per ``(timestamp, entity)`` with one column per metric. Only the
timestamp, the entity column(s) and at least one metric are required; any subset of the canonical
metrics may be present and unknown numeric columns are accepted as generic metrics.
"""

from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from network_anomaly.config import StrictModel

TIMESTAMP_COLUMN = "timestamp"
# One column identifies the monitored entity (a host by default; e.g. "service" via configuration).
DEFAULT_ENTITY_COLUMN = "host_id"

# Ground-truth columns written by the synthetic generator (or supplied with labelled data). They are
# split off at load time and never reach feature engineering.
LABEL_COLUMNS: tuple[str, ...] = (
    "is_anomaly",
    "anomaly_event_id",
    "anomaly_scenario",
    "anomaly_severity",
)


class MetricFamily(StrEnum):
    """Groups metrics for explanations and investigation playbooks."""

    TRAFFIC = "traffic"
    ERRORS = "errors"
    LATENCY = "latency"
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    CONNECTIONS = "connections"
    FLOWS = "flows"
    GENERIC = "generic"


Direction = Literal["up", "down", "both"]


class MetricSpec(StrictModel):
    """Definition of one metric column."""

    name: str
    unit: str
    description: str
    family: MetricFamily
    min_value: float | None = None
    max_value: float | None = None
    log_scale: bool = Field(
        description="Model the metric as log1p(value): volumes, latencies and counts span "
        "orders of magnitude and their noise is roughly multiplicative."
    )
    direction: Direction = Field(
        description="Which deviations are operationally meaningful: 'up' for errors, latency and "
        "saturation; 'both' where a drop can also signal trouble (e.g. traffic falling to zero)."
    )
    critical: bool = Field(description="User-facing symptom; escalates alert severity.")
    min_scale: float = Field(
        gt=0,
        description="Lower bound for the robust scale in feature space (log1p units for log-scale "
        "metrics), preventing infinite z-scores on near-constant series.",
    )

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value >= self.max_value
        ):
            raise ValueError(f"{self.name}: min_value must be below max_value")
        if self.log_scale and (self.min_value is None or self.min_value < 0):
            raise ValueError(f"{self.name}: log-scale metrics must have min_value >= 0")
        return self

    def range_text(self) -> str:
        """Human-readable valid range, e.g. ``[0, 100]`` or ``>= 0``."""
        if self.min_value is not None and self.max_value is not None:
            return f"[{self.min_value:g}, {self.max_value:g}]"
        if self.min_value is not None:
            return f">= {self.min_value:g}"
        if self.max_value is not None:
            return f"<= {self.max_value:g}"
        return "any finite value"


def _volume(
    name: str, unit: str, description: str, family: MetricFamily, **kw: object
) -> MetricSpec:
    return MetricSpec.model_validate(
        {
            "name": name,
            "unit": unit,
            "description": description,
            "family": family,
            "min_value": 0.0,
            "log_scale": True,
            **kw,
        }
    )


def _fraction(name: str, description: str, family: MetricFamily, **kw: object) -> MetricSpec:
    return MetricSpec.model_validate(
        {
            "name": name,
            "unit": "fraction",
            "description": description,
            "family": family,
            "min_value": 0.0,
            "max_value": 1.0,
            "log_scale": False,
            "direction": "up",
            **kw,
        }
    )


def _percent(name: str, description: str, family: MetricFamily, **kw: object) -> MetricSpec:
    return MetricSpec.model_validate(
        {
            "name": name,
            "unit": "%",
            "description": description,
            "family": family,
            "min_value": 0.0,
            "max_value": 100.0,
            "log_scale": False,
            "direction": "up",
            **kw,
        }
    )


CANONICAL_METRICS: tuple[MetricSpec, ...] = (
    _volume(
        "requests_per_second",
        "requests/s",
        "Requests served per second, averaged over the sampling interval.",
        MetricFamily.TRAFFIC,
        direction="both",
        critical=False,
        min_scale=0.02,
    ),
    _fraction(
        "http_4xx_rate",
        "Fraction of requests answered with a 4xx status code.",
        MetricFamily.ERRORS,
        critical=False,
        min_scale=0.002,
    ),
    _fraction(
        "http_5xx_rate",
        "Fraction of requests answered with a 5xx status code.",
        MetricFamily.ERRORS,
        critical=True,
        min_scale=0.001,
    ),
    _volume(
        "response_time_p50_ms",
        "ms",
        "Median response time over the sampling interval.",
        MetricFamily.LATENCY,
        direction="up",
        critical=False,
        min_scale=0.02,
    ),
    _volume(
        "response_time_p95_ms",
        "ms",
        "95th-percentile response time over the sampling interval.",
        MetricFamily.LATENCY,
        direction="up",
        critical=True,
        min_scale=0.03,
    ),
    _percent(
        "cpu_usage_percent",
        "CPU utilisation across all cores.",
        MetricFamily.CPU,
        critical=False,
        min_scale=0.5,
    ),
    _percent(
        "memory_usage_percent",
        "Memory in use as a share of total memory.",
        MetricFamily.MEMORY,
        critical=False,
        min_scale=0.25,
    ),
    _volume(
        "disk_io_bytes_per_second",
        "bytes/s",
        "Disk read plus write throughput.",
        MetricFamily.DISK,
        direction="up",
        critical=False,
        min_scale=0.05,
    ),
    _volume(
        "network_rx_bytes_per_second",
        "bytes/s",
        "Bytes received per second on the service interfaces.",
        MetricFamily.NETWORK,
        direction="both",
        critical=False,
        min_scale=0.03,
    ),
    _volume(
        "network_tx_bytes_per_second",
        "bytes/s",
        "Bytes sent per second on the service interfaces.",
        MetricFamily.NETWORK,
        direction="both",
        critical=False,
        min_scale=0.03,
    ),
    _volume(
        "active_connections",
        "connections",
        "Open client connections at sampling time.",
        MetricFamily.CONNECTIONS,
        direction="up",
        critical=False,
        min_scale=0.03,
    ),
    _fraction(
        "connection_error_rate",
        "Fraction of connection attempts that failed (refused, reset or timed out).",
        MetricFamily.CONNECTIONS,
        critical=True,
        min_scale=0.001,
    ),
    # Network-flow aggregates per time window (see network_anomaly.flows).
    _volume(
        "flows_per_second",
        "flows/s",
        "Flow records starting in the window, per second.",
        MetricFamily.FLOWS,
        direction="both",
        critical=False,
        min_scale=0.03,
    ),
    _volume(
        "flow_packets_per_second",
        "packets/s",
        "Packets of the window's flows, per second.",
        MetricFamily.FLOWS,
        direction="both",
        critical=False,
        min_scale=0.03,
    ),
    _volume(
        "flow_bytes_per_second",
        "bytes/s",
        "Bytes of the window's flows, per second.",
        MetricFamily.FLOWS,
        direction="both",
        critical=False,
        min_scale=0.03,
    ),
    _volume(
        "unique_sources",
        "sources",
        "Distinct source identifiers in the window.",
        MetricFamily.FLOWS,
        direction="up",
        critical=False,
        min_scale=0.05,
    ),
    _volume(
        "unique_destination_ports",
        "ports",
        "Distinct destination ports in the window.",
        MetricFamily.FLOWS,
        direction="up",
        critical=False,
        min_scale=0.05,
    ),
    _fraction(
        "syn_only_ratio",
        "Share of TCP flows with SYN but no ACK (connection attempts never completed).",
        MetricFamily.FLOWS,
        critical=False,
        min_scale=0.005,
    ),
    _fraction(
        "rst_ratio",
        "Share of TCP flows with a RST packet (connections refused or aborted).",
        MetricFamily.FLOWS,
        critical=False,
        min_scale=0.005,
    ),
)

METRIC_REGISTRY: Mapping[str, MetricSpec] = MappingProxyType(
    {spec.name: spec for spec in CANONICAL_METRICS}
)
# Server metrics (simulated by the synthetic generator) and per-window flow aggregates.
HOST_METRICS: tuple[str, ...] = tuple(
    spec.name for spec in CANONICAL_METRICS if spec.family is not MetricFamily.FLOWS
)
FLOW_AGGREGATE_METRICS: tuple[str, ...] = tuple(
    spec.name for spec in CANONICAL_METRICS if spec.family is MetricFamily.FLOWS
)


def generic_metric_spec(name: str) -> MetricSpec:
    """Spec for a numeric column that is not a canonical metric (unit unknown, no range checks)."""
    return MetricSpec(
        name=name,
        unit="unknown",
        description="Non-canonical numeric metric supplied by the user.",
        family=MetricFamily.GENERIC,
        log_scale=False,
        direction="both",
        critical=False,
        min_scale=1e-6,
    )


def metric_spec(name: str) -> MetricSpec:
    """Return the canonical spec for ``name``, or a generic spec for unknown metrics."""
    return METRIC_REGISTRY.get(name) or generic_metric_spec(name)


FlowFieldKind = Literal[
    "timestamp", "identifier", "port", "categorical", "count", "duration", "label"
]


class FlowFieldSpec(StrictModel):
    """Definition of one canonical network-flow field."""

    name: str
    kind: FlowFieldKind
    unit: str | None = None
    description: str
    required: bool = False
    sensitive: bool = Field(
        default=False,
        description="May identify people or systems (e.g. IP addresses); never used as a feature.",
    )
    min_value: float | None = None
    max_value: float | None = None


def _count(name: str, unit: str, description: str) -> FlowFieldSpec:
    return FlowFieldSpec(name=name, kind="count", unit=unit, description=description, min_value=0)


FLOW_FIELDS: tuple[FlowFieldSpec, ...] = (
    FlowFieldSpec(
        name="timestamp",
        kind="timestamp",
        description="Flow start time, ISO 8601 with a UTC offset.",
        required=True,
    ),
    FlowFieldSpec(
        name="src_addr",
        kind="identifier",
        description="Source address or pseudonymised source identifier.",
        sensitive=True,
    ),
    FlowFieldSpec(
        name="dst_addr",
        kind="identifier",
        description="Destination address or pseudonymised destination identifier.",
        sensitive=True,
    ),
    FlowFieldSpec(
        name="src_port", kind="port", description="Source port.", min_value=0, max_value=65535
    ),
    FlowFieldSpec(
        name="dst_port",
        kind="port",
        description="Destination port.",
        min_value=0,
        max_value=65535,
    ),
    FlowFieldSpec(
        name="protocol",
        kind="categorical",
        description="Transport protocol name or IANA number (e.g. tcp, udp, 6, 17).",
    ),
    FlowFieldSpec(
        name="duration_ms",
        kind="duration",
        unit="ms",
        description="Time between the first and last packet of the flow.",
        min_value=0,
    ),
    _count("packets", "packets", "Packets in both directions."),
    _count("bytes", "bytes", "Bytes in both directions."),
    _count("packets_fwd", "packets", "Packets from source to destination."),
    _count("packets_bwd", "packets", "Packets from destination to source."),
    _count("bytes_fwd", "bytes", "Bytes from source to destination."),
    _count("bytes_bwd", "bytes", "Bytes from destination to source."),
    _count("syn_count", "packets", "Packets with the TCP SYN flag set."),
    _count("ack_count", "packets", "Packets with the TCP ACK flag set."),
    _count("fin_count", "packets", "Packets with the TCP FIN flag set."),
    _count("rst_count", "packets", "Packets with the TCP RST flag set."),
    _count("psh_count", "packets", "Packets with the TCP PSH flag set."),
    _count("urg_count", "packets", "Packets with the TCP URG flag set."),
    FlowFieldSpec(
        name="conn_state",
        kind="categorical",
        description="Connection state or termination reason reported by the flow exporter.",
    ),
    FlowFieldSpec(
        name="label",
        kind="label",
        description="Original dataset label (e.g. BENIGN or an attack class); evaluation only.",
    ),
    FlowFieldSpec(
        name="is_anomaly",
        kind="label",
        description="Binary ground truth (1 = labelled anomalous); evaluation only.",
        min_value=0,
        max_value=1,
    ),
)

FLOW_FIELD_REGISTRY: Mapping[str, FlowFieldSpec] = MappingProxyType(
    {spec.name: spec for spec in FLOW_FIELDS}
)

# A flow record is only useful with at least one volume measurement.
FLOW_VOLUME_FIELDS: tuple[str, ...] = (
    "packets",
    "bytes",
    "packets_fwd",
    "packets_bwd",
    "bytes_fwd",
    "bytes_bwd",
)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in [headers, *rows]
    ]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(line.rstrip() for line in lines)


def describe_metric_schema() -> str:
    """Plain-text description of the metrics schema for the CLI and docs."""
    rows = [
        (
            spec.name,
            spec.unit,
            spec.range_text(),
            spec.direction,
            "yes" if spec.critical else "no",
            spec.description,
        )
        for spec in CANONICAL_METRICS
    ]
    header = (
        "Metrics schema: one row per (timestamp, host_id); one column per metric.\n"
        f"Required: '{TIMESTAMP_COLUMN}' (ISO 8601 with UTC offset), the entity column "
        f"('{DEFAULT_ENTITY_COLUMN}' by default) and at least one metric.\n"
        "Unknown numeric columns are accepted as generic metrics.\n"
        f"Label columns (evaluation only, never features): {', '.join(LABEL_COLUMNS)}.\n"
    )
    table = _table(("metric", "unit", "valid range", "direction", "critical", "description"), rows)
    return f"{header}\n{table}"


def describe_flow_schema() -> str:
    """Plain-text description of the network-flow schema for the CLI and docs."""
    rows = [
        (
            spec.name,
            spec.kind,
            spec.unit or "-",
            "yes" if spec.required else "no",
            "yes" if spec.sensitive else "no",
            spec.description,
        )
        for spec in FLOW_FIELDS
    ]
    header = (
        "Flow schema: one row per flow record.\n"
        f"Required: '{TIMESTAMP_COLUMN}' and at least one of: {', '.join(FLOW_VOLUME_FIELDS)}.\n"
        "Sensitive fields are reported by validation and are never used as model features.\n"
    )
    table = _table(("field", "kind", "unit", "required", "sensitive", "description"), rows)
    return f"{header}\n{table}"


# --------------------------------------------------------------------------------------------------
# Ground truth and data-quality records
# --------------------------------------------------------------------------------------------------

EventSeverity = Literal["low", "medium", "high"]


class Event(StrictModel):
    """A ground-truth anomaly event on one entity, covering the half-open interval [start, end)."""

    event_id: str
    host_id: str
    scenario: str
    severity: EventSeverity | None = None
    start: AwareDatetime
    end: AwareDatetime
    detectable_from: AwareDatetime | None = Field(
        default=None,
        description="First time any metric deviated by more than 2 noise standard deviations "
        "(synthetic data only); None if the injected effect never became that visible.",
    )
    affected_metrics: tuple[str, ...] = ()
    peak_effect_sigma: float | None = Field(
        default=None, description="Largest injected effect, in noise standard deviations."
    )
    params: dict[str, float | str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_interval(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"event {self.event_id}: end must be after start")
        return self


DataQualityKind = Literal[
    "missing_cells", "telemetry_gap", "metric_dropout", "timestamp_jitter", "noise_burst", "glitch"
]


class DataQualityEvent(StrictModel):
    """A deliberate data-quality problem in synthetic data. Never an anomaly label."""

    dq_event_id: str
    host_id: str
    kind: DataQualityKind
    start: AwareDatetime
    end: AwareDatetime
    metrics: tuple[str, ...] = ()
    detail: str = ""


# --------------------------------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------------------------------

AlertSeverity = Literal["low", "medium", "high", "critical"]
ExplanationMethod = Literal[
    "metric_terms", "reconstruction_error", "occlusion_drop_one", "occlusion_keep_one"
]
DataQualityWarningCode = Literal[
    "telemetry_gap",
    "masked_cells",
    "missing_metrics",
    "seasonal_baseline_unavailable",
    "unseen_entity",
    "in_sample",
]


class MetricContribution(StrictModel):
    """One metric at an alert's peak: observed value, baselines, deviation and contribution.

    Values are in the metric's raw units; z-scores are in robust noise units of the metric's
    modelling space (log1p for volumes). ``contribution`` is detector-specific (see
    :attr:`Alert.explanation_method`): it ranks metrics by how much they drove the score, is not
    additive across metrics and does not say that the metric *caused* anything.
    """

    metric: str
    family: MetricFamily
    unit: str
    observed: float | None
    baseline: float | None = Field(description="Median of the host's recent past (local window).")
    seasonal_baseline: float | None = Field(
        description="Median of the same time slot on previous days of the same day type."
    )
    deviation_percent: float | None = Field(
        description="100 * (observed - baseline) / baseline; None if the baseline is 0 or unknown."
    )
    robust_z: float | None = Field(description="Deviation from the recent baseline (z_local).")
    seasonal_z: float | None = Field(description="Deviation from the same-time baseline.")
    contribution: float


class DataQualityWarning(StrictModel):
    """A data problem that limits how far an alert can be trusted."""

    code: DataQualityWarningCode
    message: str
    metrics: tuple[str, ...] = ()


class Alert(StrictModel):
    """An explainable alert: one entity, one contiguous period of flagged observations.

    ``score`` is the peak score: unitless, higher = more anomalous, **not** a probability.
    ``reference_exceedance`` is the share of scored rows in the reference window (the validation
    period) that scored at least as high - an empirical frequency, not a calibrated probability.
    ``severity`` comes from rules on metric deviations and never from the score.
    """

    alert_id: str
    entity_column: str
    entity: str
    start: AwareDatetime
    fired_at: AwareDatetime
    end: AwareDatetime
    peak_at: AwareDatetime
    duration_minutes: float
    flagged_points: int
    segment: str
    detector: str
    detector_version: str
    score: float
    threshold: float
    threshold_scope: Literal["host", "global"]
    threshold_strategy: str
    reference_exceedance: float | None
    severity: AlertSeverity
    severity_reason: str
    explanation_method: ExplanationMethod
    contributions: tuple[MetricContribution, ...]
    metric_ranking: tuple[str, ...] = Field(
        default=(),
        description="Every metric with a contribution, highest first.",
    )
    peak_deviations: dict[str, float | None] = Field(
        description="Robust z (z_local) of every metric at the peak; None where unavailable."
    )
    summary: str
    suggested_checks: tuple[str, ...]
    data_quality: tuple[DataQualityWarning, ...]


# --------------------------------------------------------------------------------------------------
# Validation results
# --------------------------------------------------------------------------------------------------

IssueSeverity = Literal["error", "warning", "info"]
SchemaName = Literal["metrics", "flows"]

# Which chronological segment a frame belongs to; fitting is only allowed on "train".
SplitRole = Literal["train", "validation", "test", "all"]


class ValidationIssue(StrictModel):
    """One finding from input validation, with enough context to fix the input."""

    code: str
    severity: IssueSeverity
    message: str
    column: str | None = None
    n_affected: int = 0
    lines: tuple[int, ...] = Field(
        default=(), description="1-based line numbers in the input file (first few only)."
    )
    examples: tuple[str, ...] = ()
    hint: str | None = None


class EntityStats(StrictModel):
    """Per-entity summary of a validated metrics table."""

    entity: str
    rows: int
    start: AwareDatetime
    end: AwareDatetime
    missing_intervals: int
    missing_cells: int


class ValidationReport(StrictModel):
    """Outcome of validating one input file against a documented schema."""

    source: str
    schema_name: SchemaName
    rows: int
    columns: tuple[str, ...]
    metrics: tuple[str, ...] = ()
    label_columns: tuple[str, ...] = ()
    sensitive_columns: tuple[str, ...] = ()
    frequency_seconds: float | None = None
    entities: tuple[EntityStats, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def summary(self, max_issues: int = 20) -> str:
        """Concise human-readable summary for terminals and logs."""
        lines = [f"{self.source} ({self.schema_name} schema): {self.rows:,} data rows"]
        if self.metrics:
            lines.append(f"  metrics ({len(self.metrics)}): {', '.join(self.metrics)}")
        if self.label_columns:
            lines.append(f"  labels: {', '.join(self.label_columns)}")
        if self.sensitive_columns:
            lines.append(f"  sensitive columns: {', '.join(self.sensitive_columns)}")
        if self.entities:
            start = min(stats.start for stats in self.entities).isoformat()
            end = max(stats.end for stats in self.entities).isoformat()
            lines.append(f"  entities: {len(self.entities)}   span: {start} -> {end}")
        if self.frequency_seconds is not None:
            lines.append(f"  sampling interval: {self.frequency_seconds:g}s")
        ordered = sorted(self.issues, key=lambda i: ("error", "warning", "info").index(i.severity))
        for issue in ordered[:max_issues]:
            text = f"  [{issue.severity}] {issue.code}: {issue.message}"
            if issue.lines:
                shown = ", ".join(str(line) for line in issue.lines)
                more = "" if issue.n_affected <= len(issue.lines) else ", ..."
                text += f" (lines {shown}{more})"
            if issue.examples:
                text += f" e.g. {', '.join(repr(example) for example in issue.examples)}"
            lines.append(text)
            if issue.hint:
                lines.append(f"      hint: {issue.hint}")
        if len(ordered) > max_issues:
            lines.append(f"  ... {len(ordered) - max_issues} more issue(s); use --report for all")
        verdict = "VALID" if self.is_valid else "INVALID"
        lines.append(
            f"Result: {verdict} ({len(self.errors)} error(s), {len(self.warnings)} warning(s))"
        )
        return "\n".join(lines)
