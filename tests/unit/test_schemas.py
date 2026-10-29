import pytest
from pydantic import ValidationError

from network_anomaly.schemas import (
    CANONICAL_METRICS,
    DEFAULT_ENTITY_COLUMN,
    FLOW_FIELD_REGISTRY,
    FLOW_FIELDS,
    FLOW_VOLUME_FIELDS,
    LABEL_COLUMNS,
    METRIC_REGISTRY,
    TIMESTAMP_COLUMN,
    MetricFamily,
    MetricSpec,
    describe_flow_schema,
    describe_metric_schema,
    metric_spec,
)

DOCUMENTED_METRICS = {
    "requests_per_second",
    "http_4xx_rate",
    "http_5xx_rate",
    "response_time_p50_ms",
    "response_time_p95_ms",
    "cpu_usage_percent",
    "memory_usage_percent",
    "disk_io_bytes_per_second",
    "network_rx_bytes_per_second",
    "network_tx_bytes_per_second",
    "active_connections",
    "connection_error_rate",
}


def test_registry_covers_exactly_the_documented_metrics() -> None:
    assert set(METRIC_REGISTRY) == DOCUMENTED_METRICS
    assert len(CANONICAL_METRICS) == len(METRIC_REGISTRY)


@pytest.mark.parametrize("spec", CANONICAL_METRICS, ids=lambda spec: spec.name)
def test_unit_conventions(spec: MetricSpec) -> None:
    if spec.unit == "fraction":
        assert (spec.min_value, spec.max_value) == (0.0, 1.0)
        assert not spec.log_scale
    if spec.unit == "%":
        assert (spec.min_value, spec.max_value) == (0.0, 100.0)
        assert not spec.log_scale
    if spec.name.endswith(("_per_second", "_ms", "_connections")):
        assert spec.log_scale
        assert spec.min_value == 0.0
        assert spec.max_value is None
    assert spec.min_scale > 0


def test_reserved_columns_never_collide_with_metrics() -> None:
    reserved = {TIMESTAMP_COLUMN, DEFAULT_ENTITY_COLUMN, *LABEL_COLUMNS}
    assert not reserved & set(METRIC_REGISTRY)


def test_only_user_facing_symptoms_are_critical() -> None:
    critical = {spec.name for spec in CANONICAL_METRICS if spec.critical}
    assert critical == {"http_5xx_rate", "response_time_p95_ms", "connection_error_rate"}


def test_error_latency_and_saturation_metrics_only_count_increases() -> None:
    upward_only = {MetricFamily.ERRORS, MetricFamily.LATENCY, MetricFamily.CPU, MetricFamily.MEMORY}
    for spec in CANONICAL_METRICS:
        if spec.family in upward_only:
            assert spec.direction == "up", spec.name
    assert METRIC_REGISTRY["requests_per_second"].direction == "both"


def test_unknown_metric_gets_a_generic_spec_without_range_checks() -> None:
    spec = metric_spec("gc_pause_seconds")
    assert spec.family is MetricFamily.GENERIC
    assert spec.min_value is None
    assert spec.max_value is None
    assert metric_spec("cpu_usage_percent") is METRIC_REGISTRY["cpu_usage_percent"]


def _spec(**overrides: object) -> MetricSpec:
    fields: dict[str, object] = {
        "name": "x",
        "unit": "u",
        "description": "d",
        "family": "generic",
        "log_scale": False,
        "direction": "both",
        "critical": False,
        "min_scale": 1.0,
    }
    return MetricSpec.model_validate(fields | overrides)


def test_metric_spec_rejects_an_inverted_range() -> None:
    with pytest.raises(ValidationError, match="min_value must be below max_value"):
        _spec(min_value=5.0, max_value=1.0)


def test_metric_spec_rejects_log_scale_on_signed_metrics() -> None:
    with pytest.raises(ValidationError, match="log-scale metrics must have min_value >= 0"):
        _spec(log_scale=True, min_value=-1.0)


def test_metric_spec_rejects_unknown_fields_and_mutation() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _spec(colour="red")
    spec = _spec()
    with pytest.raises(ValidationError):
        spec.unit = "other"  # type: ignore[misc]


def test_flow_schema_marks_only_addresses_as_sensitive() -> None:
    assert {field.name for field in FLOW_FIELDS if field.sensitive} == {"src_addr", "dst_addr"}


def test_flow_schema_requires_only_the_timestamp() -> None:
    assert [field.name for field in FLOW_FIELDS if field.required] == ["timestamp"]
    assert set(FLOW_VOLUME_FIELDS) <= set(FLOW_FIELD_REGISTRY)


def test_flow_ports_are_bounded() -> None:
    for name in ("src_port", "dst_port"):
        field = FLOW_FIELD_REGISTRY[name]
        assert (field.min_value, field.max_value) == (0, 65535)


def test_schema_descriptions_document_every_column() -> None:
    metrics_text = describe_metric_schema()
    for name in METRIC_REGISTRY:
        assert name in metrics_text
    for name in LABEL_COLUMNS:
        assert name in metrics_text
    flows_text = describe_flow_schema()
    for name in FLOW_FIELD_REGISTRY:
        assert name in flows_text
