from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from network_anomaly.errors import ArtifactError, DataValidationError
from network_anomaly.ingestion.readers import read_csv_strings
from network_anomaly.ingestion.validation import (
    InvalidDataError,
    ValidatedMetrics,
    load_flows,
    load_metrics,
    validate_flows_table,
    validate_metrics_table,
)
from network_anomaly.schemas import LABEL_COLUMNS, ValidationIssue, ValidationReport

HEADER = "timestamp,host_id,requests_per_second,cpu_usage_percent,http_5xx_rate"


def _csv(tmp_path: Path, *rows: str, header: str = HEADER) -> Path:
    path = tmp_path / "input.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def _validate(path: Path, **kwargs: object) -> tuple[ValidatedMetrics | None, ValidationReport]:
    return validate_metrics_table(read_csv_strings(path), **kwargs)  # type: ignore[arg-type]


def _issue(report: ValidationReport, code: str) -> ValidationIssue:
    matches = [issue for issue in report.issues if issue.code == code]
    assert matches, f"expected issue {code!r}, got {[i.code for i in report.issues]}"
    return matches[0]


def _codes(report: ValidationReport) -> set[str]:
    return {issue.code for issue in report.issues}


def test_valid_file_is_split_into_metrics_and_labels(fixtures_dir: Path) -> None:
    validated = load_metrics(fixtures_dir / "metrics_valid.csv")
    frame = validated.metrics
    assert frame.metrics == (
        "requests_per_second",
        "cpu_usage_percent",
        "http_5xx_rate",
        "response_time_p50_ms",
        "response_time_p95_ms",
    )
    assert not set(frame.data.columns) & set(LABEL_COLUMNS)
    assert str(frame.data["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert frame.data[list(frame.metrics)].dtypes.eq(np.float64).all()
    assert list(frame.data["host_id"]) == ["web-01"] * 4 + ["web-02"] * 4
    assert validated.labels is not None
    assert validated.labels.data["is_anomaly"].tolist() == [0, 0, 1, 0, 0, 0, 0, 0]
    assert np.isnan(frame.data.loc[5, "http_5xx_rate"])  # empty cell stays missing
    report = validated.report
    assert report.is_valid
    assert report.frequency_seconds == 300
    assert [stats.rows for stats in report.entities] == [4, 4]


def test_rows_are_sorted_by_entity_then_time(tmp_path: Path) -> None:
    path = _csv(
        tmp_path,
        "2026-01-05T00:05:00Z,b,1,1,0",
        "2026-01-05T00:00:00Z,b,1,1,0",
        "2026-01-05T00:00:00Z,a,1,1,0",
    )
    validated = load_metrics(path)
    data = validated.metrics.data
    assert list(data["host_id"]) == ["a", "b", "b"]
    assert data.loc[1, "timestamp"] < data.loc[2, "timestamp"]


@pytest.mark.parametrize("missing", ["timestamp", "host_id"])
def test_missing_required_column(tmp_path: Path, missing: str) -> None:
    values = ["2026-01-05T00:00:00Z", "a", "1", "1", "0"]
    cells = dict(zip(HEADER.split(","), values, strict=True))
    del cells[missing]
    validated, report = _validate(_csv(tmp_path, ",".join(cells.values()), header=",".join(cells)))
    assert validated is None
    issue = _issue(report, "missing_column")
    assert missing in issue.message
    if missing == "host_id":
        assert issue.hint is not None
        assert "--entity-column" in issue.hint


def test_custom_entity_column(tmp_path: Path) -> None:
    path = _csv(
        tmp_path,
        "2026-01-05T00:00:00Z,checkout,1,1,0",
        header=HEADER.replace("host_id", "service"),
    )
    validated = load_metrics(path, entity_column="service")
    assert validated.metrics.entity_column == "service"


def test_non_numeric_value_reports_line_and_example(fixtures_dir: Path) -> None:
    validated, report = _validate(fixtures_dir / "metrics_malformed.csv")
    assert validated is None
    issue = _issue(report, "non_numeric")
    assert issue.severity == "error"
    assert issue.column == "requests_per_second"
    assert issue.lines == (3,)
    assert issue.examples == ("12.5x",)
    assert issue.hint is not None


def test_bad_timestamp_and_range_errors_are_all_reported_at_once(fixtures_dir: Path) -> None:
    _, report = _validate(fixtures_dir / "metrics_malformed.csv")
    assert _issue(report, "bad_timestamp").lines == (4,)
    out_of_range = [issue for issue in report.issues if issue.code == "out_of_range"]
    assert {issue.column for issue in out_of_range} == {"cpu_usage_percent", "http_5xx_rate"}
    rate_issue = next(issue for issue in out_of_range if issue.column == "http_5xx_rate")
    assert rate_issue.hint is not None
    assert "divide by 100" in rate_issue.hint
    assert not report.is_valid
    assert "Result: INVALID" in report.summary()


def test_naive_timestamps_are_rejected_unless_assumed_utc(tmp_path: Path) -> None:
    path = _csv(tmp_path, "2026-01-05T00:00:00,a,1,1,0", "2026-01-05T00:05:00,a,1,1,0")
    validated, report = _validate(path)
    assert validated is None
    assert _issue(report, "naive_timestamp").severity == "error"
    validated, report = _validate(path, assume_utc=True)
    assert validated is not None
    assert _issue(report, "naive_timestamp").severity == "warning"
    assert validated.metrics.data["timestamp"].iloc[0] == pd.Timestamp("2026-01-05T00:00:00Z")


def test_offsets_are_converted_to_utc(tmp_path: Path) -> None:
    path = _csv(tmp_path, "2026-01-05T02:00:00+02:00,a,1,1,0")
    data = load_metrics(path).metrics.data
    assert data["timestamp"].iloc[0] == pd.Timestamp("2026-01-05T00:00:00Z")


def test_missing_and_invalid_timestamps(tmp_path: Path) -> None:
    validated, report = _validate(_csv(tmp_path, ",a,1,1,0", "05/01/2026 10:00,a,1,1,0"))
    assert validated is None
    assert _issue(report, "missing_timestamp").lines == (2,)
    assert _issue(report, "bad_timestamp").examples == ("05/01/2026 10:00",)


def test_infinite_values_are_errors(tmp_path: Path) -> None:
    _, report = _validate(_csv(tmp_path, "2026-01-05T00:00:00Z,a,inf,1,0"))
    assert _issue(report, "non_finite").column == "requests_per_second"


def test_exact_duplicates_are_dropped_with_a_warning(tmp_path: Path) -> None:
    row = "2026-01-05T00:00:00Z,a,1,2,0"
    validated, report = _validate(_csv(tmp_path, row, row, "2026-01-05T00:05:00Z,a,1,2,0"))
    assert validated is not None
    assert len(validated.metrics.data) == 2
    assert _issue(report, "duplicate_rows").lines == (3,)


def test_conflicting_duplicates_are_errors(tmp_path: Path) -> None:
    validated, report = _validate(
        _csv(tmp_path, "2026-01-05T00:00:00Z,a,1,2,0", "2026-01-05T00:00:00Z,a,5,2,0")
    )
    assert validated is None
    assert _issue(report, "conflicting_duplicates").lines == (2, 3)


def test_mask_policy_blanks_invalid_cells_and_records_them(tmp_path: Path) -> None:
    path = _csv(
        tmp_path,
        "2026-01-05T00:00:00Z,a,1,140,0",
        "2026-01-05T00:05:00Z,a,oops,20,0",
        "2026-01-05T00:10:00Z,a,3,20,0",
    )
    validated, report = _validate(path, on_invalid="mask")
    assert validated is not None
    assert report.is_valid
    assert {i.severity for i in report.issues if i.code in {"out_of_range", "non_numeric"}} == {
        "warning"
    }
    data = validated.metrics.data
    mask = validated.metrics.invalid_mask
    assert mask is not None
    assert np.isnan(data.loc[0, "cpu_usage_percent"])
    assert bool(mask.loc[0, "cpu_usage_percent"])
    assert np.isnan(data.loc[1, "requests_per_second"])
    assert bool(mask.loc[1, "requests_per_second"])
    assert int(mask.to_numpy().sum()) == 2


def test_unknown_numeric_column_is_a_generic_metric_with_a_suggestion(tmp_path: Path) -> None:
    path = _csv(
        tmp_path, "2026-01-05T00:00:00Z,a,1,2", header="timestamp,host_id,cpu_usage_pct,gc_ms"
    )
    validated, report = _validate(path)
    assert validated is not None
    assert validated.metrics.metrics == ("cpu_usage_pct", "gc_ms")
    hints = [i.hint for i in report.issues if i.code == "unknown_metric"]
    assert "Did you mean 'cpu_usage_percent'?" in hints


def test_text_columns_are_ignored_not_converted(tmp_path: Path) -> None:
    path = _csv(
        tmp_path,
        "2026-01-05T00:00:00Z,a,1,eu-west",
        header="timestamp,host_id,requests_per_second,region",
    )
    validated, report = _validate(path)
    assert validated is not None
    assert validated.metrics.metrics == ("requests_per_second",)
    assert _issue(report, "ignored_column").column == "region"


def test_a_canonical_metric_with_only_text_is_an_error(tmp_path: Path) -> None:
    path = _csv(
        tmp_path, "2026-01-05T00:00:00Z,a,high", header="timestamp,host_id,cpu_usage_percent"
    )
    validated, report = _validate(path)
    assert validated is None
    assert _issue(report, "non_numeric").column == "cpu_usage_percent"


def test_file_without_metrics_is_rejected(tmp_path: Path) -> None:
    _, report = _validate(_csv(tmp_path, "2026-01-05T00:00:00Z,a", header="timestamp,host_id"))
    assert "no_metrics" in _codes(report)


@pytest.mark.parametrize("value", ["2", "yes", ""])
def test_invalid_anomaly_labels(tmp_path: Path, value: str) -> None:
    path = _csv(
        tmp_path,
        f"2026-01-05T00:00:00Z,a,1,{value}",
        header="timestamp,host_id,requests_per_second,is_anomaly",
    )
    _, report = _validate(path)
    assert {"invalid_label", "missing_label"} & _codes(report)
    assert not report.is_valid


def test_grid_collision_and_irregular_timestamps(tmp_path: Path) -> None:
    rows = [f"2026-01-05T00:{m:02d}:00Z,a,1,1,0" for m in range(0, 30, 5)]
    rows.append("2026-01-05T00:31:40Z,a,1,1,0")  # 100 s off the 5-minute grid
    rows.append("2026-01-05T00:00:40Z,a,1,1,0")  # same grid cell as 00:00
    validated, report = _validate(_csv(tmp_path, *rows))
    assert validated is None
    assert _issue(report, "grid_collision").severity == "error"
    assert _issue(report, "irregular_timestamps").lines == (8,)


def test_gaps_are_counted_per_entity(tmp_path: Path) -> None:
    rows = [f"2026-01-05T00:{m:02d}:00Z,a,1,1,0" for m in (0, 5, 10, 25, 30)]
    validated, report = _validate(_csv(tmp_path, *rows))
    assert validated is not None
    assert report.entities[0].missing_intervals == 2
    assert _issue(report, "missing_intervals").n_affected == 2


def test_explicit_frequency_overrides_inference(tmp_path: Path) -> None:
    rows = [f"2026-01-05T00:{m:02d}:00Z,a,1,1,0" for m in (0, 10, 20)]
    _, report = _validate(_csv(tmp_path, *rows), frequency=pd.Timedelta("5min"))
    assert report.frequency_seconds == 300
    assert report.entities[0].missing_intervals == 2


def test_quality_warnings(tmp_path: Path) -> None:
    header = "timestamp,host_id,cpu_usage_percent,response_time_p50_ms,response_time_p95_ms"
    path = _csv(
        tmp_path,
        "2026-01-05T00:00:00Z,a,0.4,50,40",
        "2026-01-05T00:05:00Z,a,0.4,50,90",
        header=header,
    )
    _, report = _validate(path)
    assert _issue(report, "latency_percentiles_inverted").lines == (2,)
    assert _issue(report, "constant_metric").column == "cpu_usage_percent"
    assert _issue(report, "percent_scale_suspect").column == "cpu_usage_percent"


def test_line_numbers_account_for_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "blank.csv"
    path.write_text(
        f"{HEADER}\n2026-01-05T00:00:00Z,a,1,1,0\n\n2026-01-05T00:05:00Z,a,bad,1,0\n",
        encoding="utf-8",
    )
    _, report = _validate(path)
    assert _issue(report, "non_numeric").lines == (4,)


def test_structural_problems_raise_immediately(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="do not have exactly 5 fields"):
        read_csv_strings(_csv(tmp_path, "2026-01-05T00:00:00Z,a,1,1"))
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("timestamp,host_id,cpu,cpu\n2026-01-05T00:00:00Z,a,1,2\n")
    with pytest.raises(DataValidationError, match="duplicate column names"):
        read_csv_strings(duplicate)
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DataValidationError, match="empty"):
        read_csv_strings(empty)
    header_only = tmp_path / "header.csv"
    header_only.write_text(HEADER + "\n")
    with pytest.raises(DataValidationError, match="no data rows"):
        read_csv_strings(header_only)
    with pytest.raises(ArtifactError, match="not found"):
        read_csv_strings(tmp_path / "absent.csv")


def test_load_metrics_raises_with_the_full_report(fixtures_dir: Path) -> None:
    with pytest.raises(InvalidDataError) as info:
        load_metrics(fixtures_dir / "metrics_malformed.csv")
    assert info.value.exit_code == 3
    assert len(info.value.report.errors) >= 4


def test_flows_minimal_file_is_valid_and_flags_addresses(fixtures_dir: Path) -> None:
    validated = load_flows(fixtures_dir / "flows_minimal.csv")
    report = validated.report
    assert report.is_valid
    assert report.sensitive_columns == ("src_addr", "dst_addr")
    assert report.label_columns == ("label",)
    data = validated.flows.data
    assert data["dst_port"].tolist() == [443.0, 443.0, 80.0]
    assert str(data["timestamp"].dtype) == "datetime64[ns, UTC]"


@pytest.mark.parametrize(
    ("row", "column"),
    [
        ("2026-01-05T00:00:00Z,70000,10", "dst_port"),
        ("2026-01-05T00:00:00Z,443.5,10", "dst_port"),
        ("2026-01-05T00:00:00Z,443,-1", "bytes"),
    ],
)
def test_flow_field_checks(tmp_path: Path, row: str, column: str) -> None:
    path = _csv(tmp_path, row, header="timestamp,dst_port,bytes")
    validated, report = validate_flows_table(read_csv_strings(path))
    assert validated is None
    assert _issue(report, "invalid_value").column == column


def test_flows_need_a_volume_field_and_warn_on_unknown_columns(tmp_path: Path) -> None:
    path = _csv(tmp_path, "2026-01-05T00:00:00Z,443,x", header="timestamp,dst_port,src_prot")
    _, report = validate_flows_table(read_csv_strings(path))
    assert "missing_volume_field" in _codes(report)
    unknown = _issue(report, "unknown_column")
    assert unknown.hint == "Did you mean 'src_port'?"
