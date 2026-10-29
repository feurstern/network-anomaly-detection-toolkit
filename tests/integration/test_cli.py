import json
from pathlib import Path

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from network_anomaly import __version__
from network_anomaly.cli import DetectorName, app, cli_errors
from network_anomaly.detectors import DETECTOR_TYPES
from network_anomaly.errors import ConfigError, DataValidationError, ExitCode, LeakageError

pytestmark = pytest.mark.integration

runner = CliRunner()
BASELINE_CONFIG = Path(__file__).parents[2] / "configs" / "baseline.yaml"


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"network-anomaly {__version__}"


def test_help_lists_the_available_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "schema" in result.stdout


@pytest.mark.parametrize(
    ("name", "expected"), [("metrics", "requests_per_second"), ("flows", "dst_port")]
)
def test_schema_command_describes_each_schema(name: str, expected: str) -> None:
    result = runner.invoke(app, ["schema", name])
    assert result.exit_code == 0
    assert expected in result.stdout


def test_unknown_schema_name_is_a_usage_error() -> None:
    result = runner.invoke(app, ["schema", "packets"])
    assert result.exit_code == ExitCode.USAGE


def test_invalid_global_option_is_a_usage_error() -> None:
    result = runner.invoke(app, ["--log-level", "verbose", "schema"])
    assert result.exit_code == ExitCode.USAGE


def test_log_level_is_case_insensitive() -> None:
    result = runner.invoke(app, ["--log-level", "debug", "schema", "flows"])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ConfigError("bad option"), ExitCode.CONFIG),
        (DataValidationError("bad rows"), ExitCode.DATA_VALIDATION),
        (LeakageError("fit on test data"), ExitCode.LEAKAGE),
        (RuntimeError("bug"), ExitCode.INTERNAL),
    ],
)
def test_failures_map_to_documented_exit_codes(error: Exception, expected_code: ExitCode) -> None:
    with pytest.raises(typer.Exit) as info, cli_errors():
        raise error
    assert info.value.exit_code == expected_code


SMALL_GENERATE = ["--hosts", "2", "--days", "8", "--seed", "11"]


def test_generate_then_validate(tmp_path: Path) -> None:
    output = tmp_path / "demo.csv"
    result = runner.invoke(
        app, ["generate", "--output", str(output), "--scenario", "traffic_spike", *SMALL_GENERATE]
    )
    assert result.exit_code == 0, result.output
    assert "anomaly events" in result.stdout
    for name in ("demo.csv", "demo.events.csv", "demo.dq_events.csv", "demo.metadata.json"):
        assert (tmp_path / name).is_file(), name
    metadata = json.loads((tmp_path / "demo.metadata.json").read_text())
    assert metadata["config"]["scenarios"] == ["traffic_spike"]

    report = tmp_path / "report.json"
    result = runner.invoke(app, ["validate", "--input", str(output), "--report", str(report)])
    assert result.exit_code == 0, result.output
    assert "Result: VALID" in result.stdout
    assert json.loads(report.read_text())["schema_name"] == "metrics"


def test_generate_is_deterministic(tmp_path: Path) -> None:
    for name in ("a.csv", "b.csv"):
        result = runner.invoke(app, ["generate", "--output", str(tmp_path / name), *SMALL_GENERATE])
        assert result.exit_code == 0, result.output
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_generate_several_datasets_and_a_clean_twin(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "generate",
            "--output",
            str(tmp_path / "set.csv"),
            "--n-datasets",
            "2",
            "--with-clean-twin",
            *SMALL_GENERATE,
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "set_00.csv").read_bytes() != (tmp_path / "set_01.csv").read_bytes()
    assert (tmp_path / "set_01.clean.csv").is_file()


def test_generate_reads_yaml_and_lets_options_override_it(tmp_path: Path) -> None:
    config = tmp_path / "generator.yaml"
    config.write_text("seed: 5\nhosts: 3\ndays: 8\npreset: easy\n", encoding="utf-8")
    output = tmp_path / "demo.csv"
    result = runner.invoke(
        app, ["generate", "--config", str(config), "--hosts", "1", "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    resolved = json.loads((tmp_path / "demo.metadata.json").read_text())["config"]
    assert (resolved["seed"], resolved["hosts"], resolved["preset"]) == (5, 1, "easy")


@pytest.mark.parametrize(
    ("args", "expected_code"),
    [
        (["--freq", "7min"], ExitCode.CONFIG),
        (["--scenario", "meteor_strike"], ExitCode.USAGE),
        (["--hosts", "0"], ExitCode.USAGE),
    ],
)
def test_generate_rejects_invalid_options(
    tmp_path: Path, args: list[str], expected_code: ExitCode
) -> None:
    result = runner.invoke(app, ["generate", "--output", str(tmp_path / "x.csv"), *args])
    assert result.exit_code == expected_code, result.output


def test_validate_reports_errors_and_exits_3(fixtures_dir: Path) -> None:
    result = runner.invoke(
        app, ["validate", "--input", str(fixtures_dir / "metrics_malformed.csv")]
    )
    assert result.exit_code == ExitCode.DATA_VALIDATION
    assert "[error] non_numeric" in result.stdout
    assert "(lines 3)" in result.stdout
    assert "Result: INVALID" in result.stdout


def test_validate_with_mask_policy_accepts_the_file_with_warnings(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    path.write_text(
        "timestamp,host_id,cpu_usage_percent\n"
        "2026-01-05T00:00:00Z,a,140\n2026-01-05T00:05:00Z,a,40\n",
        encoding="utf-8",
    )
    assert runner.invoke(app, ["validate", "--input", str(path)]).exit_code == 3
    result = runner.invoke(app, ["validate", "--input", str(path), "--on-invalid", "mask"])
    assert result.exit_code == 0, result.output
    assert "[warning] out_of_range" in result.stdout


def test_validate_flows(fixtures_dir: Path) -> None:
    result = runner.invoke(
        app, ["validate", "--input", str(fixtures_dir / "flows_minimal.csv"), "--schema", "flows"]
    )
    assert result.exit_code == 0, result.output
    assert "sensitive columns: src_addr, dst_addr" in result.stdout


def test_validate_missing_file_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", "--input", str(tmp_path / "absent.csv")])
    assert result.exit_code == ExitCode.USAGE


def test_validate_structural_error_exits_3(tmp_path: Path) -> None:
    path = tmp_path / "ragged.csv"
    path.write_text("timestamp,host_id,cpu_usage_percent\n2026-01-05T00:00:00Z,a\n")
    result = runner.invoke(app, ["validate", "--input", str(path)])
    assert result.exit_code == ExitCode.DATA_VALIDATION
    assert "do not have exactly 3 fields" in result.stderr


@pytest.fixture(scope="module")
def demo_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("demo") / "demo.csv"
    args = ["generate", "--output", str(path), "--hosts", "3", "--days", "14", "--seed", "3"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return path


def _detect(demo_csv: Path, output: Path, *args: str) -> str:
    result = runner.invoke(
        app, ["detect", "--input", str(demo_csv), "--output", str(output), *args]
    )
    assert result.exit_code == 0, result.output
    return result.stdout


def test_cli_detector_choices_match_the_registry() -> None:
    assert {name.value for name in DetectorName} == set(DETECTOR_TYPES)


def test_detect_writes_scores_flags_segments_and_provenance(demo_csv: Path, tmp_path: Path) -> None:
    output = tmp_path / "detections.csv"
    stdout = _detect(demo_csv, output, "--detector", "robust_zscore")
    assert "Threshold clean_quantile" in stdout
    table = pd.read_csv(output)
    assert list(table.columns[:8]) == [
        "timestamp",
        "host_id",
        "segment",
        "status",
        "score",
        "threshold",
        "threshold_scope",
        "is_flagged",
    ]
    assert set(table["segment"]) == {"train", "validation", "test"}
    assert set(table["status"]) <= {"scored", "warmup", "no_data"}
    assert len(table) == len(pd.read_csv(demo_csv))
    meta = json.loads((tmp_path / "detections.meta.json").read_text())
    assert meta["detector"]["name"] == "robust_zscore"
    assert meta["thresholds"]["fitted_on"] == "validation"
    assert len(meta["input"]["sha256"]) == 64


def test_detect_is_byte_for_byte_deterministic(demo_csv: Path, tmp_path: Path) -> None:
    for name in ("a.csv", "b.csv"):
        _detect(demo_csv, tmp_path / name, "--detector", "robust_zscore")
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_detect_reads_a_yaml_config(demo_csv: Path, tmp_path: Path) -> None:
    stdout = _detect(demo_csv, tmp_path / "d.csv", "--config", str(BASELINE_CONFIG))
    assert "Detector robust_zscore" in stdout


def test_saved_model_reproduces_the_detections(demo_csv: Path, tmp_path: Path) -> None:
    model = tmp_path / "model"
    _detect(
        demo_csv, tmp_path / "fitted.csv", "--detector", "robust_zscore", "--save-model", str(model)
    )
    assert (model / "bundle.json").is_file()
    assert (model / "detector" / "manifest.json").is_file()
    _detect(demo_csv, tmp_path / "applied.csv", "--model", str(model))
    assert (tmp_path / "fitted.csv").read_bytes() == (tmp_path / "applied.csv").read_bytes()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "choose a detector"),
        (["--detector", "static_threshold", "--config", str(BASELINE_CONFIG)], "conflicts"),
        (["--detector", "robust_zscore", "--quantile", "0.9"], "need --threshold-strategy"),
        (["--detector", "robust_zscore", "--threshold-strategy", "fixed"], "threshold.fixed.value"),
        (["--detector", "robust_zscore", "--train-until", "2030-01-01T00:00:00Z"], "outside"),
    ],
)
def test_detect_rejects_invalid_configuration(
    demo_csv: Path, tmp_path: Path, args: list[str], message: str
) -> None:
    result = runner.invoke(
        app, ["detect", "--input", str(demo_csv), "--output", str(tmp_path / "x.csv"), *args]
    )
    assert result.exit_code == ExitCode.CONFIG, result.output
    assert message in result.stderr


def test_model_cannot_be_combined_with_fitting_options(demo_csv: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "detect",
            "--input",
            str(demo_csv),
            "--model",
            str(tmp_path),
            "--detector",
            "robust_zscore",
        ],
    )
    assert result.exit_code == ExitCode.CONFIG


def test_detect_rejects_invalid_input_with_exit_code_3(fixtures_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "detect",
            "--input",
            str(fixtures_dir / "metrics_malformed.csv"),
            "--detector",
            "robust_zscore",
            "--output",
            str(tmp_path / "x.csv"),
        ],
    )
    assert result.exit_code == ExitCode.DATA_VALIDATION
    assert "non_numeric" in result.stderr


def test_evaluate_reports_point_and_event_metrics(demo_csv: Path, tmp_path: Path) -> None:
    detections = tmp_path / "detections.csv"
    _detect(demo_csv, detections, "--detector", "isolation_forest")
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--input",
            str(demo_csv),
            "--predictions",
            str(detections),
            "--output-dir",
            str(tmp_path / "eval"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Evaluation of isolation_forest on test" in result.stdout
    assert "Point level" in result.stdout
    assert "Event level" in result.stdout
    metrics = json.loads((tmp_path / "eval" / "metrics.json").read_text())
    assert metrics["segments"] == ["test"]
    assert metrics["in_sample"] is False
    assert metrics["events"]["events"] > 0
    assert 0.0 <= metrics["point"]["pr_auc"] <= 1.0


def test_evaluate_refuses_in_sample_rows_unless_allowed(demo_csv: Path, tmp_path: Path) -> None:
    detections = tmp_path / "detections.csv"
    _detect(demo_csv, detections, "--detector", "robust_zscore")
    base = ["evaluate", "--input", str(demo_csv), "--predictions", str(detections)]
    refused = runner.invoke(app, [*base, "--segment", "train"])
    assert refused.exit_code == ExitCode.LEAKAGE
    allowed = runner.invoke(app, [*base, "--segment", "train", "--allow-in-sample"])
    assert allowed.exit_code == 0, allowed.output
    assert "IN-SAMPLE" in allowed.stdout


def test_evaluate_rejects_detections_from_another_file(demo_csv: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.csv"
    args = ["generate", "--output", str(other), "--hosts", "2", "--days", "14", "--seed", "4"]
    assert runner.invoke(app, args).exit_code == 0
    detections = tmp_path / "detections.csv"
    _detect(other, detections, "--detector", "robust_zscore")
    result = runner.invoke(
        app, ["evaluate", "--input", str(demo_csv), "--predictions", str(detections)]
    )
    assert result.exit_code == ExitCode.DATA_VALIDATION
    assert "does not match" in result.stderr


def test_evaluate_needs_the_detection_metadata(demo_csv: Path, tmp_path: Path) -> None:
    detections = tmp_path / "detections.csv"
    _detect(demo_csv, detections, "--detector", "robust_zscore")
    (tmp_path / "detections.meta.json").unlink()
    result = runner.invoke(
        app, ["evaluate", "--input", str(demo_csv), "--predictions", str(detections)]
    )
    assert result.exit_code == ExitCode.ARTIFACT
