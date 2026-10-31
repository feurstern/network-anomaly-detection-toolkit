import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from network_anomaly import __version__
from network_anomaly.cli import DetectorName, app, cli_errors
from network_anomaly.detectors import DETECTOR_TYPES
from network_anomaly.errors import ConfigError, DataValidationError, ExitCode, LeakageError
from network_anomaly.reporting.export import read_alerts

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
    alerts = read_alerts(tmp_path / "detections.alerts.jsonl")
    assert alerts
    assert f"Alerts: {len(alerts)}" in stdout
    assert {a.entity for a in alerts} <= set(table["host_id"])


def test_detect_writes_alerts_where_asked(demo_csv: Path, tmp_path: Path) -> None:
    target = tmp_path / "out" / "alerts.jsonl"
    _detect(demo_csv, tmp_path / "d.csv", "--detector", "robust_zscore", "--alerts", str(target))
    assert read_alerts(target)
    assert not (tmp_path / "d.alerts.jsonl").exists()


def test_detect_is_byte_for_byte_deterministic(demo_csv: Path, tmp_path: Path) -> None:
    for name in ("a.csv", "b.csv"):
        _detect(demo_csv, tmp_path / name, "--detector", "isolation_forest")
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    alerts = [(tmp_path / f"{n}.alerts.jsonl").read_bytes() for n in ("a", "b")]
    assert alerts[0] == alerts[1]


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
    fitted, applied = (tmp_path / "fitted.alerts.jsonl", tmp_path / "applied.alerts.jsonl")
    assert fitted.read_bytes() == applied.read_bytes()


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
    report = (tmp_path / "eval" / "report.md").read_text()
    assert "## Alerts in the evaluated window" in report  # read from detections.alerts.jsonl
    figures = {path.name for path in (tmp_path / "eval" / "figures").glob("*.png")}
    assert figures == {
        "confusion_matrix.png",
        "feature_deviations.png",
        "pr_curve.png",
        "timeseries.png",
    }
    for name in figures:
        assert f"figures/{name}" in report


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


TINY_BENCHMARK = """\
name: tiny
presets: [easy]
seeds: [0, 1]
hosts: 2
days: 9
detectors:
  - type: static_threshold
  - type: robust_zscore
timing:
  warmup: 0
  repeats: 1
"""


def _benchmark(tmp_path: Path, config_text: str, *args: str) -> tuple[typer.testing.Result, Path]:
    config = tmp_path / "benchmark.yaml"
    config.write_text(config_text, encoding="utf-8")
    output = tmp_path / "bench"
    result = runner.invoke(
        app, ["benchmark", "--config", str(config), "--output", str(output), *args]
    )
    return result, output


def test_benchmark_writes_a_reproducible_run_folder(tmp_path: Path) -> None:
    result, output = _benchmark(tmp_path, TINY_BENCHMARK, "--workers", "2")
    assert result.exit_code == 0, result.output
    (run,) = output.iterdir()
    names = {path.name for path in run.iterdir()}
    assert {"manifest.json", "tasks.jsonl", "summary.csv", "per_scenario.csv", "report.md"} <= names
    records = [json.loads(line) for line in (run / "tasks.jsonl").read_text().splitlines()]
    kinds = [record["kind"] for record in records]
    assert kinds.count("accuracy") == 4  # 2 seeds x 2 detectors
    assert kinds.count("timing") == 3  # features + 2 detectors
    assert all(record["status"] == "ok" for record in records)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["config_fingerprint"] == run.name.rsplit("-", 1)[-1]
    assert manifest["environment"]["packages"]["scikit-learn"]
    report = (run / "report.md").read_text()
    assert "What these results can and cannot show" in report
    assert "do not demonstrate production effectiveness" in report
    assert (run / "figures" / "comparison_easy.png").is_file()
    assert "target_fpr operating point" in result.stdout


def test_benchmark_rejects_an_invalid_configuration(tmp_path: Path) -> None:
    result, _ = _benchmark(tmp_path, TINY_BENCHMARK.replace("seeds: [0, 1]", "seeds: []"))
    assert result.exit_code == ExitCode.CONFIG
    assert "seeds" in result.stderr


def test_benchmark_reports_failed_tasks_and_exits_non_zero(tmp_path: Path) -> None:
    broken = TINY_BENCHMARK.replace(
        "  - type: static_threshold\n", "  - type: static_threshold\n    limits: {nope: 1.0}\n"
    ).replace("seeds: [0, 1]", "seeds: [0]")
    result, output = _benchmark(tmp_path, broken, "--skip-timing")
    assert result.exit_code == ExitCode.INTERNAL
    assert "1 task(s) failed" in result.stderr
    (run,) = output.iterdir()
    assert "## Failed tasks" in (run / "report.md").read_text()


def test_detect_calibrates_on_a_known_clean_copy(tmp_path: Path) -> None:
    data = tmp_path / "pair.csv"
    args = ["generate", "--output", str(data), "--hosts", "2", "--days", "14", "--seed", "6"]
    assert runner.invoke(app, [*args, "--with-clean-twin"]).exit_code == 0
    clean = tmp_path / "pair.clean.csv"
    stdout = _detect(
        data, tmp_path / "d.csv", "--detector", "robust_zscore", "--clean-validation", str(clean)
    )
    assert "(fitted on the known-clean file)" in stdout
    meta = json.loads((tmp_path / "d.meta.json").read_text())
    assert meta["thresholds"]["source"] == "known_clean"
    model = tmp_path / "model"
    _detect(data, tmp_path / "m.csv", "--detector", "robust_zscore", "--save-model", str(model))
    combined = runner.invoke(
        app,
        ["detect", "--input", str(data), "--model", str(model), "--clean-validation", str(clean)],
    )
    assert combined.exit_code == ExitCode.CONFIG


def test_datasets_list_and_guarded_download_and_prepare(tmp_path: Path) -> None:
    listed = runner.invoke(app, ["datasets", "list"])
    assert listed.exit_code == 0
    for name in ("nab", "smd", "cicids2017-improved", "cidds-001"):
        assert name in listed.stdout
    terms = runner.invoke(app, ["datasets", "download", "cidds-001", "--data-dir", str(tmp_path)])
    assert terms.exit_code == ExitCode.CONFIG
    assert "--accept-terms" in terms.stderr
    missing = runner.invoke(app, ["datasets", "prepare", "nab", "--data-dir", str(tmp_path)])
    assert missing.exit_code == ExitCode.ARTIFACT
    assert "datasets download nab" in missing.stderr


def _synthetic_flows(path: Path, *, days: int, burst_day: int | None) -> Path:
    """Steady web traffic, optionally with a burst of SYN-only flows to many ports."""
    rng = np.random.default_rng(days)
    start = pd.Timestamp("2026-01-05T00:00:00Z")
    rows = []
    for minute in range(days * 1440):
        for _ in range(rng.poisson(4)):
            packets = int(rng.poisson(8)) + 2
            rows.append(
                (
                    start + pd.Timedelta(minutes=minute, seconds=int(rng.integers(60))),
                    "192.0.2.1",
                    443,
                    "tcp",
                    packets,
                    int(rng.normal(500, 60)) * packets,
                    1,
                    packets - 1,
                    0,
                    0,
                )
            )
    if burst_day is not None:
        burst = start + pd.Timedelta(days=burst_day, hours=12)
        for second in range(0, 600, 2):
            rows.append(
                (
                    burst + pd.Timedelta(seconds=second),
                    "192.0.2.66",
                    int(rng.integers(1, 65535)),
                    "tcp",
                    1,
                    60,
                    1,
                    0,
                    0,
                    1,
                )
            )
    columns = [
        "timestamp",
        "src_addr",
        "dst_port",
        "protocol",
        "packets",
        "bytes",
        "syn_count",
        "ack_count",
        "rst_count",
        "is_anomaly",
    ]
    frame = pd.DataFrame(rows, columns=columns).sort_values("timestamp")
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame.to_csv(path, index=False)
    return path


def test_flows_aggregate_into_metrics_that_detectors_evaluate(tmp_path: Path) -> None:
    flows = _synthetic_flows(tmp_path / "flows.csv", days=4, burst_day=3)
    metrics = tmp_path / "metrics.csv"
    result = runner.invoke(
        app,
        ["flows", "aggregate", "--input", str(flows), "--window", "5min", "--output", str(metrics)],
    )
    assert result.exit_code == 0, result.output
    assert "syn_only_ratio" in result.stdout
    assert runner.invoke(app, ["validate", "--input", str(metrics)]).exit_code == 0
    detections = tmp_path / "detections.csv"
    _detect(metrics, detections, "--detector", "robust_zscore")
    evaluated = runner.invoke(
        app, ["evaluate", "--input", str(metrics), "--predictions", str(detections)]
    )
    assert evaluated.exit_code == 0, evaluated.output
    assert "events 1, detected 1" in evaluated.stdout


def test_flows_detect_scores_flows_against_a_normal_period(tmp_path: Path) -> None:
    train = _synthetic_flows(tmp_path / "train.csv", days=1, burst_day=None)
    scored = _synthetic_flows(tmp_path / "day.csv", days=2, burst_day=1)
    output = tmp_path / "scores.csv"
    result = runner.invoke(
        app,
        ["flows", "detect", "--train", str(train), "--input", str(scored), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    assert "recall at 1% FPR 1.000" in result.stdout
    table = pd.read_csv(output)
    assert {"score", "is_flagged", "is_anomaly"} <= set(table.columns)
    assert "src_addr" not in table.columns  # addresses are not copied into reports
    meta = json.loads((tmp_path / "scores.meta.json").read_text())
    assert meta["evaluation"]["anomalous"] == 300


def test_detect_with_the_optional_autoencoder(demo_csv: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    output = tmp_path / "ae.csv"
    stdout = _detect(
        demo_csv, output, "--detector", "autoencoder", "--save-model", str(tmp_path / "m")
    )
    assert "Detector autoencoder" in stdout
    alerts = read_alerts(tmp_path / "ae.alerts.jsonl")
    assert alerts
    assert {a.explanation_method for a in alerts} == {"reconstruction_error"}
    assert (tmp_path / "m" / "detector" / "model.pt").is_file()
