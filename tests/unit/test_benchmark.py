"""Benchmark building blocks, run in-process: configuration rules, tasks, aggregation."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from network_anomaly.benchmarks.config import BenchmarkConfig
from network_anomaly.benchmarks.environment import environment, peak_rss_bytes, reset_peak_rss
from network_anomaly.benchmarks.report import flatten, summarise
from network_anomaly.benchmarks.runner import accuracy_task, prepare_dataset, timing_task

TINY: dict[str, Any] = {
    "name": "tiny",
    "presets": ["easy"],
    "seeds": [0],
    "hosts": 2,
    "days": 9,
    "detectors": [{"type": "static_threshold"}, {"type": "robust_zscore"}],
    "timing": {"warmup": 0, "repeats": 2},
}


@pytest.fixture(scope="module")
def tiny() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(TINY)


@pytest.fixture(scope="module")
def dataset(tiny: BenchmarkConfig, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    directory = tmp_path_factory.mktemp("datasets")
    return prepare_dataset(tiny.model_dump_json(), "easy", 0, str(directory))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"thresholds": {"headline": "target_fpr", "operating_points": ["fixed"]}}, "headline"),
        ({"thresholds": {"headline": "fixed"}}, "cannot be the headline"),
        ({"seeds": [1, 1]}, "seeds"),
        ({"detectors": [{"type": "robust_zscore"}, {"type": "robust_zscore"}]}, "one entry per"),
        ({"split": {"train_until": "2026-01-10T00:00:00Z"}}, "fractions"),
        ({"days": 3}, "greater than or equal to 7"),
        ({"freq": "7min"}, "7min"),
    ],
)
def test_invalid_benchmark_configurations_are_rejected(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        BenchmarkConfig.model_validate({**TINY, **change})


def test_fingerprint_identifies_the_resolved_configuration(tiny: BenchmarkConfig) -> None:
    assert tiny.fingerprint() == BenchmarkConfig.model_validate(TINY).fingerprint()
    assert (
        tiny.fingerprint() != BenchmarkConfig.model_validate({**TINY, "seeds": [1]}).fingerprint()
    )


def test_fixed_thresholds_only_apply_to_static_limits(tiny: BenchmarkConfig) -> None:
    assert "fixed" in tiny.operating_points_for("static_threshold")
    assert "fixed" not in tiny.operating_points_for("robust_zscore")


def test_dataset_preparation_describes_the_generated_files(dataset: dict[str, Any]) -> None:
    assert Path(dataset["paths"]["clean"]).is_file()
    assert len(dataset["sha256"]) == 64
    assert dataset["sha256"] != dataset["clean_sha256"] or dataset["events"] == 0
    assert dataset["rows"] > 0


def test_accuracy_task_is_deterministic_and_never_calibrates_on_test(
    tiny: BenchmarkConfig, dataset: dict[str, Any]
) -> None:
    first = accuracy_task(tiny.model_dump_json(), dataset, 1)
    second = accuracy_task(tiny.model_dump_json(), dataset, 1)
    assert first["status"] == "ok", first.get("error")
    assert first["operating_points"] == second["operating_points"]
    points = first["operating_points"]
    assert set(points) == set(tiny.operating_points_for("robust_zscore"))
    assert {p["thresholds"]["fitted_on"] for p in points.values()} <= {"train", "validation"}
    assert points["clean_quantile"]["thresholds"]["source"] == "known_clean"
    assert points["validation_quantile"]["thresholds"]["source"] == "input"
    for result in points.values():
        assert result["test"]["events"]["events"] >= 0
        assert (
            result["clean_test"]["alerts"] >= result["clean_test"]["alerts_in_data_quality_windows"]
        )


def test_a_failing_task_is_recorded_instead_of_raised(
    tiny: BenchmarkConfig, dataset: dict[str, Any]
) -> None:
    broken = BenchmarkConfig.model_validate(
        {**TINY, "detectors": [{"type": "static_threshold", "limits": {"no_such_metric": 1.0}}]}
    )
    record = accuracy_task(broken.model_dump_json(), dataset, 0)
    assert record["status"] == "error"
    assert "none of the static limits apply" in record["error"]


def test_timing_task_reports_medians_throughput_and_peak_memory(
    tiny: BenchmarkConfig, dataset: dict[str, Any]
) -> None:
    features = timing_task(tiny.model_dump_json(), dataset, None)
    detector = timing_task(tiny.model_dump_json(), dataset, 1)
    assert features["status"] == "ok", features.get("error")
    assert detector["status"] == "ok", detector.get("error")
    assert set(features["phases"]) == {"features"}
    assert set(detector["phases"]) == {"fit", "score"}
    score = detector["phases"]["score"]
    assert len(score["seconds"]) == 2
    assert score["rows_per_second"] > 0
    assert detector["memory"]["peak_rss_bytes"] > 0
    assert detector["memory"]["method"]


def test_environment_and_memory_probes() -> None:
    env = environment()
    assert {"cpu", "python", "packages", "threadpools", "platform"} <= set(env)
    assert env["packages"]["numpy"]
    assert isinstance(reset_peak_rss(), bool)
    peak, method = peak_rss_bytes()
    assert peak > 0
    assert method in ("VmHWM", "ru_maxrss")


def _record(seed: int, detector: str, recall: float) -> dict[str, Any]:
    return {
        "kind": "accuracy",
        "status": "ok",
        "preset": "easy",
        "seed": seed,
        "detector": detector,
        "operating_points": {
            "target_fpr": {"test": {"events": {"recall": recall}}, "clean_test": {}}
        },
    }


def test_summary_uses_sample_standard_deviation_over_seeds() -> None:
    records = [_record(0, "a", 0.5), _record(1, "a", 1.0), _record(0, "b", 0.25)]
    summary = summarise(flatten(records))
    recall = summary[summary["metric"] == "event_recall"].set_index("detector")
    assert recall.loc["a", "mean"] == pytest.approx(0.75)
    assert recall.loc["a", "std"] == pytest.approx(pd.Series([0.5, 1.0]).std(ddof=1))
    assert recall.loc["a", "n"] == 2
    assert recall.loc["b", "n"] == 1
    assert pd.isna(recall.loc["b", "std"])
