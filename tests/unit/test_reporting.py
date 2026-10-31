"""Report figures render headless and reproducibly; the Markdown report states its caveats."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from network_anomaly.evaluation.point import point_metrics
from network_anomaly.evaluation.runner import evaluate_inputs, load_evaluation_inputs
from network_anomaly.ingestion.validation import load_metrics
from network_anomaly.pipelines import DetectionConfig, run_detection
from network_anomaly.reporting.export import read_alerts, write_alerts, write_detections
from network_anomaly.reporting.markdown import write_evaluation_report
from network_anomaly.reporting.plots import (
    confusion_matrix_figure,
    pr_curve_figure,
    save_figure,
    timeseries_figure,
)
from network_anomaly.synthetic.generator import GeneratorConfig, generate_dataset, write_dataset

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
STEP = pd.Timedelta("5min")
T0 = pd.Timestamp("2026-01-05T00:00:00Z")


def _labels_and_scores() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = (rng.random(500) < 0.05).astype(np.int8)
    scores = rng.normal(0.0, 1.0, 500) + 3.0 * labels
    return labels, scores, scores >= 2.0


def _timeseries() -> object:
    times = pd.Series(T0 + STEP * np.r_[0:100, 110:288])  # a gap of ten intervals
    rng = np.random.default_rng(1)
    scores = rng.normal(0.4, 0.05, len(times))
    flagged = scores > 0.5
    return timeseries_figure(
        entity="web-01",
        detector="robust_zscore",
        times=times,
        scores=scores,
        thresholds=np.full(len(times), 0.5),
        flagged=flagged,
        metrics={"http_5xx_rate": rng.random(len(times)) * 0.01},
        frequency=STEP,
        window=(T0, T0 + 288 * STEP),
        events=[(T0 + 40 * STEP, T0 + 50 * STEP)],
        alerts=[(T0 + 41 * STEP, T0 + 48 * STEP)],
    )


def test_figures_render_to_png_and_are_byte_identical_on_rerun(tmp_path: Path) -> None:
    labels, scores, flagged = _labels_and_scores()
    point = point_metrics(labels, flagged, scores)
    for name, build in {
        "pr": lambda: pr_curve_figure(labels, scores, point=point, detector="test"),
        "confusion": lambda: confusion_matrix_figure(point, detector="test"),
        "timeseries": _timeseries,
    }.items():
        first = save_figure(build(), tmp_path / f"{name}-1.png").read_bytes()
        second = save_figure(build(), tmp_path / f"{name}-2.png").read_bytes()
        assert first.startswith(PNG_SIGNATURE)
        assert len(first) > 10_000
        assert first == second


def test_pr_curve_is_skipped_when_only_one_class_is_present() -> None:
    labels = np.zeros(10, dtype=np.int8)
    scores = np.arange(10.0)
    point = point_metrics(labels, scores > 5, scores)
    assert pr_curve_figure(labels, scores, point=point, detector="test") is None


def test_plots_work_without_a_display_and_never_load_pyplot(tmp_path: Path) -> None:
    script = (
        "import sys, numpy as np\n"
        "from pathlib import Path\n"
        "from network_anomaly.evaluation.point import point_metrics\n"
        "from network_anomaly.reporting.plots import confusion_matrix_figure, save_figure\n"
        "p = point_metrics(np.array([0, 1], dtype=np.int8), np.array([0, 1], bool), "
        "np.array([0.0, 1.0]))\n"
        "out = Path(sys.argv[1]) / 'c.png'\n"
        "save_figure(confusion_matrix_figure(p, detector='t'), out)\n"
        "print('matplotlib.pyplot' in sys.modules)\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["MPLBACKEND"] = "TkAgg"  # a GUI backend: must not matter, pyplot is never used
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"
    assert (tmp_path / "c.png").read_bytes().startswith(PNG_SIGNATURE)


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("report")
    data = directory / "demo.csv"
    write_dataset(generate_dataset(GeneratorConfig(seed=5, hosts=3, days=14)), data)
    config = DetectionConfig.model_validate({"detector": {"type": "robust_zscore"}})
    result = run_detection(load_metrics(data), config, STEP)
    predictions, _ = write_detections(result, directory / "detections.csv", data)
    write_alerts(result.alerts(), directory / "detections.alerts.jsonl")
    return data, predictions


def test_evaluation_report_writes_metrics_figures_and_caveats(
    evaluated: tuple[Path, Path], tmp_path: Path
) -> None:
    data, predictions = evaluated
    inputs = load_evaluation_inputs(data, predictions)
    outcome = evaluate_inputs(inputs)
    alerts = read_alerts(predictions.with_name("detections.alerts.jsonl"))
    written = write_evaluation_report(
        tmp_path, inputs, outcome, input_path=data, predictions_path=predictions, alerts=alerts
    )
    assert {"metrics", "report", "confusion_matrix", "timeseries"} <= set(written)
    text = written["report"].read_text()
    for heading in ("## Event level", "## Point level", "## Alerts in the evaluated window"):
        assert heading in text
    assert "not probabilities" in text
    assert "**Synthetic data.**" in text
    assert "In-sample" not in text
    for figure in written.values():
        assert figure.is_file()


def test_in_sample_reports_carry_a_warning(evaluated: tuple[Path, Path], tmp_path: Path) -> None:
    data, predictions = evaluated
    inputs = load_evaluation_inputs(data, predictions)
    outcome = evaluate_inputs(inputs, segments=("train",), allow_in_sample=True)
    written = write_evaluation_report(
        tmp_path, inputs, outcome, input_path=data, predictions_path=predictions
    )
    text = written["report"].read_text()
    assert "**In-sample.**" in text
    assert "## Alerts in the evaluated window" not in text  # no alerts file given
