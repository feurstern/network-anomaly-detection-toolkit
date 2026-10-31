"""Pipeline rules: what is fitted on which segment, and saved models reproduce results."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from network_anomaly.detectors.base import Detector
from network_anomaly.errors import ArtifactError, ConfigError, DataValidationError
from network_anomaly.features.builder import FeatureFrame
from network_anomaly.ingestion.validation import ValidatedMetrics, load_metrics
from network_anomaly.pipelines import DetectionConfig, ModelBundle, apply_bundle, run_detection
from network_anomaly.synthetic.generator import GeneratorConfig, generate_dataset, write_dataset

FIVE_MINUTES = pd.Timedelta("5min")


@pytest.fixture(scope="module")
def data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "demo.csv"
    dataset = generate_dataset(GeneratorConfig(seed=21, hosts=3, days=21))
    write_dataset(dataset, path, with_clean_twin=True)
    return path


@pytest.fixture(scope="module")
def validated(data_path: Path) -> ValidatedMetrics:
    return load_metrics(data_path)


def _config(**fields: Any) -> DetectionConfig:
    return DetectionConfig.model_validate({"detector": {"type": "robust_zscore"}, **fields})


def test_detector_and_features_only_see_training_rows(
    validated: ValidatedMetrics, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    original = Detector.fit

    def spy(self: Detector, ff: FeatureFrame) -> Detector:
        seen["role"], seen["latest"] = ff.role, ff.timestamps.max()
        return original(self, ff)

    monkeypatch.setattr(Detector, "fit", spy)
    result = run_detection(validated, _config(), FIVE_MINUTES)
    split = result.bundle.split
    assert seen["role"] == "train"
    assert seen["latest"] < split.train_end
    assert pd.Timestamp(result.bundle.features.training_end) < split.train_end


def test_default_thresholds_are_calibrated_on_the_validation_segment(
    validated: ValidatedMetrics,
) -> None:
    result = run_detection(validated, _config(), FIVE_MINUTES)
    thresholds = result.bundle.thresholds
    assert (thresholds.strategy, thresholds.fitted_on) == ("clean_quantile", "validation")
    assert pd.Timestamp(thresholds.window_start) == result.bundle.split.train_end
    assert pd.Timestamp(thresholds.window_end) == result.bundle.split.validation_end
    counts = result.segment_counts()
    assert all(counts[segment]["rows"] > 0 for segment in ("train", "validation", "test"))


def test_static_limits_default_to_their_natural_fixed_threshold(
    validated: ValidatedMetrics,
) -> None:
    config = DetectionConfig.model_validate({"detector": {"type": "static_threshold"}})
    thresholds = run_detection(validated, config, FIVE_MINUTES).bundle.thresholds
    assert (thresholds.strategy, thresholds.global_threshold) == ("fixed", 1.0)


def test_target_fpr_needs_and_uses_validation_labels(validated: ValidatedMetrics) -> None:
    config = _config(threshold={"strategy": "target_fpr", "target_fpr": 0.01})
    result = run_detection(validated, config, FIVE_MINUTES)
    assert result.bundle.thresholds.params["target_fpr"] == 0.01
    unlabelled = ValidatedMetrics(validated.metrics, None, validated.report)
    with pytest.raises(ConfigError, match="needs anomaly labels"):
        run_detection(unlabelled, config, FIVE_MINUTES)


def test_saved_bundle_reproduces_scores_flags_and_segments(
    validated: ValidatedMetrics, tmp_path: Path
) -> None:
    result = run_detection(validated, _config(), FIVE_MINUTES)
    result.bundle.save(tmp_path / "model")
    reloaded = apply_bundle(validated, ModelBundle.load(tmp_path / "model"))
    np.testing.assert_array_equal(reloaded.scores, result.scores)
    np.testing.assert_array_equal(reloaded.flagged, result.flagged)
    assert (reloaded.segments == result.segments).all()


def test_corrupt_or_unknown_bundles_are_rejected(
    validated: ValidatedMetrics, tmp_path: Path
) -> None:
    run_detection(validated, _config(), FIVE_MINUTES).bundle.save(tmp_path / "model")
    bundle_file = tmp_path / "model" / "bundle.json"
    payload = json.loads(bundle_file.read_text())
    payload["format"] = 99
    bundle_file.write_text(json.dumps(payload))
    with pytest.raises(ArtifactError, match="unsupported bundle format"):
        ModelBundle.load(tmp_path / "model")
    with pytest.raises(ArtifactError, match="cannot load model bundle"):
        ModelBundle.load(tmp_path / "nowhere")


def test_known_clean_copy_calibrates_clean_quantile_on_its_validation_window(
    validated: ValidatedMetrics, data_path: Path
) -> None:
    clean = load_metrics(data_path.with_name("demo.clean.csv"))
    result = run_detection(validated, _config(), FIVE_MINUTES, clean=clean)
    thresholds = result.bundle.thresholds
    assert (thresholds.source, thresholds.fitted_on) == ("known_clean", "validation")
    assert pd.Timestamp(thresholds.window_start) == result.bundle.split.train_end
    assert pd.Timestamp(thresholds.window_end) == result.bundle.split.validation_end
    assert run_detection(validated, _config(), FIVE_MINUTES).bundle.thresholds.source == "input"


def test_a_known_clean_file_must_be_clean_and_only_serves_clean_quantile(
    validated: ValidatedMetrics, data_path: Path
) -> None:
    with pytest.raises(DataValidationError, match="labelled anomalous"):
        run_detection(validated, _config(), FIVE_MINUTES, clean=validated)
    clean = load_metrics(data_path.with_name("demo.clean.csv"))
    config = _config(threshold={"strategy": "target_fpr"})
    with pytest.raises(ConfigError, match="only applies to clean_quantile"):
        run_detection(validated, config, FIVE_MINUTES, clean=clean)
