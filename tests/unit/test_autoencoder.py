"""The optional autoencoder: training history, early stopping, determinism and persistence."""

import importlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from network_anomaly.detectors import load_detector
from network_anomaly.detectors.autoencoder import AutoencoderConfig, AutoencoderDetector
from network_anomaly.errors import ConfigError
from network_anomaly.features.builder import FeatureBuilder, FeatureFrame
from network_anomaly.ingestion.validation import MetricsFrame

FIVE_MINUTES = pd.Timedelta("5min")
TRAIN_END = pd.Timestamp("2026-01-08T00:00:00Z")


@pytest.fixture(scope="module")
def frames(make_metrics: Callable[..., MetricsFrame]) -> tuple[FeatureFrame, FeatureFrame]:
    pytest.importorskip("torch")
    metrics = make_metrics(seed=11, days=5)
    builder = FeatureBuilder().fit(metrics.until(TRAIN_END, FIVE_MINUTES), FIVE_MINUTES)
    ff = builder.transform(metrics)
    return ff.select((ff.timestamps < TRAIN_END).to_numpy(), "train"), ff


def _fit(train: FeatureFrame, **settings: object) -> AutoencoderDetector:
    return AutoencoderDetector(AutoencoderConfig.model_validate(settings)).fit(train)


def test_training_records_history_and_stops_early(
    frames: tuple[FeatureFrame, FeatureFrame],
) -> None:
    detector = _fit(frames[0], epochs=300, patience=2)
    assert detector.best_epoch is not None
    assert 1 <= detector.best_epoch <= len(detector.history) < 300
    assert {"epoch", "train_loss", "validation_loss"} <= set(detector.history[0])
    best = min(h["validation_loss"] for h in detector.history)
    assert detector.history[detector.best_epoch - 1]["validation_loss"] == best


def test_training_is_reproducible_and_survives_save_and_load(
    frames: tuple[FeatureFrame, FeatureFrame], tmp_path: Path
) -> None:
    train, ff = frames
    first, second = _fit(train, epochs=5), _fit(train, epochs=5)
    np.testing.assert_array_equal(first.score(ff), second.score(ff))
    first.save(tmp_path)
    assert {"manifest.json", "model.pt", "state.json"} <= {p.name for p in tmp_path.iterdir()}
    restored = load_detector(tmp_path)
    np.testing.assert_array_equal(restored.score(ff), first.score(ff))
    assert isinstance(restored, AutoencoderDetector)
    assert restored.history == first.history


def test_per_metric_terms_add_up_to_the_score(frames: tuple[FeatureFrame, FeatureFrame]) -> None:
    train, ff = frames
    detector = _fit(train, epochs=5)
    positions = np.flatnonzero(detector.rows(ff))[:200]
    terms = detector.metric_terms(ff, positions)
    assert terms is not None
    np.testing.assert_allclose(terms.sum(axis=1), detector.score(ff)[positions], rtol=1e-6)
    assert detector.term_method == "reconstruction_error"


def test_a_missing_torch_install_gives_an_actionable_error(
    frames: tuple[FeatureFrame, FeatureFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = importlib.import_module

    def without_torch(name: str, package: str | None = None) -> object:
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", without_torch)
    with pytest.raises(ConfigError, match="uv sync --extra torch"):
        AutoencoderDetector().fit(frames[0])
