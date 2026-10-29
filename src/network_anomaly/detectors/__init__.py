"""Anomaly detectors and the registry that builds them from configuration."""

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from network_anomaly.detectors.base import MANIFEST, Detector
from network_anomaly.detectors.sklearn_models import (
    IsolationForestConfig,
    IsolationForestDetector,
    OneClassSVMConfig,
    OneClassSVMDetector,
)
from network_anomaly.detectors.statistical import (
    RobustZScoreConfig,
    RobustZScoreDetector,
    StaticThresholdConfig,
    StaticThresholdDetector,
)
from network_anomaly.errors import ArtifactError, ConfigError

DetectorConfig = Annotated[
    StaticThresholdConfig | RobustZScoreConfig | IsolationForestConfig | OneClassSVMConfig,
    Field(discriminator="type"),
]

DETECTOR_TYPES: dict[str, tuple[type[Detector], type[Any]]] = {
    StaticThresholdDetector.name: (StaticThresholdDetector, StaticThresholdConfig),
    RobustZScoreDetector.name: (RobustZScoreDetector, RobustZScoreConfig),
    IsolationForestDetector.name: (IsolationForestDetector, IsolationForestConfig),
    OneClassSVMDetector.name: (OneClassSVMDetector, OneClassSVMConfig),
}


def default_config(name: str) -> Any:
    """Default configuration for the detector called ``name``."""
    if name not in DETECTOR_TYPES:
        raise ConfigError(
            f"unknown detector {name!r}", hint=f"Available: {', '.join(sorted(DETECTOR_TYPES))}."
        )
    return DETECTOR_TYPES[name][1]()


def create_detector(config: Any) -> Detector:
    """Instantiate the detector described by ``config`` (one of the DetectorConfig models)."""
    detector_class, _ = DETECTOR_TYPES[config.type]
    return detector_class(config)


def load_detector(directory: Path) -> Detector:
    """Load a detector saved with :meth:`Detector.save`, checking its manifest."""
    try:
        manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {directory / MANIFEST}: {exc}") from exc
    kind = manifest.get("kind")
    if kind not in DETECTOR_TYPES:
        raise ArtifactError(f"{directory}: unknown detector kind {kind!r}")
    detector_class, config_class = DETECTOR_TYPES[kind]
    detector = detector_class(config_class.model_validate(manifest["config"]))
    return detector.restore(directory, manifest)


__all__ = [
    "DETECTOR_TYPES",
    "Detector",
    "DetectorConfig",
    "create_detector",
    "default_config",
    "load_detector",
]
