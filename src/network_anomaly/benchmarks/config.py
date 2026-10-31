"""Benchmark configuration: which datasets, detectors and operating points to compare.

Every detector sees the same datasets, splits, features, alerting and matching rules. Each
operating point is fitted without looking at the test segment; the headline is chosen *before*
the run (``thresholds.headline``), never by comparing test results.
"""

import hashlib
from typing import Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from network_anomaly.config import StrictModel, parse_frequency
from network_anomaly.detectors import DetectorConfig, default_config
from network_anomaly.errors import ConfigError
from network_anomaly.evaluation.events import EventMatchingConfig
from network_anomaly.evaluation.splits import SplitConfig
from network_anomaly.explanations.alerts import AlertingConfig
from network_anomaly.features.builder import FeatureConfig
from network_anomaly.pipelines import DetectionConfig
from network_anomaly.synthetic.generator import GeneratorConfig
from network_anomaly.synthetic.scenarios import Preset, ScenarioName

OperatingPoint = Literal[
    "train_quantile", "validation_quantile", "clean_quantile", "target_fpr", "fixed"
]

OPERATING_POINTS: dict[OperatingPoint, str] = {
    "train_quantile": "q-quantile of training scores (unlabelled; in-sample for model detectors)",
    "validation_quantile": "q-quantile of the validation window, assumed normal but containing "
    "the dataset's anomalies (the `detect` default; no labels)",
    "clean_quantile": "q-quantile of a known-clean copy of the validation window (the clean twin)",
    "target_fpr": "(1 - fpr)-quantile of validation rows labelled normal (needs labels)",
    "fixed": "fixed value 1.0, the natural operating point of static limits (static only)",
}

DEFAULT_DETECTORS = ("static_threshold", "robust_zscore", "isolation_forest", "one_class_svm")


class ThresholdSettings(StrictModel):
    quantile: float = Field(default=0.995, gt=0.5, lt=1.0)
    target_fpr: float = Field(default=0.005, gt=0.0, lt=0.5)
    per_host: bool = True
    operating_points: tuple[OperatingPoint, ...] = tuple(OPERATING_POINTS)
    headline: OperatingPoint = Field(
        default="target_fpr",
        description="Operating point used for the headline comparison, fixed before the run.",
    )

    @model_validator(mode="after")
    def _headline_is_run(self) -> Self:
        if self.headline not in self.operating_points:
            raise ValueError(f"headline {self.headline!r} must be one of operating_points")
        if self.headline == "fixed":
            raise ValueError("'fixed' only applies to static limits and cannot be the headline")
        if len(set(self.operating_points)) != len(self.operating_points):
            raise ValueError("operating_points must not repeat")
        return self


class TimingSettings(StrictModel):
    enabled: bool = True
    warmup: int = Field(default=1, ge=0, le=10)
    repeats: int = Field(default=3, ge=1, le=20)


def _default_detectors() -> tuple[DetectorConfig, ...]:
    return tuple(default_config(name) for name in DEFAULT_DETECTORS)


class BenchmarkConfig(StrictModel):
    """A reproducible comparison: presets x seeds x detectors on synthetic data."""

    name: str = "default"
    presets: tuple[Preset, ...] = (Preset.MODERATE,)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    hosts: int = Field(default=20, ge=1, le=500)
    days: float = Field(default=35.0, ge=7.0, le=366.0)
    freq: str = "5min"
    scenarios: tuple[ScenarioName, ...] = tuple(ScenarioName)
    detectors: tuple[DetectorConfig, ...] = Field(default_factory=_default_detectors)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    matching: EventMatchingConfig = Field(default_factory=EventMatchingConfig)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    timing: TimingSettings = Field(default_factory=TimingSettings)

    @field_validator("freq")
    @classmethod
    def _check_freq(cls, value: str) -> str:
        try:
            parse_frequency(value)
        except ConfigError as exc:
            raise ValueError(exc.message) from exc
        return value

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty list without repeats")
        if not self.presets or len(set(self.presets)) != len(self.presets):
            raise ValueError("presets must be a non-empty list without repeats")
        names = [d.type for d in self.detectors]
        if not names or len(set(names)) != len(names):
            raise ValueError("detectors must be a non-empty list with one entry per type")
        if self.split.train_until is not None or self.split.validation_until is not None:
            raise ValueError(
                "benchmarks split by fractions; train_until/validation_until are "
                "specific to one dataset"
            )
        try:  # the generator has its own rules (e.g. the interval must divide a day)
            self.generator_config(self.presets[0], self.seeds[0])
        except ValidationError as exc:
            raise ValueError(f"invalid dataset settings: {exc.errors()[0]['msg']}") from exc
        return self

    def generator_config(self, preset: Preset, seed: int) -> GeneratorConfig:
        return GeneratorConfig(
            seed=seed,
            preset=preset,
            hosts=self.hosts,
            days=self.days,
            freq=self.freq,
            scenarios=self.scenarios,
        )

    def detection_config(self, detector: DetectorConfig) -> DetectionConfig:
        return DetectionConfig(
            detector=detector,
            features=self.features,
            split=self.split,
            alerting=self.alerting,
        )

    def fingerprint(self) -> str:
        """Short hash of the resolved configuration, used to name run directories."""
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()[:8]

    def operating_points_for(self, detector: str) -> tuple[OperatingPoint, ...]:
        """Operating points that apply to a detector (``fixed`` only suits static limits)."""
        return tuple(
            point
            for point in self.thresholds.operating_points
            if point != "fixed" or detector == "static_threshold"
        )
