"""End-to-end detection: fit on the training segment, calibrate, then score every row.

This module is shared by the CLI and the benchmark runner so that both follow the same rules:

* features and the detector are fitted on training rows only;
* thresholds are fitted on the segment their strategy requires, never on test rows;
* every output row carries its segment, so evaluation can exclude in-sample rows.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pydantic import Field, ValidationError, model_validator

from network_anomaly.config import StrictModel
from network_anomaly.detectors import Detector, DetectorConfig, create_detector, load_detector
from network_anomaly.errors import ArtifactError
from network_anomaly.evaluation.splits import SplitConfig, SplitPlan, plan_split
from network_anomaly.explanations.alerts import AlertingConfig
from network_anomaly.features.builder import (
    FeatureBuilder,
    FeatureConfig,
    FeatureFrame,
    FeatureState,
)
from network_anomaly.ingestion.validation import LabelFrame, ValidatedMetrics
from network_anomaly.preprocessing import SOURCE_TIMESTAMP
from network_anomaly.schemas import TIMESTAMP_COLUMN
from network_anomaly.scoring.thresholds import (
    CALIBRATION_ROLE,
    FixedThresholdConfig,
    HostThresholds,
    QuantileThresholdConfig,
    TargetFprConfig,
    ThresholdConfig,
    fit_thresholds,
)

logger = logging.getLogger(__name__)

BUNDLE_FILE = "bundle.json"
BUNDLE_FORMAT = 1
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
AnyThreshold = FixedThresholdConfig | QuantileThresholdConfig | TargetFprConfig


class DetectionConfig(StrictModel):
    """Everything needed to fit and run one detector; loadable from YAML."""

    detector: DetectorConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    threshold: ThresholdConfig | None = Field(
        default=None,
        description="Operating point; defaults to fixed 1.0 for static limits, otherwise "
        "the 0.995 quantile of validation scores.",
    )
    split: SplitConfig = Field(default_factory=SplitConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)

    @model_validator(mode="after")
    def _one_operating_point(self) -> Self:
        if getattr(self.detector, "contamination", None) is not None and self.threshold is not None:
            raise ValueError("set either detector.contamination or threshold, not both")
        return self

    def resolved_threshold(self) -> AnyThreshold:
        if self.threshold is not None:
            return self.threshold
        contamination = getattr(self.detector, "contamination", None)
        if contamination is not None:
            return QuantileThresholdConfig(strategy="train_quantile", quantile=1.0 - contamination)
        if self.detector.type == "static_threshold":
            return FixedThresholdConfig(value=1.0)
        return QuantileThresholdConfig()


@dataclass(frozen=True)
class ModelBundle:
    """A deployable model: feature statistics, fitted detector, thresholds and split windows."""

    config: DetectionConfig
    features: FeatureState
    detector: Detector
    thresholds: HostThresholds
    split: SplitPlan

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": BUNDLE_FORMAT,
            "config": self.config.model_dump(mode="json"),
            "features": self.features.model_dump(mode="json"),
            "thresholds": self.thresholds.model_dump(mode="json"),
            "split": self.split.as_dict(),
        }
        (directory / BUNDLE_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.detector.save(directory / "detector")

    @classmethod
    def load(cls, directory: Path) -> "ModelBundle":
        try:
            payload: dict[str, Any] = json.loads((directory / BUNDLE_FILE).read_text("utf-8"))
            if payload.get("format") != BUNDLE_FORMAT:
                raise ArtifactError(
                    f"{directory}: unsupported bundle format {payload.get('format')}"
                )
            split = {key: pd.Timestamp(value) for key, value in payload["split"].items()}
            return cls(
                config=DetectionConfig.model_validate(payload["config"]),
                features=FeatureState.model_validate(payload["features"]),
                detector=load_detector(directory / "detector"),
                thresholds=HostThresholds.model_validate(payload["thresholds"]),
                split=SplitPlan(**split),
            )
        except (OSError, KeyError, json.JSONDecodeError, ValidationError) as exc:
            raise ArtifactError(f"cannot load model bundle from {directory}: {exc}") from exc


@dataclass(frozen=True)
class DetectionResult:
    """Scores and flags for every grid row, with the model that produced them."""

    features: FeatureFrame
    scores: FloatArray
    flagged: BoolArray
    segments: npt.NDArray[np.object_]
    bundle: ModelBundle

    def segment_counts(self) -> dict[str, dict[str, int]]:
        observed = self.features.observed
        counts: dict[str, dict[str, int]] = {}
        for segment in ("train", "validation", "test"):
            rows = observed & (self.segments == segment)
            counts[segment] = {
                "rows": int(rows.sum()),
                "scored": int((rows & np.isfinite(self.scores)).sum()),
                "flagged": int((rows & self.flagged).sum()),
            }
        return counts


def align_labels(ff: FeatureFrame, labels: LabelFrame | None) -> npt.NDArray[np.int8] | None:
    """``is_anomaly`` for each feature row (0 for gap rows), or None without labels."""
    if labels is None or "is_anomaly" not in labels.data.columns:
        return None
    entity = ff.entity_column
    keys = ff.keys[[entity, SOURCE_TIMESTAMP]].rename(columns={SOURCE_TIMESTAMP: TIMESTAMP_COLUMN})
    merged = keys.merge(
        labels.data[[TIMESTAMP_COLUMN, entity, "is_anomaly"]],
        on=[entity, TIMESTAMP_COLUMN],
        how="left",
        validate="many_to_one",
    )
    return merged["is_anomaly"].fillna(0).to_numpy(dtype=np.int8)


def _calibrate(
    config: AnyThreshold,
    scores: FloatArray,
    ff: FeatureFrame,
    segments: npt.NDArray[np.object_],
    split: SplitPlan,
    labels: npt.NDArray[np.int8] | None,
) -> HostThresholds:
    role = CALIBRATION_ROLE[config.strategy]
    if role is None:
        return fit_thresholds(config, scores, ff.entities, role="train")
    rows = (segments == role) & ff.observed
    window = (
        (split.start, split.train_end)
        if role == "train"
        else (split.train_end, split.validation_end)
    )
    return fit_thresholds(
        config,
        scores[rows],
        ff.entities[rows],
        role=role,
        labels=None if labels is None else labels[rows],
        window=(window[0].to_pydatetime(), window[1].to_pydatetime()),
    )


def run_detection(
    validated: ValidatedMetrics, config: DetectionConfig, frequency: pd.Timedelta
) -> DetectionResult:
    """Fit features and the detector on the training segment, calibrate, score every row."""
    metrics = validated.metrics
    grid_times = metrics.data[TIMESTAMP_COLUMN].dt.round(to_offset(frequency))
    split = plan_split(grid_times, config.split, frequency)
    builder = FeatureBuilder(config.features).fit(
        metrics.until(split.train_end, frequency), frequency
    )
    ff = builder.transform(metrics)
    segments = split.segment(ff.timestamps)
    detector = create_detector(config.detector).fit(ff.select(segments == "train", "train"))
    scores = detector.score(ff)
    labels = align_labels(ff, validated.labels)
    thresholds = _calibrate(config.resolved_threshold(), scores, ff, segments, split, labels)
    flagged = thresholds.apply(scores, ff.entities)
    bundle = ModelBundle(config, builder.state, detector, thresholds, split)
    return DetectionResult(ff, scores, flagged, segments, bundle)


def apply_bundle(validated: ValidatedMetrics, bundle: ModelBundle) -> DetectionResult:
    """Score data with a saved model. Segments follow the model's own training windows."""
    ff = FeatureBuilder.from_state(bundle.features).transform(validated.metrics)
    scores = bundle.detector.score(ff)
    flagged = bundle.thresholds.apply(scores, ff.entities)
    return DetectionResult(ff, scores, flagged, bundle.split.segment(ff.timestamps), bundle)
