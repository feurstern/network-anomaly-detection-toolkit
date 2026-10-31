"""End-to-end detection: fit on the training segment, calibrate, then score every row.

This module is shared by the CLI and the benchmark runner so that both follow the same rules:

* features and the detector are fitted on training rows only;
* thresholds are fitted on the segment their strategy requires, never on test rows - or, for
  ``clean_quantile``, optionally on a known-clean copy of the validation window;
* every output row carries its segment, so evaluation can exclude in-sample rows.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pydantic import Field, ValidationError, model_validator

from network_anomaly.config import StrictModel
from network_anomaly.detectors import Detector, DetectorConfig, create_detector, load_detector
from network_anomaly.errors import ArtifactError, ConfigError, DataValidationError
from network_anomaly.evaluation.splits import SplitConfig, SplitPlan, plan_split
from network_anomaly.explanations.alerts import AlertingConfig, assemble_alerts
from network_anomaly.features.builder import (
    FeatureBuilder,
    FeatureConfig,
    FeatureFrame,
    FeatureState,
)
from network_anomaly.ingestion.validation import LabelFrame, MetricsFrame, ValidatedMetrics
from network_anomaly.preprocessing import SOURCE_TIMESTAMP
from network_anomaly.schemas import TIMESTAMP_COLUMN, Alert
from network_anomaly.scoring.thresholds import (
    CALIBRATION_ROLE,
    FixedThresholdConfig,
    HostThresholds,
    QuantileThresholdConfig,
    ScoreReference,
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
    """A deployable model: feature statistics, fitted detector, thresholds and split windows.

    ``reference`` summarises validation-period scores so alerts can say how rare their score was.
    """

    config: DetectionConfig
    features: FeatureState
    detector: Detector
    thresholds: HostThresholds
    split: SplitPlan
    reference: ScoreReference | None = None

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": BUNDLE_FORMAT,
            "config": self.config.model_dump(mode="json"),
            "features": self.features.model_dump(mode="json"),
            "thresholds": self.thresholds.model_dump(mode="json"),
            "split": self.split.as_dict(),
            "reference": None if self.reference is None else self.reference.model_dump(mode="json"),
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
            reference = payload.get("reference")
            return cls(
                config=DetectionConfig.model_validate(payload["config"]),
                features=FeatureState.model_validate(payload["features"]),
                detector=load_detector(directory / "detector"),
                thresholds=HostThresholds.model_validate(payload["thresholds"]),
                split=SplitPlan(**split),
                reference=None if reference is None else ScoreReference.model_validate(reference),
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

    def alerts(self) -> list[Alert]:
        """Explainable alerts for every segment (each alert records its segment)."""
        bundle = self.bundle
        return assemble_alerts(
            self.features,
            self.scores,
            self.flagged,
            self.segments,
            detector=bundle.detector,
            thresholds=bundle.thresholds,
            feature_medians=bundle.features.feature_medians,
            frequency=bundle.features.frequency,
            config=bundle.config.alerting,
            reference=bundle.reference,
        )

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


def _window(split: SplitPlan, role: str) -> tuple[datetime, datetime]:
    bounds = (
        (split.start, split.train_end)
        if role == "train"
        else (split.train_end, split.validation_end)
    )
    return bounds[0].to_pydatetime(), bounds[1].to_pydatetime()


def calibrate(
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
    return fit_thresholds(
        config,
        scores[rows],
        ff.entities[rows],
        role=role,
        labels=None if labels is None else labels[rows],
        window=_window(split, role),
    )


@dataclass(frozen=True)
class FittedScores:
    """Features and a detector fitted on the training segment, with scores for every grid row."""

    builder: FeatureBuilder
    detector: Detector
    split: SplitPlan
    features: FeatureFrame
    segments: npt.NDArray[np.object_]
    scores: FloatArray

    def score_other(
        self, metrics: MetricsFrame
    ) -> tuple[FeatureFrame, FloatArray, npt.NDArray[np.object_]]:
        """Features, scores and segments for another dataset scored with the same fitted model."""
        ff = self.builder.transform(metrics)
        return ff, self.detector.score(ff), self.split.segment(ff.timestamps)


def split_for(metrics: MetricsFrame, config: SplitConfig, frequency: pd.Timedelta) -> SplitPlan:
    """Chronological cut times for ``metrics`` on its sampling grid."""
    grid_times = metrics.data[TIMESTAMP_COLUMN].dt.round(to_offset(frequency))
    return plan_split(grid_times, config, frequency)


def fit_and_score(
    metrics: MetricsFrame, config: DetectionConfig, frequency: pd.Timedelta
) -> FittedScores:
    """Plan the split, fit features and the detector on training rows, score every row."""
    split = split_for(metrics, config.split, frequency)
    builder = FeatureBuilder(config.features).fit(
        metrics.until(split.train_end, frequency), frequency
    )
    ff = builder.transform(metrics)
    segments = split.segment(ff.timestamps)
    detector = create_detector(config.detector).fit(ff.select(segments == "train", "train"))
    return FittedScores(builder, detector, split, ff, segments, detector.score(ff))


def calibrate_on_clean(
    fitted: FittedScores, config: AnyThreshold, clean: ValidatedMetrics
) -> tuple[HostThresholds, FloatArray]:
    """Fit ``clean_quantile`` thresholds on a known-clean copy of the validation window.

    Only the clean file's rows inside the validation window are used (never test-period rows);
    its earlier rows only serve as history for the causal baselines. Returns the thresholds and
    the clean validation scores.
    """
    if config.strategy != "clean_quantile":
        raise ConfigError(
            f"a known-clean validation file only applies to clean_quantile, not {config.strategy}",
            hint="Use --threshold-strategy clean_quantile with --clean-validation.",
        )
    ff, scores, segments = fitted.score_other(clean.metrics)
    labels = align_labels(ff, clean.labels)
    rows = (segments == "validation") & ff.observed
    if labels is not None and labels[rows].any():
        raise DataValidationError(
            f"{clean.metrics.source}: {int(labels[rows].sum())} row(s) in the validation window "
            "are labelled anomalous; a known-clean file must not contain anomalies"
        )
    thresholds = fit_clean_thresholds(
        config, ff, scores, segments, fitted.split, source=clean.metrics.source
    )
    return thresholds, scores[rows]


def fit_clean_thresholds(
    config: AnyThreshold,
    ff: FeatureFrame,
    scores: FloatArray,
    segments: npt.NDArray[np.object_],
    split: SplitPlan,
    *,
    source: str,
) -> HostThresholds:
    """Thresholds from the validation-window rows of an already scored known-clean dataset."""
    rows = (segments == "validation") & ff.observed
    start, end = _window(split, "validation")
    if not np.isfinite(scores[rows]).any():
        raise DataValidationError(
            f"{source} has no scorable rows in the validation window "
            f"({start.isoformat()} to {end.isoformat()})",
            hint="Use a clean copy of the same period, e.g. <name>.clean.csv from "
            "`generate --with-clean-twin`.",
        )
    thresholds = fit_thresholds(
        config, scores[rows], ff.entities[rows], role="validation", window=(start, end)
    )
    return thresholds.model_copy(update={"source": "known_clean"})


def run_detection(
    validated: ValidatedMetrics,
    config: DetectionConfig,
    frequency: pd.Timedelta,
    *,
    clean: ValidatedMetrics | None = None,
) -> DetectionResult:
    """Fit features and the detector on the training segment, calibrate, score every row.

    With ``clean``, ``clean_quantile`` thresholds are fitted on that known-clean copy of the
    validation window instead of the input's own (possibly contaminated) validation rows.
    """
    fitted = fit_and_score(validated.metrics, config, frequency)
    ff, segments, scores = fitted.features, fitted.segments, fitted.scores
    threshold_config = config.resolved_threshold()
    if clean is None:
        labels = align_labels(ff, validated.labels)
        thresholds = calibrate(threshold_config, scores, ff, segments, fitted.split, labels)
        reference_scores = scores[(segments == "validation") & ff.observed]
    else:
        thresholds, reference_scores = calibrate_on_clean(fitted, threshold_config, clean)
    flagged = thresholds.apply(scores, ff.entities)
    reference = ScoreReference.from_scores(reference_scores, window="validation")
    bundle = ModelBundle(
        config, fitted.builder.state, fitted.detector, thresholds, fitted.split, reference
    )
    return DetectionResult(ff, scores, flagged, segments, bundle)


def apply_bundle(validated: ValidatedMetrics, bundle: ModelBundle) -> DetectionResult:
    """Score data with a saved model. Segments follow the model's own training windows."""
    ff = FeatureBuilder.from_state(bundle.features).transform(validated.metrics)
    scores = bundle.detector.score(ff)
    flagged = bundle.thresholds.apply(scores, ff.entities)
    return DetectionResult(ff, scores, flagged, bundle.split.segment(ff.timestamps), bundle)
