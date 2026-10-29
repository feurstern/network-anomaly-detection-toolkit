"""Multivariate detectors from scikit-learn: Isolation Forest and One-Class SVM.

Both see one pooled model across hosts, trained on *host-normalised* features (robust z-scores
against each host's own causal baselines, see the feature builder). Raw levels and host IDs are
never inputs: either would make every host unseen in training look anomalous. Features are
compressed with ``sign(z) * log1p(|z|)`` so a handful of extreme values cannot dominate the
geometry.

Scores are the negated ``score_samples`` of the underlying estimator, so higher always means more
anomalous. scikit-learn's own ``predict``/``contamination`` cut-offs are never used for decisions;
thresholds come from :mod:`network_anomaly.scoring.thresholds`.
"""

import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from pydantic import Field
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from network_anomaly.config import StrictModel
from network_anomaly.detectors.base import BoolArray, Detector, FloatArray
from network_anomaly.errors import ArtifactError, NotFittedError
from network_anomaly.features.builder import FeatureFrame

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
ModelFeatureKind = Literal["z_local", "z_season", "roc", "calendar"]
DEFAULT_KINDS: tuple[ModelFeatureKind, ...] = ("z_local", "z_season", "roc")


def compress(values: FloatArray) -> FloatArray:
    """Signed log compression: keeps sign and order, tames extreme deviations."""
    result: FloatArray = np.sign(values) * np.log1p(np.abs(values))
    return result


class IsolationForestConfig(StrictModel):
    type: Literal["isolation_forest"] = "isolation_forest"
    n_estimators: int = Field(default=200, ge=10, le=5000)
    max_samples: int = Field(default=256, ge=16, description="Rows drawn to grow each tree.")
    max_features: float = Field(default=1.0, gt=0.0, le=1.0)
    random_state: int = 0
    contamination: float | None = Field(
        default=None,
        gt=0.0,
        lt=0.5,
        description="Expected share of anomalies in training data. Only used to pick a "
        "train_quantile threshold (q = 1 - contamination) when no threshold is configured.",
    )
    feature_kinds: tuple[ModelFeatureKind, ...] = DEFAULT_KINDS


class OneClassSVMConfig(StrictModel):
    type: Literal["one_class_svm"] = "one_class_svm"
    nu: float = Field(default=0.02, gt=0.0, le=1.0, description="Upper bound on training outliers.")
    gamma: Literal["scale", "auto"] | float = "scale"
    max_train_samples: int = Field(
        default=10_000,
        ge=100,
        description="Training subsample size: kernel methods cost O(n^2) memory and O(n^2..n^3) "
        "time, so training uses a host-stratified sample of at most this many rows.",
    )
    random_state: int = 0
    feature_kinds: tuple[ModelFeatureKind, ...] = DEFAULT_KINDS


class _SklearnDetector(Detector):
    """Shared persistence for detectors whose state is a picklable estimator."""

    def _load_joblib(self, directory: Path) -> object:
        try:
            return joblib.load(directory / MODEL_FILE)
        except (OSError, EOFError, ValueError) as exc:
            raise ArtifactError(f"cannot read {directory / MODEL_FILE}: {exc}") from exc


class IsolationForestDetector(_SklearnDetector):
    """Isolation Forest on compressed host-normalised features.

    Explanation note: trees isolate points along random features, so the model has no additive
    per-feature attribution; alerts explain IF scores by perturbing one metric at a time.
    """

    name = "isolation_forest"
    config: IsolationForestConfig

    def __init__(self, config: IsolationForestConfig | None = None) -> None:
        super().__init__(config or IsolationForestConfig())
        self.feature_kinds = tuple(self.config.feature_kinds)
        self.model: IsolationForest | None = None

    def _fit(self, ff: FeatureFrame, rows: BoolArray) -> None:
        features = compress(self.matrix(ff, rows))
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=min(self.config.max_samples, len(features)),
            max_features=self.config.max_features,
            random_state=self.config.random_state,
            n_jobs=1,
        ).fit(features)

    def _score(self, ff: FeatureFrame, rows: BoolArray) -> FloatArray:
        if self.model is None:
            raise NotFittedError("isolation forest is not fitted")
        scores: FloatArray = -self.model.score_samples(compress(self.matrix(ff, rows)))
        return scores

    def _state(self, directory: Path) -> None:
        joblib.dump(self.model, directory / MODEL_FILE)

    def _restore(self, directory: Path) -> None:
        model = self._load_joblib(directory)
        if not isinstance(model, IsolationForest):
            raise ArtifactError(f"{directory / MODEL_FILE} does not hold an IsolationForest")
        self.model = model


def stratified_sample(entities: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    """Row positions of a sample of at most ``limit`` rows, proportional per entity."""
    if len(entities) <= limit:
        return np.arange(len(entities))
    chosen = []
    for entity in sorted(set(entities.tolist())):
        positions = np.flatnonzero(entities == entity)
        take = max(1, round(limit * len(positions) / len(entities)))
        chosen.append(rng.choice(positions, size=min(take, len(positions)), replace=False))
    return np.sort(np.concatenate(chosen))


class OneClassSVMDetector(_SklearnDetector):
    """RBF One-Class SVM on robust-scaled, compressed host-normalised features.

    Needs feature scaling (the RBF kernel is distance-based), so a RobustScaler is fitted on the
    training sample. Training cost grows quadratically with rows, hence the capped subsample;
    scoring cost grows with the number of support vectors, which is logged after fitting.
    """

    name = "one_class_svm"
    config: OneClassSVMConfig

    def __init__(self, config: OneClassSVMConfig | None = None) -> None:
        super().__init__(config or OneClassSVMConfig())
        self.feature_kinds = tuple(self.config.feature_kinds)
        self.scaler: RobustScaler | None = None
        self.model: OneClassSVM | None = None

    def _fit(self, ff: FeatureFrame, rows: BoolArray) -> None:
        features = compress(self.matrix(ff, rows))
        rng = np.random.default_rng(self.config.random_state)
        sample = stratified_sample(ff.entities[rows], self.config.max_train_samples, rng)
        self.scaler = RobustScaler().fit(features[sample])
        self.model = OneClassSVM(kernel="rbf", nu=self.config.nu, gamma=self.config.gamma)
        self.model.fit(self.scaler.transform(features[sample]))
        logger.info(
            "fitted one-class svm",
            extra={"train_rows": len(sample), "support_vectors": len(self.model.support_)},
        )

    def _score(self, ff: FeatureFrame, rows: BoolArray) -> FloatArray:
        if self.model is None or self.scaler is None:
            raise NotFittedError("one-class svm is not fitted")
        features = self.scaler.transform(compress(self.matrix(ff, rows)))
        scores: FloatArray = -self.model.score_samples(features)
        return scores

    def _state(self, directory: Path) -> None:
        joblib.dump({"scaler": self.scaler, "model": self.model}, directory / MODEL_FILE)

    def _restore(self, directory: Path) -> None:
        state = self._load_joblib(directory)
        if not isinstance(state, dict) or not isinstance(state.get("model"), OneClassSVM):
            raise ArtifactError(f"{directory / MODEL_FILE} does not hold a One-Class SVM")
        self.scaler, self.model = state["scaler"], state["model"]
