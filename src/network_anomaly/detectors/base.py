"""The detector interface shared by every algorithm.

A detector turns a :class:`~network_anomaly.features.builder.FeatureFrame` into one anomaly score
per row. Contract for every implementation:

* **Score direction:** higher always means more anomalous. Scores are unitless and are *not*
  probabilities; compare them only with thresholds fitted for the same detector.
* **Unscorable rows** (gap rows, warm-up) get NaN and are never flagged.
* **Leakage guard:** ``fit`` only accepts frames tagged ``role="train"``.
* **Persistence:** ``save`` writes a directory with ``manifest.json`` (kind, version, parameters,
  feature schema, library versions) plus a state file; ``load_detector`` verifies the manifest.
  Pickled scikit-learn state can execute code when loaded: only load models you created.

Subclasses implement ``_fit``, ``_score``, ``_state`` and ``_restore``. For explanations they
either expose exact per-metric score terms (:meth:`Detector.metric_terms`) or can score modified
feature rows (:meth:`Detector.score_features`), see :mod:`network_anomaly.explanations.attribution`.
"""

import json
import logging
from abc import ABC, abstractmethod
from importlib.metadata import version
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from network_anomaly import __version__
from network_anomaly.errors import (
    ArtifactError,
    DataValidationError,
    LeakageError,
    NotFittedError,
)
from network_anomaly.features.builder import FeatureFrame
from network_anomaly.schemas import ExplanationMethod

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]
MANIFEST = "manifest.json"


def library_versions(
    names: tuple[str, ...] = ("numpy", "pandas", "scikit-learn"),
) -> dict[str, str]:
    return {"network-anomaly-toolkit": __version__} | {name: version(name) for name in names}


class Detector(ABC):
    """Base class for anomaly detectors (see the module docstring for the contract)."""

    name: ClassVar[str]
    version: ClassVar[str] = "1"
    default_feature_kinds: ClassVar[tuple[str, ...]] = ()
    requires_baseline: ClassVar[bool] = True
    min_training_rows: ClassVar[int] = 50
    libraries: ClassVar[tuple[str, ...]] = ("numpy", "pandas", "scikit-learn")
    term_method: ClassVar[ExplanationMethod] = "metric_terms"

    def __init__(self, config: BaseModel) -> None:
        self.config = config
        self.feature_kinds: tuple[str, ...] = self.default_feature_kinds
        self.feature_schema: tuple[str, ...] = ()
        self.metrics: tuple[str, ...] = ()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def input_metrics(self) -> tuple[str, ...]:
        """Metrics whose values this detector reads (all fitted metrics unless overridden)."""
        return self.metrics

    def rows(self, ff: FeatureFrame) -> BoolArray:
        """Rows this detector can score: observed, and with a ready baseline if it needs one."""
        return ff.scorable if self.requires_baseline else ff.observed

    def fit(self, ff: FeatureFrame) -> Self:
        if ff.role != "train":
            raise LeakageError(
                f"{self.name} can only be fitted on training data, not on a {ff.role!r} frame"
            )
        rows = self.rows(ff)
        if int(rows.sum()) < self.min_training_rows:
            raise DataValidationError(
                f"{self.name} needs at least {self.min_training_rows} scorable training rows, "
                f"got {int(rows.sum())}",
                hint="Use a longer training period; early rows of each host are warm-up.",
            )
        self.feature_schema = tuple(ff.columns(self.feature_kinds))
        self.metrics = ff.metrics
        self._fit(ff, rows)
        self._fitted = True
        logger.info("fitted detector", extra={"detector": self.name, "rows": int(rows.sum())})
        return self

    def score(self, ff: FeatureFrame) -> FloatArray:
        """One score per row of ``ff``; NaN for rows that cannot be scored."""
        if not self._fitted:
            raise NotFittedError(f"{self.name} must be fitted (or loaded) before scoring")
        missing = [column for column in self.feature_schema if column not in ff.X.columns]
        if missing or ff.metrics != self.metrics:
            raise DataValidationError(
                f"{self.name} was fitted on different features; missing: {missing[:5]}"
            )
        scores = np.full(len(ff), np.nan)
        rows = self.rows(ff)
        if rows.any():
            scores[rows] = self._score(ff, rows)
        return scores

    def matrix(self, ff: FeatureFrame, rows: BoolArray | IntArray) -> FloatArray:
        """The fitted feature columns for ``rows`` (a mask or positions) as a float64 array."""
        return ff.X[list(self.feature_schema)].to_numpy(dtype=np.float64)[rows]

    def metric_terms(self, ff: FeatureFrame, positions: IntArray) -> FloatArray | None:
        """(rows x metrics) terms that make up the score exactly, or None.

        The score is the maximum of the terms for the statistical detectors and their sum for the
        autoencoder (see ``term_method``). Detectors without such terms are explained by
        occlusion.
        """
        return None

    def score_features(self, features: FloatArray) -> FloatArray:
        """Score feature rows given as a matrix in ``feature_schema`` order (used for occlusion)."""
        raise NotImplementedError(f"{self.name} cannot score modified feature rows")

    def save(self, directory: Path) -> None:
        if not self._fitted:
            raise NotFittedError(f"cannot save an unfitted {self.name} detector")
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "kind": self.name,
            "version": self.version,
            "config": self.config.model_dump(mode="json"),
            "feature_schema": list(self.feature_schema),
            "metrics": list(self.metrics),
            "libraries": library_versions(self.libraries),
        }
        (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        self._state(directory)

    def restore(self, directory: Path, manifest: dict[str, Any]) -> Self:
        """Load fitted state written by :meth:`save` (called by ``load_detector``)."""
        if manifest.get("kind") != self.name or str(manifest.get("version")) != self.version:
            raise ArtifactError(
                f"{directory} holds {manifest.get('kind')} v{manifest.get('version')}, "
                f"not {self.name} v{self.version}"
            )
        saved = manifest.get("libraries", {})
        current = library_versions(self.libraries)
        changed = {k: (saved.get(k), v) for k, v in current.items() if saved.get(k) != v}
        if changed:
            logger.warning(
                "model was saved with different library versions", extra={"changed": changed}
            )
        self.feature_schema = tuple(manifest["feature_schema"])
        self.metrics = tuple(manifest["metrics"])
        self._restore(directory)
        self._fitted = True
        return self

    @abstractmethod
    def _fit(self, ff: FeatureFrame, rows: BoolArray) -> None: ...

    @abstractmethod
    def _score(self, ff: FeatureFrame, rows: BoolArray) -> FloatArray: ...

    @abstractmethod
    def _state(self, directory: Path) -> None:
        """Write fitted state into ``directory``."""

    @abstractmethod
    def _restore(self, directory: Path) -> None:
        """Read fitted state written by :meth:`_state`."""
