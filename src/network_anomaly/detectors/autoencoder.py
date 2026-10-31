"""Optional autoencoder detector (PyTorch, CPU only): anomaly score = reconstruction error.

Install the optional dependency with ``uv sync --extra torch``; everything else in the toolkit
works without it. The network sees the same host-normalised, compressed features as the
Isolation Forest (robust deviations from each host's causal baselines), standardised on training
rows. A small MLP squeezes them through a bottleneck; rows it cannot reconstruct score high.

Training: Adam on mean squared error, mini-batches shuffled by a seeded generator, early stopping
on the chronologically last ``validation_fraction`` of the *training* window (never on the
validation or test segments), restoring the weights of the best epoch. The per-epoch training and
early-stopping losses are kept as the training history and saved with the model.

Limitations:

* Training data contains anomalies; a flexible network can learn to reconstruct them, which lowers
  their scores. The bottleneck and early stopping limit this but do not prevent it.
* With little data (a few hosts, a few days) the network overfits easily; the statistical
  detectors are the better choice for small datasets.
* Reconstruction error depends on feature scaling, so scores are only comparable within one
  fitted model.
* Results are reproducible for the same PyTorch version on the same kind of CPU (fixed seed,
  deterministic algorithms, one thread); they can differ across versions or instruction sets.

Scores are unitless and not probabilities. Because the score is the mean of squared errors over
features, it splits exactly into one term per metric, which is how alerts explain it.
"""

import copy
import importlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from network_anomaly.config import StrictModel
from network_anomaly.detectors.base import BoolArray, Detector, FloatArray, IntArray
from network_anomaly.detectors.sklearn_models import (
    DEFAULT_KINDS,
    ModelFeatureKind,
    compress,
    stratified_sample,
)
from network_anomaly.errors import ArtifactError, ConfigError, NotFittedError
from network_anomaly.features.builder import FeatureFrame, feature_groups
from network_anomaly.schemas import ExplanationMethod

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "model.pt"
STATE_FILE = "state.json"
MIN_STD = 1e-6


def require_torch() -> ModuleType:
    """Import PyTorch or explain how to install it."""
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ConfigError(
            "the autoencoder detector needs PyTorch, which is not installed",
            hint="Install the optional CPU build with `uv sync --extra torch`.",
        ) from exc


class AutoencoderConfig(StrictModel):
    type: Literal["autoencoder"] = "autoencoder"
    hidden_units: int = Field(default=32, ge=2, le=1024)
    bottleneck: int = Field(default=8, ge=1, le=256)
    epochs: int = Field(default=60, ge=1, le=1000)
    batch_size: int = Field(default=256, ge=8, le=65536)
    learning_rate: float = Field(default=1e-3, gt=0, le=1)
    weight_decay: float = Field(default=1e-5, ge=0)
    validation_fraction: float = Field(
        default=0.1,
        gt=0,
        lt=0.5,
        description="Chronologically last share of the training rows used for early stopping.",
    )
    patience: int = Field(default=6, ge=1, description="Epochs without improvement before stop.")
    max_train_samples: int = Field(
        default=50_000, ge=100, description="Host-stratified subsample size for training."
    )
    random_state: int = 0
    feature_kinds: tuple[ModelFeatureKind, ...] = DEFAULT_KINDS


@contextmanager
def _reproducible(torch: ModuleType, seed: int) -> Iterator[None]:
    """Seeded, deterministic, single-threaded PyTorch; previous settings are restored."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.set_num_threads(threads)


def _network(torch: ModuleType, features: int, hidden: int, bottleneck: int) -> Any:
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, bottleneck),
        nn.ReLU(),
        nn.Linear(bottleneck, hidden),
        nn.ReLU(),
        nn.Linear(hidden, features),
    )


class AutoencoderDetector(Detector):
    """MLP autoencoder on standardised, compressed host-normalised features."""

    name = "autoencoder"
    libraries: ClassVar[tuple[str, ...]] = (*Detector.libraries, "torch")
    term_method: ClassVar[ExplanationMethod] = "reconstruction_error"
    config: AutoencoderConfig

    def __init__(self, config: AutoencoderConfig | None = None) -> None:
        super().__init__(config or AutoencoderConfig())
        self.feature_kinds = tuple(self.config.feature_kinds)
        self.center: FloatArray = np.zeros(0)
        self.scale: FloatArray = np.ones(0)
        self.network: Any = None
        self.history: list[dict[str, float]] = []
        self.best_epoch: int | None = None

    def _standardise(self, features: FloatArray) -> FloatArray:
        result: FloatArray = (compress(features) - self.center) / self.scale
        return result

    def _fit(self, ff: FeatureFrame, rows: BoolArray) -> None:
        torch = require_torch()
        config = self.config
        features = compress(self.matrix(ff, rows))
        times = ff.timestamps.to_numpy(dtype="datetime64[ns]")[rows]
        cut = np.quantile(times.astype(np.int64), 1.0 - config.validation_fraction)
        stopping = times.astype(np.int64) >= cut
        rng = np.random.default_rng(config.random_state)
        sample = stratified_sample(ff.entities[rows][~stopping], config.max_train_samples, rng)
        train, held_out = features[~stopping][sample], features[stopping]
        if len(held_out) == 0 or len(train) == 0:
            raise ConfigError(
                "the training period is too short to hold out rows for early stopping"
            )
        self.center = train.mean(axis=0)
        self.scale = np.maximum(train.std(axis=0), MIN_STD)
        train_tensor = torch.as_tensor((train - self.center) / self.scale, dtype=torch.float32)
        held_tensor = torch.as_tensor((held_out - self.center) / self.scale, dtype=torch.float32)
        with _reproducible(torch, config.random_state):
            network = _network(torch, features.shape[1], config.hidden_units, config.bottleneck)
            optimizer = torch.optim.Adam(
                network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
            generator = torch.Generator().manual_seed(config.random_state)
            best_loss, best_state, waited = float("inf"), None, 0
            self.history = []
            for epoch in range(1, config.epochs + 1):
                network.train()
                order = torch.randperm(len(train_tensor), generator=generator)
                total = 0.0
                for start in range(0, len(order), config.batch_size):
                    batch = train_tensor[order[start : start + config.batch_size]]
                    optimizer.zero_grad()
                    loss = torch.mean((network(batch) - batch) ** 2)
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(batch)
                network.eval()
                with torch.no_grad():
                    held_loss = float(torch.mean((network(held_tensor) - held_tensor) ** 2))
                self.history.append(
                    {
                        "epoch": epoch,
                        "train_loss": round(total / len(train_tensor), 8),
                        "validation_loss": round(held_loss, 8),
                    }
                )
                if held_loss < best_loss - 1e-6:
                    best_loss, best_state, waited = (
                        held_loss,
                        copy.deepcopy(network.state_dict()),
                        0,
                    )
                    self.best_epoch = epoch
                else:
                    waited += 1
                    if waited >= config.patience:
                        break
            network.load_state_dict(best_state)
            network.eval()
        self.network = network
        logger.info(
            "trained autoencoder",
            extra={
                "epochs": len(self.history),
                "best_epoch": self.best_epoch,
                "early_stopping_loss": round(best_loss, 6),
                "train_rows": len(train),
            },
        )

    def _errors(self, features: FloatArray) -> FloatArray:
        """Squared reconstruction error per standardised feature (rows x features)."""
        if self.network is None:
            raise NotFittedError("autoencoder is not fitted")
        torch = require_torch()
        inputs = torch.as_tensor(self._standardise(features), dtype=torch.float32)
        with _reproducible(torch, self.config.random_state), torch.no_grad():
            outputs = self.network(inputs)
        errors: FloatArray = ((outputs - inputs) ** 2).numpy().astype(np.float64)
        return errors

    def score_features(self, features: FloatArray) -> FloatArray:
        scores: FloatArray = self._errors(features).mean(axis=1)
        return scores

    def _score(self, ff: FeatureFrame, rows: BoolArray) -> FloatArray:
        return self.score_features(self.matrix(ff, rows))

    def metric_terms(self, ff: FeatureFrame, positions: IntArray) -> FloatArray | None:
        """Each metric's share of the mean squared error; the terms sum to the score."""
        errors = self._errors(self.matrix(ff, positions)) / len(self.feature_schema)
        groups = feature_groups(self.feature_schema, self.metrics)
        return np.column_stack(
            [
                errors[:, groups[m]].sum(axis=1) if groups[m] else np.full(len(errors), np.nan)
                for m in self.metrics
            ]
        )

    def _state(self, directory: Path) -> None:
        torch = require_torch()
        torch.save(self.network.state_dict(), directory / WEIGHTS_FILE)
        state = {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "history": self.history,
            "best_epoch": self.best_epoch,
        }
        (directory / STATE_FILE).write_text(json.dumps(state, indent=2) + "\n")

    def _restore(self, directory: Path) -> None:
        torch = require_torch()
        try:
            state = json.loads((directory / STATE_FILE).read_text())
            weights = torch.load(directory / WEIGHTS_FILE, weights_only=True)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            raise ArtifactError(f"cannot read autoencoder state in {directory}: {exc}") from exc
        self.center = np.asarray(state["center"], dtype=np.float64)
        self.scale = np.asarray(state["scale"], dtype=np.float64)
        self.history = list(state["history"])
        self.best_epoch = state["best_epoch"]
        network = _network(
            torch, len(self.center), self.config.hidden_units, self.config.bottleneck
        )
        network.load_state_dict(weights)
        network.eval()
        self.network = network
