"""Operating points: turning a detector's unitless scores into flags.

A threshold is a cut-off on one detector's own score scale, fitted on a designated period and then
frozen. It is **not** a probability, and it is never fitted on test data (:class:`LeakageError`).

Strategies:

* ``fixed`` - an operator-chosen value, e.g. 1.0 for static limits.
* ``train_quantile`` - the q-quantile of training scores. Training data may contain anomalies and
  in-sample scores of model detectors are optimistic, so prefer a validation-based strategy.
* ``clean_quantile`` - the q-quantile of scores in a validation window that the operator has vetted
  as normal. Needs no labels, but anomalies inside the window raise the threshold.
* ``target_fpr`` - the (1 - fpr)-quantile of validation rows *labelled normal*. Needs labels.

With ``per_host`` enabled, hosts with at least ``ceil(10 / (1 - q))`` scored rows in the fitting
window get their own quantile (estimated from >= 10 exceedances); other hosts, including hosts
never seen in the window, fall back to the global value.
"""

import math
from datetime import datetime
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import AwareDatetime, Field

from network_anomaly.config import StrictModel
from network_anomaly.errors import ConfigError, DataValidationError, LeakageError
from network_anomaly.schemas import SplitRole

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
ObjectArray = npt.NDArray[np.object_]

MIN_EXCEEDANCES = 10


class FixedThresholdConfig(StrictModel):
    strategy: Literal["fixed"] = "fixed"
    value: float


class QuantileThresholdConfig(StrictModel):
    strategy: Literal["train_quantile", "clean_quantile"] = "clean_quantile"
    quantile: float = Field(default=0.995, gt=0.5, lt=1.0)
    per_host: bool = True


class TargetFprConfig(StrictModel):
    strategy: Literal["target_fpr"] = "target_fpr"
    target_fpr: float = Field(default=0.005, gt=0.0, lt=0.5)
    per_host: bool = True


ThresholdConfig = Annotated[
    FixedThresholdConfig | QuantileThresholdConfig | TargetFprConfig,
    Field(discriminator="strategy"),
]

# Which segment each strategy is calibrated on; fixed thresholds are not fitted at all.
CALIBRATION_ROLE: dict[str, SplitRole | None] = {
    "fixed": None,
    "train_quantile": "train",
    "clean_quantile": "validation",
    "target_fpr": "validation",
}


class HostThresholds(StrictModel):
    """A fitted operating point: one global threshold plus optional per-host thresholds."""

    strategy: str
    fitted_on: SplitRole | None
    window_start: AwareDatetime | None = None
    window_end: AwareDatetime | None = None
    global_threshold: float
    per_host: dict[str, float] = Field(default_factory=dict)
    hosts_using_global: tuple[str, ...] = ()
    n_samples: int = 0
    params: dict[str, float | bool] = Field(default_factory=dict)

    def thresholds_for(self, entities: ObjectArray) -> FloatArray:
        mapped = pd.Series(entities, dtype=object).map(self.per_host)
        filled = mapped.astype(np.float64).fillna(self.global_threshold)
        return np.asarray(filled, dtype=np.float64)

    def scope_for(self, entities: ObjectArray) -> ObjectArray:
        known = pd.Series(entities, dtype=object).isin(list(self.per_host)).to_numpy(dtype=bool)
        return np.where(known, "host", "global").astype(object)

    def apply(self, scores: FloatArray, entities: ObjectArray) -> BoolArray:
        """Flag rows whose score reaches their threshold; NaN scores are never flagged."""
        return np.isfinite(scores) & (scores >= self.thresholds_for(entities))


def _quantile(values: FloatArray, q: float) -> float:
    return float(np.quantile(values, q))


def fit_thresholds(
    config: FixedThresholdConfig | QuantileThresholdConfig | TargetFprConfig,
    scores: FloatArray,
    entities: ObjectArray,
    *,
    role: SplitRole,
    labels: npt.NDArray[np.int8] | None = None,
    window: tuple[datetime, datetime] | None = None,
) -> HostThresholds:
    """Fit an operating point on the scores of one segment.

    Raises:
        LeakageError: if asked to fit on test data.
        ConfigError: if the strategy needs a different segment or labels that are not available.
        DataValidationError: if the segment has no scored rows.
    """
    if role == "test":
        raise LeakageError("thresholds must never be fitted on test data")
    expected = CALIBRATION_ROLE[config.strategy]
    if expected is not None and role != expected:
        raise ConfigError(f"{config.strategy} thresholds are fitted on {expected} data, not {role}")
    window_fields = {"window_start": window[0], "window_end": window[1]} if window else {}
    if isinstance(config, FixedThresholdConfig):
        return HostThresholds(
            strategy="fixed",
            fitted_on=None,
            global_threshold=config.value,
            params={"value": config.value},
        )

    usable = np.isfinite(scores)
    if isinstance(config, TargetFprConfig):
        if labels is None:
            raise ConfigError(
                "target_fpr needs anomaly labels for the validation period",
                hint="Provide is_anomaly labels or use the clean_quantile strategy.",
            )
        usable &= labels == 0
        q = 1.0 - config.target_fpr
        params: dict[str, float | bool] = {"target_fpr": config.target_fpr}
    else:
        q = config.quantile
        params = {"quantile": q}
    params["per_host"] = config.per_host
    values = scores[usable]
    if values.size == 0:
        raise DataValidationError(
            f"no scored rows to fit {config.strategy} thresholds on the {role} segment",
            hint="Check the split fractions and that the segment is longer than the warm-up.",
        )
    per_host: dict[str, float] = {}
    using_global: list[str] = []
    if config.per_host:
        hosts = entities[usable]
        minimum = math.ceil(MIN_EXCEEDANCES / (1.0 - q))
        for host in sorted({str(h) for h in hosts}):
            host_values = values[hosts == host]
            if host_values.size >= minimum:
                per_host[host] = _quantile(host_values, q)
            else:
                using_global.append(host)
    return HostThresholds(
        strategy=config.strategy,
        fitted_on=role,
        global_threshold=_quantile(values, q),
        per_host=per_host,
        hosts_using_global=tuple(using_global),
        n_samples=int(values.size),
        params=params,
        **window_fields,
    )
