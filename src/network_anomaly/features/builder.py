"""Causal, per-entity features with an explicit fit/transform split.

:meth:`FeatureBuilder.fit` sees training rows only. It learns the sampling interval, a scale floor
per (entity, metric) with a pooled fallback for entities never seen in training, and the median of
every feature on training rows (the "normal" reference used when explaining model detectors).
Everything else is computed online from each row's own past, so transforming the whole timeline
once and then slicing it is equivalent to processing it as a stream.

Per metric, in the metric's modelling space (log1p for volumes, latencies and counts):

* ``z_local``  = (x - m) / s, where m is the median of the previous ``local_window`` and s the
  robust scale of recent one-step residuals (:func:`~.rolling.causal_residual_scale`);
* ``z_season`` = (x - median of the same slot on previous days of the same day type) / s;
* ``roc``      = (x - previous x) / s, the rate of change in noise units.

Identifiers, timestamps and labels never become features: :class:`FeatureFrame` refuses them.
"""

import logging
import math
from dataclasses import dataclass
from typing import Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import AwareDatetime, Field, field_validator

from network_anomaly.config import StrictModel, parse_frequency
from network_anomaly.errors import ConfigError, DataValidationError, NotFittedError
from network_anomaly.features.rolling import (
    causal_residual_scale,
    causal_rolling_median,
    causal_seasonal_median,
    causal_shift,
)
from network_anomaly.ingestion.validation import MetricsFrame
from network_anomaly.preprocessing import OBSERVED, SOURCE_TIMESTAMP, GridFrame, regularize
from network_anomaly.schemas import (
    DEFAULT_ENTITY_COLUMN,
    LABEL_COLUMNS,
    TIMESTAMP_COLUMN,
    MetricSpec,
    SplitRole,
    metric_spec,
)

logger = logging.getLogger(__name__)

FeatureKind = Literal["z_local", "z_season", "roc", "calendar"]
MODEL_FEATURE_KINDS: tuple[FeatureKind, ...] = ("z_local", "z_season", "roc")
CALENDAR_COLUMNS: tuple[str, ...] = (
    "calendar__hour_sin",
    "calendar__hour_cos",
    "calendar__weekend",
)
_FORBIDDEN = frozenset({TIMESTAMP_COLUMN, DEFAULT_ENTITY_COLUMN, SOURCE_TIMESTAMP, OBSERVED})
_FORBIDDEN |= frozenset(LABEL_COLUMNS)

BoolArray = npt.NDArray[np.bool_]


class FeatureConfig(StrictModel):
    """How baselines are computed. Windows are durations, converted to steps of the data."""

    local_window: str = Field(default="1h", description="History for the rolling median.")
    local_min_fraction: float = Field(default=0.5, gt=0, le=1)
    scale_window: str = Field(default="1D", description="History for the residual scale.")
    scale_min_fraction: float = Field(default=0.125, gt=0, le=1)
    seasonal_days: int = Field(default=7, ge=1, le=28)
    seasonal_min_values: int = Field(default=2, ge=1, le=28)
    calendar_tz: str = Field(default="UTC", description="Time zone for day types and hours.")
    calendar_features: bool = False
    clip: float = Field(default=50.0, gt=0)
    scale_floor_quantile: float = Field(default=0.10, ge=0, lt=0.5)

    @field_validator("local_window", "scale_window")
    @classmethod
    def _check_window(cls, value: str) -> str:
        try:
            parse_frequency(value)
        except ConfigError as exc:
            raise ValueError(exc.message) from exc
        return value

    @field_validator("calendar_tz")
    @classmethod
    def _check_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown time zone {value!r}") from exc
        return value

    def window_steps(self, window: str, frequency: pd.Timedelta) -> int:
        return max(1, round(parse_frequency(window) / frequency))


class FeatureState(StrictModel):
    """Everything :meth:`FeatureBuilder.fit` learned; JSON-serialisable for saved models."""

    config: FeatureConfig
    entity_column: str
    metrics: tuple[str, ...]
    frequency_seconds: float
    scale_floors: dict[str, dict[str, float]]
    pooled_floors: dict[str, float]
    feature_medians: dict[str, float]
    training_entities: tuple[str, ...]
    training_start: AwareDatetime
    training_end: AwareDatetime
    training_rows: int

    @property
    def frequency(self) -> pd.Timedelta:
        return pd.Timedelta(seconds=self.frequency_seconds)


@dataclass(frozen=True)
class FeatureFrame:
    """Model inputs plus row-aligned keys, raw-unit context and data-quality flags.

    * ``X``: float64 features named ``<metric>__<kind>``; unavailable values are neutral (0).
    * ``keys``: grid ``timestamp``, entity, ``source_timestamp`` and ``observed``.
    * ``context``: per metric ``__value``, ``__baseline``, ``__seasonal_baseline`` (raw units)
      and ``__scale`` (modelling space), for explanations and reports only.
    * ``available``: per metric, whether its local features could be computed on the row.
    * ``quality``: ``local_ready``, ``season_ready``, ``entity_seen``, ``unavailable_metrics``,
      ``masked_cells``.
    """

    X: pd.DataFrame
    keys: pd.DataFrame
    context: pd.DataFrame
    available: pd.DataFrame
    quality: pd.DataFrame
    entity_column: str
    metrics: tuple[str, ...]
    role: SplitRole = "all"

    def __post_init__(self) -> None:
        frames = (self.keys, self.context, self.available, self.quality)
        if any(not frame.index.equals(self.X.index) for frame in frames):
            raise ValueError("FeatureFrame parts must share the same row index")
        forbidden = {self.entity_column, *_FORBIDDEN}
        for column in self.X.columns:
            if str(column).split("__")[0] in forbidden:
                raise ValueError(f"identifier or label column cannot be a feature: {column!r}")
        if not all(dtype == np.float64 for dtype in self.X.dtypes):
            raise ValueError("FeatureFrame.X must contain float64 columns only")
        if not np.isfinite(self.X.to_numpy()).all():
            raise ValueError("FeatureFrame.X must be finite (neutral-fill unavailable values)")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def observed(self) -> BoolArray:
        return self.keys[OBSERVED].to_numpy(dtype=bool)

    @property
    def scorable(self) -> BoolArray:
        """Observed rows whose local baseline is ready for at least one metric."""
        return self.observed & self.quality["local_ready"].to_numpy(dtype=bool)

    @property
    def entities(self) -> npt.NDArray[np.object_]:
        return self.keys[self.entity_column].to_numpy(dtype=object)

    @property
    def timestamps(self) -> pd.Series:
        return self.keys[TIMESTAMP_COLUMN]

    def columns(self, kinds: tuple[str, ...]) -> list[str]:
        """Feature columns of the given kinds, ordered by kind then metric."""
        names: list[str] = []
        for kind in kinds:
            if kind == "calendar":
                names.extend(c for c in CALENDAR_COLUMNS if c in self.X.columns)
            else:
                names.extend(f"{metric}__{kind}" for metric in self.metrics)
        return names

    def select(self, mask: BoolArray, role: SplitRole) -> "FeatureFrame":
        """Subset of rows, tagged with the split role they belong to."""

        def take(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.loc[mask].reset_index(drop=True)

        return FeatureFrame(
            X=take(self.X),
            keys=take(self.keys),
            context=take(self.context),
            available=take(self.available),
            quality=take(self.quality),
            entity_column=self.entity_column,
            metrics=self.metrics,
            role=role,
        )


def _to_model_space(values: npt.NDArray[np.float64], spec: MetricSpec) -> npt.NDArray[np.float64]:
    return np.log1p(values) if spec.log_scale else values


def _to_raw_units(values: npt.NDArray[np.float64], spec: MetricSpec) -> npt.NDArray[np.float64]:
    return np.expm1(values) if spec.log_scale else values


@dataclass(frozen=True)
class _Parts:
    """Intermediate causal statistics, before scale floors are applied."""

    x: pd.DataFrame
    local_baseline: pd.DataFrame
    raw_scale: pd.DataFrame
    seasonal_baseline: pd.DataFrame
    previous: pd.DataFrame
    local_time: pd.Series


def _causal_parts(grid: GridFrame, config: FeatureConfig) -> _Parts:
    data = grid.data
    groups = data[grid.entity_column]
    x = pd.DataFrame(
        {
            m: _to_model_space(data[m].to_numpy(dtype=np.float64), metric_spec(m))
            for m in grid.metrics
        }
    )
    local = config.window_steps(config.local_window, grid.frequency)
    scale = config.window_steps(config.scale_window, grid.frequency)
    local_baseline = causal_rolling_median(
        x, groups, local, max(1, math.ceil(local * config.local_min_fraction))
    )
    raw_scale = causal_residual_scale(
        x, local_baseline, groups, scale, max(1, math.ceil(scale * config.scale_min_fraction))
    )
    local_time = data[TIMESTAMP_COLUMN].dt.tz_convert(ZoneInfo(config.calendar_tz))
    step_seconds = grid.frequency.total_seconds()
    if 86_400 % step_seconds:
        logger.warning(
            "sampling interval does not divide a day; seasonal features are disabled",
            extra={"interval_seconds": step_seconds},
        )
        seasonal = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    else:
        seasonal = causal_seasonal_median(
            x,
            groups,
            steps_per_day=round(86_400 / step_seconds),
            days=config.seasonal_days,
            min_values=config.seasonal_min_values,
            weekend=(local_time.dt.day_of_week >= 5).to_numpy(dtype=bool),
            weekend_lagged=[
                ((local_time - pd.Timedelta(days=k)).dt.day_of_week >= 5).to_numpy(dtype=bool)
                for k in range(1, config.seasonal_days + 1)
            ],
        )
    previous = causal_shift(x, groups, 1)
    return _Parts(x, local_baseline, raw_scale, seasonal, previous, local_time)


def _assemble(grid: GridFrame, parts: _Parts, state: FeatureState) -> FeatureFrame:
    data = grid.data
    entities = data[grid.entity_column]
    floors = pd.DataFrame(
        {
            m: entities.map({e: f[m] for e, f in state.scale_floors.items()})
            .astype(np.float64)
            .fillna(state.pooled_floors[m])
            .to_numpy(dtype=np.float64)
            for m in grid.metrics
        }
    )
    scale = parts.raw_scale.clip(lower=floors)  # NaN (not ready) stays NaN
    limit = state.config.clip
    z_local = ((parts.x - parts.local_baseline) / scale).clip(-limit, limit)
    z_season = ((parts.x - parts.seasonal_baseline) / scale).clip(-limit, limit)
    roc = ((parts.x - parts.previous) / scale).clip(-limit, limit)
    available = z_local.notna()

    features: dict[str, npt.NDArray[np.float64]] = {}
    context: dict[str, npt.NDArray[np.float64]] = {}
    for metric in grid.metrics:
        spec = metric_spec(metric)
        for kind, frame in (("z_local", z_local), ("z_season", z_season), ("roc", roc)):
            features[f"{metric}__{kind}"] = frame[metric].fillna(0.0).to_numpy(dtype=np.float64)
        context[f"{metric}__value"] = data[metric].to_numpy(dtype=np.float64)
        context[f"{metric}__baseline"] = _to_raw_units(
            parts.local_baseline[metric].to_numpy(dtype=np.float64), spec
        )
        context[f"{metric}__seasonal_baseline"] = _to_raw_units(
            parts.seasonal_baseline[metric].to_numpy(dtype=np.float64), spec
        )
        context[f"{metric}__scale"] = scale[metric].to_numpy(dtype=np.float64)
    if state.config.calendar_features:
        hours = (parts.local_time.dt.hour + parts.local_time.dt.minute / 60).to_numpy(np.float64)
        features["calendar__hour_sin"] = np.sin(2 * np.pi * hours / 24)
        features["calendar__hour_cos"] = np.cos(2 * np.pi * hours / 24)
        weekend = (parts.local_time.dt.day_of_week >= 5).to_numpy(dtype=np.float64)
        features["calendar__weekend"] = weekend

    quality = pd.DataFrame(
        {
            "local_ready": available.any(axis=1).to_numpy(dtype=bool),
            "season_ready": z_season.notna().any(axis=1).to_numpy(dtype=bool),
            "entity_seen": entities.isin(state.training_entities).to_numpy(dtype=bool),
            "unavailable_metrics": (~available).sum(axis=1).to_numpy(dtype=np.int64),
            "masked_cells": grid.masked.sum(axis=1).to_numpy(dtype=np.int64),
        }
    )
    keys = data[[TIMESTAMP_COLUMN, grid.entity_column, SOURCE_TIMESTAMP, OBSERVED]]
    return FeatureFrame(
        X=pd.DataFrame(features),
        keys=keys.reset_index(drop=True),
        context=pd.DataFrame(context),
        available=available.reset_index(drop=True),
        quality=quality,
        entity_column=grid.entity_column,
        metrics=grid.metrics,
    )


def _align_metrics(frame: MetricsFrame, metrics: tuple[str, ...]) -> MetricsFrame:
    """Restrict ``frame`` to the training metrics; missing ones become all-missing columns."""
    absent = [m for m in metrics if m not in frame.metrics]
    extra = [m for m in frame.metrics if m not in metrics]
    if absent:
        logger.warning("metrics seen in training are missing", extra={"metrics": absent})
    if extra:
        logger.info("ignoring metrics not seen in training", extra={"metrics": extra})
    if not absent and not extra:
        return frame
    data = frame.data[[TIMESTAMP_COLUMN, frame.entity_column]].copy()
    mask = None if frame.invalid_mask is None else pd.DataFrame(index=frame.data.index)
    for metric in metrics:
        present = metric in frame.metrics
        data[metric] = frame.data[metric] if present else np.nan
        if mask is not None and frame.invalid_mask is not None:
            mask[metric] = frame.invalid_mask[metric] if present else False
    return MetricsFrame(data, frame.entity_column, metrics, mask, frame.source)


class FeatureBuilder:
    """Fit causal feature statistics on training data, then transform any period."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self._state: FeatureState | None = None

    @classmethod
    def from_state(cls, state: FeatureState) -> "FeatureBuilder":
        builder = cls(state.config)
        builder._state = state
        return builder

    @property
    def state(self) -> FeatureState:
        if self._state is None:
            raise NotFittedError("the feature builder has not been fitted")
        return self._state

    def fit(self, train: MetricsFrame, frequency: pd.Timedelta) -> Self:
        """Learn scale floors and training medians from training rows only."""
        if train.data.empty:
            raise DataValidationError("the training period contains no rows")
        grid = regularize(train, frequency)
        parts = _causal_parts(grid, self.config)
        entities = grid.data[train.entity_column]
        observed = grid.observed
        floors: dict[str, dict[str, float]] = {}
        for entity in pd.unique(entities):
            rows = observed & (entities == entity).to_numpy(dtype=bool)
            floors[str(entity)] = {}
            for metric in train.metrics:
                values = parts.raw_scale[metric].to_numpy(dtype=np.float64)[rows]
                values = values[np.isfinite(values)]
                learned = (
                    float(np.quantile(values, self.config.scale_floor_quantile))
                    if values.size
                    else 0.0
                )
                floors[str(entity)][metric] = max(metric_spec(metric).min_scale, learned)
        pooled = {
            m: float(np.median([floors[e][m] for e in floors])) if floors else 0.0
            for m in train.metrics
        }
        pooled = {m: max(metric_spec(m).min_scale, value) for m, value in pooled.items()}
        timestamps = train.data[TIMESTAMP_COLUMN]
        provisional = FeatureState(
            config=self.config,
            entity_column=train.entity_column,
            metrics=train.metrics,
            frequency_seconds=frequency.total_seconds(),
            scale_floors=floors,
            pooled_floors=pooled,
            feature_medians={},
            training_entities=tuple(floors),
            training_start=timestamps.min().to_pydatetime(),
            training_end=timestamps.max().to_pydatetime(),
            training_rows=int(observed.sum()),
        )
        frame = _assemble(grid, parts, provisional)
        rows = frame.scorable
        medians = {
            str(column): float(np.median(frame.X[column].to_numpy()[rows])) if rows.any() else 0.0
            for column in frame.X.columns
        }
        self._state = provisional.model_copy(update={"feature_medians": medians})
        logger.info(
            "fitted feature builder",
            extra={"entities": len(floors), "rows": provisional.training_rows},
        )
        return self

    def transform(self, metrics: MetricsFrame) -> FeatureFrame:
        """Compute features for every row; each row only uses its own entity's past."""
        state = self.state
        if metrics.entity_column != state.entity_column:
            raise DataValidationError(
                f"entity column {metrics.entity_column!r} differs from the training data's "
                f"{state.entity_column!r}"
            )
        grid = regularize(_align_metrics(metrics, state.metrics), state.frequency)
        return _assemble(grid, _causal_parts(grid, state.config), state)
