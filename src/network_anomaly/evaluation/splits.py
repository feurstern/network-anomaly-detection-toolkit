"""Chronological train / validation / test segments with one set of cut times for all hosts.

Cuts are shared by every host (rather than per-host fractions) so a pooled model never learns from
one host's future while being evaluated on another host's past. Cut times are aligned to the
sampling grid. Rolling features of validation and test rows may look back into earlier segments:
that is the past, available in production, and not leakage.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import AwareDatetime, Field, model_validator

from network_anomaly.config import StrictModel
from network_anomaly.errors import ConfigError

SEGMENTS: tuple[str, ...] = ("train", "validation", "test")


class SplitConfig(StrictModel):
    train_fraction: float = Field(default=0.5, gt=0, lt=1)
    validation_fraction: float = Field(default=0.2, ge=0, lt=1)
    train_until: AwareDatetime | None = Field(
        default=None, description="Explicit end of training (exclusive); overrides the fraction."
    )
    validation_until: AwareDatetime | None = Field(
        default=None, description="Explicit end of validation (exclusive)."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.train_until is None and self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must leave room for a test set")
        if (
            self.train_until is not None
            and self.validation_until is not None
            and self.validation_until < self.train_until
        ):
            raise ValueError("validation_until must not be before train_until")
        return self


@dataclass(frozen=True)
class SplitPlan:
    """Cut times: train is [start, train_end), validation [train_end, validation_end), then test."""

    start: pd.Timestamp
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    end: pd.Timestamp

    def segment(self, timestamps: pd.Series) -> npt.NDArray[np.object_]:
        in_train = (timestamps < self.train_end).to_numpy(dtype=bool)
        in_validation = (timestamps < self.validation_end).to_numpy(dtype=bool)
        return np.where(in_train, "train", np.where(in_validation, "validation", "test")).astype(
            object
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "start": self.start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "end": self.end.isoformat(),
        }


def plan_split(timestamps: pd.Series, config: SplitConfig, frequency: pd.Timedelta) -> SplitPlan:
    """Compute grid-aligned cut times covering ``timestamps`` (grid timestamps, UTC)."""
    start = timestamps.min()
    end = timestamps.max() + frequency
    span = end - start
    train_end = (
        pd.Timestamp(config.train_until)
        if config.train_until is not None
        else (start + span * config.train_fraction).floor(frequency)
    )
    validation_end = (
        pd.Timestamp(config.validation_until)
        if config.validation_until is not None
        else (train_end + span * config.validation_fraction).floor(frequency)
    )
    if not start < train_end <= validation_end <= end:
        raise ConfigError(
            f"split cut times are outside the data ({start} .. {end}): "
            f"train_end={train_end}, validation_end={validation_end}",
            hint="Adjust --train-until/--validation-until or the split fractions.",
        )
    return SplitPlan(start, train_end, min(validation_end, end), end)
