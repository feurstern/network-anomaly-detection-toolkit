"""Regular per-entity time grid: the shared preparation step before feature engineering.

Timestamps are rounded to an epoch-anchored grid (``Timestamp.round(freq)``), so the grid built from
a slice of the data is identical to the grid of the whole dataset. Each entity gets one row per
interval from its first to its last observation; intervals without an observation become explicit
gap rows (``observed == False``) whose metrics are NaN.

Nothing is forward-filled or interpolated: forward-filling invents flat values that shrink the
robust scale (and then cause false alarms after a gap), and interpolation uses future values.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.tseries.frequencies import to_offset

from network_anomaly.config import interval_text, snap_interval
from network_anomaly.errors import DataValidationError
from network_anomaly.ingestion.validation import MetricsFrame
from network_anomaly.schemas import TIMESTAMP_COLUMN

SOURCE_TIMESTAMP = "source_timestamp"
OBSERVED = "observed"
_MASK_PREFIX = "__mask__"


@dataclass(frozen=True)
class GridFrame:
    """Metrics on a complete regular grid per entity, sorted by (entity, grid timestamp).

    ``data`` columns: ``timestamp`` (grid cell), the entity column, ``source_timestamp`` (original
    timestamp, NaT for gap rows), ``observed`` and the metric columns. ``masked`` marks metric cells
    that were invalid in the input and masked during validation.
    """

    data: pd.DataFrame
    entity_column: str
    metrics: tuple[str, ...]
    frequency: pd.Timedelta
    masked: pd.DataFrame

    @property
    def observed(self) -> npt.NDArray[np.bool_]:
        return self.data[OBSERVED].to_numpy(dtype=bool)


def infer_frequency(frame: MetricsFrame) -> pd.Timedelta:
    """Median spacing between consecutive observations of the same entity, snapped to a common
    interval (see :func:`~network_anomaly.config.snap_interval`)."""
    data = frame.data
    deltas = data.groupby(frame.entity_column, sort=False)[TIMESTAMP_COLUMN].diff().dropna()
    if deltas.empty:
        raise DataValidationError(
            "cannot infer the sampling interval: every entity has a single observation",
            hint="Pass the interval explicitly, e.g. --freq 5min.",
        )
    return snap_interval(float(deltas.dt.total_seconds().median()))


def regularize(frame: MetricsFrame, frequency: pd.Timedelta) -> GridFrame:
    """Place every observation on the epoch-anchored grid and insert explicit gap rows."""
    entity = frame.entity_column
    data = frame.data.copy()
    data[SOURCE_TIMESTAMP] = data[TIMESTAMP_COLUMN]
    data[TIMESTAMP_COLUMN] = data[TIMESTAMP_COLUMN].dt.round(to_offset(frequency))
    if frame.invalid_mask is not None:
        for metric in frame.metrics:
            data[_MASK_PREFIX + metric] = frame.invalid_mask[metric].to_numpy(dtype=bool)
    collisions = data.duplicated([entity, TIMESTAMP_COLUMN], keep=False)
    if collisions.any():
        raise DataValidationError(
            f"{int(collisions.sum())} observations share a {interval_text(frequency)} "
            "interval with another observation of the same entity",
            hint="Aggregate them to one row per interval or use a shorter interval.",
        )

    bounds = data.groupby(entity, sort=False)[TIMESTAMP_COLUMN].agg(["min", "max"])
    grid = pd.concat(
        [
            pd.DataFrame(
                {
                    TIMESTAMP_COLUMN: pd.date_range(first, last, freq=frequency).as_unit("ns"),
                    entity: name,
                }
            )
            for name, (first, last) in bounds.iterrows()
        ],
        ignore_index=True,
    )
    merged = grid.merge(data, on=[entity, TIMESTAMP_COLUMN], how="left", validate="one_to_one")
    merged[OBSERVED] = merged[SOURCE_TIMESTAMP].notna()
    masked = pd.DataFrame(
        {
            metric: merged[_MASK_PREFIX + metric].eq(True).to_numpy(dtype=bool)
            if frame.invalid_mask is not None
            else np.zeros(len(merged), dtype=bool)
            for metric in frame.metrics
        }
    )
    columns = [TIMESTAMP_COLUMN, entity, SOURCE_TIMESTAMP, OBSERVED, *frame.metrics]
    return GridFrame(
        data=merged[columns].reset_index(drop=True),
        entity_column=entity,
        metrics=frame.metrics,
        frequency=frequency,
        masked=masked,
    )
