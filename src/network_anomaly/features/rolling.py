"""Grouped, causal rolling kernels used by feature engineering.

Every statistic at row ``t`` uses only earlier rows of the same entity (``closed="left"``), so a
value never influences its own baseline and nothing leaks between entities or from the future.
Inputs are frames sorted by (entity, time) on a complete regular grid (see
:mod:`network_anomaly.preprocessing`), so shifting by ``k`` rows means "``k`` intervals earlier".
Outputs keep the input row order and index.

These functions are the documented boundary for an optional compiled implementation: a
replacement must reproduce their output on the golden fixtures in the test suite.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

MAD_TO_SIGMA = 1.4826  # makes the median absolute deviation consistent with a normal std


def causal_rolling_median(
    values: pd.DataFrame, groups: pd.Series, window: int, min_periods: int
) -> pd.DataFrame:
    """Median of the previous ``window`` rows of the same group, excluding the current row.

    Missing values are skipped; fewer than ``min_periods`` valid values gives NaN.
    """
    rolled = (
        values.groupby(groups, sort=False)
        .rolling(window, min_periods=min_periods, closed="left")
        .median()
        .droplevel(0)
    )
    # groupby emits groups in order of first appearance; restore the caller's row order.
    return rolled.reindex(values.index)


def causal_shift(values: pd.DataFrame, groups: pd.Series, periods: int) -> pd.DataFrame:
    """Value ``periods`` rows earlier within the same group (NaN before the group starts)."""
    return values.groupby(groups, sort=False).shift(periods)


def causal_residual_scale(
    values: pd.DataFrame,
    baseline: pd.DataFrame,
    groups: pd.Series,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    """Robust scale of one-step-ahead residuals: 1.4826 x rolling median of past |x - baseline|.

    The residual at row ``t`` involves ``x_t``, so the scale at ``t`` uses residuals of rows before
    ``t`` only. This is the median absolute deviation of forecast errors, the right yardstick for a
    deviation from a rolling baseline, and it needs two fast rolling medians instead of an exact
    windowed MAD.
    """
    residuals = (values - baseline).abs()
    return MAD_TO_SIGMA * causal_rolling_median(residuals, groups, window, min_periods)


def causal_seasonal_median(
    values: pd.DataFrame,
    groups: pd.Series,
    *,
    steps_per_day: int,
    days: int,
    min_values: int,
    weekend: npt.NDArray[np.bool_],
    weekend_lagged: list[npt.NDArray[np.bool_]],
) -> pd.DataFrame:
    """Median of the same time slot on up to ``days`` previous days of the same day type.

    ``weekend`` flags each row's day type and ``weekend_lagged[k - 1]`` the day type ``k`` days
    earlier, both in the calendar time zone, so weekdays are compared with weekdays and weekends
    with weekends. Rows with fewer than ``min_values`` comparable days get NaN.
    """
    lagged = [causal_shift(values, groups, k * steps_per_day) for k in range(1, days + 1)]
    same_type = np.column_stack([lag_type == weekend for lag_type in weekend_lagged])
    result: dict[str, npt.NDArray[np.float64]] = {}
    for column in values.columns:
        stack = np.column_stack([lag[column].to_numpy(dtype=np.float64) for lag in lagged])
        stack = np.where(same_type, stack, np.nan)
        enough = np.isfinite(stack).sum(axis=1) >= min_values
        median = np.full(len(values), np.nan)
        if enough.any():
            median[enough] = np.nanmedian(stack[enough], axis=1)
        result[str(column)] = median
    return pd.DataFrame(result, index=values.index)
