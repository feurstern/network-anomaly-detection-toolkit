"""The causal kernels must match straightforward reference implementations exactly."""

import numpy as np
import pandas as pd
import pytest

from network_anomaly.features.rolling import (
    MAD_TO_SIGMA,
    causal_residual_scale,
    causal_rolling_median,
    causal_seasonal_median,
    causal_shift,
)


def _series_with_groups(
    seed: int, n_per_group: int = 40, groups: int = 3
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_per_group * groups, 2))
    values[rng.random(values.shape) < 0.15] = np.nan  # missing cells
    frame = pd.DataFrame(values, columns=["a", "b"])
    group = pd.Series(np.repeat([f"g{i}" for i in range(groups)], n_per_group))
    return frame, group


def _naive_previous(values: np.ndarray, group: np.ndarray, i: int, window: int) -> np.ndarray:
    """Values in the ``window`` positions before ``i`` that belong to the same group."""
    start = max(0, i - window)
    positions = [j for j in range(start, i) if group[j] == group[i]]
    return values[positions]


def _naive_median(
    values: np.ndarray, group: np.ndarray, window: int, min_periods: int
) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        previous = _naive_previous(values, group, i, window)
        previous = previous[np.isfinite(previous)]
        if len(previous) >= min_periods:
            out[i] = np.median(previous)
    return out


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(("window", "min_periods"), [(1, 1), (5, 3), (12, 6)])
def test_rolling_median_matches_reference(seed: int, window: int, min_periods: int) -> None:
    frame, group = _series_with_groups(seed)
    result = causal_rolling_median(frame, group, window, min_periods)
    for column in frame.columns:
        expected = _naive_median(frame[column].to_numpy(), group.to_numpy(), window, min_periods)
        np.testing.assert_allclose(result[column].to_numpy(), expected, equal_nan=True)


def test_rolling_median_never_uses_the_current_or_later_rows() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0, 5.0]})
    group = pd.Series(["g"] * 5)
    result = causal_rolling_median(frame, group, window=3, min_periods=1)["a"].tolist()
    assert result[0] != result[0]  # NaN: nothing before the first row
    assert result[1:] == [1.0, 1.5, 2.0, 3.0]


def test_rolling_median_restarts_at_group_boundaries_and_keeps_row_order() -> None:
    frame = pd.DataFrame({"a": [1.0, 10.0, 2.0, 20.0]})
    group = pd.Series(["x", "y", "x", "y"])  # interleaved groups
    result = causal_rolling_median(frame, group, window=5, min_periods=1)
    assert result.index.equals(frame.index)
    assert result["a"].tolist()[2:] == [1.0, 10.0]


def test_residual_scale_matches_reference() -> None:
    frame, group = _series_with_groups(7)
    baseline = causal_rolling_median(frame, group, 6, 3)
    result = causal_residual_scale(frame, baseline, group, window=10, min_periods=4)
    for column in frame.columns:
        residuals = (frame[column] - baseline[column]).abs().to_numpy()
        expected = MAD_TO_SIGMA * _naive_median(residuals, group.to_numpy(), 10, 4)
        np.testing.assert_allclose(result[column].to_numpy(), expected, equal_nan=True)


def test_shift_stays_within_groups() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    group = pd.Series(["x", "x", "y", "y"])
    assert causal_shift(frame, group, 1)["a"].tolist()[1::2] == [1.0, 3.0]
    assert np.isnan(causal_shift(frame, group, 1)["a"].to_numpy()[[0, 2]]).all()


def test_seasonal_median_matches_reference() -> None:
    rng = np.random.default_rng(3)
    steps_per_day, days, n = 4, 3, 4 * 10
    frame = pd.DataFrame({"a": rng.normal(size=2 * n)})
    group = pd.Series(np.repeat(["g0", "g1"], n))
    weekend = rng.random(2 * n) < 0.3
    lagged = [rng.random(2 * n) < 0.3 for _ in range(days)]
    result = causal_seasonal_median(
        frame,
        group,
        steps_per_day=steps_per_day,
        days=days,
        min_values=2,
        weekend=weekend,
        weekend_lagged=lagged,
    )["a"].to_numpy()
    values, groups = frame["a"].to_numpy(), group.to_numpy()
    for i in range(2 * n):
        same = []
        for k in range(1, days + 1):
            j = i - k * steps_per_day
            if j >= 0 and groups[j] == groups[i] and lagged[k - 1][i] == weekend[i]:
                same.append(values[j])
        expected = np.median(same) if len(same) >= 2 else np.nan
        np.testing.assert_allclose(result[i], expected, equal_nan=True)
