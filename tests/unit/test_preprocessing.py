from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from network_anomaly.errors import DataValidationError
from network_anomaly.ingestion.validation import MetricsFrame
from network_anomaly.preprocessing import infer_frequency, regularize

FIVE_MINUTES = pd.Timedelta("5min")


def _frame(rows: list[tuple[str, str, float]], mask: list[bool] | None = None) -> MetricsFrame:
    data = pd.DataFrame(rows, columns=["timestamp", "host_id", "cpu_usage_percent"])
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True).dt.as_unit("ns")
    invalid = None if mask is None else pd.DataFrame({"cpu_usage_percent": mask})
    return MetricsFrame(data, "host_id", ("cpu_usage_percent",), invalid)


def test_gaps_become_explicit_unobserved_rows_without_filling() -> None:
    frame = _frame(
        [
            ("2026-01-05T00:00:00Z", "a", 1.0),
            ("2026-01-05T00:05:00Z", "a", 2.0),
            ("2026-01-05T00:20:00Z", "a", 5.0),
        ]
    )
    grid = regularize(frame, FIVE_MINUTES)
    assert len(grid.data) == 5
    assert grid.observed.tolist() == [True, True, False, False, True]
    values = grid.data["cpu_usage_percent"].to_numpy()
    assert np.isnan(values[2:4]).all()  # never forward-filled or interpolated
    assert grid.data["source_timestamp"].isna().tolist() == [False, False, True, True, False]


def test_jittered_timestamps_snap_to_the_epoch_anchored_grid() -> None:
    frame = _frame([("2026-01-05T00:01:10Z", "a", 1.0), ("2026-01-05T00:04:20Z", "b", 2.0)])
    grid = regularize(frame, FIVE_MINUTES)
    assert grid.data["timestamp"].dt.strftime("%H:%M").tolist() == ["00:00", "00:05"]
    original = grid.data["source_timestamp"].dt.strftime("%H:%M:%S").tolist()
    assert original == ["00:01:10", "00:04:20"]


def test_each_entity_spans_only_its_own_observations() -> None:
    frame = _frame(
        [
            ("2026-01-05T00:00:00Z", "a", 1.0),
            ("2026-01-05T00:10:00Z", "a", 1.0),
            ("2026-01-05T00:05:00Z", "late", 1.0),
        ]
    )
    grid = regularize(frame, FIVE_MINUTES)
    assert grid.data.groupby("host_id").size().to_dict() == {"a": 3, "late": 1}


def test_grid_of_a_slice_matches_the_grid_of_the_whole(
    make_metrics: Callable[..., MetricsFrame],
) -> None:
    frame = make_metrics(seed=4)
    whole = regularize(frame, FIVE_MINUTES).data
    cut = pd.Timestamp("2026-01-06T00:00:00Z")
    part = regularize(frame.until(cut, FIVE_MINUTES), FIVE_MINUTES).data
    shared = whole[whole["timestamp"] < cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(part[part["timestamp"] < cut].reset_index(drop=True), shared)


def test_masked_cells_travel_with_their_rows() -> None:
    frame = _frame(
        [("2026-01-05T00:00:00Z", "a", 1.0), ("2026-01-05T00:10:00Z", "a", np.nan)],
        mask=[False, True],
    )
    grid = regularize(frame, FIVE_MINUTES)
    assert grid.masked["cpu_usage_percent"].tolist() == [False, False, True]


def test_two_observations_in_one_interval_are_rejected() -> None:
    frame = _frame([("2026-01-05T00:00:00Z", "a", 1.0), ("2026-01-05T00:01:00Z", "a", 2.0)])
    with pytest.raises(DataValidationError, match="share a 5min interval"):
        regularize(frame, FIVE_MINUTES)


def test_frequency_inference(make_metrics: Callable[..., MetricsFrame]) -> None:
    assert infer_frequency(make_metrics(seed=1)) == FIVE_MINUTES
    single = _frame([("2026-01-05T00:00:00Z", "a", 1.0)])
    with pytest.raises(DataValidationError, match="cannot infer"):
        infer_frequency(single)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(301.0, 300), (59.2, 60), (299.4, 300), (3540.0, 3600), (420.0, 420), (0.97, 1)],
)
def test_jittered_spacing_snaps_to_a_common_interval(seconds: float, expected: int) -> None:
    from network_anomaly.config import snap_interval

    assert snap_interval(seconds) == pd.Timedelta(seconds=expected)
