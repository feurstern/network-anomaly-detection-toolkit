import numpy as np
import pandas as pd

from network_anomaly.explanations.alerts import AlertingConfig, build_alert_intervals

STEP = pd.Timedelta("5min")
T0 = pd.Timestamp("2026-01-05T00:00:00Z")


def _alerts(flags: list[int], *, config: AlertingConfig, times: list[int] | None = None) -> list:
    steps = times if times is not None else list(range(len(flags)))
    return build_alert_intervals(
        np.array(["web-1"] * len(flags), dtype=object),
        pd.Series([T0 + s * STEP for s in steps]),
        np.array(flags, dtype=bool),
        np.arange(len(flags), dtype=float),
        STEP,
        config,
    )


def test_alert_fires_on_its_min_points_th_flag_and_ends_after_its_last() -> None:
    (alert,) = _alerts([0, 1, 1, 1, 0], config=AlertingConfig(merge_gap=0, min_points=2))
    assert alert.start == T0 + 1 * STEP
    assert alert.fired_at == T0 + 2 * STEP
    assert alert.end == T0 + 4 * STEP
    assert alert.flagged_points == 3
    assert alert.peak_at == T0 + 3 * STEP  # scores increase with time in this fixture


def test_single_flags_never_fire_with_min_points_two() -> None:
    assert _alerts([1, 0, 0, 0, 1], config=AlertingConfig(merge_gap=2, min_points=2)) == []


def test_short_interruptions_are_merged_long_ones_split() -> None:
    config = AlertingConfig(merge_gap=2, min_points=2)
    assert len(_alerts([1, 0, 0, 1], config=config)) == 1  # two quiet intervals: merged
    assert len(_alerts([1, 1, 0, 0, 0, 1, 1], config=config)) == 2  # three: split


def test_missing_intervals_count_as_gaps() -> None:
    config = AlertingConfig(merge_gap=1, min_points=2)
    # Flags at steps 0, 1 and 5: steps 2-4 have no row at all, so the last flag stands alone.
    alerts = _alerts([1, 1, 1], config=config, times=[0, 1, 5])
    assert len(alerts) == 1
    assert alerts[0].end == T0 + 2 * STEP


def test_entities_are_never_merged() -> None:
    alerts = build_alert_intervals(
        np.array(["a", "b", "a", "b"], dtype=object),
        pd.Series([T0, T0, T0 + STEP, T0 + STEP]),
        np.ones(4, dtype=bool),
        np.ones(4),
        STEP,
        AlertingConfig(merge_gap=0, min_points=2),
    )
    assert [alert.entity for alert in alerts] == ["a", "b"]
