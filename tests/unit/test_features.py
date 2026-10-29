"""Feature engineering must be causal: no row may depend on its own entity's future, on other
entities, or on labels. These tests corrupt the future and check the past is bit-identical; the
negative controls prove the tests would catch a leaky implementation."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from network_anomaly.features import builder as builder_module
from network_anomaly.features.builder import FeatureBuilder, FeatureConfig, FeatureFrame
from network_anomaly.ingestion.validation import MetricsFrame
from network_anomaly.schemas import TIMESTAMP_COLUMN

FIVE_MINUTES = pd.Timedelta("5min")
TRAIN_END = pd.Timestamp("2026-01-06T00:00:00Z")
CONFIG = FeatureConfig(local_window="1h", scale_window="6h", seasonal_days=2)
PARTS = ("X", "context", "available", "quality", "keys")
MakeMetrics = Callable[..., MetricsFrame]


def _fit(frame: MetricsFrame) -> FeatureBuilder:
    return FeatureBuilder(CONFIG).fit(frame.until(TRAIN_END, FIVE_MINUTES), FIVE_MINUTES)


def _keyed(ff: FeatureFrame, part: str) -> pd.DataFrame:
    frame = getattr(ff, part).copy()
    index = pd.MultiIndex.from_frame(ff.keys[["host_id", TIMESTAMP_COLUMN]])
    frame.index = index
    return frame.drop(columns=["host_id", TIMESTAMP_COLUMN], errors="ignore")


def _corrupt_after(
    frame: MetricsFrame, cut: pd.Timestamp, rng: np.random.Generator
) -> MetricsFrame:
    """Garbage values, dropped rows and a brand-new host, all strictly after ``cut``."""
    data = frame.data.copy()
    after = (data[TIMESTAMP_COLUMN] > cut + FIVE_MINUTES).to_numpy()
    for metric in frame.metrics:
        noise = rng.uniform(0, 1000, size=after.sum())
        data.loc[after, metric] = np.where(rng.random(after.sum()) < 0.1, np.nan, noise)
    dropped = after & (rng.random(len(data)) < 0.3)
    data = data.loc[~dropped]
    later_h0 = (data["host_id"] == "h0") & (data[TIMESTAMP_COLUMN] > cut + FIVE_MINUTES)
    newcomer = data.loc[later_h0].head(50).copy()
    newcomer["host_id"] = "newcomer"
    data = pd.concat([data, newcomer], ignore_index=True)
    return MetricsFrame(data.reset_index(drop=True), "host_id", frame.metrics)


def _past_unchanged(
    builder: FeatureBuilder, clean: MetricsFrame, dirty: MetricsFrame, cut: pd.Timestamp
) -> bool:
    before, after = builder.transform(clean), builder.transform(dirty)
    for part in PARTS:
        expected = _keyed(before, part)
        expected = expected[expected.index.get_level_values(TIMESTAMP_COLUMN) <= cut]
        actual = _keyed(after, part).loc[expected.index]
        if not actual.equals(expected):
            return False
    return True


def _cut(ff: FeatureFrame, rng: np.random.Generator) -> pd.Timestamp:
    later = ff.timestamps[ff.timestamps > TRAIN_END + pd.Timedelta(hours=6)].unique()
    return pd.Timestamp(later[rng.integers(len(later) - 12)])


@pytest.mark.parametrize("seed", range(10))
def test_corrupting_the_future_leaves_past_features_bit_identical(
    seed: int, make_metrics: MakeMetrics
) -> None:
    frame = make_metrics(seed=seed)
    builder = _fit(frame)
    rng = np.random.default_rng(seed)
    cut = _cut(builder.transform(frame), rng)
    assert _past_unchanged(builder, frame, _corrupt_after(frame, cut, rng), cut)


def test_negative_control_a_centered_window_is_caught(
    make_metrics: MakeMetrics, monkeypatch: pytest.MonkeyPatch
) -> None:
    def leaky(
        values: pd.DataFrame, groups: pd.Series, window: int, min_periods: int
    ) -> pd.DataFrame:
        rolled = values.groupby(groups, sort=False).rolling(
            window, min_periods=min_periods, center=True
        )
        return rolled.median().droplevel(0).reindex(values.index)

    monkeypatch.setattr(builder_module, "causal_rolling_median", leaky)
    frame = make_metrics(seed=0)
    builder = _fit(frame)
    rng = np.random.default_rng(0)
    cut = _cut(builder.transform(frame), rng)
    assert not _past_unchanged(builder, frame, _corrupt_after(frame, cut, rng), cut)


def _current_point_test(builder: FeatureBuilder, frame: MetricsFrame) -> tuple[bool, bool, bool]:
    """Change one observation and report, in order: the baseline at t is unchanged, z at t
    changed, and the baseline at t + 1 changed."""
    ff = builder.transform(frame)
    position = int(
        np.flatnonzero(
            (ff.timestamps > TRAIN_END + pd.Timedelta(hours=8)).to_numpy() & ff.observed
        )[0]
    )
    host, time = ff.keys.loc[position, "host_id"], ff.keys.loc[position, TIMESTAMP_COLUMN]
    row = frame.data.index[
        (frame.data["host_id"] == host)
        & (frame.data[TIMESTAMP_COLUMN].dt.round(FIVE_MINUTES) == time)
    ][0]
    data = frame.data.copy()
    following_baseline = ff.context.loc[position + 1, "requests_per_second__baseline"]
    current = data.loc[row, "requests_per_second"]
    data.loc[row, "requests_per_second"] = 0.0 if current > following_baseline else current * 1000
    changed = builder.transform(MetricsFrame(data, "host_id", frame.metrics))
    column = "requests_per_second__baseline"
    return (
        ff.context.loc[position, column] == changed.context.loc[position, column],
        ff.X.loc[position, "requests_per_second__z_local"]
        != changed.X.loc[position, "requests_per_second__z_local"],
        ff.context.loc[position + 1, column] != changed.context.loc[position + 1, column],
    )


def test_a_value_never_influences_its_own_baseline(make_metrics: MakeMetrics) -> None:
    frame = make_metrics(seed=2, gap_fraction=0.0)
    assert _current_point_test(_fit(frame), frame) == (True, True, True)


def test_negative_control_b_including_the_current_point_is_caught(
    make_metrics: MakeMetrics, monkeypatch: pytest.MonkeyPatch
) -> None:
    def inclusive(
        values: pd.DataFrame, groups: pd.Series, window: int, min_periods: int
    ) -> pd.DataFrame:
        rolled = values.groupby(groups, sort=False).rolling(window, min_periods=min_periods)
        return rolled.median().droplevel(0).reindex(values.index)

    monkeypatch.setattr(builder_module, "causal_rolling_median", inclusive)
    frame = make_metrics(seed=2, gap_fraction=0.0)
    unchanged_at_t, _, _ = _current_point_test(_fit(frame), frame)
    assert not unchanged_at_t


def test_hosts_never_influence_each_other(make_metrics: MakeMetrics) -> None:
    frame = make_metrics(seed=5)
    builder = _fit(frame)
    data = frame.data.copy()
    is_h0 = (data["host_id"] == "h0").to_numpy()
    data.loc[is_h0, "requests_per_second"] *= 7.0
    before = builder.transform(frame)
    after = builder.transform(MetricsFrame(data, "host_id", frame.metrics))
    others = (before.keys["host_id"] != "h0").to_numpy()
    for part in PARTS:
        assert getattr(before, part).loc[others].equals(getattr(after, part).loc[others]), part


def test_row_order_of_the_input_does_not_matter(make_metrics: MakeMetrics) -> None:
    frame = make_metrics(seed=6)
    builder = _fit(frame)
    shuffled = frame.data.sample(frac=1.0, random_state=1).reset_index(drop=True)
    expected = builder.transform(frame)
    actual = builder.transform(MetricsFrame(shuffled, "host_id", frame.metrics))
    for part in PARTS:
        assert getattr(expected, part).equals(getattr(actual, part)), part


def test_fit_only_sees_training_rows(make_metrics: MakeMetrics) -> None:
    frame = make_metrics(seed=8)
    state = _fit(frame).state
    assert pd.Timestamp(state.training_end) < TRAIN_END
    data = frame.data.copy()
    data.loc[data[TIMESTAMP_COLUMN] >= TRAIN_END + FIVE_MINUTES, "requests_per_second"] *= 50
    assert _fit(MetricsFrame(data, "host_id", frame.metrics)).state == state


def test_warmup_rows_are_not_scorable(make_metrics: MakeMetrics) -> None:
    frame = make_metrics(seed=9, gap_fraction=0.0)
    ff = _fit(frame).transform(frame)
    first = ff.keys.groupby("host_id").head(1).index
    assert not ff.scorable[first].any()
    # 1h local window needing half of it, then 6h scale window needing 12.5 %: ready within ~2h.
    later = ff.timestamps > pd.Timestamp("2026-01-05T03:00:00Z")
    assert ff.scorable[later.to_numpy()].mean() > 0.99


def test_unseen_hosts_use_pooled_floors_and_are_flagged(make_metrics: MakeMetrics) -> None:
    frame = make_metrics(seed=10)
    builder = _fit(frame)
    data = frame.data.copy()
    data["host_id"] = data["host_id"].replace({"h2": "brand-new"})
    ff = builder.transform(MetricsFrame(data, "host_id", frame.metrics))
    unseen = (ff.keys["host_id"] == "brand-new").to_numpy()
    assert not ff.quality.loc[unseen, "entity_seen"].any()
    assert ff.quality.loc[~unseen, "entity_seen"].all()
    assert ff.scorable[unseen].any()


def test_missing_metric_cells_become_neutral_features(make_metrics: MakeMetrics) -> None:
    frame = make_metrics(seed=11, gap_fraction=0.0)
    builder = _fit(frame)
    data = frame.data.copy()
    hole = data.index[(data[TIMESTAMP_COLUMN] > TRAIN_END + pd.Timedelta(hours=12))][:5]
    data.loc[hole, "cpu_usage_percent"] = np.nan
    ff = builder.transform(MetricsFrame(data, "host_id", frame.metrics))
    rows = ff.context["cpu_usage_percent__value"].isna().to_numpy() & ff.observed
    assert rows.sum() == 5
    assert (ff.X.loc[rows, "cpu_usage_percent__z_local"] == 0.0).all()
    assert not ff.available.loc[rows, "cpu_usage_percent"].any()
    assert (ff.quality.loc[rows, "unavailable_metrics"] >= 1).all()


def test_feature_frame_rejects_identifier_and_label_columns(make_metrics: MakeMetrics) -> None:
    frame = make_metrics(seed=12)
    ff = _fit(frame).transform(frame)
    for name in ("host_id__z_local", "is_anomaly", "timestamp"):
        X = ff.X.assign(**{name: 0.0})
        with pytest.raises(ValueError, match="cannot be a feature"):
            FeatureFrame(X, ff.keys, ff.context, ff.available, ff.quality, "host_id", ff.metrics)
    with pytest.raises(ValueError, match="label columns can never be metrics"):
        MetricsFrame(
            frame.data.rename(columns={"http_5xx_rate": "is_anomaly"}),
            "host_id",
            ("requests_per_second", "cpu_usage_percent", "is_anomaly"),
        )


def test_seasonal_baseline_compares_like_with_like() -> None:
    # Two weeks of a host whose traffic doubles at weekends: the seasonal baseline on a Saturday
    # should come from previous weekend days, not from the weekdays just before it.
    times = pd.date_range("2026-01-05T00:00:00Z", periods=14 * 24, freq="1h")
    rps = np.where(times.day_of_week >= 5, 200.0, 100.0)
    data = pd.DataFrame(
        {"timestamp": times.as_unit("ns"), "host_id": "a", "requests_per_second": rps}
    )
    frame = MetricsFrame(data, "host_id", ("requests_per_second",))
    config = FeatureConfig(local_window="3h", scale_window="1D", seasonal_days=7)
    hour = pd.Timedelta("1h")
    ff = (
        FeatureBuilder(config)
        .fit(frame.until(pd.Timestamp("2026-01-12T00:00:00Z"), hour), hour)
        .transform(frame)
    )
    second_saturday = (ff.timestamps == pd.Timestamp("2026-01-17T12:00:00Z")).to_numpy()
    assert ff.context.loc[
        second_saturday, "requests_per_second__seasonal_baseline"
    ].item() == pytest.approx(200.0)
    assert ff.X.loc[second_saturday, "requests_per_second__z_season"].item() == pytest.approx(0.0)
