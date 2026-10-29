from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from network_anomaly.errors import ConfigError
from network_anomaly.evaluation.splits import SplitConfig, plan_split

FIVE_MINUTES = pd.Timedelta("5min")
TIMES = pd.Series(pd.date_range("2026-01-05T00:00:00Z", periods=10 * 288, freq="5min"))


def test_fractions_give_grid_aligned_cuts_shared_by_all_hosts() -> None:
    plan = plan_split(TIMES, SplitConfig(train_fraction=0.5, validation_fraction=0.2), FIVE_MINUTES)
    assert plan.start == pd.Timestamp("2026-01-05T00:00:00Z")
    assert plan.train_end == pd.Timestamp("2026-01-10T00:00:00Z")
    assert plan.validation_end == pd.Timestamp("2026-01-12T00:00:00Z")
    assert plan.end == pd.Timestamp("2026-01-15T00:00:00Z")
    segments = pd.Series(plan.segment(TIMES)).value_counts().to_dict()
    assert segments == {"train": 5 * 288, "validation": 2 * 288, "test": 3 * 288}


def test_explicit_cut_times_override_fractions() -> None:
    config = SplitConfig(
        train_until=datetime(2026, 1, 8, tzinfo=UTC),
        validation_until=datetime(2026, 1, 9, tzinfo=UTC),
    )
    plan = plan_split(TIMES, config, FIVE_MINUTES)
    assert plan.segment(pd.Series([pd.Timestamp("2026-01-08T00:00:00Z")])).tolist() == [
        "validation"
    ]
    assert plan.segment(pd.Series([pd.Timestamp("2026-01-09T00:00:00Z")])).tolist() == ["test"]


def test_cut_times_outside_the_data_are_rejected() -> None:
    config = SplitConfig(train_until=datetime(2027, 1, 1, tzinfo=UTC))
    with pytest.raises(ConfigError, match="outside the data"):
        plan_split(TIMES, config, FIVE_MINUTES)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"train_fraction": 0.8, "validation_fraction": 0.2}, "leave room for a test set"),
        (
            {
                "train_until": datetime(2026, 1, 9, tzinfo=UTC),
                "validation_until": datetime(2026, 1, 8, tzinfo=UTC),
            },
            "must not be before train_until",
        ),
        ({"train_fraction": 0.0}, "greater than 0"),
    ],
)
def test_invalid_split_configs(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        SplitConfig.model_validate(fields)
