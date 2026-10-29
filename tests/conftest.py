import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from network_anomaly.ingestion.validation import MetricsFrame
from network_anomaly.logging_setup import PACKAGE_LOGGER

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SMALL_METRICS = ("requests_per_second", "cpu_usage_percent", "http_5xx_rate")


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(autouse=True)
def _reset_package_logger() -> Iterator[None]:
    """CLI invocations configure the package logger; restore defaults so caplog keeps working."""
    yield
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def random_metrics(
    seed: int = 0,
    *,
    hosts: int = 3,
    days: float = 3.0,
    gap_fraction: float = 0.02,
    jitter_seconds: int = 40,
    start: str = "2026-01-05T00:00:00Z",
) -> MetricsFrame:
    """Small multi-host metrics: random-walk traffic, daily CPU cycles, gaps and clock jitter."""
    rng = np.random.default_rng(seed)
    steps = round(days * 288)
    frames = []
    for host in range(hosts):
        times = pd.date_range(start, periods=steps, freq="5min")
        times = times + pd.to_timedelta(
            rng.integers(-jitter_seconds, jitter_seconds + 1, steps), "s"
        )
        rps = 100.0 * (host + 1) * np.exp(np.cumsum(rng.normal(0.0, 0.02, steps)))
        cpu = 30 + 10 * np.sin(np.arange(steps) / 288 * 2 * np.pi) + rng.normal(0, 2, steps)
        errors = np.clip(rng.normal(0.002, 0.001, steps), 0.0, 1.0)
        keep = rng.random(steps) >= gap_fraction
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": times[keep],
                    "host_id": f"h{host}",
                    "requests_per_second": rps[keep],
                    "cpu_usage_percent": np.clip(cpu, 0, 100)[keep],
                    "http_5xx_rate": errors[keep],
                }
            )
        )
    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = data["timestamp"].dt.as_unit("ns")
    data = data.sort_values(["host_id", "timestamp"]).reset_index(drop=True)
    return MetricsFrame(data=data, entity_column="host_id", metrics=SMALL_METRICS)


@pytest.fixture(scope="session")
def make_metrics() -> Callable[..., MetricsFrame]:
    return random_metrics
