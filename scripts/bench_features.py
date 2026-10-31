"""Measure the Python feature kernels before considering a Rust implementation.

The optional Rust boundary (docs/architecture.md, "Rust boundary") would replace the grouped causal
kernels in ``network_anomaly.features.rolling``. This script measures, on a synthetic dataset:

* each kernel's median time over repeated runs, and its share of the whole feature step;
* the full feature step (``FeatureBuilder.fit`` + ``transform``);
* the conversion a Rust boundary would add: pandas frame -> contiguous float64 matrix plus int64
  group offsets, and back;
* the peak resident memory of the process.

It changes nothing; it prints a table (and writes JSON with ``--json``). Usage:

    uv run python scripts/bench_features.py --hosts 20 --days 35 --repeats 3
"""

import argparse
import gc
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from network_anomaly.benchmarks.environment import peak_rss_bytes
from network_anomaly.features.builder import FeatureBuilder, FeatureConfig
from network_anomaly.features.rolling import (
    causal_residual_scale,
    causal_rolling_median,
    causal_seasonal_median,
    causal_shift,
)
from network_anomaly.ingestion.validation import MetricsFrame
from network_anomaly.preprocessing import regularize
from network_anomaly.schemas import HOST_METRICS, TIMESTAMP_COLUMN, metric_spec
from network_anomaly.synthetic.generator import GeneratorConfig, generate_dataset

FREQUENCY = pd.Timedelta("5min")


def median_seconds(action: Callable[[], object], repeats: int) -> float:
    action()  # warm-up
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter_ns()
        action()
        times.append((time.perf_counter_ns() - start) / 1e9)
    return statistics.median(times)


def metrics_frame(hosts: int, days: float) -> MetricsFrame:
    data = generate_dataset(
        GeneratorConfig(seed=0, hosts=hosts, days=days, data_quality=False)
    ).data
    data = data[[TIMESTAMP_COLUMN, "host_id", *HOST_METRICS]].copy()
    data[TIMESTAMP_COLUMN] = pd.to_datetime(data[TIMESTAMP_COLUMN], utc=True).dt.as_unit("ns")
    data = data.sort_values(["host_id", TIMESTAMP_COLUMN]).reset_index(drop=True)
    return MetricsFrame(data, "host_id", HOST_METRICS)


def run(hosts: int, days: float, repeats: int) -> dict[str, Any]:
    metrics = metrics_frame(hosts, days)
    config = FeatureConfig()
    grid = regularize(metrics, FREQUENCY)
    groups = grid.data["host_id"]
    x = pd.DataFrame(
        {
            m: np.log1p(grid.data[m]) if metric_spec(m).log_scale else grid.data[m]
            for m in grid.metrics
        }
    )
    local = config.window_steps(config.local_window, FREQUENCY)
    scale = config.window_steps(config.scale_window, FREQUENCY)
    baseline = causal_rolling_median(x, groups, local, math.ceil(local / 2))
    times = grid.data[TIMESTAMP_COLUMN]
    weekend = (times.dt.day_of_week >= 5).to_numpy()
    lagged = [((times - pd.Timedelta(days=k)).dt.day_of_week >= 5).to_numpy() for k in range(1, 8)]
    kernels: dict[str, Callable[[], object]] = {
        "rolling median (local baseline)": lambda: causal_rolling_median(
            x, groups, local, math.ceil(local / 2)
        ),
        "residual scale (rolling median of absolute residuals)": lambda: causal_residual_scale(
            x, baseline, groups, scale, math.ceil(scale / 8)
        ),
        "seasonal median (same slot, 7 days)": lambda: causal_seasonal_median(
            x,
            groups,
            steps_per_day=288,
            days=7,
            min_values=2,
            weekend=weekend,
            weekend_lagged=lagged,
        ),
        "shift (previous value)": lambda: causal_shift(x, groups, 1),
    }
    results: dict[str, Any] = {"rows": len(grid.data), "metrics": len(grid.metrics), "phases": {}}
    for name, action in kernels.items():
        results["phases"][name] = median_seconds(action, repeats)

    train_end = times.min() + (times.max() - times.min()) / 2
    train = metrics.until(train_end.floor(to_offset(FREQUENCY)), FREQUENCY)

    def full() -> object:
        return FeatureBuilder(config).fit(train, FREQUENCY).transform(metrics)

    results["full_feature_step"] = median_seconds(full, repeats)

    codes = pd.factorize(groups)[0]

    def to_contract() -> tuple[np.ndarray, np.ndarray]:
        matrix = np.ascontiguousarray(x.to_numpy(dtype=np.float64))
        offsets = np.flatnonzero(np.diff(codes, prepend=-1)).astype(np.int64)
        return matrix, offsets

    matrix, _ = to_contract()

    def from_contract() -> pd.DataFrame:
        return pd.DataFrame(matrix, columns=x.columns, index=x.index)

    results["conversion_to_contract"] = median_seconds(to_contract, repeats)
    results["conversion_from_contract"] = median_seconds(from_contract, repeats)
    peak, method = peak_rss_bytes()
    results["peak_rss_mib"] = round(peak / 2**20, 1)
    results["peak_rss_method"] = method
    results["python"] = platform.python_version()
    results["pandas"] = pd.__version__
    return results


def report(results: dict[str, Any]) -> str:
    rows = results["rows"]
    kernel_total = sum(results["phases"].values())
    lines = [
        f"{rows:,} grid rows x {results['metrics']} metrics "
        f"(Python {results['python']}, pandas {results['pandas']})",
        "",
        "| Step | Median s | Rows / s | Share of kernels |",
        "|---|---|---|---|",
    ]
    for name, seconds in results["phases"].items():
        lines.append(
            f"| {name} | {seconds:.3f} | {rows / seconds:,.0f} | {seconds / kernel_total:.0%} |"
        )
    full = results["full_feature_step"]
    convert = results["conversion_to_contract"] + results["conversion_from_contract"]
    lines += [
        f"| full feature step (fit + transform) | {full:.3f} | {rows / full:,.0f} | - |",
        f"| conversion to and from the Rust contract | {convert:.4f} | - | "
        f"{convert / kernel_total:.1%} of kernel time |",
        "",
        f"Peak resident memory: {results['peak_rss_mib']} MiB ({results['peak_rss_method']}).",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hosts", type=int, default=20)
    parser.add_argument("--days", type=float, default=35.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path, help="Also write the measurements as JSON.")
    args = parser.parse_args()
    results = run(args.hosts, args.days, args.repeats)
    print(report(results))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
