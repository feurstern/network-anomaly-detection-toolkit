"""Generate paired synthetic datasets with point labels, event labels and full provenance.

Every dataset is produced twice from the same random draws: once with anomaly scenarios injected
and once without (the *clean twin*). Labels come from comparing the two copies, so each event's
label interval covers the injection plus any after-effect that still moves a metric by more than
one noise standard deviation (for example memory recovering after a restart). The clean twin
doubles as normal-only data for measuring false positives.

Randomness comes from independent streams keyed by ``(dataset_index, stream, host_index)``, so
adding a host or changing the anomaly schedule never alters the noise of other hosts.
"""

import hashlib
import json
import logging
import math
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import AwareDatetime, Field, field_validator

from network_anomaly import __version__
from network_anomaly.config import StrictModel, parse_frequency
from network_anomaly.errors import ConfigError
from network_anomaly.ingestion.events import (
    format_timestamp,
    sibling_path,
    write_data_quality_events,
    write_events,
)
from network_anomaly.schemas import (
    LABEL_COLUMNS,
    METRIC_REGISTRY,
    TIMESTAMP_COLUMN,
    DataQualityEvent,
    Event,
    MetricSpec,
)
from network_anomaly.synthetic.scenarios import (
    METRIC_ORDER,
    PRESETS,
    SCENARIOS,
    DataQualityPlan,
    PlannedEvent,
    Preset,
    PresetParams,
    ScenarioName,
    ScenarioParams,
    plan_data_quality,
    schedule_events,
)
from network_anomaly.synthetic.system import (
    BaselineShift,
    Drivers,
    FloatArray,
    baseline_demand,
    draw_noise,
    sample_host_profile,
    simulate_host,
)

logger = logging.getLogger(__name__)

GENERATOR_VERSION = "1.0.0"
ENTITY_COLUMN = "host_id"
STREAM_HOST, STREAM_NOISE, STREAM_EVENTS, STREAM_DATA_QUALITY = 0, 1, 2, 3
LABEL_EXTENSION_SIGMA = 1.0
DETECTABLE_SIGMA = 2.0
MAX_LABEL_EXTENSION = pd.Timedelta(hours=24)
DEFAULT_START = datetime(2026, 1, 5, tzinfo=UTC)  # a Monday, so weeks start cleanly

# Decimal places written per metric: compact files without losing meaningful precision.
DECIMALS: dict[str, int] = {
    "requests_per_second": 3,
    "http_4xx_rate": 6,
    "http_5xx_rate": 6,
    "response_time_p50_ms": 3,
    "response_time_p95_ms": 3,
    "cpu_usage_percent": 3,
    "memory_usage_percent": 3,
    "disk_io_bytes_per_second": 1,
    "network_rx_bytes_per_second": 1,
    "network_tx_bytes_per_second": 1,
    "active_connections": 0,
    "connection_error_rate": 6,
}


class GeneratorConfig(StrictModel):
    """Parameters of one synthetic dataset. Every field is recorded in the dataset metadata."""

    seed: int = Field(default=42, ge=0)
    preset: Preset = Preset.MODERATE
    hosts: int = Field(default=6, ge=1, le=1000)
    days: float = Field(default=35.0, gt=0, le=731)
    freq: str = "5min"
    start: AwareDatetime = DEFAULT_START
    scenarios: tuple[ScenarioName, ...] = tuple(ScenarioName)
    events_per_host_week: float = Field(default=1.5, ge=0, le=50)
    min_event_gap_hours: float = Field(default=6.0, ge=0)
    data_quality: bool = True
    exclusion_zones: tuple[tuple[AwareDatetime, AwareDatetime], ...] = Field(
        default=(),
        description="Intervals in which no event may occur, e.g. around benchmark split points.",
    )
    exclusion_margin_hours: float = Field(default=2.0, ge=0)

    @field_validator("freq")
    @classmethod
    def _check_freq(cls, value: str) -> str:
        try:
            seconds = parse_frequency(value).total_seconds()
        except ConfigError as exc:
            raise ValueError(exc.message) from exc
        if not 60 <= seconds <= 3600 or 86_400 % seconds:
            raise ValueError("must be between 1min and 1h and divide one day evenly")
        return value

    @field_validator("scenarios")
    @classmethod
    def _dedupe(cls, value: tuple[ScenarioName, ...]) -> tuple[ScenarioName, ...]:
        return tuple(dict.fromkeys(value))

    @property
    def step(self) -> pd.Timedelta:
        return parse_frequency(self.freq)

    @property
    def n_steps(self) -> int:
        return round(self.days * 86_400 / self.step.total_seconds())


@dataclass(frozen=True)
class SyntheticDataset:
    """One generated dataset: anomalous data with labels, its clean twin and ground truth."""

    data: pd.DataFrame
    clean: pd.DataFrame
    events: tuple[Event, ...]
    data_quality_events: tuple[DataQualityEvent, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _HostResult:
    data: pd.DataFrame
    clean: pd.DataFrame
    events: list[Event]
    data_quality_events: list[DataQualityEvent]
    metadata: dict[str, Any]


def _rng(seed: int, dataset_index: int, stream: int, host_index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(seed, spawn_key=(dataset_index, stream, host_index))
    return np.random.default_rng(sequence)


def _transform(values: FloatArray, spec: MetricSpec) -> FloatArray:
    return np.log1p(values) if spec.log_scale else values


def noise_sigma(values: FloatArray, spec: MetricSpec) -> float:
    """Robust step-to-step noise scale of a series, in the metric's modelling space."""
    diffs = np.diff(_transform(values, spec))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return spec.min_scale
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    return max(1.4826 * mad / math.sqrt(2.0), spec.min_scale)


def _time_at(times: pd.DatetimeIndex, step: int, delta: pd.Timedelta) -> datetime:
    stamp = times[step] if step < len(times) else times[-1] + delta
    return stamp.to_pydatetime()


def _sample_shifts(
    rng: np.random.Generator,
    preset: PresetParams,
    n_steps: int,
    first_seen: int,
    steps_per_day: int,
) -> list[BaselineShift]:
    count = int(rng.poisson(preset.baseline_shifts_per_host))
    low, high = first_seen + 2 * steps_per_day, n_steps - steps_per_day
    shifts = []
    for _ in range(count):
        if high <= low:
            break
        sign = 1.0 if rng.random() < 0.5 else -1.0
        width = rng.uniform(30, 120) * 60 * steps_per_day / 86_400  # 30-120 minutes, in steps
        shifts.append(
            BaselineShift(int(rng.integers(low, high)), sign * rng.uniform(0.1, 0.3), width)
        )
    return sorted(shifts, key=lambda shift: shift.step)


def _label_events(
    host_id: str,
    planned: list[PlannedEvent],
    params: list[ScenarioParams],
    clean: dict[str, FloatArray],
    anomalous: dict[str, FloatArray],
    sigma: dict[str, float],
    times: pd.DatetimeIndex,
    step: pd.Timedelta,
) -> tuple[dict[str, npt.NDArray[Any]], list[Event]]:
    n_steps = len(times)
    effect = np.vstack(
        [
            np.abs(
                _transform(anomalous[m], METRIC_REGISTRY[m])
                - _transform(clean[m], METRIC_REGISTRY[m])
            )
            / sigma[m]
            for m in METRIC_ORDER
        ]
    )
    strongest = effect.max(axis=0)
    max_extension = int(MAX_LABEL_EXTENSION / step)
    labels: dict[str, npt.NDArray[Any]] = {
        "is_anomaly": np.zeros(n_steps, dtype=np.int8),
        "anomaly_event_id": np.full(n_steps, "", dtype=object),
        "anomaly_scenario": np.full(n_steps, "", dtype=object),
        "anomaly_severity": np.full(n_steps, "", dtype=object),
    }
    events: list[Event] = []
    for index, (plan, plan_params) in enumerate(zip(planned, params, strict=True)):
        next_start = planned[index + 1].start if index + 1 < len(planned) else n_steps
        limit = min(n_steps, plan.stop + max_extension, next_start)
        end = plan.stop
        while end < limit and strongest[end] > LABEL_EXTENSION_SIGMA:
            end += 1
        window = slice(plan.start, end)
        peak_by_metric = effect[:, window].max(axis=1)
        order = np.argsort(-peak_by_metric, kind="stable")
        affected = tuple(METRIC_ORDER[j] for j in order if peak_by_metric[j] > DETECTABLE_SIGMA)
        visible = np.flatnonzero(strongest[window] > DETECTABLE_SIGMA)
        event_id = f"{host_id}-ev{index + 1:03d}"
        labels["is_anomaly"][window] = 1
        labels["anomaly_event_id"][window] = event_id
        labels["anomaly_scenario"][window] = plan.scenario.value
        labels["anomaly_severity"][window] = plan.severity
        events.append(
            Event(
                event_id=event_id,
                host_id=host_id,
                scenario=plan.scenario.value,
                severity=plan.severity,
                start=_time_at(times, plan.start, step),
                end=_time_at(times, end, step),
                detectable_from=_time_at(times, plan.start + int(visible[0]), step)
                if visible.size
                else None,
                affected_metrics=affected,
                peak_effect_sigma=round(float(peak_by_metric.max()), 2),
                params={
                    **{
                        key: round(value, 4) if isinstance(value, float) else value
                        for key, value in plan_params.items()
                    },
                    "magnitude": round(plan.magnitude, 4),
                    "injection_end": format_timestamp(_time_at(times, plan.stop, step)),
                },
            )
        )
    return labels, events


def _apply_data_quality(
    values: dict[str, FloatArray], plan: DataQualityPlan, sigma: dict[str, float]
) -> dict[str, FloatArray]:
    observed: dict[str, FloatArray] = {}
    for j, metric in enumerate(METRIC_ORDER):
        spec = METRIC_REGISTRY[metric]
        series = values[metric].copy()
        burst = plan.noise_scale[:, j] > 0
        if burst.any():
            shift = plan.noise_scale[burst, j] * sigma[metric] * plan.noise_draws[burst, j]
            if spec.log_scale:
                series[burst] = np.expm1(np.log1p(series[burst]) + shift)
            else:
                series[burst] = series[burst] + shift
        spike = plan.glitch_mode[:, j] == 1
        if spike.any():
            level = plan.glitch_level[spike, j]
            if spec.unit == "%":
                series[spike] = 100.0
            elif spec.unit == "fraction":
                series[spike] = 0.3 + 0.7 * level
            else:
                series[spike] = series[spike] * (5.0 + 15.0 * level)
        series[plan.glitch_mode[:, j] == 2] = 0.0
        series = np.clip(series, spec.min_value, spec.max_value)
        series[plan.missing[:, j]] = np.nan
        observed[metric] = series
    return observed


def _host_frame(
    host_id: str,
    stamps: pd.DatetimeIndex,
    values: dict[str, FloatArray],
    labels: dict[str, npt.NDArray[Any]] | None,
    keep: npt.NDArray[np.bool_],
) -> pd.DataFrame:
    columns: dict[str, Any] = {TIMESTAMP_COLUMN: stamps[keep], ENTITY_COLUMN: host_id}
    for metric in METRIC_ORDER:
        columns[metric] = np.round(values[metric][keep], DECIMALS[metric])
    if labels is None:
        columns["is_anomaly"] = np.zeros(int(keep.sum()), dtype=np.int8)
        for column in LABEL_COLUMNS[1:]:
            columns[column] = ""
    else:
        for column in LABEL_COLUMNS:
            columns[column] = labels[column][keep]
    return pd.DataFrame(columns)


def _generate_host(
    config: GeneratorConfig,
    preset: PresetParams,
    dataset_index: int,
    host_index: int,
    times: pd.DatetimeIndex,
    blocked: npt.NDArray[np.bool_],
) -> _HostResult:
    n_steps = len(times)
    step = config.step
    step_seconds = step.total_seconds()
    steps_per_day = round(86_400 / step_seconds)
    host_id = f"web-{host_index + 1:02d}"

    host_rng = _rng(config.seed, dataset_index, STREAM_HOST, host_index)
    late = bool(host_rng.random() < preset.late_host_share)
    profile = sample_host_profile(host_id, host_rng, n_steps=n_steps, late_start=late)
    shifts = _sample_shifts(host_rng, preset, n_steps, profile.first_seen_step, steps_per_day)

    local = times + pd.Timedelta(hours=profile.phase_hours)
    local_hour = np.asarray(local.hour + local.minute / 60 + local.second / 3600, dtype=np.float64)
    local_weekday = np.asarray(local.day_of_week, dtype=np.int64)

    noise_rng = _rng(config.seed, dataset_index, STREAM_NOISE, host_index)
    noise = draw_noise(noise_rng, n_steps, heavy_tails=preset.noise.heavy_tails)
    demand = baseline_demand(profile, local_hour, local_weekday, shifts, noise, preset.noise)

    events_rng = _rng(config.seed, dataset_index, STREAM_EVENTS, host_index)
    planned = schedule_events(
        events_rng,
        n_steps=n_steps,
        first_seen_step=profile.first_seen_step,
        step_seconds=step_seconds,
        scenarios=config.scenarios,
        preset=preset,
        events_per_host_week=config.events_per_host_week,
        min_gap_steps=round(config.min_event_gap_hours * 3600 / step_seconds),
        blocked=blocked,
    )
    drivers = Drivers.neutral(n_steps)
    params = [
        SCENARIOS[event.scenario].apply(
            drivers, event.start, event.stop, event.magnitude, events_rng
        )
        for event in planned
    ]

    clean, anomalous = (
        simulate_host(profile, demand, host_drivers, noise, preset.noise, local_hour, step_seconds)
        for host_drivers in (Drivers.neutral(n_steps), drivers)
    )
    active = slice(profile.first_seen_step, n_steps)
    sigma = {m: noise_sigma(clean[m][active], METRIC_REGISTRY[m]) for m in METRIC_ORDER}
    labels, events = _label_events(host_id, planned, params, clean, anomalous, sigma, times, step)

    dq_rng = _rng(config.seed, dataset_index, STREAM_DATA_QUALITY, host_index)
    plan = plan_data_quality(
        dq_rng,
        n_steps=n_steps,
        first_seen_step=profile.first_seen_step,
        step_seconds=step_seconds,
        rates=preset.data_quality,
        enabled=config.data_quality,
    )
    keep = np.zeros(n_steps, dtype=bool)
    keep[profile.first_seen_step :] = True
    keep &= ~plan.drop_rows
    stamps = times + pd.to_timedelta(plan.jitter_seconds, unit="s")
    data = _host_frame(host_id, stamps, _apply_data_quality(anomalous, plan, sigma), labels, keep)
    clean_frame = _host_frame(host_id, stamps, _apply_data_quality(clean, plan, sigma), None, keep)
    dq_events = [
        DataQualityEvent(
            dq_event_id=f"{host_id}-dq{index:03d}",
            host_id=host_id,
            kind=event.kind,
            start=_time_at(times, event.start, step),
            end=_time_at(times, event.stop, step),
            metrics=event.metrics,
            detail=event.detail,
        )
        for index, event in enumerate(plan.events, start=1)
    ]
    profile_fields = {k: v for k, v in asdict(profile).items() if k != "host_id"}
    metadata = {
        "host_id": host_id,
        "first_seen": format_timestamp(_time_at(times, profile.first_seen_step, step)),
        "late_start": late,
        "profile": profile_fields,
        "baseline_shifts": [
            {
                "at": format_timestamp(_time_at(times, shift.step, step)),
                "relative_change": round(shift.factor, 4),
            }
            for shift in shifts
        ],
        "noise_sigma": {metric: round(value, 6) for metric, value in sigma.items()},
    }
    return _HostResult(data, clean_frame, events, dq_events, metadata)


def _blocked_steps(config: GeneratorConfig, times: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
    blocked = np.zeros(len(times), dtype=bool)
    margin = pd.Timedelta(hours=config.exclusion_margin_hours)
    for start, end in config.exclusion_zones:
        blocked |= (times >= pd.Timestamp(start) - margin) & (times < pd.Timestamp(end) + margin)
    return blocked


def generate_dataset(config: GeneratorConfig, dataset_index: int = 0) -> SyntheticDataset:
    """Generate one dataset. Identical ``config`` and ``dataset_index`` give identical output."""
    preset = PRESETS[config.preset]
    start = pd.Timestamp(config.start).tz_convert("UTC")
    times = pd.date_range(start, periods=config.n_steps, freq=config.step).as_unit("ns")
    blocked = _blocked_steps(config, times)
    hosts = [
        _generate_host(config, preset, dataset_index, index, times, blocked)
        for index in range(config.hosts)
    ]
    order = [TIMESTAMP_COLUMN, ENTITY_COLUMN]
    data = pd.concat([host.data for host in hosts], ignore_index=True)
    data = data.sort_values(order, kind="stable").reset_index(drop=True)
    clean = pd.concat([host.clean for host in hosts], ignore_index=True)
    clean = clean.sort_values(order, kind="stable").reset_index(drop=True)
    events = tuple(
        sorted((e for host in hosts for e in host.events), key=lambda e: (e.start, e.host_id))
    )
    dq_events = tuple(
        sorted(
            (e for host in hosts for e in host.data_quality_events),
            key=lambda e: (e.start, e.host_id),
        )
    )
    metadata = {
        "generator": {"version": GENERATOR_VERSION, "toolkit_version": __version__},
        "seed": config.seed,
        "dataset_index": dataset_index,
        "config": config.model_dump(mode="json"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "time_range": {
            "start": format_timestamp(times[0].to_pydatetime()),
            "end": format_timestamp((times[-1] + config.step).to_pydatetime()),
            "freq": config.freq,
            "steps": config.n_steps,
        },
        "rows": len(data),
        "anomalous_rows": int(data["is_anomaly"].sum()),
        "events_by_scenario": dict(sorted(Counter(e.scenario for e in events).items())),
        "data_quality_events_by_kind": dict(sorted(Counter(e.kind for e in dq_events).items())),
        "units": {m: METRIC_REGISTRY[m].unit for m in METRIC_ORDER},
        "metric_definitions": {m: METRIC_REGISTRY[m].description for m in METRIC_ORDER},
        "label_definitions": {
            "is_anomaly": "1 while an injected scenario visibly changes any metric (> 1 noise sd)",
            "anomaly_event_id": "Event identifier, matching <name>.events.csv",
            "anomaly_scenario": "Injected scenario name",
            "anomaly_severity": "Injected magnitude tier: low, medium or high",
        },
        "hosts": [host.metadata for host in hosts],
    }
    logger.info(
        "generated synthetic dataset",
        extra={"rows": len(data), "events": len(events), "dq_events": len(dq_events)},
    )
    return SyntheticDataset(data, clean, events, dq_events, metadata)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_metrics_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%SZ", na_rep=""
    )


def write_dataset(
    dataset: SyntheticDataset, path: Path, *, with_clean_twin: bool = False
) -> dict[str, Path]:
    """Write the dataset CSV, its events, data-quality events, optional clean twin and metadata."""
    if path.suffix != ".csv":
        raise ConfigError(f"output path must end in .csv, got {path}")
    paths = {
        "data": path,
        "events": sibling_path(path, "events.csv"),
        "data_quality_events": sibling_path(path, "dq_events.csv"),
    }
    if with_clean_twin:
        paths["clean"] = sibling_path(path, "clean.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(dataset.data, paths["data"])
    write_events(dataset.events, paths["events"])
    write_data_quality_events(dataset.data_quality_events, paths["data_quality_events"])
    if with_clean_twin:
        _write_metrics_csv(dataset.clean, paths["clean"])
    metadata = {
        **dataset.metadata,
        "files": {name: {"file": p.name, "sha256": _sha256(p)} for name, p in paths.items()},
    }
    metadata_path = sibling_path(path, "metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    paths["metadata"] = metadata_path
    return paths


def dataset_paths(output: Path, n_datasets: int) -> list[Path]:
    """One path per dataset: ``demo.csv`` or ``demo_00.csv, demo_01.csv, ...``."""
    if n_datasets == 1:
        return [output]
    stem = output.name.removesuffix(".csv")
    return [output.with_name(f"{stem}_{index:02d}.csv") for index in range(n_datasets)]
