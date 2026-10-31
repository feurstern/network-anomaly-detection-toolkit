"""Anomaly scenarios, data-quality problems, difficulty presets and event scheduling.

Scenarios edit :class:`~network_anomaly.synthetic.system.Drivers` over a window of steps; the system
model turns those edits into correlated symptoms. Data-quality problems instead edit the *observed*
table (dropped rows, missing cells, extra noise, glitches, clock jitter). They are recorded as
data-quality events and are never anomaly labels.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from network_anomaly.schemas import HOST_METRICS, DataQualityKind, EventSeverity
from network_anomaly.synthetic.system import Drivers, FloatArray, NoiseLevels

BoolArray = npt.NDArray[np.bool_]
SEVERITIES: tuple[EventSeverity, ...] = ("low", "medium", "high")


class ScenarioName(StrEnum):
    TRAFFIC_SPIKE = "traffic_spike"
    TRAFFIC_RAMP = "traffic_ramp"
    TRAFFIC_BURST = "traffic_burst"
    HTTP_4XX_SURGE = "http_4xx_surge"
    HTTP_5XX_SURGE = "http_5xx_surge"
    LATENCY_DEGRADATION = "latency_degradation"
    CPU_SATURATION = "cpu_saturation"
    MEMORY_PRESSURE = "memory_pressure"
    CONNECTION_SURGE = "connection_surge"
    THROUGHPUT_SPIKE = "throughput_spike"
    CORRELATED_DEGRADATION = "correlated_degradation"


ScenarioParams = dict[str, float | str]
Apply = Callable[[Drivers, int, int, float, np.random.Generator], ScenarioParams]


@dataclass(frozen=True)
class ScenarioSpec:
    name: ScenarioName
    description: str
    duration_minutes: tuple[float, float]
    magnitude: Mapping[EventSeverity, tuple[float, float]]
    apply: Apply
    gradual: bool = False


def _traffic_spike(
    d: Drivers, start: int, stop: int, m: float, _: np.random.Generator
) -> ScenarioParams:
    d.demand_mult[start:stop] *= m
    return {"demand_multiplier": m}


def _traffic_ramp(
    d: Drivers, start: int, stop: int, m: float, _: np.random.Generator
) -> ScenarioParams:
    length = stop - start
    ramp = max(1, round(0.7 * length))
    profile = np.concatenate([np.linspace(1.0, m, ramp + 1)[1:], np.full(length - ramp, m)])
    d.demand_mult[start:stop] *= profile
    return {"demand_multiplier": m, "ramp_steps": float(ramp)}


def _http_4xx(
    d: Drivers, start: int, stop: int, m: float, _: np.random.Generator
) -> ScenarioParams:
    d.p_4xx_add[start:stop] += m
    return {"added_4xx_probability": m}


def _http_5xx(
    d: Drivers, start: int, stop: int, m: float, _: np.random.Generator
) -> ScenarioParams:
    d.p_5xx_add[start:stop] += m
    return {"added_5xx_probability": m}


def _latency(d: Drivers, start: int, stop: int, m: float, _: np.random.Generator) -> ScenarioParams:
    d.service_time_mult[start:stop] *= m
    return {"service_time_multiplier": m}


def _cpu(d: Drivers, start: int, stop: int, m: float, _: np.random.Generator) -> ScenarioParams:
    d.cpu_background_pct[start:stop] += m
    return {"background_cpu_percent": m}


def _memory(d: Drivers, start: int, stop: int, m: float, _: np.random.Generator) -> ScenarioParams:
    # A leak grows linearly until the process is restarted at the end of the window.
    d.memory_leak_pct[start:stop] += np.linspace(0.0, m, stop - start)
    return {"leaked_memory_percent": m}


def _connections(
    d: Drivers, start: int, stop: int, m: float, _: np.random.Generator
) -> ScenarioParams:
    d.hold_time_mult[start:stop] *= m
    return {"hold_time_multiplier": m}


def _throughput(
    d: Drivers, start: int, stop: int, m: float, rng: np.random.Generator
) -> ScenarioParams:
    if rng.random() < 0.7:
        d.tx_bytes_mult[start:stop] *= m
        return {"bytes_multiplier": m, "direction": "tx"}
    d.rx_bytes_mult[start:stop] *= m
    return {"bytes_multiplier": m, "direction": "rx"}


def _correlated(
    d: Drivers, start: int, stop: int, m: float, _: np.random.Generator
) -> ScenarioParams:
    d.service_time_mult[start:stop] *= m
    d.cpu_cost_mult[start:stop] *= 1.0 + 0.4 * (m - 1.0)
    d.p_5xx_add[start:stop] += 0.015 * (m - 1.0)
    d.demand_mult[start:stop] *= 0.92  # clients back off while the service struggles
    return {"service_time_multiplier": m, "cpu_cost_multiplier": 1.0 + 0.4 * (m - 1.0)}


def _spec(
    name: ScenarioName,
    description: str,
    minutes: tuple[float, float],
    low: tuple[float, float],
    medium: tuple[float, float],
    high: tuple[float, float],
    apply: Apply,
    *,
    gradual: bool = False,
) -> ScenarioSpec:
    tiers: dict[EventSeverity, tuple[float, float]] = {"low": low, "medium": medium, "high": high}
    return ScenarioSpec(name, description, minutes, MappingProxyType(tiers), apply, gradual)


_S = ScenarioName
SCENARIOS: Mapping[ScenarioName, ScenarioSpec] = MappingProxyType(
    {
        spec.name: spec
        for spec in (
            _spec(
                _S.TRAFFIC_SPIKE,
                "Sudden multiplicative jump in request demand.",
                (20, 120),
                (1.5, 2.0),
                (2.0, 3.0),
                (3.0, 6.0),
                _traffic_spike,
            ),
            _spec(
                _S.TRAFFIC_RAMP,
                "Demand rises gradually over hours, then plateaus.",
                (180, 600),
                (1.3, 1.6),
                (1.6, 2.2),
                (2.2, 3.5),
                _traffic_ramp,
                gradual=True,
            ),
            _spec(
                _S.TRAFFIC_BURST,
                "Very short, very large burst of requests.",
                (5, 15),
                (2.5, 4.0),
                (4.0, 8.0),
                (8.0, 15.0),
                _traffic_spike,
            ),
            _spec(
                _S.HTTP_4XX_SURGE,
                "More requests rejected with 4xx (e.g. broken client).",
                (15, 90),
                (0.02, 0.05),
                (0.05, 0.12),
                (0.12, 0.35),
                _http_4xx,
            ),
            _spec(
                _S.HTTP_5XX_SURGE,
                "Server-side failures returning 5xx.",
                (10, 60),
                (0.005, 0.015),
                (0.015, 0.05),
                (0.05, 0.2),
                _http_5xx,
            ),
            _spec(
                _S.LATENCY_DEGRADATION,
                "Slower request handling (e.g. a slow dependency).",
                (20, 120),
                (1.4, 1.8),
                (1.8, 3.0),
                (3.0, 6.0),
                _latency,
            ),
            _spec(
                _S.CPU_SATURATION,
                "Background CPU load steals capacity (e.g. runaway job).",
                (20, 120),
                (10, 20),
                (20, 35),
                (35, 60),
                _cpu,
            ),
            _spec(
                _S.MEMORY_PRESSURE,
                "Memory leak until restart; swapping near the limit.",
                (120, 480),
                (10, 20),
                (20, 35),
                (35, 55),
                _memory,
                gradual=True,
            ),
            _spec(
                _S.CONNECTION_SURGE,
                "Connections held far longer (e.g. slow clients).",
                (15, 90),
                (2.0, 4.0),
                (4.0, 10.0),
                (10.0, 30.0),
                _connections,
            ),
            _spec(
                _S.THROUGHPUT_SPIKE,
                "Unusual network volume per request (rx or tx).",
                (15, 120),
                (1.8, 2.5),
                (2.5, 5.0),
                (5.0, 12.0),
                _throughput,
            ),
            _spec(
                _S.CORRELATED_DEGRADATION,
                "Slower, costlier requests with rising 5xx.",
                (30, 180),
                (1.3, 1.7),
                (1.7, 2.5),
                (2.5, 4.0),
                _correlated,
            ),
        )
    }
)


class Preset(StrEnum):
    EASY = "easy"
    MODERATE = "moderate"
    DIFFICULT = "difficult"


@dataclass(frozen=True)
class DataQualityRates:
    missing_cell_probability: float
    telemetry_gaps_per_host_week: float
    metric_dropouts_per_host_week: float
    jitter_row_fraction: float
    noise_bursts_per_host_week: float
    glitches_per_host_week: float


@dataclass(frozen=True)
class PresetParams:
    noise: NoiseLevels
    severity_weights: Mapping[EventSeverity, float]
    gradual_weight: float  # sampling weight of gradual scenarios relative to abrupt ones
    baseline_shifts_per_host: float  # expected legitimate level shifts per host per dataset
    late_host_share: float  # probability that a host only starts reporting late
    data_quality: DataQualityRates


PRESETS: Mapping[Preset, PresetParams] = MappingProxyType(
    {
        Preset.EASY: PresetParams(
            noise=NoiseLevels(0.03, 0.90, 0.8, 0.2, 0.04, 0.03, 0.02, heavy_tails=False),
            severity_weights={"low": 0.1, "medium": 0.3, "high": 0.6},
            gradual_weight=0.5,
            baseline_shifts_per_host=0.0,
            late_host_share=0.0,
            data_quality=DataQualityRates(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        Preset.MODERATE: PresetParams(
            noise=NoiseLevels(0.06, 0.93, 1.5, 0.4, 0.08, 0.06, 0.04, heavy_tails=False),
            severity_weights={"low": 0.3, "medium": 0.4, "high": 0.3},
            gradual_weight=1.0,
            baseline_shifts_per_host=1.0,
            late_host_share=0.1,
            data_quality=DataQualityRates(0.001, 0.5, 0.3, 0.02, 0.3, 0.5),
        ),
        Preset.DIFFICULT: PresetParams(
            noise=NoiseLevels(0.10, 0.95, 2.5, 0.8, 0.14, 0.10, 0.07, heavy_tails=True),
            severity_weights={"low": 0.55, "medium": 0.35, "high": 0.10},
            gradual_weight=2.0,
            baseline_shifts_per_host=2.0,
            late_host_share=0.2,
            data_quality=DataQualityRates(0.005, 1.5, 1.0, 0.10, 1.0, 2.0),
        ),
    }
)


@dataclass(frozen=True)
class PlannedEvent:
    scenario: ScenarioName
    severity: EventSeverity
    start: int
    stop: int  # end of the injection (exclusive); labels may extend past it
    magnitude: float


def schedule_events(
    rng: np.random.Generator,
    *,
    n_steps: int,
    first_seen_step: int,
    step_seconds: float,
    scenarios: Sequence[ScenarioName],
    preset: PresetParams,
    events_per_host_week: float,
    min_gap_steps: int,
    blocked: BoolArray,
) -> list[PlannedEvent]:
    """Place non-overlapping events on one host, at least ``min_gap_steps`` apart.

    Events start no earlier than one day after the host first reports and never touch ``blocked``
    steps (e.g. around benchmark split boundaries). Events that cannot be placed are skipped.
    """
    steps_per_day = round(86_400 / step_seconds)
    earliest = first_seen_step + steps_per_day
    if not scenarios or events_per_host_week <= 0 or earliest >= n_steps - 1:
        return []
    weeks = (n_steps - first_seen_step) / (7 * steps_per_day)
    count = int(rng.poisson(events_per_host_week * weeks))
    weights = np.array([preset.gradual_weight if SCENARIOS[s].gradual else 1.0 for s in scenarios])
    weights = weights / weights.sum()
    severity_weights = np.array([preset.severity_weights[s] for s in SEVERITIES])
    severity_weights = severity_weights / severity_weights.sum()

    planned: list[PlannedEvent] = []
    for _ in range(count):
        scenario = scenarios[int(rng.choice(len(scenarios), p=weights))]
        spec = SCENARIOS[scenario]
        severity = SEVERITIES[int(rng.choice(len(SEVERITIES), p=severity_weights))]
        magnitude = float(rng.uniform(*spec.magnitude[severity]))
        minutes = float(rng.uniform(*spec.duration_minutes))
        length = max(1, round(minutes * 60 / step_seconds))
        latest = n_steps - length - 1
        if latest <= earliest:
            continue
        for _attempt in range(50):
            start = int(rng.integers(earliest, latest))
            stop = start + length
            if blocked[start:stop].any():
                continue
            if any(
                start < other.stop + min_gap_steps and other.start - min_gap_steps < stop
                for other in planned
            ):
                continue
            planned.append(PlannedEvent(scenario, severity, start, stop, magnitude))
            break
    return sorted(planned, key=lambda event: event.start)


# --------------------------------------------------------------------------------------------------
# Data-quality problems
# --------------------------------------------------------------------------------------------------

METRIC_ORDER: tuple[str, ...] = HOST_METRICS


@dataclass(frozen=True)
class PlannedDataQuality:
    kind: DataQualityKind
    start: int
    stop: int
    metrics: tuple[str, ...] = ()
    detail: str = ""


@dataclass
class DataQualityPlan:
    """Edits to the observed table of one host, applied identically to both paired copies."""

    drop_rows: BoolArray
    missing: BoolArray  # (steps, metrics)
    noise_scale: FloatArray  # (steps, metrics): extra noise in noise-sigma units, 0 = none
    noise_draws: FloatArray  # (steps, metrics)
    glitch_mode: npt.NDArray[np.int8]  # (steps, metrics): 0 none, 1 spike, 2 drop to zero
    glitch_level: FloatArray  # (steps, metrics): uniform draw setting the spike size
    jitter_seconds: FloatArray  # (steps,)
    events: list[PlannedDataQuality] = field(default_factory=list)


def plan_data_quality(
    rng: np.random.Generator,
    *,
    n_steps: int,
    first_seen_step: int,
    step_seconds: float,
    rates: DataQualityRates,
    enabled: bool,
) -> DataQualityPlan:
    """Plan data-quality problems for one host. Draw order is fixed for reproducibility."""
    n_metrics = len(METRIC_ORDER)
    plan = DataQualityPlan(
        drop_rows=np.zeros(n_steps, dtype=bool),
        missing=np.zeros((n_steps, n_metrics), dtype=bool),
        noise_scale=np.zeros((n_steps, n_metrics)),
        noise_draws=rng.standard_normal((n_steps, n_metrics)),
        glitch_mode=np.zeros((n_steps, n_metrics), dtype=np.int8),
        glitch_level=rng.uniform(size=(n_steps, n_metrics)),
        jitter_seconds=np.zeros(n_steps),
    )
    if not enabled:
        return plan
    active = n_steps - first_seen_step
    steps_per_hour = 3600 / step_seconds
    weeks = active / (7 * 24 * steps_per_hour)

    def windows(per_week: float, hours: tuple[float, float]) -> list[tuple[int, int]]:
        found = []
        for _ in range(int(rng.poisson(per_week * weeks))):
            length = max(1, round(rng.uniform(*hours) * steps_per_hour))
            start = int(rng.integers(first_seen_step, max(first_seen_step + 1, n_steps - length)))
            found.append((start, min(n_steps, start + length)))
        return found

    for start, stop in windows(rates.telemetry_gaps_per_host_week, (10 / 60, 2.0)):
        plan.drop_rows[start:stop] = True
        plan.events.append(PlannedDataQuality("telemetry_gap", start, stop))
    for start, stop in windows(rates.metric_dropouts_per_host_week, (1.0, 6.0)):
        chosen = np.sort(rng.choice(n_metrics, size=int(rng.integers(2, 5)), replace=False))
        plan.missing[start:stop, chosen] = True
        names = tuple(METRIC_ORDER[i] for i in chosen)
        plan.events.append(PlannedDataQuality("metric_dropout", start, stop, names))
    for start, stop in windows(rates.noise_bursts_per_host_week, (0.5, 3.0)):
        chosen = np.sort(rng.choice(n_metrics, size=int(rng.integers(3, 6)), replace=False))
        plan.noise_scale[start:stop, chosen] = 3.0
        names = tuple(METRIC_ORDER[i] for i in chosen)
        plan.events.append(PlannedDataQuality("noise_burst", start, stop, names))
    for start, _ in windows(rates.glitches_per_host_week, (0.0, 0.0)):
        column = int(rng.integers(n_metrics))
        mode = 1 if rng.random() < 0.7 else 2
        plan.glitch_mode[start, column] = mode
        detail = "spike" if mode == 1 else "drop to zero"
        plan.events.append(
            PlannedDataQuality("glitch", start, start + 1, (METRIC_ORDER[column],), detail)
        )

    eligible = np.zeros(n_steps, dtype=bool)
    eligible[first_seen_step:] = True
    missing_cells = rng.uniform(size=(n_steps, n_metrics)) < rates.missing_cell_probability
    missing_cells &= eligible[:, None]
    if missing_cells.any():
        plan.missing |= missing_cells
        rows = np.flatnonzero(missing_cells.any(axis=1))
        plan.events.append(
            PlannedDataQuality(
                "missing_cells",
                int(rows[0]),
                int(rows[-1]) + 1,
                detail=f"{int(missing_cells.sum())} isolated cells left empty",
            )
        )
    jittered = (rng.uniform(size=n_steps) < rates.jitter_row_fraction) & eligible
    offsets = np.round(rng.uniform(-0.15, 0.15, size=n_steps) * step_seconds)
    plan.jitter_seconds = np.where(jittered, offsets, 0.0)
    if jittered.any():
        rows = np.flatnonzero(jittered)
        plan.events.append(
            PlannedDataQuality(
                "timestamp_jitter",
                int(rows[0]),
                int(rows[-1]) + 1,
                detail=f"{len(rows)} timestamps shifted by up to 15% of the interval",
            )
        )
    plan.events.sort(key=lambda event: (event.start, event.kind))
    return plan
