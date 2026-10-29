"""Hidden drivers and the coupled observation model behind synthetic web-service metrics.

Each host is modelled as a small queueing system. Latent *drivers* (demand, service time, CPU cost
per request, background CPU load, memory leak, connection hold time, bytes per request and
error-probability offsets) feed deterministic equations that produce the twelve canonical metrics:

* application utilisation ``rho = demand * cpu_cost / capacity``; CPU% grows with ``rho`` plus any
  background load, which also steals capacity from request handling;
* latency follows an M/M/1 queue: mean sojourn ``W = S / (1 - rho)``, ``p50 = d0 + ln(2) W`` and
  ``p95 = d0 + ln(20) W``, with a swap penalty once memory is nearly full;
* active connections follow Little's law, ``demand * (W + keep-alive hold time)``, up to a cap;
* memory relaxes towards a target driven by open connections and any leak;
* error rates are binomial proportions whose probabilities rise near saturation and out-of-memory.

Anomaly scenarios only modify drivers, so their symptoms propagate through these couplings the way
they would on a real server. All randomness is drawn up front (:class:`NoiseDraws`), which lets a
clean copy and an anomalous copy of the same host share identical noise.
"""

import math
from dataclasses import dataclass, field, fields

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

LN2 = math.log(2.0)
LN20 = math.log(20.0)
MAX_QUEUE_UTILISATION = 0.97
MEMORY_RELAXATION = 0.25  # fraction of the gap to the memory target closed per step

NOISE_CHANNELS: tuple[str, ...] = (
    "demand",
    "rps",
    "cpu",
    "memory",
    "latency_p50",
    "latency_p95",
    "connections",
    "rx",
    "tx",
    "disk",
    "err_4xx",
    "err_5xx",
    "err_connection",
)
# Channels whose innovations become heavy-tailed in the difficult preset.
_HEAVY_TAIL_CHANNELS = frozenset({"demand", "latency_p50", "latency_p95", "rx", "tx"})


@dataclass(frozen=True)
class NoiseLevels:
    """Noise amplitudes; standard-normal draws are scaled by these values."""

    demand_sigma: float  # stationary std of the log-demand AR(1) process
    ar_phi: float  # persistence of demand noise per step
    cpu_pct: float
    memory_pct: float
    latency_log: float
    bytes_log: float
    connections_log: float
    heavy_tails: bool


@dataclass(frozen=True)
class HostProfile:
    """Static characteristics of one simulated host."""

    host_id: str
    peak_rps: float
    capacity_rps: float
    cpu_idle_fraction: float
    service_time_ms: float
    latency_floor_ms: float
    keepalive_seconds: float
    connection_cap: float
    request_bytes: float
    response_bytes: float
    disk_base_bps: float
    disk_bytes_per_request: float
    backup_bps: float
    backup_hour: float
    memory_base_pct: float
    memory_connection_pct: float
    p_4xx: float
    p_5xx: float
    p_connection_error: float
    phase_hours: float
    weekend_factor: float
    first_seen_step: int


def sample_host_profile(
    host_id: str, rng: np.random.Generator, *, n_steps: int, late_start: bool
) -> HostProfile:
    """Draw a plausible web-host profile. Every call consumes the same number of draws."""
    peak = float(np.clip(rng.lognormal(math.log(200.0), 0.9), 20.0, 2000.0))
    peak_utilisation = rng.uniform(0.55, 0.65)
    service_time = rng.uniform(15.0, 60.0)
    keepalive = rng.uniform(2.0, 10.0)
    peak_sojourn_s = service_time / 1000.0 / (1.0 - peak_utilisation)
    peak_connections = peak * (peak_sojourn_s + keepalive)
    late_fraction = rng.uniform(0.55, 0.75)
    return HostProfile(
        host_id=host_id,
        peak_rps=peak,
        capacity_rps=peak / peak_utilisation,
        cpu_idle_fraction=rng.uniform(0.03, 0.08),
        service_time_ms=service_time,
        latency_floor_ms=rng.uniform(1.0, 5.0),
        keepalive_seconds=keepalive,
        connection_cap=peak_connections * rng.uniform(2.5, 4.0),
        request_bytes=rng.uniform(800.0, 3000.0),
        response_bytes=rng.uniform(10_000.0, 80_000.0),
        disk_base_bps=rng.uniform(0.5e6, 5e6),
        disk_bytes_per_request=rng.uniform(2_000.0, 20_000.0),
        backup_bps=rng.uniform(20e6, 60e6),
        backup_hour=rng.uniform(1.0, 4.0),
        memory_base_pct=rng.uniform(30.0, 55.0),
        memory_connection_pct=rng.uniform(5.0, 15.0),
        p_4xx=rng.uniform(0.01, 0.04),
        p_5xx=rng.uniform(0.0005, 0.003),
        p_connection_error=rng.uniform(0.0005, 0.002),
        phase_hours=rng.uniform(-2.0, 2.0),
        weekend_factor=rng.uniform(0.55, 0.85),
        first_seen_step=int(late_fraction * n_steps) if late_start else 0,
    )


@dataclass
class Drivers:
    """Per-step latent inputs. Neutral values reproduce normal behaviour; scenarios edit copies."""

    demand_mult: FloatArray
    service_time_mult: FloatArray
    cpu_cost_mult: FloatArray
    cpu_background_pct: FloatArray
    memory_leak_pct: FloatArray
    hold_time_mult: FloatArray
    rx_bytes_mult: FloatArray
    tx_bytes_mult: FloatArray
    p_4xx_add: FloatArray
    p_5xx_add: FloatArray
    p_connection_error_add: FloatArray

    @classmethod
    def neutral(cls, n_steps: int) -> "Drivers":
        ones = {"demand_mult", "service_time_mult", "cpu_cost_mult", "hold_time_mult"}
        ones |= {"rx_bytes_mult", "tx_bytes_mult"}
        return cls(
            **{
                f.name: np.ones(n_steps) if f.name in ones else np.zeros(n_steps)
                for f in fields(cls)
            }
        )


@dataclass(frozen=True)
class NoiseDraws:
    """Standard-normal (or unit-variance Student-t) draws per channel, shared by paired copies."""

    channels: dict[str, FloatArray] = field(default_factory=dict)

    def __getitem__(self, name: str) -> FloatArray:
        return self.channels[name]


def draw_noise(rng: np.random.Generator, n_steps: int, *, heavy_tails: bool) -> NoiseDraws:
    """Draw every noise channel in a fixed order so results do not depend on the call site."""
    channels: dict[str, FloatArray] = {}
    for name in NOISE_CHANNELS:
        if heavy_tails and name in _HEAVY_TAIL_CHANNELS:
            # Student-t with 3 degrees of freedom has variance 3; rescale to unit variance.
            channels[name] = rng.standard_t(3, size=n_steps) / math.sqrt(3.0)
        else:
            channels[name] = rng.standard_normal(n_steps)
    return NoiseDraws(channels)


def sigmoid(x: FloatArray) -> FloatArray:
    """Logistic function in a form that cannot overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _diurnal_bump(hour: FloatArray, center: float, width: float) -> FloatArray:
    distance = (hour - center + 12.0) % 24.0 - 12.0
    return np.exp(-0.5 * (distance / width) ** 2)


def _raw_daily_shape(hour: FloatArray) -> FloatArray:
    morning = _diurnal_bump(hour, 11.0, 3.0)
    evening = _diurnal_bump(hour, 20.5, 2.3)
    return 0.22 + 0.78 * (0.9 * morning + 0.75 * evening)


_DAILY_PEAK = float(_raw_daily_shape(np.linspace(0.0, 24.0, 24 * 60, endpoint=False)).max())


def daily_shape(hour: FloatArray) -> FloatArray:
    """Two-hump diurnal demand profile (late morning and evening), scaled to a peak of 1."""
    return _raw_daily_shape(hour) / _DAILY_PEAK


@dataclass(frozen=True)
class BaselineShift:
    """A legitimate, permanent change in demand level (e.g. a migration or a new customer)."""

    step: int
    factor: float  # relative change, e.g. +0.2 for 20 % more traffic
    width_steps: float  # sigmoid transition width


def baseline_demand(
    profile: HostProfile,
    local_hour: FloatArray,
    local_weekday: npt.NDArray[np.int64],
    shifts: list[BaselineShift],
    noise: NoiseDraws,
    levels: NoiseLevels,
) -> FloatArray:
    """Normal demand in requests/s: seasonality x legitimate level shifts x AR(1) noise."""
    n_steps = len(local_hour)
    seasonal = profile.peak_rps * daily_shape(local_hour)
    seasonal = seasonal * np.where(local_weekday >= 5, profile.weekend_factor, 1.0)
    steps = np.arange(n_steps, dtype=np.float64)
    level = np.ones(n_steps)
    for shift in shifts:
        level *= 1.0 + shift.factor * sigmoid((steps - shift.step) / shift.width_steps)
    innovation_scale = levels.demand_sigma * math.sqrt(1.0 - levels.ar_phi**2)
    log_noise = np.empty(n_steps)
    innovations = noise["demand"]
    log_noise[0] = levels.demand_sigma * innovations[0]
    for i in range(1, n_steps):
        log_noise[i] = levels.ar_phi * log_noise[i - 1] + innovation_scale * innovations[i]
    return seasonal * level * np.exp(log_noise)


def _relax(target: FloatArray, rate: float) -> FloatArray:
    out = np.empty_like(target)
    out[0] = target[0]
    for i in range(1, len(target)):
        out[i] = out[i - 1] + rate * (target[i] - out[i - 1])
    return out


def _binomial_rate(probability: FloatArray, requests: FloatArray, draws: FloatArray) -> FloatArray:
    """Observed failure fraction for ``requests`` trials, via a normal approximation.

    The approximation keeps the draw monotone in ``probability`` for a fixed noise value, so the
    clean and anomalous copies differ only by the injected effect.
    """
    p = np.clip(probability, 0.0, 1.0)
    count = requests * p + np.sqrt(requests * p * (1.0 - p)) * draws
    return np.clip(count / requests, 0.0, 1.0)


def simulate_host(
    profile: HostProfile,
    demand: FloatArray,
    drivers: Drivers,
    noise: NoiseDraws,
    levels: NoiseLevels,
    local_hour: FloatArray,
    step_seconds: float,
) -> dict[str, FloatArray]:
    """Compute the twelve canonical metrics for one host from its drivers and shared noise."""
    rate = demand * drivers.demand_mult
    rho_app = rate * drivers.cpu_cost_mult / profile.capacity_rps
    background = drivers.cpu_background_pct
    idle = profile.cpu_idle_fraction
    cpu = 100.0 * (idle + (1.0 - idle) * np.minimum(rho_app, 1.0)) + background
    cpu = np.clip(cpu + levels.cpu_pct * noise["cpu"], 0.0, 100.0)

    # Background load steals capacity from request handling, so queueing sees a higher utilisation.
    headroom = np.maximum(0.05, 1.0 - background / 100.0)
    rho_queue = np.clip(rho_app / headroom, 0.0, MAX_QUEUE_UTILISATION)
    sojourn_ms = profile.service_time_ms * drivers.service_time_mult / (1.0 - rho_queue)

    expected_connections = rate * (
        sojourn_ms / 1000.0 + profile.keepalive_seconds * drivers.hold_time_mult
    )
    connection_load = expected_connections / profile.connection_cap
    connections = np.minimum(expected_connections, profile.connection_cap)
    connections = connections * np.exp(levels.connections_log * noise["connections"])

    memory_target = (
        profile.memory_base_pct
        + profile.memory_connection_pct * np.minimum(connection_load, 1.0)
        + drivers.memory_leak_pct
    )
    memory = _relax(memory_target, MEMORY_RELAXATION) + levels.memory_pct * noise["memory"]
    memory = np.clip(memory, 0.0, 100.0)

    swap_penalty = 1.0 + 4.0 * np.clip((memory - 90.0) / 10.0, 0.0, 1.0) ** 2
    sojourn_ms = sojourn_ms * swap_penalty
    p50 = profile.latency_floor_ms + LN2 * sojourn_ms
    p95 = profile.latency_floor_ms + LN20 * sojourn_ms
    p50 = p50 * np.exp(levels.latency_log * noise["latency_p50"])
    p95 = np.maximum(p95 * np.exp(1.3 * levels.latency_log * noise["latency_p95"]), p50)

    overload = np.clip((rho_queue - 0.9) / (MAX_QUEUE_UTILISATION - 0.9), 0.0, 1.0)
    out_of_memory = np.clip((memory - 95.0) / 5.0, 0.0, 1.0)
    connection_saturation = sigmoid((connection_load - 0.97) * 60.0)
    p_4xx = profile.p_4xx + drivers.p_4xx_add
    p_5xx = (
        profile.p_5xx
        + 0.15 * overload**2
        + 0.25 * out_of_memory
        + 0.05 * connection_saturation
        + drivers.p_5xx_add
    )
    p_connection_error = (
        profile.p_connection_error
        + 0.4 * connection_saturation
        + 0.05 * overload
        + drivers.p_connection_error_add
    )
    requests = np.maximum(rate * step_seconds, 1.0)

    backup = (local_hour >= profile.backup_hour) & (local_hour < profile.backup_hour + 0.75)
    disk = profile.disk_base_bps + rate * profile.disk_bytes_per_request
    disk = (disk + profile.backup_bps * backup) * np.exp(levels.bytes_log * noise["disk"])
    rx = rate * profile.request_bytes * drivers.rx_bytes_mult
    tx = rate * profile.response_bytes * drivers.tx_bytes_mult

    return {
        "requests_per_second": np.maximum(rate + np.sqrt(rate / step_seconds) * noise["rps"], 0.0),
        "http_4xx_rate": _binomial_rate(p_4xx, requests, noise["err_4xx"]),
        "http_5xx_rate": _binomial_rate(p_5xx, requests, noise["err_5xx"]),
        "response_time_p50_ms": p50,
        "response_time_p95_ms": p95,
        "cpu_usage_percent": cpu,
        "memory_usage_percent": memory,
        "disk_io_bytes_per_second": disk,
        "network_rx_bytes_per_second": rx * np.exp(levels.bytes_log * noise["rx"]),
        "network_tx_bytes_per_second": tx * np.exp(levels.bytes_log * noise["tx"]),
        "active_connections": np.round(connections),
        "connection_error_rate": _binomial_rate(
            p_connection_error, requests, noise["err_connection"]
        ),
    }
