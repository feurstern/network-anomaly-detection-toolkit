import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from network_anomaly.config import parse_config
from network_anomaly.errors import ConfigError
from network_anomaly.ingestion.events import read_events, sibling_path
from network_anomaly.ingestion.validation import load_metrics
from network_anomaly.schemas import METRIC_REGISTRY
from network_anomaly.synthetic.generator import (
    DECIMALS,
    GeneratorConfig,
    SyntheticDataset,
    dataset_paths,
    generate_dataset,
    write_dataset,
)
from network_anomaly.synthetic.scenarios import PRESETS, Preset, ScenarioName
from network_anomaly.synthetic.system import (
    Drivers,
    draw_noise,
    sample_host_profile,
    simulate_host,
)

# Small but non-trivial: two hosts for nine days, every scenario, data-quality problems enabled.
SMALL = GeneratorConfig(seed=7, hosts=2, days=9, events_per_host_week=4)

# sha256 of the CSV text for SMALL. NumPy only guarantees Generator streams within a version, so a
# deliberate dependency upgrade may require updating this value; any other change to it means the
# generator's output changed and the generator version (and docs) should be bumped.
GOLDEN_SHA256 = "ec1053d4d7d87b4437d859b38c3ead2b2bfa1c78e8b541bab6b5d337e39f66e8"


def _digest(config: GeneratorConfig) -> str:
    dataset = generate_dataset(config)
    csv = dataset.data.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%SZ")
    return hashlib.sha256(csv.encode()).hexdigest()


@pytest.fixture(scope="module")
def small() -> SyntheticDataset:
    return generate_dataset(SMALL)


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left.reset_index(drop=True).equals(right.reset_index(drop=True))


def test_same_seed_writes_identical_files(tmp_path: Path) -> None:
    first = write_dataset(generate_dataset(SMALL), tmp_path / "a.csv")
    second = write_dataset(generate_dataset(SMALL), tmp_path / "b.csv")
    for name in ("data", "events", "data_quality_events"):
        assert first[name].read_bytes() == second[name].read_bytes(), name


def test_golden_hash_pins_the_generator_output() -> None:
    assert _digest(SMALL) == GOLDEN_SHA256


def test_different_seeds_and_dataset_indices_differ(small: SyntheticDataset) -> None:
    other_seed = generate_dataset(SMALL.model_copy(update={"seed": 8}))
    other_index = generate_dataset(SMALL, dataset_index=1)
    assert not _frames_equal(small.data, other_seed.data)
    assert not _frames_equal(small.data, other_index.data)


def test_adding_a_host_leaves_existing_hosts_unchanged(small: SyntheticDataset) -> None:
    bigger = generate_dataset(SMALL.model_copy(update={"hosts": 3}))
    for host in ("web-01", "web-02"):
        before = small.data[small.data["host_id"] == host]
        after = bigger.data[bigger.data["host_id"] == host]
        assert _frames_equal(before, after), host
    assert [e for e in bigger.events if e.host_id != "web-03"] == list(small.events)


def test_point_labels_agree_with_event_intervals(small: SyntheticDataset) -> None:
    data = small.data
    assert small.events, "the configuration should produce events"
    covered = np.zeros(len(data), dtype=bool)
    for event in small.events:
        inside = (
            (data["host_id"] == event.host_id)
            & (data["timestamp"] >= event.start - pd.Timedelta(minutes=1))
            & (data["timestamp"] < event.end - pd.Timedelta(minutes=1))
        ).to_numpy()
        labelled = data.loc[inside]
        assert (labelled["is_anomaly"] == 1).all(), event.event_id
        assert (labelled["anomaly_event_id"] == event.event_id).all()
        assert (labelled["anomaly_scenario"] == event.scenario).all()
        covered |= inside
    assert (data.loc[~covered, "is_anomaly"] == 0).all()


def test_events_file_round_trips(tmp_path: Path, small: SyntheticDataset) -> None:
    paths = write_dataset(small, tmp_path / "demo.csv")
    assert read_events(paths["events"]) == list(small.events)


def test_labels_cover_every_visible_effect(small: SyntheticDataset) -> None:
    """Outside labelled rows the injected effect stays within one noise standard deviation."""
    sigma_by_host = {h["host_id"]: h["noise_sigma"] for h in small.metadata["hosts"]}
    normal = small.data["is_anomaly"].to_numpy() == 0
    for metric, spec in METRIC_REGISTRY.items():
        anomalous = small.data[metric].to_numpy()
        clean = small.clean[metric].to_numpy()
        # Written values are rounded, which can move a just-below-threshold effect slightly.
        quantum = 10.0 ** -DECIMALS[metric]
        if spec.log_scale:
            allowance = quantum / (1.0 + np.minimum(anomalous, clean))
            difference = np.abs(np.log1p(anomalous) - np.log1p(clean))
        else:
            allowance = np.full(len(anomalous), quantum)
            difference = np.abs(anomalous - clean)
        sigma = small.data["host_id"].map(lambda h, m=metric: sigma_by_host[h][m]).to_numpy()
        visible = normal & np.isfinite(difference) & (difference > sigma + allowance)
        assert not visible.any(), (
            f"{metric}: unlabelled effect at rows {np.flatnonzero(visible)[:5]}"
        )


def test_clean_twin_has_identical_rows_and_no_labels(small: SyntheticDataset) -> None:
    assert small.clean["timestamp"].equals(small.data["timestamp"])
    assert small.clean["host_id"].equals(small.data["host_id"])
    assert (small.clean["is_anomaly"] == 0).all()
    assert small.data["is_anomaly"].sum() > 0


def test_normal_only_data_keeps_data_quality_problems_unlabelled() -> None:
    config = SMALL.model_copy(update={"scenarios": (), "preset": Preset.DIFFICULT})
    dataset = generate_dataset(config)
    assert dataset.events == ()
    assert dataset.data_quality_events, "difficult preset should inject data-quality problems"
    assert (dataset.data["is_anomaly"] == 0).all()
    assert _frames_equal(dataset.data, dataset.clean)


def test_selected_scenarios_only(tmp_path: Path) -> None:
    config = SMALL.model_copy(update={"scenarios": (ScenarioName.TRAFFIC_SPIKE,)})
    dataset = generate_dataset(config)
    assert {event.scenario for event in dataset.events} == {"traffic_spike"}


def test_events_keep_their_distance_and_avoid_exclusion_zones() -> None:
    zone = (datetime(2026, 1, 9, tzinfo=UTC), datetime(2026, 1, 10, tzinfo=UTC))
    config = SMALL.model_copy(update={"exclusion_zones": (zone,), "events_per_host_week": 10})
    dataset = generate_dataset(config)
    margin = pd.Timedelta(hours=config.exclusion_margin_hours)
    for event in dataset.events:
        injection_end = pd.Timestamp(str(event.params["injection_end"]))
        assert injection_end <= pd.Timestamp(zone[0]) - margin or (
            pd.Timestamp(event.start) >= pd.Timestamp(zone[1]) + margin
        ), event.event_id
    for host in ("web-01", "web-02"):
        starts = sorted(e.start for e in dataset.events if e.host_id == host)
        ends = sorted(
            pd.Timestamp(str(e.params["injection_end"]))
            for e in dataset.events
            if e.host_id == host
        )
        for previous_end, next_start in zip(ends, starts[1:], strict=False):
            assert pd.Timestamp(next_start) - previous_end >= pd.Timedelta(hours=6)


def test_late_hosts_start_reporting_after_the_first_half() -> None:
    config = GeneratorConfig(seed=1, hosts=12, days=10, preset=Preset.DIFFICULT)
    dataset = generate_dataset(config)
    first_seen = dataset.data.groupby("host_id")["timestamp"].min()
    start = pd.Timestamp(config.start)
    late = first_seen[first_seen > start + pd.Timedelta(days=5)]
    flagged = {h["host_id"] for h in dataset.metadata["hosts"] if h["late_start"]}
    assert set(late.index) == flagged
    assert flagged, "seed 1 with 12 hosts includes late hosts"


def test_generated_files_validate_cleanly(tmp_path: Path, small: SyntheticDataset) -> None:
    paths = write_dataset(small, tmp_path / "demo.csv", with_clean_twin=True)
    for name in ("data", "clean"):
        validated = load_metrics(paths[name])
        assert validated.report.is_valid
        assert not validated.report.warnings, validated.report.warnings
        assert validated.metrics.metrics == tuple(METRIC_REGISTRY)


def test_metadata_records_parameters_and_checksums(tmp_path: Path, small: SyntheticDataset) -> None:
    paths = write_dataset(small, tmp_path / "demo.csv")
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["seed"] == SMALL.seed
    assert GeneratorConfig.model_validate(metadata["config"]) == SMALL
    assert metadata["units"]["cpu_usage_percent"] == "%"
    assert set(metadata["label_definitions"]) == {
        "is_anomaly",
        "anomaly_event_id",
        "anomaly_scenario",
        "anomaly_severity",
    }
    for name, entry in metadata["files"].items():
        digest = hashlib.sha256(paths[name].read_bytes()).hexdigest()
        assert entry["sha256"] == digest, name
    assert sibling_path(paths["data"], "metadata.json") == paths["metadata"]


def test_dataset_paths_for_multiple_datasets() -> None:
    assert dataset_paths(Path("d/demo.csv"), 1) == [Path("d/demo.csv")]
    assert dataset_paths(Path("d/demo.csv"), 2) == [Path("d/demo_00.csv"), Path("d/demo_01.csv")]


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"freq": "7min"}, "divide one day evenly"),
        ({"freq": "soon"}, "invalid sampling interval"),
        ({"hosts": 0}, "greater than or equal to 1"),
        ({"scenarios": ["meteor_strike"]}, "scenarios.0"),
        ({"colour": "red"}, "unknown option"),
    ],
)
def test_invalid_generator_config(update: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config({**SMALL.model_dump(), **update}, GeneratorConfig)


def test_presets_get_harder() -> None:
    easy, moderate, hard = (PRESETS[p] for p in (Preset.EASY, Preset.MODERATE, Preset.DIFFICULT))
    for name in ("demand_sigma", "cpu_pct", "latency_log", "bytes_log"):
        assert getattr(easy.noise, name) < getattr(moderate.noise, name) < getattr(hard.noise, name)
    assert easy.data_quality.missing_cell_probability == 0
    assert hard.severity_weights["low"] > easy.severity_weights["low"]
    assert hard.noise.heavy_tails
    assert not easy.noise.heavy_tails


# --------------------------------------------------------------------------------------------------
# System model: injected drivers propagate to correlated symptoms
# --------------------------------------------------------------------------------------------------


def _simulate(drivers: Drivers) -> dict[str, np.ndarray]:
    steps = 48
    rng = np.random.default_rng(0)
    profile = sample_host_profile("web-01", rng, n_steps=steps, late_start=False)
    noise = draw_noise(np.random.default_rng(1), steps, heavy_tails=False)
    levels = PRESETS[Preset.EASY].noise
    hour = np.full(steps, 12.0)
    demand = np.full(steps, profile.peak_rps * 0.6)
    return simulate_host(profile, demand, drivers, noise, levels, hour, step_seconds=300.0)


def _window_mean(values: dict[str, np.ndarray], metric: str) -> float:
    return float(values[metric][20:30].mean())


def test_traffic_spike_propagates_to_load_metrics() -> None:
    baseline = _simulate(Drivers.neutral(48))
    drivers = Drivers.neutral(48)
    drivers.demand_mult[20:30] *= 2.0
    spiked = _simulate(drivers)
    for metric in (
        "requests_per_second",
        "cpu_usage_percent",
        "response_time_p95_ms",
        "active_connections",
        "network_tx_bytes_per_second",
    ):
        assert _window_mean(spiked, metric) > 1.2 * _window_mean(baseline, metric), metric
    np.testing.assert_array_equal(
        spiked["requests_per_second"][:20], baseline["requests_per_second"][:20]
    )


def test_cpu_saturation_slows_requests_without_changing_traffic() -> None:
    baseline = _simulate(Drivers.neutral(48))
    drivers = Drivers.neutral(48)
    drivers.cpu_background_pct[20:30] += 35.0
    saturated = _simulate(drivers)
    assert (
        _window_mean(saturated, "cpu_usage_percent")
        > _window_mean(baseline, "cpu_usage_percent") + 25
    )
    assert _window_mean(saturated, "response_time_p50_ms") > 1.3 * _window_mean(
        baseline, "response_time_p50_ms"
    )
    np.testing.assert_array_equal(saturated["requests_per_second"], baseline["requests_per_second"])


def test_connection_surge_saturates_the_connection_limit() -> None:
    baseline = _simulate(Drivers.neutral(48))
    drivers = Drivers.neutral(48)
    drivers.hold_time_mult[20:30] *= 30.0
    surged = _simulate(drivers)
    assert _window_mean(surged, "active_connections") > 2 * _window_mean(
        baseline, "active_connections"
    )
    assert _window_mean(surged, "connection_error_rate") > 0.2
