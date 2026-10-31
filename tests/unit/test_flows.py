"""Flow analysis: window aggregation into metrics, per-flow features, scoring and evaluation."""

import numpy as np
import pandas as pd
import pytest

from network_anomaly.errors import DataValidationError
from network_anomaly.flows import (
    FlowDetectionConfig,
    FlowModel,
    aggregate_flows,
    evaluate_flows,
    flow_features,
    protocol_names,
)

T0 = pd.Timestamp("2026-01-05T00:00:00Z")
MINUTE = pd.Timedelta("1min")


def _flows(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.as_unit("ns")
    return frame


def _flow(minute: float, **fields: object) -> dict[str, object]:
    base: dict[str, object] = {
        "timestamp": T0 + minute * MINUTE,
        "src_addr": "198.51.100.1",
        "dst_addr": "192.0.2.10",
        "dst_port": 443.0,
        "protocol": "tcp",
        "packets": 10.0,
        "bytes": 5000.0,
        "syn_count": 1.0,
        "ack_count": 9.0,
        "rst_count": 0.0,
        "is_anomaly": 0,
    }
    return base | fields


def test_windows_count_rates_sources_ports_and_handshakes() -> None:
    flows = _flows(
        [
            _flow(0.1),
            _flow(0.5, src_addr="198.51.100.2", dst_port=80.0),
            _flow(0.9, ack_count=0.0, packets=1.0, bytes=60.0),  # SYN without ACK
            _flow(1.2, rst_count=1.0),
            _flow(1.4, protocol="udp", syn_count=0.0, ack_count=0.0, is_anomaly=1),
        ]
    )
    table = aggregate_flows(flows, window=MINUTE)
    first, second = table.iloc[0], table.iloc[1]
    assert first["flows_per_second"] == pytest.approx(3 / 60)
    assert first["flow_bytes_per_second"] == pytest.approx(10_060 / 60)
    assert (first["unique_sources"], first["unique_destination_ports"]) == (2, 2)
    assert first["syn_only_ratio"] == pytest.approx(1 / 3)
    assert second["rst_ratio"] == pytest.approx(1.0)  # one TCP flow, reset; UDP is not counted
    assert list(table["is_anomaly"]) == [0, 1]


def test_quiet_windows_are_zero_traffic_not_gaps() -> None:
    flows = _flows([_flow(0.5), _flow(3.5)])
    table = aggregate_flows(flows, window=MINUTE)
    assert len(table) == 4
    assert list(table["flows_per_second"] * 60) == [1, 0, 0, 1]
    assert table["syn_only_ratio"].isna().tolist() == [False, True, True, False]


def test_entities_and_the_busiest_n() -> None:
    flows = _flows([_flow(0.1 * (i + 1), dst_addr=name) for i, name in enumerate("aabc")])
    table = aggregate_flows(flows, window=MINUTE, entity="dst_addr", top_entities=1)
    assert dict(zip(table["host_id"], table["flows_per_second"] * 60, strict=True)) == {
        "a": 2,
        "other": 2,
    }
    with pytest.raises(DataValidationError, match="no such column"):
        aggregate_flows(flows, window=MINUTE, entity="vlan")


def test_flow_features_never_use_identifiers_times_or_labels() -> None:
    flows = _flows([_flow(0.1, label="BENIGN"), _flow(0.2, protocol="17", dst_port=53.0)])
    features = flow_features(flows)
    forbidden = {"timestamp", "src_addr", "dst_addr", "label", "is_anomaly"}
    assert not forbidden & set(features.columns)
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[1, "protocol_udp"] == 1.0
    assert list(protocol_names(pd.Series(["TCP ", "6", "17.0", "icmp", "gre"]))) == [
        "tcp",
        "tcp",
        "udp",
        "icmp",
        "other",
    ]


def _traffic(seed: int, count: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [
        _flow(
            i / 10,
            packets=float(rng.integers(5, 20)),
            bytes=float(rng.normal(5000, 500)),
            duration_ms=float(rng.normal(200, 20)),
        )
        for i in range(count)
    ]
    return _flows(rows)


def test_flow_model_ranks_unusual_flows_above_normal_ones() -> None:
    train = _traffic(0, 2000)
    config = FlowDetectionConfig(target_fpr=0.02)
    model = FlowModel.fit(train, config)
    assert (model.score(train) >= model.threshold).mean() == pytest.approx(0.02, abs=0.005)
    odd = _flows([_flow(500, packets=4000.0, bytes=6e7, dst_port=31337.0, duration_ms=9e5)])
    assert model.score(odd)[0] > np.quantile(model.score(_traffic(1, 500)), 0.99)
    again = FlowModel.fit(train, config)
    np.testing.assert_array_equal(again.score(odd), model.score(odd))
    with pytest.raises(DataValidationError, match="at least 100 training flows"):
        FlowModel.fit(train.head(50), config)


def test_flow_evaluation_by_hand() -> None:
    labels = np.array([1, 1, 0, 0, 0], dtype=np.int8)
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    flagged = scores >= 0.5
    names = pd.Series(["DoS", "Scan", "BENIGN", "BENIGN", "BENIGN"])
    result = evaluate_flows(labels, scores, flagged, names)
    assert (result.precision, result.recall) == (0.5, 0.5)
    assert result.false_positive_rate == pytest.approx(1 / 3)
    assert result.recall_by_label == {"DoS": 1.0, "Scan": 0.0}


def test_identical_training_flows_are_rejected() -> None:
    same = _flows([_flow(i / 10) for i in range(200)])
    with pytest.raises(DataValidationError, match="all identical"):
        FlowModel.fit(same, FlowDetectionConfig())
