"""Public-dataset support, offline: pinned files, verified downloads and the converters."""

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from network_anomaly.errors import ArtifactError, ConfigError
from network_anomaly.ingestion.datasets import (
    DATASETS,
    describe_datasets,
    download_dataset,
    flow_sources,
    nab,
    remote_files,
    smd,
)
from network_anomaly.ingestion.datasets import download as download_module
from network_anomaly.ingestion.datasets.download import RemoteFile, fetch, verify
from network_anomaly.ingestion.events import read_events
from network_anomaly.ingestion.validation import load_flows, load_metrics

CONTENT = b"timestamp,value\n2014-02-14 14:27:00,51.8\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def test_every_dataset_is_described_and_every_file_is_pinned() -> None:
    text = describe_datasets()
    for name in DATASETS:
        assert name in text
        files = remote_files(name)
        assert files
        assert all(f.url.startswith("https://") for f in files)
        assert all(f.sha256 or f.git_sha1 for f in files)
    assert len(remote_files("nab")) == 22
    assert len(remote_files("smd")) == 5  # licence + four files of the default machine
    assert len(remote_files("smd", machines="all")) == 113
    with pytest.raises(ConfigError, match="unknown SMD machine"):
        remote_files("smd", machines="machine-9-9")
    with pytest.raises(ConfigError, match="unknown dataset"):
        remote_files("kdd99")


@pytest.mark.parametrize("pin", ["sha256", "git_sha1"])
def test_checksums_accept_exact_content_only(tmp_path: Path, pin: str) -> None:
    path = tmp_path / "file.csv"
    path.write_bytes(CONTENT)
    digest = _sha256(CONTENT) if pin == "sha256" else _git_sha1(CONTENT)
    remote = RemoteFile(url="https://example.org/f", target="f", size=len(CONTENT), **{pin: digest})
    assert verify(path, remote)
    path.write_bytes(CONTENT.replace(b"51.8", b"51.9"))
    assert not verify(path, remote)


def test_only_pinned_https_files_can_be_described() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        RemoteFile(url="http://example.org/f", target="f", size=1, sha256="0" * 64)
    with pytest.raises(ValueError, match="pinned checksum"):
        RemoteFile(url="https://example.org/f", target="f", size=1)


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self


def test_downloads_are_verified_before_they_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = {"body": CONTENT}
    calls: list[str] = []

    def fake_urlopen(request: Any, timeout: float) -> _Response:
        calls.append(request.full_url)
        return _Response(served["body"])

    monkeypatch.setattr(download_module.urllib.request, "urlopen", fake_urlopen)
    remote = RemoteFile(
        url="https://example.org/a.csv",
        target="dir/a.csv",
        size=len(CONTENT),
        sha256=_sha256(CONTENT),
    )
    assert fetch(remote, tmp_path).read_bytes() == CONTENT
    fetch(remote, tmp_path)  # already verified: no second request
    assert len(calls) == 1
    served["body"] = CONTENT.replace(b"51.8", b"99.9")
    altered = RemoteFile(
        url="https://example.org/b.csv", target="b.csv", size=len(CONTENT), sha256=_sha256(CONTENT)
    )
    with pytest.raises(ArtifactError, match="does not match its pinned checksum"):
        fetch(altered, tmp_path)
    assert not (tmp_path / "b.csv").exists()
    assert not (tmp_path / "b.csv.part").exists()


def test_datasets_with_terms_need_explicit_acceptance(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="terms of use"):
        download_dataset("cidds-001", tmp_path)


def test_address_policies(monkeypatch: pytest.MonkeyPatch) -> None:
    flows = pd.DataFrame({"src_addr": ["192.0.2.1", "192.0.2.1"], "dst_addr": ["192.0.2.9", None]})
    monkeypatch.delenv(flow_sources.ANON_KEY_ENV, raising=False)
    with pytest.raises(ConfigError, match=flow_sources.ANON_KEY_ENV):
        flow_sources.anonymisation_key("hash")
    monkeypatch.setenv(flow_sources.ANON_KEY_ENV, "k" * 32)
    hashed = flow_sources.apply_ip_policy(
        flows.copy(), "hash", flow_sources.anonymisation_key("hash")
    )
    assert hashed.loc[0, "src_addr"] == hashed.loc[1, "src_addr"] != "192.0.2.1"
    assert hashed.loc[0, "src_addr"].startswith("h-")
    assert pd.isna(hashed.loc[1, "dst_addr"])
    other = flow_sources.apply_ip_policy(flows.copy(), "hash", b"another-key-of-16b")
    assert other.loc[0, "src_addr"] != hashed.loc[0, "src_addr"]
    assert list(flow_sources.apply_ip_policy(flows.copy(), "drop", None).columns) == []


def test_nab_series_become_labelled_metrics_with_window_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    series = raw / "data/realAWSCloudwatch/elb_request_count_test.csv"
    series.parent.mkdir(parents=True)
    times = pd.date_range("2014-04-10 00:02", periods=3000, freq="5min")
    values = np.r_[np.full(2999, 300.0), 600.0]
    rows = [f"{t:%Y-%m-%d %H:%M:%S},{v}" for t, v in zip(times, values, strict=True)]
    rows.insert(5, rows[4])  # a repeated timestamp, as in three NAB files
    series.write_text("timestamp,value\n" + "\n".join(rows) + "\n")
    windows = {
        "realAWSCloudwatch/elb_request_count_test.csv": [
            ["2014-04-20 00:02:00.000000", "2014-04-20 01:02:00.000000"]
        ]
    }
    (raw / "labels").mkdir()
    (raw / nab.WINDOWS).write_text(json.dumps(windows))
    monkeypatch.setattr(
        nab, "NAB_FILES", {"data/realAWSCloudwatch/elb_request_count_test.csv": (0, "")}
    )
    prepared = nab.prepare(raw, out)
    assert any("dropped 1 repeated" in note for note in prepared.notes)
    validated = load_metrics(prepared.files[0])
    assert validated.metrics.metrics == ("requests_per_second",)
    assert validated.metrics.data["requests_per_second"].iloc[0] == pytest.approx(
        1.0
    )  # 300 / 300 s
    (event,) = read_events(out / "elb_request_count_test.events.csv")
    assert event.end - event.start == pd.Timedelta("65min")  # window end is inclusive
    assert int(validated.labels.data["is_anomaly"].sum()) == 13


def test_smd_machines_get_a_grid_labels_and_interpreted_events(tmp_path: Path) -> None:
    root = tmp_path / "raw" / smd.ROOT
    rng = np.random.default_rng(0)
    for part, rows in (("train", 300), ("test", 200)):
        (root / part).mkdir(parents=True)
        np.savetxt(
            root / part / "machine-1-1.txt", rng.random((rows, 3)), fmt="%.6f", delimiter=","
        )
    labels = np.zeros(200, dtype=int)
    labels[50:60] = 1
    (root / "test_label").mkdir()
    np.savetxt(root / "test_label/machine-1-1.txt", labels, fmt="%d")
    (root / "interpretation_label").mkdir()
    (root / "interpretation_label/machine-1-1.txt").write_text("50-58:1,3\n")
    prepared = smd.prepare(tmp_path / "raw", tmp_path / "out")
    table = load_metrics(prepared.files[0], frequency=pd.Timedelta("1min"))
    assert table.metrics.metrics == ("dim_01", "dim_02", "dim_03")
    assert len(table.metrics.data) == 500
    (event,) = read_events(tmp_path / "out/machine-1-1.events.csv")
    assert event.affected_metrics == ("dim_01", "dim_03")
    assert pd.Timestamp(event.start) == pd.Timestamp(smd.START) + 350 * smd.STEP
    assert event.end - event.start == 10 * smd.STEP
    assert "--train-until" in prepared.commands[0]


CICIDS_HEADER = list(flow_sources.CICIDS_COLUMNS)


def _cicids_row(label: str, **values: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(CICIDS_HEADER, 0)
    row |= {
        "Timestamp": "2017-07-04 13:00:01.123456",
        "Src IP": "192.0.2.5",
        "Dst IP": "198.51.100.7",
        "Src Port": 50000,
        "Dst Port": 21,
        "Protocol": 6,
        "Flow Duration": 2500,
        "Total Fwd Packet": 4,
        "Total Bwd packets": 3,
        "Label": label,
    }
    return row | values


def test_cicids_days_are_converted_from_the_archive(tmp_path: Path) -> None:
    rows = [
        _cicids_row("BENIGN"),
        _cicids_row("FTP-Patator"),
        _cicids_row("FTP-Patator - Attempted"),
    ]
    buffer = io.StringIO()
    pd.DataFrame(rows, columns=["id", *CICIDS_HEADER, "Attempted Category"]).fillna(0).to_csv(
        buffer, index=False
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / flow_sources.CICIDS_ARCHIVE, "w") as archive:
        archive.writestr("monday.csv", buffer.getvalue())
    prepared = flow_sources.prepare_cicids(
        raw, tmp_path / "out", ip_policy="drop", days=("monday",)
    )
    flows = load_flows(prepared.files[0]).flows.data
    assert "src_addr" not in flows
    assert flows["is_anomaly"].tolist() == [0, 1, 0]  # attempted attacks are not positives
    assert flows["duration_ms"].iloc[0] == pytest.approx(2.5)
    assert flows["timestamp"].iloc[0] == pd.Timestamp("2017-07-04T13:00:01.123456Z")


def test_cidds_flows_are_converted_to_utc_with_flags_and_attack_events(tmp_path: Path) -> None:
    header = (
        "Date first seen,Duration,Proto,Src IP Addr,Src Pt,Dst IP Addr,Dst Pt,Packets,Bytes,"
        "Flows,Flags,Tos,class,attackType,attackID,attackDescription"
    )  # exactly as in the published files
    lines = [
        "2017-03-23 09:46:10.123,0.004,TCP  ,ATTACKER1,4242,EXT_SERVER,22,     1,     60,"
        "    1,.....S,  0,attacker,portScan,1,---",
        "2017-03-23 09:46:11.000,1.500,TCP  ,EXT_SERVER,22,OPENSTACK_NET,5000,  3057,   2.1 M,"
        "    1,.AP.SF,  0,normal,---,---,---",
        "2017-03-23 09:47:00.000,0.000,ICMP ,10000_34,0,EXT_SERVER,3.3,     1,     84,"
        "    1,......,  0,suspicious,---,---,---",
    ]
    log = "IP,start,end,activity,activityID,description\n"
    log += "ATTACKER1,2017-03-23 09:46:00.000000,2017-03-23 09:48:00.000000,portScan,1,---\n"
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / flow_sources.CIDDS_ARCHIVE, "w") as archive:
        for week in range(1, 5):
            name = f"CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week{week}.csv"
            archive.writestr(name, header + "\n" + "\n".join(lines) + "\n")
        archive.writestr("CIDDS-001/attack_logs/attack_logs_extern.csv", log)
    prepared = flow_sources.prepare_cidds(raw, tmp_path / "out", ip_policy="drop")
    flows = load_flows(prepared.files[0]).flows.data
    assert flows["timestamp"].iloc[0] == pd.Timestamp("2017-03-23T08:46:10.123Z")  # CET -> UTC
    assert flows["is_anomaly"].tolist() == [1, 0, 0]
    assert flows["label"].tolist() == ["attacker/portScan", "normal", "suspicious"]
    assert flows["bytes"].iloc[1] == pytest.approx(2.1e6)
    assert flows[["syn_count", "ack_count", "fin_count"]].iloc[1].tolist() == [1, 1, 1]
    assert pd.isna(flows["dst_port"].iloc[2])  # ICMP type.code is not a port
    (event,) = read_events(tmp_path / "out/cidds-001-external.events.csv")
    assert (event.host_id, event.scenario) == ("all", "portScan")
    assert pd.Timestamp(event.start) == pd.Timestamp("2017-03-23T08:46:00Z")
