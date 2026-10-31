"""Supported public datasets: what they are, how to fetch them and how they are converted.

``network-anomaly datasets list|download|prepare``. Nothing is redistributed with the toolkit:
``download`` fetches pinned files from the original publishers into ``data/raw/<name>/`` and
verifies their checksums; ``prepare`` converts them into the canonical schemas under
``data/processed/<name>/``. Licences, citation requests and caveats are listed per dataset.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from network_anomaly.errors import ConfigError
from network_anomaly.ingestion.datasets import flow_sources, nab, smd
from network_anomaly.ingestion.datasets.common import Prepared
from network_anomaly.ingestion.datasets.download import RemoteFile, fetch


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    title: str
    kind: Literal["metrics", "flows"]
    licence: str
    source: str
    download: str
    use: str
    terms: str | None = None  # shown before download; requires --accept-terms
    caveats: tuple[str, ...] = ()


DATASETS: dict[str, DatasetInfo] = {
    "nab": DatasetInfo(
        name="nab",
        title="Numenta Anomaly Benchmark: AWS CloudWatch and known-cause server series",
        kind="metrics",
        licence="MIT (pinned commit; older NAB releases are AGPL-3.0)",
        source="https://github.com/numenta/NAB",
        download="22 files, 2.6 MB",
        use="single-metric event detection on real labelled server metrics",
        caveats=(
            "units inferred from CloudWatch metric names",
            "label windows are wide; point-level precision is not meaningful",
        ),
    ),
    "smd": DatasetInfo(
        name="smd",
        title="Server Machine Dataset (OmniAnomaly)",
        kind="metrics",
        licence="MIT",
        source="https://github.com/NetManAIOps/OmniAnomaly",
        download="17 MB per machine (28 machines, 486 MB); default machine-1-1",
        use="multivariate detection and explanation hit rates (interpretation labels)",
        caveats=("no timestamps (1-minute grid assigned)", "38 unnamed, pre-scaled dimensions"),
    ),
    "cicids2017-improved": DatasetInfo(
        name="cicids2017-improved",
        title="CICIDS2017, improved labels (DistriNet, KU Leuven)",
        kind="flows",
        licence="no explicit licence; citation requested",
        source="https://intrusion-detection.distrinet-research.be/CNS2022/",
        download="344 MB zip (1.15 GB of CSV)",
        use="per-flow scoring (train on benign Monday, score other days)",
        terms="Research use. Cite G. Engelen, V. Rimmer and W. Joosen, 'Troubleshooting an "
        "Intrusion Detection Dataset: the CICIDS2017 Case Study', IEEE SPW 2021. The data "
        "contains IP addresses; prepare pseudonymises them by default.",
        caveats=("lab traffic with scripted attacks", "some flows remain mislabelled"),
    ),
    "cidds-001": DatasetInfo(
        name="cidds-001",
        title="CIDDS-001 (Coburg Intrusion Detection Data Sets)",
        kind="flows",
        licence="no explicit licence; citation requested",
        source="https://www.hs-coburg.de/forschung/forschungsprojekte-oeffentlich/"
        "informationstechnologie/cidds-coburg-intrusion-detection-data-sets.html",
        download="403 MB zip (4.4 GB of CSV; the external-server part is 87 MB)",
        use="window aggregation of flows with attack-log events",
        terms="Research use. Cite M. Ring, S. Wunderlich, D. Gruedl, D. Landes and A. Hotho, "
        "'Flow-based benchmark data sets for intrusion detection', ECCWS 2017. The data contains "
        "IP addresses; prepare pseudonymises them by default.",
        caveats=("emulated environment", "internet traffic to the external server is unlabelled"),
    ),
}


def dataset_info(name: str) -> DatasetInfo:
    if name not in DATASETS:
        raise ConfigError(f"unknown dataset {name!r}", hint=f"Choose from {', '.join(DATASETS)}.")
    return DATASETS[name]


def describe_datasets() -> str:
    lines = []
    for info in DATASETS.values():
        lines += [
            f"{info.name} ({info.kind}): {info.title}",
            f"  licence: {info.licence}",
            f"  download: {info.download}; source: {info.source}",
            f"  used for: {info.use}",
        ]
        if info.terms:
            lines.append("  terms must be accepted (--accept-terms) before download")
        lines += [f"  caveat: {caveat}" for caveat in info.caveats]
    return "\n".join(lines)


def remote_files(name: str, *, machines: str | None = None) -> list[RemoteFile]:
    files: dict[str, Callable[[], list[RemoteFile]]] = {
        "nab": nab.remote_files,
        "smd": lambda: smd.remote_files(machines),
        "cicids2017-improved": flow_sources.cicids_files,
        "cidds-001": flow_sources.cidds_files,
    }
    dataset_info(name)
    return files[name]()


def download_dataset(
    name: str,
    data_dir: Path,
    *,
    accept_terms: bool = False,
    machines: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Fetch and verify every file of ``name`` into ``data_dir/<name>/``."""
    info = dataset_info(name)
    if info.terms and not accept_terms:
        raise ConfigError(
            f"{name} has terms of use: {info.terms}",
            hint="Re-run with --accept-terms if you agree to them.",
        )
    return [fetch(f, data_dir / name, progress) for f in remote_files(name, machines=machines)]


def prepare_dataset(
    name: str,
    data_dir: Path,
    output_dir: Path,
    *,
    machines: str | None = None,
    ip_policy: flow_sources.IpPolicy = "hash",
    days: tuple[str, ...] | None = None,
    part: Literal["external", "internal"] = "external",
    attempted_as_anomalous: bool = False,
) -> Prepared:
    """Convert downloaded files of ``name`` into canonical files under ``output_dir/<name>/``."""
    dataset_info(name)
    raw, output = data_dir / name, output_dir / name
    if name == "nab":
        return nab.prepare(raw, output)
    if name == "smd":
        return smd.prepare(raw, output, machines)
    if name == "cicids2017-improved":
        return flow_sources.prepare_cicids(
            raw,
            output,
            ip_policy=ip_policy,
            days=days or flow_sources.CICIDS_DAYS,
            attempted_as_anomalous=attempted_as_anomalous,
        )
    return flow_sources.prepare_cidds(raw, output, ip_policy=ip_policy, part=part)
