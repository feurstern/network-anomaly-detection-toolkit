"""Public network-flow datasets converted to the canonical flow schema.

* **CICIDS2017 (improved)** - DistriNet's corrected re-labelling of CIC-IDS2017 (Engelen et al.,
  2021): five capture days of bidirectional flows with UTC timestamps (microseconds), flag counts
  and labels. Flows labelled ``... - Attempted`` (attacks that did not get through) are *not*
  positives by default. Monday contains only benign traffic, which makes it the natural training
  day for per-flow scoring.
* **CIDDS-001** - four weeks of unidirectional NetFlow from an emulated small business (Ring et
  al., 2017). The adapter uses the external server's traffic by default (``--part external``);
  the OpenStack part is 4 GB. Timestamps are local German time and are converted to UTC (the
  report's "missing hour" on 2017-03-26 is the daylight-saving jump, not lost traffic). Byte
  counts such as ``2.1 M`` are expanded (factor 1000); TCP flags are a set of letters per flow, so
  ``*_count`` fields hold 1 when a flag occurred at least once, not a packet count. Flows of class
  ``attacker`` or ``victim`` are positives; ``suspicious`` and ``unknown`` (internet traffic of
  unknown intent) are not, and are kept as ``label``. The attack logs become events for
  window-level evaluation of the aggregated traffic (entity ``all``).

Neither dataset carries a formal licence; both ask to be cited, and ``datasets download`` shows
those terms before it fetches anything. Both contain IP addresses. ``ip_policy`` decides what
happens to them: ``hash`` (default) replaces each address with a keyed HMAC-SHA256 pseudonym,
``drop`` removes the columns and ``keep`` leaves them as published. Addresses are never features.
"""

import hashlib
import hmac
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from network_anomaly.errors import ArtifactError, ConfigError, DataValidationError
from network_anomaly.ingestion.datasets.common import Prepared, write_table
from network_anomaly.ingestion.datasets.download import RemoteFile
from network_anomaly.ingestion.datasets.pins import CICIDS2017_IMPROVED, CIDDS_001
from network_anomaly.ingestion.events import write_events
from network_anomaly.schemas import TIMESTAMP_COLUMN, Event

IpPolicy = Literal["hash", "drop", "keep"]
ANON_KEY_ENV = "NETWORK_ANOMALY_ANON_KEY"
ADDRESS_COLUMNS = ("src_addr", "dst_addr")
CHUNK_ROWS = 500_000

CICIDS_ARCHIVE = "CICIDS2017_improved.zip"
CICIDS_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")
CICIDS_COLUMNS = {
    "Timestamp": TIMESTAMP_COLUMN,
    "Src IP": "src_addr",
    "Src Port": "src_port",
    "Dst IP": "dst_addr",
    "Dst Port": "dst_port",
    "Protocol": "protocol",
    "Flow Duration": "duration_us",
    "Total Fwd Packet": "packets_fwd",
    "Total Bwd packets": "packets_bwd",
    "Total Length of Fwd Packet": "bytes_fwd",
    "Total Length of Bwd Packet": "bytes_bwd",
    "FIN Flag Count": "fin_count",
    "SYN Flag Count": "syn_count",
    "RST Flag Count": "rst_count",
    "PSH Flag Count": "psh_count",
    "ACK Flag Count": "ack_count",
    "URG Flag Count": "urg_count",
    "Label": "label",
}

CIDDS_ARCHIVE = "CIDDS-001.zip"
CIDDS_PARTS = {
    "external": ("ExternalServer", "CIDDS-001-external-week{}.csv", "attack_logs_extern.csv"),
    "internal": ("OpenStack", "CIDDS-001-internal-week{}.csv", "attack_logs_intern.csv"),
}
CIDDS_COLUMNS = {
    "Date first seen": TIMESTAMP_COLUMN,
    "Duration": "duration_s",
    "Proto": "protocol",
    "Src IP Addr": "src_addr",
    "Src Pt": "src_port",
    "Dst IP Addr": "dst_addr",
    "Dst Pt": "dst_port",
    "Packets": "packets",
    "Bytes": "bytes",
    "Flags": "flags",
    "class": "class",
    "attackType": "attack_type",
}
# nfdump prints TCP flags as six positions, "UAPRSF", with "." where a flag is absent.
CIDDS_FLAGS = {"urg_count": 0, "ack_count": 1, "psh_count": 2, "rst_count": 3, "syn_count": 4}
CIDDS_FLAGS |= {"fin_count": 5}
CIDDS_TIMEZONE = "Europe/Berlin"


def cicids_files() -> list[RemoteFile]:
    url, size, sha256 = CICIDS2017_IMPROVED
    return [RemoteFile(url=url, target=CICIDS_ARCHIVE, size=size, sha256=sha256)]


def cidds_files() -> list[RemoteFile]:
    url, size, sha256 = CIDDS_001
    return [RemoteFile(url=url, target=CIDDS_ARCHIVE, size=size, sha256=sha256)]


# --------------------------------------------------------------------------------------------------
# Address policy
# --------------------------------------------------------------------------------------------------


def anonymisation_key(policy: IpPolicy) -> bytes | None:
    """The HMAC key for ``hash``, read from the environment (never stored with the data)."""
    if policy != "hash":
        return None
    key = os.environ.get(ANON_KEY_ENV, "")
    if len(key) < 16:
        raise ConfigError(
            f"--ip-policy hash needs a secret key of at least 16 characters in {ANON_KEY_ENV}",
            hint=f"export {ANON_KEY_ENV}=$(openssl rand -hex 32), or use --ip-policy drop.",
        )
    return key.encode("utf-8")


def apply_ip_policy(frame: pd.DataFrame, policy: IpPolicy, key: bytes | None) -> pd.DataFrame:
    """Pseudonymise (``hash``), remove (``drop``) or keep the address columns."""
    present = [c for c in ADDRESS_COLUMNS if c in frame]
    if policy == "drop":
        return frame.drop(columns=present)
    if policy == "hash":
        if key is None:
            raise ConfigError("hashing addresses needs a key")
        for column in present:
            values = frame[column].astype("string")
            unique = values.dropna().unique()
            codes = {
                value: "h-" + hmac.new(key, value.encode(), hashlib.sha256).hexdigest()[:16]
                for value in unique
            }
            frame[column] = values.map(codes)
    return frame


def _archive(raw: Path, name: str, dataset: str) -> zipfile.ZipFile:
    path = raw / name
    if not path.is_file():
        raise ArtifactError(
            f"{path} not found", hint=f"Run `network-anomaly datasets download {dataset}` first."
        )
    return zipfile.ZipFile(path)


def _chunks(
    handle: zipfile.ZipFile, member: str, columns: dict[str, str]
) -> Iterator[pd.DataFrame]:
    with handle.open(member) as stream:
        header = pd.read_csv(stream, nrows=0, skipinitialspace=True).columns
    absent = [c for c in columns if c not in header]
    if absent:
        raise DataValidationError(
            f"{member}: expected columns are missing: {', '.join(absent)}",
            hint="The archive differs from the pinned release; re-download it.",
        )
    with handle.open(member) as stream:
        yield from pd.read_csv(
            stream,
            usecols=list(columns),
            dtype=str,
            chunksize=CHUNK_ROWS,
            skipinitialspace=True,
            keep_default_na=False,
        )


# --------------------------------------------------------------------------------------------------
# CICIDS2017 (improved)
# --------------------------------------------------------------------------------------------------


def convert_cicids(chunk: pd.DataFrame, *, attempted_as_anomalous: bool) -> pd.DataFrame:
    """One chunk of a CICIDS2017-improved day file in the canonical flow schema."""
    frame = chunk.rename(columns=CICIDS_COLUMNS)
    out = pd.DataFrame(
        {TIMESTAMP_COLUMN: pd.to_datetime(frame[TIMESTAMP_COLUMN], format="ISO8601", utc=True)}
    )
    for column in ("src_addr", "dst_addr", "protocol", "label"):
        out[column] = frame[column].str.strip()
    numeric = [
        c for c in CICIDS_COLUMNS.values() if c.endswith(("_port", "_fwd", "_bwd", "_count"))
    ]
    for column in numeric:
        out[column] = pd.to_numeric(frame[column], errors="raise")
    out["duration_ms"] = pd.to_numeric(frame["duration_us"], errors="raise") / 1000.0
    attempted = out["label"].str.endswith("Attempted")
    attack = out["label"].ne("BENIGN") & (attempted_as_anomalous | ~attempted)
    out["is_anomaly"] = attack.astype(np.int8)
    return out


def prepare_cicids(
    raw: Path,
    output: Path,
    *,
    ip_policy: IpPolicy,
    days: tuple[str, ...] = CICIDS_DAYS,
    attempted_as_anomalous: bool = False,
) -> Prepared:
    unknown = [d for d in days if d not in CICIDS_DAYS]
    if unknown:
        raise ConfigError(f"unknown day(s): {', '.join(unknown)}", hint=f"Choose {CICIDS_DAYS}.")
    key = anonymisation_key(ip_policy)
    prepared = Prepared("cicids2017-improved")
    with _archive(raw, CICIDS_ARCHIVE, "cicids2017-improved") as archive:
        for day in days:
            target = output / f"{day}.flows.csv"
            for index, chunk in enumerate(_chunks(archive, f"{day}.csv", CICIDS_COLUMNS)):
                flows = convert_cicids(chunk, attempted_as_anomalous=attempted_as_anomalous)
                write_table(apply_ip_policy(flows, ip_policy, key), target, append=index > 0)
            prepared.files.append(target)
    prepared.notes.append(
        f"addresses: {ip_policy}"
        + (f" (HMAC-SHA256 keyed by ${ANON_KEY_ENV})" if ip_policy == "hash" else "")
    )
    if not attempted_as_anomalous:
        prepared.notes.append("'... - Attempted' flows are labelled but not counted as anomalies")
    if "monday" in days:
        rest = [d for d in days if d != "monday"]
        day = rest[0] if rest else "monday"
        prepared.commands.append(
            f"network-anomaly flows detect --train {output / 'monday.flows.csv'} "
            f"--input {output / f'{day}.flows.csv'} --output reports/cicids/{day}.scores.csv"
        )
    return prepared


# --------------------------------------------------------------------------------------------------
# CIDDS-001
# --------------------------------------------------------------------------------------------------


def parse_byte_counts(values: pd.Series) -> pd.Series:
    """nfdump byte counts such as ``1234``, ``2.1 M`` or ``3.5 G`` (factor 1000)."""
    text = values.str.strip()
    scale = text.str.extract(r"([KMG])$", expand=False).map({"K": 1e3, "M": 1e6, "G": 1e9})
    number = pd.to_numeric(text.str.replace(r"\s*[KMG]$", "", regex=True), errors="raise")
    return number * scale.fillna(1.0)


def _local_to_utc(values: pd.Series) -> pd.Series:
    local = pd.to_datetime(values.str.strip(), format="%Y-%m-%d %H:%M:%S.%f")
    zoned = local.dt.tz_localize(CIDDS_TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")
    return zoned.dt.tz_convert("UTC")


def convert_cidds(chunk: pd.DataFrame) -> pd.DataFrame:
    """One chunk of a CIDDS-001 traffic file in the canonical flow schema."""
    frame = chunk.rename(columns=CIDDS_COLUMNS)
    out = pd.DataFrame({TIMESTAMP_COLUMN: _local_to_utc(frame[TIMESTAMP_COLUMN])})
    out["src_addr"] = frame["src_addr"].str.strip()
    out["dst_addr"] = frame["dst_addr"].str.strip()
    out["protocol"] = frame["protocol"].str.strip().str.lower()
    # nfdump writes ICMP type.code (e.g. "3.3") into the port fields; ICMP has no ports.
    icmp = out["protocol"].eq("icmp")
    for column in ("src_port", "dst_port"):
        out[column] = pd.to_numeric(frame[column].where(~icmp), errors="raise")
    out["duration_ms"] = pd.to_numeric(frame["duration_s"], errors="raise") * 1000.0
    out["packets"] = pd.to_numeric(frame["packets"], errors="raise")
    out["bytes"] = parse_byte_counts(frame["bytes"])
    flags = frame["flags"].str.strip().str.pad(6, side="right", fillchar=".")
    for column, position in CIDDS_FLAGS.items():
        out[column] = flags.str[position].ne(".").astype(np.int8)
    attack = frame["attack_type"].str.strip()
    cls = frame["class"].str.strip()
    out["label"] = cls.where(attack.isin(["---", ""]), cls + "/" + attack)
    out["is_anomaly"] = cls.isin(["attacker", "victim"]).astype(np.int8)
    return out.dropna(subset=[TIMESTAMP_COLUMN])  # times that repeat when clocks go back


def _cidds_events(archive: zipfile.ZipFile, log: str) -> list[Event]:
    with archive.open(f"CIDDS-001/attack_logs/{log}") as stream:
        attacks = pd.read_csv(stream, dtype=str, keep_default_na=False)
    starts, ends = _local_to_utc(attacks["start"]), _local_to_utc(attacks["end"])
    return [
        Event(
            event_id=f"cidds-{row['activityID']}",
            host_id="all",
            scenario=str(row["activity"]),
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
        )
        for (_, row), start, end in zip(attacks.iterrows(), starts, ends, strict=True)
        if end > start
    ]


def prepare_cidds(
    raw: Path,
    output: Path,
    *,
    ip_policy: IpPolicy,
    part: Literal["external", "internal"] = "external",
) -> Prepared:
    key = anonymisation_key(ip_policy)
    folder, pattern, log = CIDDS_PARTS[part]
    prepared = Prepared("cidds-001")
    with _archive(raw, CIDDS_ARCHIVE, "cidds-001") as archive:
        for week in range(1, 5):
            name = pattern.format(week)
            target = output / name.replace(".csv", ".flows.csv")
            member = f"CIDDS-001/traffic/{folder}/{name}"
            for index, chunk in enumerate(_chunks(archive, member, CIDDS_COLUMNS)):
                flows = apply_ip_policy(convert_cidds(chunk), ip_policy, key)
                write_table(flows, target, append=index > 0)
            prepared.files.append(target)
        events_path = output / f"cidds-001-{part}.events.csv"
        write_events(_cidds_events(archive, log), events_path)
    prepared.notes += [
        f"timestamps converted from {CIDDS_TIMEZONE} local time to UTC",
        "TCP flag fields hold 1 when the flag occurred in the flow (CIDDS records flag sets)",
        "suspicious/unknown internet traffic is kept but not labelled anomalous",
        f"attack logs written to {events_path.name} for the aggregated entity 'all'",
    ]
    second_week = output / pattern.format(2).replace(".csv", ".flows.csv")
    prepared.commands.append(
        f"network-anomaly flows aggregate --input {second_week} --window 5min "
        f"--output data/processed/cidds-001/{part}-week2.metrics.csv"
    )
    return prepared
