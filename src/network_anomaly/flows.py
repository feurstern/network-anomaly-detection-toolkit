"""Network-flow analysis: windowed aggregation into the metrics table, and per-flow scoring.

Two complementary views of flow records that passed the flow schema
(:func:`network_anomaly.ingestion.validation.load_flows`):

* **Windowed aggregation** turns flows into the wide metrics table used by every detector: one row
  per (window, entity) with flows, packets and bytes per second, distinct sources and destination
  ports, and the share of TCP flows that never completed a handshake (SYN without ACK) or were
  reset. Alerts, evaluation and benchmarks then work on flow data unchanged. A window between an
  entity's first and last flow that holds no flow is an observation of zero traffic, written as
  zeros (ratios stay empty); it is not missing data.
* **Per-flow scoring** fits an Isolation Forest on the features of flows from a period assumed to
  be normal (for example a capture day without attacks) and scores the flows of another period.

Identifiers (addresses), timestamps and labels are never model features. Source addresses are
only counted per window; the aggregated table holds counts, not addresses.
"""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from network_anomaly.config import StrictModel
from network_anomaly.errors import ConfigError, DataValidationError
from network_anomaly.schemas import DEFAULT_ENTITY_COLUMN, TIMESTAMP_COLUMN

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

TCP = "tcp"
_PROTOCOLS = {"tcp": TCP, "6": TCP, "udp": "udp", "17": "udp", "icmp": "icmp", "1": "icmp"}
_PROTOCOLS |= {"icmpv6": "icmp", "ipv6-icmp": "icmp", "58": "icmp"}
OTHER_ENTITY = "other"
FLAGS = ("syn", "ack", "fin", "rst", "psh")


def protocol_names(values: pd.Series) -> pd.Series:
    """Normalised protocol names (tcp, udp, icmp, other) from names or IANA numbers."""
    text = values.astype("string").str.strip().str.lower()
    text = text.str.replace(r"\.0$", "", regex=True)  # numbers parsed as floats elsewhere
    return text.map(_PROTOCOLS).fillna("other").astype(str)


def _total(data: pd.DataFrame, field: str) -> pd.Series:
    """``packets``/``bytes`` if present, else forward + backward (NaN when unknown)."""
    if field in data:
        return data[field].astype(np.float64)
    parts = [f"{field}_fwd", f"{field}_bwd"]
    if all(part in data for part in parts):
        return data[parts[0]].astype(np.float64) + data[parts[1]].astype(np.float64)
    return pd.Series(np.nan, index=data.index)


def _flag(data: pd.DataFrame, name: str) -> pd.Series:
    column = f"{name}_count"
    if column not in data:
        return pd.Series(np.nan, index=data.index)
    return data[column].astype(np.float64)


# --------------------------------------------------------------------------------------------------
# Windowed aggregation
# --------------------------------------------------------------------------------------------------


def aggregate_flows(
    data: pd.DataFrame,
    *,
    window: pd.Timedelta,
    entity: str | None = None,
    top_entities: int | None = None,
) -> pd.DataFrame:
    """Aggregate flow records into one metrics row per (window start, entity).

    ``entity`` names the flow field to group by (e.g. ``dst_addr``); None aggregates everything
    into one entity, ``all``. With ``top_entities``, entities beyond the busiest N are merged into
    ``other``. The result has ``timestamp`` (window start, UTC), ``host_id`` (the entity), the
    flow-aggregate metrics available from the input and, for labelled flows, ``is_anomaly``
    (1 if any flow of the window is labelled anomalous).
    """
    if window <= pd.Timedelta(0):
        raise ConfigError("the aggregation window must be positive")
    if entity is not None and entity not in data:
        raise DataValidationError(f"cannot aggregate by {entity!r}: the flows have no such column")
    frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: data[TIMESTAMP_COLUMN].dt.floor(window),
            DEFAULT_ENTITY_COLUMN: "all" if entity is None else data[entity].astype(str),
            "packets": _total(data, "packets"),
            "bytes": _total(data, "bytes"),
        }
    )
    if top_entities is not None and entity is not None:
        busiest = frame[DEFAULT_ENTITY_COLUMN].value_counts().index[:top_entities]
        others = ~frame[DEFAULT_ENTITY_COLUMN].isin(busiest)
        frame.loc[others, DEFAULT_ENTITY_COLUMN] = OTHER_ENTITY
    if "src_addr" in data:
        frame["source"] = data["src_addr"].astype("string")
    if "dst_port" in data:
        frame["dst_port"] = data["dst_port"].astype(np.float64)
    if "protocol" in data and {"syn_count", "ack_count"} <= set(data.columns):
        tcp = protocol_names(data["protocol"]).eq(TCP).to_numpy()
        syn_only = (_flag(data, "syn") > 0) & ~(_flag(data, "ack") > 0)
        frame["syn_only"] = np.where(tcp, syn_only.astype(np.float64), np.nan)
        if "rst_count" in data:
            frame["rst"] = np.where(tcp, (_flag(data, "rst") > 0).astype(np.float64), np.nan)
    if "is_anomaly" in data:
        frame["is_anomaly"] = data["is_anomaly"].astype(np.float64)

    keys = [DEFAULT_ENTITY_COLUMN, TIMESTAMP_COLUMN]
    grouped = frame.groupby(keys, sort=True)
    seconds = window.total_seconds()
    table = pd.DataFrame({"flows_per_second": grouped.size() / seconds})
    table["flow_packets_per_second"] = grouped["packets"].sum(min_count=1) / seconds
    table["flow_bytes_per_second"] = grouped["bytes"].sum(min_count=1) / seconds
    if "source" in frame:
        table["unique_sources"] = grouped["source"].nunique()
    if "dst_port" in frame:
        table["unique_destination_ports"] = grouped["dst_port"].nunique()
    if "syn_only" in frame:
        table["syn_only_ratio"] = grouped["syn_only"].mean()
    if "rst" in frame:
        table["rst_ratio"] = grouped["rst"].mean()
    if "is_anomaly" in frame:
        table["is_anomaly"] = grouped["is_anomaly"].max()
    table = table.dropna(axis=1, how="all")

    # Windows without flows between an entity's first and last flow observed zero traffic.
    complete = []
    for name, rows in table.groupby(level=0, sort=True):
        times = rows.index.get_level_values(1)
        grid = pd.date_range(times.min(), times.max(), freq=window)
        index = pd.MultiIndex.from_product([[name], grid], names=keys)
        complete.append(rows.reindex(index))
    table = pd.concat(complete)
    counts = [c for c in table.columns if not c.endswith("_ratio")]
    table[counts] = table[counts].fillna(0.0)
    if "is_anomaly" in table:
        table["is_anomaly"] = table["is_anomaly"].astype(np.int8)
    table = table.reset_index()[[TIMESTAMP_COLUMN, *keys[:1], *table.columns]]
    return table.sort_values(keys, kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# Per-flow scoring
# --------------------------------------------------------------------------------------------------


def flow_features(data: pd.DataFrame) -> pd.DataFrame:
    """Numeric per-flow features. Addresses, timestamps and labels are never included."""
    packets = _total(data, "packets")
    size = _total(data, "bytes")
    duration = (
        data["duration_ms"].astype(np.float64)
        if "duration_ms" in data
        else pd.Series(np.nan, index=data.index)
    )
    features = pd.DataFrame(
        {
            "log_packets": np.log1p(packets.clip(lower=0)),
            "log_bytes": np.log1p(size.clip(lower=0)),
            "log_duration_ms": np.log1p(duration.clip(lower=0)),
            "log_bytes_per_packet": np.log1p((size / packets.where(packets > 0)).clip(lower=0)),
        }
    )
    for name in FLAGS:
        features[f"{name}_share"] = (_flag(data, name) / packets.where(packets > 0)).clip(0, 1)
    port = data["dst_port"].astype(np.float64) if "dst_port" in data else None
    if port is not None:
        features["port_well_known"] = (port < 1024).astype(np.float64)
        features["port_registered"] = ((port >= 1024) & (port < 49152)).astype(np.float64)
        features["port_dynamic"] = (port >= 49152).astype(np.float64)
    if "protocol" in data:
        names = protocol_names(data["protocol"])
        for name in ("tcp", "udp", "icmp", "other"):
            features[f"protocol_{name}"] = names.eq(name).astype(np.float64)
    # Unknown values become 0 so that a missing field never looks extreme.
    return features.fillna(0.0).astype(np.float64)


class FlowDetectionConfig(StrictModel):
    n_estimators: int = Field(default=200, ge=10, le=5000)
    max_samples: int = Field(default=256, ge=16)
    max_train_flows: int = Field(default=200_000, ge=1000)
    target_fpr: float = Field(
        default=0.01,
        gt=0,
        lt=0.5,
        description="Share of training flows allowed above the threshold (train_quantile).",
    )
    random_state: int = 0


@dataclass(frozen=True)
class FlowModel:
    """An Isolation Forest over per-flow features with a training-quantile threshold."""

    forest: IsolationForest
    features: tuple[str, ...]
    threshold: float
    training_flows: int

    @classmethod
    def fit(cls, train: pd.DataFrame, config: FlowDetectionConfig) -> "FlowModel":
        features = flow_features(train)
        if len(features) < 100:
            raise DataValidationError(
                f"need at least 100 training flows, got {len(features)}",
                hint="Use a longer period of normal traffic for --train.",
            )
        matrix = features.to_numpy(dtype=np.float64)
        if not np.ptp(matrix, axis=0).any():
            raise DataValidationError(
                "the training flows are all identical in every feature; nothing can stand out",
                hint="Train on a realistic period of normal traffic.",
            )
        rng = np.random.default_rng(config.random_state)
        rows = (
            np.sort(rng.choice(len(features), config.max_train_flows, replace=False))
            if len(features) > config.max_train_flows
            else np.arange(len(features))
        )
        forest = IsolationForest(
            n_estimators=config.n_estimators,
            max_samples=min(config.max_samples, len(rows)),
            random_state=config.random_state,
            n_jobs=1,
        ).fit(matrix[rows])
        scores = -forest.score_samples(matrix[rows])
        threshold = float(np.quantile(scores, 1.0 - config.target_fpr))
        return cls(forest, tuple(features.columns), threshold, len(rows))

    def score(self, data: pd.DataFrame) -> FloatArray:
        """Higher = more unusual than the training flows; unitless, not a probability."""
        features = flow_features(data).reindex(columns=list(self.features), fill_value=0.0)
        scores: FloatArray = -self.forest.score_samples(features.to_numpy(dtype=np.float64))
        return scores


class FlowEvaluation(StrictModel):
    flows: int
    anomalous: int
    flagged: int
    precision: float | None
    recall: float | None
    false_positive_rate: float | None
    pr_auc: float | None
    recall_at_1pct_fpr: float | None = Field(
        description="Recall at the score that 1% of normal flows exceed (threshold-free summary)."
    )
    recall_by_label: dict[str, float]


def evaluate_flows(
    labels: npt.NDArray[np.int8],
    scores: FloatArray,
    flagged: BoolArray,
    names: pd.Series | None = None,
) -> FlowEvaluation:
    """Per-flow metrics at the deployed threshold plus threshold-free ranking quality."""
    actual = labels.astype(bool)
    tp = int((flagged & actual).sum())
    fp = int((flagged & ~actual).sum())
    positives, negatives = int(actual.sum()), int((~actual).sum())
    both = 0 < positives < len(actual)
    recall_at_1pct = None
    if both:
        cut = float(np.quantile(scores[~actual], 0.99))
        recall_at_1pct = float((scores[actual] > cut).mean())
    by_label: dict[str, float] = {}
    if names is not None:
        names = names.reset_index(drop=True).astype("string").fillna("unknown")
        for name in sorted(set(names[actual])):
            rows = actual & names.eq(name).to_numpy()
            by_label[str(name)] = float(flagged[rows].mean())
    return FlowEvaluation(
        flows=len(actual),
        anomalous=positives,
        flagged=int(flagged.sum()),
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / positives if positives else None,
        false_positive_rate=fp / negatives if negatives else None,
        pr_auc=float(average_precision_score(actual, scores)) if both else None,
        recall_at_1pct_fpr=recall_at_1pct,
        recall_by_label=by_label,
    )
