"""Do alert explanations point at the metrics an anomaly actually affected?

Ground truth comes from events that list their affected metrics: the synthetic generator records
every metric whose injected effect exceeded two noise standard deviations, and SMD's
interpretation labels list the dimensions engineers attributed each anomaly to.

For a detected event with ground-truth set G, the first alert that detected it ranks metrics by
contribution (:attr:`~network_anomaly.schemas.Alert.metric_ranking`):

* **HitRate@P%** (Su et al., KDD 2019) - the share of G found among the top floor(P% x |G|)
  ranked metrics, averaged over events; reported for P = 100 % and 150 %.
* **Top-metric hit rate** - the share of events whose highest-ranked metric is in G.

Contributions rank metrics that moved with the anomaly; a high hit rate says the explanation
points where an engineer would look, not that it identifies a cause.
"""

import math
from collections.abc import Sequence

import pandas as pd

from network_anomaly.config import StrictModel
from network_anomaly.evaluation.events import EventMatch
from network_anomaly.schemas import Alert


class ExplanationMetrics(StrictModel):
    events: int  # detected events with listed affected metrics and an explained alert
    hit_rate_at_100: float | None
    hit_rate_at_150: float | None
    top_metric_hit_rate: float | None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def explanation_hit_rates(
    matches: Sequence[EventMatch], alerts: Sequence[Alert]
) -> ExplanationMetrics:
    """Hit rates of alert explanations against the affected metrics of detected events."""
    explained = {(a.entity, pd.Timestamp(a.start)): a for a in alerts}
    hits: dict[int, list[float]] = {100: [], 150: []}
    top: list[float] = []
    for match in matches:
        truth = set(match.event.affected_metrics)
        if not match.alerts or not truth:
            continue
        first = min(match.alerts, key=lambda interval: interval.fired_at)
        alert = explained.get((first.entity, first.start))
        if alert is None or not alert.metric_ranking:
            continue
        for percent, values in hits.items():
            ranked = alert.metric_ranking[: math.floor(percent / 100 * len(truth))]
            values.append(len(set(ranked) & truth) / len(truth))
        top.append(float(alert.metric_ranking[0] in truth))
    return ExplanationMetrics(
        events=len(top),
        hit_rate_at_100=_mean(hits[100]),
        hit_rate_at_150=_mean(hits[150]),
        top_metric_hit_rate=_mean(top),
    )
