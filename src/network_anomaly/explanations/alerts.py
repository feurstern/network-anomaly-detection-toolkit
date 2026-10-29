"""From flagged points to alerts.

A single flagged point is a weak signal: one noisy sample or a scrape glitch can produce it.
Alerts therefore group flagged points per entity, tolerate short interruptions, and only fire once
enough points have been flagged, like an alert rule with a ``for:`` duration in production:

* flagged points of one entity separated by at most ``merge_gap`` unflagged or missing intervals
  belong to the same alert;
* an alert needs ``min_points`` flagged points. It *fires* at its ``min_points``-th point
  (``fired_at``, when an operator would be notified) and is *active* from ``fired_at`` until
  ``end``, one interval after its last flagged point.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field

from network_anomaly.config import StrictModel


class AlertingConfig(StrictModel):
    merge_gap: int = Field(
        default=2, ge=0, description="Unflagged or missing intervals allowed inside one alert."
    )
    min_points: int = Field(
        default=2, ge=1, description="Flagged points required before an alert fires."
    )
    
    
@dataclass(frozen=True)
class AlertInterval:
    """One alert on one entity. Times are grid timestamps (UTC); ``end`` is exclusive."""

    entity: str
    start: pd.Timestamp
    fired_at: pd.Timestamp
    end: pd.Timestamp
    peak_at: pd.Timestamp
    peak_score: float
    flagged_points: int


def build_alert_intervals(
    entities: npt.NDArray[np.object_],
    times: pd.Series,
    flagged: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    frequency: pd.Timedelta,
    config: AlertingConfig,
) -> list[AlertInterval]:
    """Group flagged rows into alerts, ordered by firing time then entity."""
    rows = pd.DataFrame(
        {
            "entity": entities,
            "time": times.to_numpy(),
            "score": scores,
        }
    ).loc[flagged]
    rows["time"] = pd.to_datetime(rows["time"], utc=True)
    rows = rows.sort_values(["entity", "time"], kind="stable")
    largest_step = (config.merge_gap + 1) * frequency
    alerts: list[AlertInterval] = []
    for entity, group in rows.groupby("entity", sort=True):
        block = (group["time"].diff() > largest_step).cumsum()
        for _, points in group.groupby(block, sort=True):
            if len(points) < config.min_points:
                continue
            peak = points.iloc[int(np.argmax(points["score"].to_numpy()))]
            alerts.append(
                AlertInterval(
                    entity=str(entity),
                    start=points["time"].iloc[0],
                    fired_at=points["time"].iloc[config.min_points - 1],
                    end=points["time"].iloc[-1] + frequency,
                    peak_at=peak["time"],
                    peak_score=float(peak["score"]),
                    flagged_points=len(points),
                )
            )
    return sorted(alerts, key=lambda alert: (alert.fired_at, alert.entity))
