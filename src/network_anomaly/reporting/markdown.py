"""The evaluation report: ``metrics.json``, figures and a Markdown summary in one directory.

Every number in the figures also appears in a table, so the report reads without them. The report
says what scores and alerts are not (probabilities, proof of an attack), whether rows were
in-sample, and - for synthetic data - that results do not demonstrate production effectiveness.
"""

import json
import math
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from network_anomaly import __version__
from network_anomaly.evaluation.events import EventMatch, EventMatchingConfig, event_window
from network_anomaly.evaluation.explanations import ExplanationMetrics, explanation_hit_rates
from network_anomaly.evaluation.runner import EvaluationInputs, EvaluationOutcome
from network_anomaly.ingestion.events import sibling_path
from network_anomaly.reporting.plots import (
    confusion_matrix_figure,
    feature_deviation_figure,
    pr_curve_figure,
    save_figure,
    timeseries_figure,
)
from network_anomaly.schemas import Alert, Event
from network_anomaly.scoring.severity import SEVERITY_ORDER

DEFAULT_PLOT_METRICS = (
    "requests_per_second",
    "response_time_p95_ms",
    "http_5xx_rate",
    "cpu_usage_percent",
)
MAX_PLOT_METRICS = 3
MAX_LISTED = 15


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _when(value: datetime | pd.Timestamp) -> str:
    return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%d %H:%M")


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def is_synthetic(input_path: Path) -> bool:
    """Whether ``input_path`` was written by the synthetic generator (per its metadata file)."""
    path = sibling_path(input_path, "metadata.json")
    try:
        return "generator" in json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def alert_matches_event(
    alert: Alert,
    events: Sequence[Event],
    frequency: pd.Timedelta,
    matching: EventMatchingConfig,
) -> bool:
    """Whether the alert is active inside the matching window of an event on its entity."""
    for event in events:
        if event.host_id != alert.entity:
            continue
        start, end = event_window(event, frequency, matching)
        if pd.Timestamp(alert.fired_at) < end and pd.Timestamp(alert.end) > start:
            return True
    return False


def _plot_entity(
    inputs: EvaluationInputs, outcome: EvaluationOutcome, requested: str | None
) -> str | None:
    table = inputs.table.loc[outcome.selected]
    if requested is not None:
        return requested if requested in set(table["entity"]) else None
    if table.empty:
        return None
    events = Counter(match.event.host_id for match in outcome.matches)
    flagged = table.loc[table["flagged"], "entity"].value_counts()
    return min(
        sorted(set(table["entity"])),
        key=lambda entity: (-events.get(entity, 0), -int(flagged.get(entity, 0)), entity),
    )


def _plot_metrics(inputs: EvaluationInputs, alerts: Sequence[Alert], entity: str) -> list[str]:
    available = list(inputs.metric_values.columns)
    counts = Counter(c.metric for a in alerts if a.entity == entity for c in a.contributions)
    chosen = [m for m, _ in counts.most_common() if m in available][:MAX_PLOT_METRICS]
    for metric in (*DEFAULT_PLOT_METRICS, *available):
        if len(chosen) >= MAX_PLOT_METRICS:
            break
        if metric in available and metric not in chosen:
            chosen.append(metric)
    return chosen


def _figures(
    directory: Path,
    inputs: EvaluationInputs,
    outcome: EvaluationOutcome,
    alerts: Sequence[Alert],
    matched: Sequence[bool],
    plot_entity: str | None,
) -> dict[str, str]:
    report = outcome.report
    detector = report.detector
    table = inputs.table.loc[outcome.selected]
    written: dict[str, str] = {}

    def save(name: str, figure: Any) -> None:
        if figure is not None:
            save_figure(figure, directory / "figures" / f"{name}.png")
            written[name] = f"figures/{name}.png"

    save("confusion_matrix", confusion_matrix_figure(report.point, detector=detector))
    save(
        "pr_curve",
        pr_curve_figure(
            table["is_anomaly"].to_numpy(dtype=np.int8),
            table["score"].to_numpy(dtype=np.float64),
            point=report.point,
            detector=detector,
        ),
    )
    save("feature_deviations", feature_deviation_figure(alerts, matched, detector=detector))
    entity = _plot_entity(inputs, outcome, plot_entity)
    if entity is not None:
        rows = table[table["entity"] == entity].sort_values("time")
        window = (pd.Timestamp(report.window_start), pd.Timestamp(report.window_end))
        metrics = _plot_metrics(inputs, alerts, entity)
        events = [
            (pd.Timestamp(m.event.start), pd.Timestamp(m.event.end))
            for m in outcome.matches
            if m.event.host_id == entity
        ]
        intervals = [
            (max(a.fired_at, window[0]), min(a.end, window[1]))
            for a in outcome.alerts
            if a.entity == entity and a.fired_at < window[1] and a.end > window[0]
        ]
        save(
            "timeseries",
            timeseries_figure(
                entity=entity,
                detector=detector,
                times=rows["time"],
                scores=rows["score"].to_numpy(dtype=np.float64),
                thresholds=rows["threshold"].to_numpy(dtype=np.float64),
                flagged=rows["flagged"].to_numpy(dtype=bool),
                metrics={
                    m: inputs.metric_values.loc[rows.index, m].to_numpy(np.float64) for m in metrics
                },
                frequency=inputs.frequency,
                window=window,
                events=events,
                alerts=intervals,
            ),
        )
        written["timeseries_entity"] = entity
    return written


def _threshold_text(meta: dict[str, Any]) -> str:
    thresholds = meta.get("thresholds", {})
    params = ", ".join(
        f"{k.replace('_', ' ')} {v}" if not isinstance(v, bool) else ("per host" if v else "global")
        for k, v in thresholds.get("params", {}).items()
    )
    fitted = thresholds.get("fitted_on")
    copy = "a known-clean copy of " if thresholds.get("source") == "known_clean" else ""
    where = (
        f", fitted on {copy}the {fitted} segment ({_when(thresholds['window_start'])} to "
        f"{_when(thresholds['window_end'])} UTC)"
        if fitted and thresholds.get("window_start")
        else (f", fitted on the {fitted} segment" if fitted else "")
    )
    hosts = len(thresholds.get("per_host", {}))
    scope = f"; per-host thresholds for {hosts} host(s)" if hosts else ""
    return f"`{thresholds.get('strategy', 'unknown')}` ({params}){where}{scope}"


def _event_rows(matches: Sequence[EventMatch], detected: bool) -> list[list[object]]:
    rows: list[list[object]] = []
    for match in matches:
        if bool(match.alerts) != detected:
            continue
        event = match.event
        minutes = (pd.Timestamp(event.end) - pd.Timestamp(event.start)).total_seconds() / 60
        row: list[object] = [
            event.event_id,
            event.host_id,
            event.scenario,
            event.severity or "-",
            _when(event.start),
            f"{minutes:.0f}",
        ]
        if detected:
            row.append(_num(match.delay_minutes, 0))
        rows.append(row)
    return rows[:MAX_LISTED]


def render_markdown(
    inputs: EvaluationInputs,
    outcome: EvaluationOutcome,
    *,
    alerts: Sequence[Alert] | None,
    matched: Sequence[bool],
    figures: dict[str, str],
    input_path: Path,
    predictions_path: Path,
    explanations: ExplanationMetrics | None = None,
) -> str:
    report = outcome.report
    p, e = report.point, report.events
    meta = inputs.meta
    days = (report.window_end - report.window_start).total_seconds() / 86_400
    segments = ", ".join(report.segments)
    sha = (meta.get("input") or {}).get("sha256", "")
    lines = [f"# Evaluation report: {report.detector}", ""]
    if report.in_sample:
        lines += [
            "> **In-sample.** The evaluated rows include data the detector or its threshold was "
            "fitted on. These numbers are optimistic and must not be compared with test results.",
            "",
        ]
    lines += [
        f"Generated by network-anomaly {__version__} from `{input_path.name}` and "
        f"`{predictions_path.name}`.",
        "",
        _table(
            ("Item", "Value"),
            [
                ("Input", f"`{input_path.name}`" + (f" (SHA-256 `{sha[:16]}…`)" if sha else "")),
                ("Detector", f"`{report.detector}` v{meta['detector'].get('version', '?')}"),
                (
                    "Evaluated",
                    f"{segments} segment: {_when(report.window_start)} to "
                    f"{_when(report.window_end)} UTC ({days:.1f} days)",
                ),
                ("Threshold", _threshold_text(meta)),
                (
                    "Alerting",
                    f"an alert fires on flagged point {report.alerting.min_points}; flagged points "
                    f"up to {report.alerting.merge_gap} intervals apart are merged",
                ),
                (
                    "Event matching",
                    f"detected if an alert is active from {report.matching.early_tolerance} "
                    f"interval(s) before an event to {report.matching.late_tolerance} after it",
                ),
            ],
        ),
        "",
        "> **How to read this.** Scores are unitless (higher = more anomalous) and are not "
        "probabilities; thresholds are cut-offs on that scale, not calibrated risks. An alert "
        "is a signal for investigation, not proof of an attack or of its cause.",
        "",
    ]
    if is_synthetic(input_path):
        lines += [
            "> **Synthetic data.** These results describe generated data with known labels. They "
            "do not demonstrate effectiveness on production traffic.",
            "",
        ]
    lines += [
        "## Event level",
        "",
        "Each labelled event is one incident; alerts are grouped flagged points.",
        "",
        _table(
            ("Metric", "Value"),
            [
                ("Events evaluated", e.events),
                ("Detected (recall)", f"{e.detected} ({_num(e.recall)})"),
                ("Alerts fired in the window", e.alerts),
                ("False alerts (precision)", f"{e.false_alerts} ({_num(e.precision)})"),
                ("Event F1", _num(e.f1)),
                (
                    "Detection delay, median / p90 (min)",
                    f"{_num(e.median_delay_minutes, 1)} / {_num(e.p90_delay_minutes, 1)}",
                ),
                (
                    "Delay from first visible effect, median (min)",
                    _num(e.median_delay_from_detectable_minutes, 1),
                ),
                (
                    "False alerts per day / per host-day",
                    f"{_num(e.false_alerts_per_day, 2)} / {_num(e.false_alerts_per_host_day, 3)}",
                ),
                ("Alerts per detected event (fragmentation)", _num(e.fragmentation, 2)),
                ("Alerts covering several events", e.merged_detections),
                ("Events started before the window (not evaluated)", e.excluded_events),
            ],
        ),
        "",
    ]
    if e.by_scenario:
        lines += [
            "Recall by scenario:",
            "",
            _table(
                ("Scenario", "Events", "Detected", "Recall"),
                [(k, v.events, v.detected, _num(v.recall)) for k, v in e.by_scenario.items()],
            ),
            "",
        ]
    if any(key != "unknown" for key in e.by_severity):
        lines += [
            "Recall by injected magnitude (event severity label, not alert severity):",
            "",
            _table(
                ("Magnitude", "Events", "Detected", "Recall"),
                [(k, v.events, v.detected, _num(v.recall)) for k, v in e.by_severity.items()],
            ),
            "",
        ]
    missed = _event_rows(outcome.matches, detected=False)
    if missed:
        lines += [
            f"Missed events (first {MAX_LISTED}):",
            "",
            _table(("Event", "Entity", "Scenario", "Magnitude", "Start (UTC)", "Minutes"), missed),
            "",
        ]
    lines += [
        "## Point level",
        "",
        "Every scored interval is one prediction at the deployed thresholds. Unscored rows "
        "(warm-up, missing data) count as not flagged; point-adjusted scores are not reported.",
        "",
        _table(
            ("Metric", "Value"),
            [
                ("Rows (scored)", f"{p.rows:,} ({p.scored:,})"),
                ("Labelled anomalous", f"{p.positives:,} ({p.positives / max(p.rows, 1):.2%})"),
                ("Precision", _num(p.precision)),
                ("Recall", _num(p.recall)),
                ("F1", _num(p.f1)),
                ("False-positive rate", _num(p.false_positive_rate, 4)),
                ("PR-AUC (average precision)", _num(p.pr_auc)),
            ],
        ),
        "",
        _table(
            ("", "Flagged", "Not flagged"),
            [
                ("Labelled anomalous", f"{p.true_positives:,}", f"{p.false_negatives:,}"),
                ("Labelled normal", f"{p.false_positives:,}", f"{p.true_negatives:,}"),
            ],
        ),
        "",
    ]
    for name, caption in (
        ("confusion_matrix", "Confusion matrix at the deployed thresholds."),
        ("pr_curve", "Precision-recall curve over all thresholds; the dot is the deployed one."),
    ):
        if name in figures:
            lines += [f"![{caption}]({figures[name]})", "", f"*{caption}*", ""]
    if alerts is not None:
        lines += ["## Alerts in the evaluated window", ""]
        if not alerts:
            lines += ["No alert fired in the evaluated window.", ""]
        else:
            by_severity = Counter(a.severity for a in alerts)
            lines += [
                f"{len(alerts)} alert(s): "
                + ", ".join(f"{by_severity[s]} {s}" for s in SEVERITY_ORDER if by_severity[s])
                + f"; {sum(matched)} match a labelled event. Severity comes from metric "
                "deviations, not from the score.",
                "",
            ]
            ranked = sorted(
                zip(alerts, matched, strict=True),
                key=lambda item: (SEVERITY_ORDER.index(item[0].severity), -item[0].score),
            )
            lines += [
                _table(
                    ("Fired (UTC)", "Entity", "Severity", "Score / threshold", "Event", "Summary"),
                    [
                        (
                            _when(a.fired_at),
                            a.entity,
                            a.severity,
                            f"{a.score:.3g} / {a.threshold:.3g}",
                            "yes" if hit else "no",
                            a.summary,
                        )
                        for a, hit in ranked[:MAX_LISTED]
                    ],
                ),
                "",
            ]
            if "feature_deviations" in figures:
                caption = (
                    "How far each metric was from its recent baseline at alert peaks, for alerts "
                    "with and without a labelled event. Deviations describe co-occurrence, not "
                    "cause."
                )
                lines += [
                    f"![Feature deviations]({figures['feature_deviations']})",
                    "",
                    f"*{caption}*",
                    "",
                ]
    if explanations is not None and explanations.events:
        lines += [
            "### Explanation quality",
            "",
            f"For the {explanations.events} detected event(s) whose ground truth lists the "
            "affected metrics, how often the first alert's metric ranking points at them "
            "(HitRate@P%: share of affected metrics among the top P% x |affected| ranked metrics).",
            "",
            _table(
                ("Measure", "Value"),
                [
                    ("HitRate@100%", _num(explanations.hit_rate_at_100)),
                    ("HitRate@150%", _num(explanations.hit_rate_at_150)),
                    ("Top-ranked metric is affected", _num(explanations.top_metric_hit_rate)),
                ],
            ),
            "",
        ]
    if "timeseries" in figures:
        entity = figures["timeseries_entity"]
        caption = (
            f"{entity} during the evaluated window: anomaly score against its threshold (orange "
            "dots are flagged points), alerts and labelled events (top strip, grey bands) and the "
            "metrics most involved in its alerts. Gaps are missing data."
        )
        lines += [
            "## Time series",
            "",
            f"![Time series]({figures['timeseries']})",
            "",
            f"*{caption}*",
            "",
        ]
    lines += [
        "## Method notes",
        "",
        "- Point metrics treat every interval separately; event metrics treat each labelled "
        "incident once and count each alert once, so many alerts on one incident cannot inflate "
        "precision.",
        "- Detection delay runs from the event start to the first moment an alert is active "
        "(alerts fire on their configured flagged point, as an operator would see them).",
        "- Thresholds were fixed before the evaluated window; nothing here was tuned on it.",
        "",
    ]
    return "\n".join(lines)


def write_evaluation_report(
    directory: Path,
    inputs: EvaluationInputs,
    outcome: EvaluationOutcome,
    *,
    input_path: Path,
    predictions_path: Path,
    alerts: Sequence[Alert] | None = None,
    plot_entity: str | None = None,
) -> dict[str, Path]:
    """Write ``metrics.json``, ``report.md`` and ``figures/*.png``; returns the written files."""
    directory.mkdir(parents=True, exist_ok=True)
    report = outcome.report
    window = (report.window_start, report.window_end)
    in_window = (
        None if alerts is None else [a for a in alerts if window[0] <= a.fired_at < window[1]]
    )
    matched = [
        alert_matches_event(a, inputs.events, inputs.frequency, report.matching)
        for a in in_window or []
    ]
    figures = _figures(directory, inputs, outcome, in_window or [], matched, plot_entity)
    explanations = None if alerts is None else explanation_hit_rates(outcome.matches, alerts)
    metrics_path = directory / "metrics.json"
    payload = report.model_dump(mode="json") | {
        "explanations": None if explanations is None else explanations.model_dump(mode="json")
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path = directory / "report.md"
    report_path.write_text(
        render_markdown(
            inputs,
            outcome,
            alerts=in_window,
            matched=matched,
            figures=figures,
            input_path=input_path,
            predictions_path=predictions_path,
            explanations=explanations,
        ),
        encoding="utf-8",
    )
    written = {"metrics": metrics_path, "report": report_path}
    written |= {
        name: directory / relative
        for name, relative in figures.items()
        if name != "timeseries_entity"
    }
    return written
