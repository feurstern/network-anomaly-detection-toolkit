"""Benchmark outputs: raw task records, summary tables, figures and a Markdown report.

``tasks.jsonl`` keeps every number; ``summary.csv`` aggregates them per (preset, detector,
operating point, metric) as mean, sample standard deviation (ddof=1) and the number of seeds.
"""

import json
import math
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from network_anomaly import __version__
from network_anomaly.benchmarks.config import OPERATING_POINTS, BenchmarkConfig
from network_anomaly.benchmarks.environment import environment
from network_anomaly.reporting.plots import (
    detector_comparison_figure,
    operating_points_figure,
    save_figure,
)

Record = dict[str, Any]

METRICS: dict[str, tuple[tuple[str, ...], str]] = {
    "event_recall": (("test", "events", "recall"), "Event recall"),
    "event_precision": (("test", "events", "precision"), "Event precision"),
    "event_f1": (("test", "events", "f1"), "Event F1"),
    "false_alerts_per_host_day": (
        ("test", "events", "false_alerts_per_host_day"),
        "False alerts / host-day",
    ),
    "median_delay_minutes": (("test", "events", "median_delay_minutes"), "Median delay (min)"),
    "point_precision": (("test", "point", "precision"), "Point precision"),
    "point_recall": (("test", "point", "recall"), "Point recall"),
    "point_f1": (("test", "point", "f1"), "Point F1"),
    "point_fpr": (("test", "point", "false_positive_rate"), "Point FPR"),
    "pr_auc": (("test", "point", "pr_auc"), "PR-AUC"),
    "clean_false_alerts_per_host_day": (
        ("clean_test", "false_alerts_per_host_day"),
        "Clean-twin false alerts / host-day",
    ),
    "clean_point_fpr": (("clean_test", "false_positive_rate"), "Clean-twin point FPR"),
    "explanation_hit_rate_100": (
        ("test", "explanations", "hit_rate_at_100"),
        "Explanation HitRate@100%",
    ),
    "explanation_hit_rate_150": (
        ("test", "explanations", "hit_rate_at_150"),
        "Explanation HitRate@150%",
    ),
    "explanation_top_metric": (
        ("test", "explanations", "top_metric_hit_rate"),
        "Top-ranked metric affected",
    ),
}
HEADLINE_METRICS = (
    "event_recall",
    "event_precision",
    "false_alerts_per_host_day",
    "clean_false_alerts_per_host_day",
    "median_delay_minutes",
    "pr_auc",
)


def _get(record: Record, path: tuple[str, ...]) -> float | None:
    value: Any = record
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return None if value is None else float(value)


def flatten(records: Sequence[Record]) -> pd.DataFrame:
    """One row per (preset, seed, detector, operating point, metric) of successful tasks."""
    rows = [
        {
            "preset": record["preset"],
            "seed": record["seed"],
            "detector": record["detector"],
            "operating_point": point,
            "metric": metric,
            "value": _get(result, path),
        }
        for record in records
        if record["kind"] == "accuracy" and record["status"] == "ok"
        for point, result in record["operating_points"].items()
        for metric, (path, _) in METRICS.items()
    ]
    columns = ["preset", "seed", "detector", "operating_point", "metric", "value"]
    return pd.DataFrame(rows, columns=columns)


def summarise(flat: pd.DataFrame) -> pd.DataFrame:
    """Mean, sample standard deviation and seed count of every metric."""
    keys = ["preset", "detector", "operating_point", "metric"]
    grouped = flat.dropna(subset=["value"]).groupby(keys, sort=False)["value"]
    return grouped.agg(mean="mean", std=lambda v: v.std(ddof=1), n="count").reset_index()


def per_scenario(records: Sequence[Record]) -> pd.DataFrame:
    """Event recall per scenario, pooled over seeds."""
    rows = [
        {
            "preset": record["preset"],
            "seed": record["seed"],
            "detector": record["detector"],
            "operating_point": point,
            "scenario": scenario,
            "events": group["events"],
            "detected": group["detected"],
        }
        for record in records
        if record["kind"] == "accuracy" and record["status"] == "ok"
        for point, result in record["operating_points"].items()
        for scenario, group in result["test"]["events"]["by_scenario"].items()
    ]
    if not rows:
        return pd.DataFrame(
            columns=["preset", "detector", "operating_point", "scenario", "events", "detected"]
        )
    frame = pd.DataFrame(rows)
    keys = ["preset", "detector", "operating_point", "scenario"]
    pooled = frame.groupby(keys, sort=False)[["events", "detected"]].sum().reset_index()
    pooled["recall"] = pooled["detected"] / pooled["events"]
    return pooled


def timing_table(records: Sequence[Record]) -> pd.DataFrame:
    rows = []
    for record in records:
        if record["kind"] != "timing" or record["status"] != "ok":
            continue
        memory = record.get("memory", {})
        for phase, values in record["phases"].items():
            rows.append(
                {
                    "component": record["detector"],
                    "phase": phase,
                    "rows": values["rows"],
                    "median_seconds": values["median_seconds"],
                    "microseconds_per_row": values["microseconds_per_row"],
                    "rows_per_second": values["rows_per_second"],
                    "repeats": record["repeats"],
                    "peak_rss_mib": round(memory.get("peak_rss_bytes", 0) / 2**20, 1),
                    "peak_increase_mib": round(  # garbage collection can make it negative
                        max(
                            0,
                            memory.get("peak_rss_bytes", 0)
                            - (memory.get("baseline_rss_bytes") or 0),
                        )
                        / 2**20,
                        1,
                    ),
                    "memory_method": memory.get("method"),
                }
            )
    return pd.DataFrame(rows)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _mean_std(summary: pd.DataFrame, digits: int = 3) -> dict[tuple[str, str, str, str], str]:
    cells = {}
    for row in summary.to_dict("records"):
        spread = float(row["std"])
        std = "" if not math.isfinite(spread) else f" ± {spread:.{digits}f}"
        key = (str(row["preset"]), str(row["detector"]), str(row["operating_point"]))
        cells[(*key, str(row["metric"]))] = f"{float(row['mean']):.{digits}f}{std}"
    return cells


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def render_report(
    config: BenchmarkConfig,
    *,
    run_name: str,
    datasets: Sequence[Record],
    records: Sequence[Record],
    summary: pd.DataFrame,
    scenarios: pd.DataFrame,
    timing: pd.DataFrame,
    env: Record,
    figures: dict[str, str],
) -> str:
    headline = config.thresholds.headline
    detectors = [d.type for d in config.detectors]
    cells = _mean_std(summary)
    failures = [r for r in records if r["status"] != "ok"]
    seeds = ", ".join(str(seed) for seed in config.seeds)
    validation_days = config.days * config.split.validation_fraction
    lines = [
        f"# Benchmark report: {config.name}",
        "",
        f"Run `{run_name}`, network-anomaly {__version__}. Every detector saw the same datasets, "
        "chronological split, features, alerting and event-matching rules.",
        "",
        "## What these results can and cannot show",
        "",
        "- The data is **synthetic**: generated by this toolkit's own model of a web service, "
        "with anomaly types designed by its authors. Real systems have other failure modes, "
        "noisier labels and drift. **These numbers do not demonstrate production "
        "effectiveness.**",
        "- No detector is declared best. The headline operating point "
        f"(`{headline}`) was fixed in the configuration before the run, and nothing was tuned on "
        "the test segment. Rankings can change with the preset, scenario mix, feature windows "
        "and threshold strategy.",
        f"- Values are mean ± sample standard deviation over {len(config.seeds)} seed(s) "
        f"({seeds}); with so few seeds, treat differences within one standard deviation as "
        "noise.",
        "- `target_fpr` uses validation labels, which most deployments do not have; "
        "`validation_quantile` (the `detect` default) needs no labels but is raised by anomalies "
        "in the validation window; `clean_quantile` needs a window known to be normal.",
        "- Timings describe one machine (see Environment) with single-threaded workers.",
        *(
            [
                f"- The validation window is {validation_days:.1f} days, shorter than a week, so "
                "thresholds are calibrated without a full weekly cycle; expect unstable results."
            ]
            if validation_days < 7
            else []
        ),
        "",
        "## Setup",
        "",
        _table(
            ("Item", "Value"),
            [
                ("Presets", ", ".join(p.value for p in config.presets)),
                ("Seeds", seeds),
                (
                    "Datasets",
                    f"{config.hosts} hosts x {config.days:g} days at {config.freq}; "
                    f"{sum(d['rows'] for d in datasets):,} rows, "
                    f"{sum(d['events'] for d in datasets)} events in total",
                ),
                ("Scenarios", ", ".join(s.value for s in config.scenarios)),
                ("Detectors", ", ".join(detectors)),
                (
                    "Split",
                    f"train {config.split.train_fraction:.0%}, validation "
                    f"{config.split.validation_fraction:.0%}, test the rest",
                ),
                (
                    "Alerting",
                    f"fires on flagged point {config.alerting.min_points}; gaps <= "
                    f"{config.alerting.merge_gap} intervals",
                ),
                (
                    "Event matching",
                    f"-{config.matching.early_tolerance}/+{config.matching.late_tolerance} "
                    "intervals around each event",
                ),
                (
                    "Operating points",
                    f"q = {config.thresholds.quantile}, target FPR = "
                    f"{config.thresholds.target_fpr}, per-host = {config.thresholds.per_host}",
                ),
            ],
        ),
        "",
        _table(
            ("Operating point", "How the threshold is fitted"),
            [
                (f"`{p}`" + (" (headline)" if p == headline else ""), OPERATING_POINTS[p])
                for p in config.thresholds.operating_points
            ],
        ),
        "",
    ]
    for preset in config.presets:
        name = preset.value
        lines += [
            f"## Results: {name} preset, `{headline}` operating point",
            "",
            _table(
                ("Detector", *(METRICS[m][1] for m in HEADLINE_METRICS)),
                [
                    (
                        detector,
                        *(
                            cells.get((name, detector, headline, m), "n/a")
                            for m in HEADLINE_METRICS
                        ),
                    )
                    for detector in detectors
                ],
            ),
            "",
        ]
        if f"comparison_{name}" in figures:
            lines += [
                f"![Detector comparison, {name}]({figures[f'comparison_{name}']})",
                "",
                f"*Detector comparison on the {name} preset at the `{headline}` operating point.*",
                "",
            ]
        explanation_metrics = (
            "explanation_hit_rate_100",
            "explanation_hit_rate_150",
            "explanation_top_metric",
        )
        lines += [
            f"### Explanation quality ({name}, `{headline}`)",
            "",
            "For detected events, how often the first alert's metric ranking points at the "
            "metrics the generator actually changed (more than two noise standard deviations). "
            "HitRate@P% is the share of affected metrics among the top P% x |affected| ranked "
            "metrics.",
            "",
            _table(
                ("Detector", *(METRICS[m][1] for m in explanation_metrics)),
                [
                    (
                        detector,
                        *(
                            cells.get((name, detector, headline, m), "n/a")
                            for m in explanation_metrics
                        ),
                    )
                    for detector in detectors
                ],
            ),
            "",
        ]
        lines += [
            f"### All operating points ({name})",
            "",
            _table(
                (
                    "Detector",
                    "Operating point",
                    "Event recall",
                    "Event precision",
                    "False alerts / host-day",
                    "Clean-twin false alerts / host-day",
                ),
                [
                    (
                        detector,
                        point,
                        *(
                            cells.get((name, detector, point, m), "n/a")
                            for m in (
                                "event_recall",
                                "event_precision",
                                "false_alerts_per_host_day",
                                "clean_false_alerts_per_host_day",
                            )
                        ),
                    )
                    for detector in detectors
                    for point in config.operating_points_for(detector)
                ],
            ),
            "",
        ]
        if f"operating_points_{name}" in figures:
            lines += [
                f"![Operating points, {name}]({figures[f'operating_points_{name}']})",
                "",
                "*Recall against false alerts for each way of choosing the threshold.*",
                "",
            ]
        subset = scenarios[
            (scenarios["preset"] == name) & (scenarios["operating_point"] == headline)
        ]
        if not subset.empty:
            table = subset.pivot_table(
                index="scenario", columns="detector", values="recall", sort=True
            )
            events = subset.groupby("scenario")["events"].max()
            lines += [
                f"### Event recall by scenario ({name}, `{headline}`, pooled over seeds)",
                "",
                _table(
                    ("Scenario", "Events", *detectors),
                    [
                        (
                            scenario,
                            int(events[scenario]),
                            *(
                                _fmt(table.loc[scenario, d]) if d in table.columns else "n/a"
                                for d in detectors
                            ),
                        )
                        for scenario in table.index
                    ],
                ),
                "",
            ]
        quiet: dict[str, list[float]] = {}
        for record in records:
            if record["kind"] != "accuracy" or record["status"] != "ok":
                continue
            if record["preset"] != name:
                continue
            clean = record["operating_points"][headline]["clean_test"]
            shares = record["clean_test_time_share"]
            totals = quiet.setdefault(record["detector"], [0.0] * 6)
            totals[0] += clean["alerts"]
            totals[1] += clean["alerts_in_data_quality_windows"]
            totals[2] += clean["alerts_after_baseline_shifts"]
            totals[3] += shares["data_quality"]
            totals[4] += shares["after_baseline_shifts"]
            totals[5] += 1
        if quiet:

            def share(count: float, total: float) -> str:
                return f"{count:.0f} ({count / total:.0%})" if total else "0"

            lines += [
                f"### False alerts on normal-only data ({name}, `{headline}`, all seeds)",
                "",
                "The clean twin has the same noise, data-quality problems and legitimate level "
                "shifts but no anomalies, so every alert in its test period is false. Counts are "
                "followed by their share of all alerts; compare it with the share of host-time "
                "those periods cover. Local data-quality problems are telemetry gaps, metric "
                "dropouts, noise bursts and glitches.",
                "",
                _table(
                    (
                        "Detector",
                        "Alerts",
                        "During local data-quality problems",
                        "Host-time share",
                        "Within 6 h after a level shift",
                        "Host-time share",
                    ),
                    [
                        (
                            detector,
                            int(t[0]),
                            share(t[1], t[0]),
                            f"{t[3] / t[5]:.1%}",
                            share(t[2], t[0]),
                            f"{t[4] / t[5]:.1%}",
                        )
                        for detector in detectors
                        if (t := quiet.get(detector))
                    ],
                ),
                "",
            ]
    if not timing.empty:
        lines += [
            "## Runtime and memory",
            "",
            "Median of repeated runs on the first dataset, one thread per process; feature "
            "building is timed separately from detector fitting and scoring. Peak memory is the "
            "process's peak resident set size during one run of that component (fitting and "
            "scoring together for detectors); the increase is over the resident size just before "
            "that run, which already holds the libraries and the dataset.",
            "",
            _table(
                (
                    "Component",
                    "Phase",
                    "Rows",
                    "Median s",
                    "µs / row",
                    "Rows / s",
                    "Peak RSS MiB (increase)",
                ),
                [
                    (
                        row["component"],
                        row["phase"],
                        f"{int(row['rows']):,}",
                        _fmt(row["median_seconds"]),
                        _fmt(row["microseconds_per_row"], 2),
                        f"{row['rows_per_second']:,.0f}" if row["rows_per_second"] else "n/a",
                        f"{_fmt(row['peak_rss_mib'], 1)} (+{_fmt(row['peak_increase_mib'], 1)})",
                    )
                    for row in timing.to_dict("records")
                ],
            ),
            "",
        ]
    git = env.get("git") or {}
    lines += [
        "## Environment",
        "",
        _table(
            ("Item", "Value"),
            [
                ("CPU", f"{env['cpu']['model']} ({env['cpu']['logical_cores']} logical cores)"),
                ("CPU governor", env["cpu"]["governor"] or "unknown"),
                (
                    "Memory",
                    f"{env['memory_bytes'] / 2**30:.1f} GiB" if env["memory_bytes"] else "unknown",
                ),
                ("OS", f"{env['platform']['system']} {env['platform']['release']}"),
                ("Python", env["python"]["version"]),
                (
                    "Packages",
                    ", ".join(f"{k} {v}" for k, v in env["packages"].items() if v is not None),
                ),
                (
                    "Source",
                    f"git {git['commit'][:12]}{' (uncommitted changes)' if git['dirty'] else ''}"
                    if git
                    else "not a git checkout",
                ),
            ],
        ),
        "",
        "Raw results: `tasks.jsonl`; aggregates: `summary.csv`, `per_scenario.csv`, "
        "`timing.csv`; full configuration and environment: `manifest.json`.",
        "",
    ]
    if failures:
        lines += [
            "## Failed tasks",
            "",
            *(
                f"- {r['kind']} {r['detector']} {r['preset']} seed {r['seed']}: "
                f"`{r['error'].strip().splitlines()[-1]}`"
                for r in failures
            ),
            "",
        ]
    return "\n".join(lines)


def write_outputs(
    directory: Path,
    config: BenchmarkConfig,
    *,
    datasets: Sequence[Record],
    records: Sequence[Record],
    started: datetime,
    finished: datetime,
) -> dict[str, Path]:
    """Write the manifest, raw records, aggregates, figures and report into ``directory``."""
    env = environment()
    manifest = {
        "toolkit_version": __version__,
        "name": config.name,
        "config_fingerprint": config.fingerprint(),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "command": sys.argv,
        "config": config.model_dump(mode="json"),
        "operating_points": {p: OPERATING_POINTS[p] for p in config.thresholds.operating_points},
        "datasets": [{k: v for k, v in d.items() if k != "paths"} for d in datasets],
        "environment": env,
        "tasks": {"total": len(records), "failed": sum(r["status"] != "ok" for r in records)},
    }
    written = {"manifest": directory / "manifest.json"}
    written["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    written["tasks"] = directory / "tasks.jsonl"
    with written["tasks"].open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    flat = flatten(records)
    summary = summarise(flat)
    scenarios = per_scenario(records)
    timing = timing_table(records)
    for name, frame in (("summary", summary), ("per_scenario", scenarios), ("timing", timing)):
        written[name] = directory / f"{name}.csv"
        frame.to_csv(written[name], index=False, lineterminator="\n")
    detectors = [d.type for d in config.detectors]
    headline = config.thresholds.headline
    figures: dict[str, str] = {}
    for preset in config.presets:
        name = preset.value
        values = flat[(flat["preset"] == name) & (flat["operating_point"] == headline)]
        if values.empty:
            continue
        path = save_figure(
            detector_comparison_figure(
                values,
                detectors=detectors,
                title=f"Detector comparison: {name} preset, {headline} operating point "
                f"({len(config.seeds)} seeds)",
            ),
            directory / "figures" / f"comparison_{name}.png",
        )
        figures[f"comparison_{name}"] = f"figures/{path.name}"
        means = (
            flat[flat["preset"] == name]
            .pivot_table(
                index=["detector", "operating_point"],
                columns="metric",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
        )
        if {"event_recall", "false_alerts_per_host_day"} <= set(means.columns):
            path = save_figure(
                operating_points_figure(
                    means, detectors=detectors, title=f"Operating points: {name} preset"
                ),
                directory / "figures" / f"operating_points_{name}.png",
            )
            figures[f"operating_points_{name}"] = f"figures/{path.name}"
    written |= {name: directory / relative for name, relative in figures.items()}
    written["report"] = directory / "report.md"
    written["report"].write_text(
        render_report(
            config,
            run_name=directory.name,
            datasets=datasets,
            records=records,
            summary=summary,
            scenarios=scenarios,
            timing=timing,
            env=env,
            figures=figures,
        ),
        encoding="utf-8",
    )
    return written
