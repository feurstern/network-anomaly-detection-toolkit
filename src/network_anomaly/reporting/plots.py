"""Static report figures, rendered headless.

Figures use Matplotlib's object-oriented API on the Agg canvas - never ``pyplot`` - so they work on
servers without a display and leave no global state behind. One small palette is used throughout:
data in blue, detections in orange (the pair passes colour-vision-deficiency checks), labelled
ground truth as a neutral grey wash; text stays in ink colours. Gaps in the data stay gaps: lines
are broken, never interpolated across missing intervals.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import EngFormatter, FuncFormatter, PercentFormatter
from matplotlib.typing import RcKeyType
from sklearn.metrics import precision_recall_curve

from network_anomaly.evaluation.point import PointMetrics, ranking_scores
from network_anomaly.schemas import Alert, metric_spec

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Interval = tuple[pd.Timestamp, pd.Timestamp]

DPI = 150
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DATA = "#2a78d6"
DETECTION = "#eb6834"
TRUTH = "#898781"
TRUTH_ALPHA = 0.18
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SCORE_NOTE = "Scores are unitless (higher = more anomalous) and are not probabilities."

STYLE: dict[RcKeyType, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 8,
}


def _figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), dpi=DPI, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def save_figure(figure: Figure, path: Path) -> Path:
    """Write ``figure`` as PNG (no timestamp in the metadata, so reruns give identical files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, facecolor=SURFACE, metadata={"Software": None})
    return path


def _naive_utc(values: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    return index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index


def _unit_label(unit: str) -> str:
    return "%" if unit in ("fraction", "%") else unit


def _value_formatter(unit: str) -> FuncFormatter | PercentFormatter | EngFormatter:
    if unit == "fraction":
        return PercentFormatter(xmax=1.0, decimals=1)
    if unit == "%":
        return PercentFormatter(xmax=100.0, decimals=0)
    if unit == "bytes/s":
        return EngFormatter(unit="B/s", places=0)
    return FuncFormatter(lambda value, _: f"{value:,.4g}")


def _shade(axes: Axes, intervals: Sequence[Interval]) -> None:
    for start, end in intervals:
        left, right = mdates.date2num(_naive_utc(pd.DatetimeIndex([start, end])))
        axes.axvspan(float(left), float(right), color=TRUTH, alpha=TRUTH_ALPHA, linewidth=0)


def _bars(axes: Axes, intervals: Sequence[Interval], row: float, color: str) -> None:
    ranges = []
    for start, end in intervals:
        left, right = mdates.date2num(_naive_utc(pd.DatetimeIndex([start, end])))
        ranges.append((left, max(right - left, 1e-4)))
    if ranges:
        axes.broken_barh(ranges, (row - 0.32, 0.64), facecolors=color, linewidth=0)


def timeseries_figure(
    *,
    entity: str,
    detector: str,
    times: pd.Series,
    scores: FloatArray,
    thresholds: FloatArray,
    flagged: BoolArray,
    metrics: Mapping[str, FloatArray],
    frequency: pd.Timedelta,
    window: Interval,
    events: Sequence[Interval],
    alerts: Sequence[Interval],
) -> Figure:
    """Score, threshold and selected metrics for one entity, with events and alerts marked."""
    grid = pd.date_range(window[0], window[1] - frequency, freq=frequency)
    frame = pd.DataFrame(
        {"score": scores, "threshold": thresholds, "flagged": flagged, **metrics},
        index=pd.DatetimeIndex(times),
    ).reindex(grid)  # missing intervals become NaN, so lines break at gaps
    x = _naive_utc(grid)
    with mpl.rc_context(STYLE):
        heights = [0.55, 1.8, *([1.15] * len(metrics))]
        figure = _figure(11.0, 1.1 + sum(heights))
        axes = figure.subplots(len(heights), 1, sharex=True, gridspec_kw={"height_ratios": heights})
        strip, score_axes, *metric_axes = axes
        _bars(strip, events, 1.0, TRUTH)
        _bars(strip, alerts, 0.0, DETECTION)
        strip.set_yticks([0.0, 1.0], ["alerts", "labelled events"])
        strip.set_ylim(-0.6, 1.6)
        strip.grid(False)
        strip.tick_params(axis="y", length=0)
        for side in ("left", "bottom"):
            strip.spines[side].set_visible(False)

        score_axes.plot(x, frame["score"], color=DATA)
        score_axes.plot(x, frame["threshold"], color=INK_SECONDARY, linewidth=0.8)
        marks = frame["flagged"].eq(True).to_numpy()
        score_axes.scatter(
            x[marks],
            frame["score"].to_numpy()[marks],
            s=18,
            color=DETECTION,
            edgecolors=SURFACE,
            linewidths=0.8,
            zorder=3,
        )
        last = frame["threshold"].dropna()
        if not last.empty:
            score_axes.annotate(
                "threshold",
                xy=(1.0, float(last.iloc[-1])),
                xycoords=("axes fraction", "data"),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=INK_SECONDARY,
                fontsize=8,
                annotation_clip=False,
            )
        score_axes.set_title("anomaly score")
        for name, metric_axes_item in zip(metrics, metric_axes, strict=True):
            unit = metric_spec(name).unit
            metric_axes_item.plot(x, frame[name], color=DATA)
            metric_axes_item.set_title(
                f"{name} ({_unit_label(unit)})", fontsize=9, color=INK_SECONDARY
            )
            metric_axes_item.yaxis.set_major_formatter(_value_formatter(unit))
        for panel in (score_axes, *metric_axes):
            _shade(panel, events)
        bottom = axes[-1]
        locator = mdates.AutoDateLocator()
        bottom.xaxis.set_major_locator(locator)
        bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        bottom.set_xlim(x[0], x[-1] + frequency)
        strip.legend(
            handles=[
                Line2D([], [], color=DATA, label="anomaly score / metric"),
                Line2D([], [], color=INK_SECONDARY, linewidth=0.8, label="threshold"),
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=5,
                    markerfacecolor=DETECTION,
                    markeredgecolor=SURFACE,
                    label="flagged point",
                ),
                Patch(color=DETECTION, label="alert (fired to end)"),
                Patch(color=TRUTH, alpha=0.5, label="labelled anomaly event"),
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.05),
            ncols=5,
            borderaxespad=0.0,
        )
        start, end = window
        figure.suptitle(
            f"{entity}: {detector} detections, {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC",
            x=0.01,
            ha="left",
            fontsize=11,
        )
        figure.supxlabel(f"Time (UTC). {SCORE_NOTE}", fontsize=8, color=INK_SECONDARY)
    return figure


def pr_curve_figure(
    labels: npt.NDArray[np.int8],
    scores: FloatArray,
    *,
    point: PointMetrics,
    detector: str,
) -> Figure | None:
    """Point-level precision-recall curve with the deployed operating point; None if undefined."""
    if point.pr_auc is None:
        return None
    precision, recall, _ = precision_recall_curve(labels.astype(bool), ranking_scores(scores))
    prevalence = point.positives / point.rows
    with mpl.rc_context(STYLE):
        figure = _figure(6.0, 4.8)
        axes = figure.subplots()
        axes.plot(recall, precision, color=DATA, drawstyle="steps-post", linewidth=1.4)
        axes.axhline(prevalence, color=INK_MUTED, linewidth=0.8)
        axes.annotate(
            f"random ranking ({prevalence:.1%} anomalous)",
            xy=(0.0, prevalence),
            xytext=(4, 4),
            textcoords="offset points",
            ha="left",
            color=INK_SECONDARY,
            fontsize=8,
        )
        if point.precision is not None and point.recall is not None:
            axes.scatter(
                [point.recall],
                [point.precision],
                s=60,
                color=DETECTION,
                edgecolors=SURFACE,
                linewidths=1.2,
                zorder=3,
            )
            axes.annotate(
                f"deployed threshold\nprecision {point.precision:.2f}, recall {point.recall:.2f}",
                xy=(point.recall, point.precision),
                xytext=(12, 12),
                textcoords="offset points",
                va="bottom",
                color=INK,
                fontsize=8,
            )
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.03)
        axes.set_xlabel("recall: share of anomalous points that were flagged")
        axes.set_ylabel("precision: share of flagged points that are anomalous")
        axes.set_title(
            f"Precision-recall, point level ({detector}); average precision {point.pr_auc:.3f}"
        )
    return figure


def confusion_matrix_figure(point: PointMetrics, *, detector: str) -> Figure:
    """2x2 confusion matrix at the deployed thresholds; colour is the share of each true class."""
    counts = np.array(
        [
            [point.true_negatives, point.false_positives],
            [point.false_negatives, point.true_positives],
        ]
    )
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)
    colormap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL)
    with mpl.rc_context(STYLE):
        figure = _figure(5.6, 4.4)
        axes = figure.subplots()
        axes.imshow(shares, cmap=colormap, vmin=0.0, vmax=1.0)
        for (row, column), count in np.ndenumerate(counts):
            share = shares[row, column]
            axes.text(
                column,
                row,
                f"{count:,}\n{share:.1%} of row",
                ha="center",
                va="center",
                fontsize=10,
                color="#ffffff" if share > 0.55 else INK,
            )
        axes.set_xticks([0, 1], ["not flagged", "flagged"])
        axes.set_yticks([0, 1], ["labelled normal", "labelled anomalous"])
        axes.set_xticks([0.5], minor=True)
        axes.set_yticks([0.5], minor=True)
        axes.grid(False)
        axes.grid(which="minor", color=SURFACE, linewidth=3)
        axes.tick_params(which="both", length=0)
        for spine in axes.spines.values():
            spine.set_visible(False)
        axes.set_xlabel("decision at the deployed threshold")
        axes.set_title(f"Confusion matrix, point level ({detector})")
    return figure


def feature_deviation_figure(
    alerts: Sequence[Alert], matched: Sequence[bool], *, detector: str
) -> Figure | None:
    """Median |robust z| of every metric at alert peaks, for alerts with and without an event."""
    if not alerts:
        return None
    groups = {
        True: [a for a, hit in zip(alerts, matched, strict=True) if hit],
        False: [a for a, hit in zip(alerts, matched, strict=True) if not hit],
    }
    metrics = sorted({metric for alert in alerts for metric in alert.peak_deviations})

    def summary(selected: Sequence[Alert], metric: str) -> tuple[float, float, float] | None:
        values = [abs(z) for a in selected if (z := a.peak_deviations.get(metric)) is not None]
        if not values:
            return None
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        return float(q1), float(median), float(q3)

    def order_key(metric: str) -> float:
        stats = summary(groups[True] or groups[False], metric)
        return -stats[1] if stats else 0.0

    metrics.sort(key=order_key)
    styles = {
        True: (DATA, f"alerts matching a labelled event (n={len(groups[True])})", -0.17),
        False: (DETECTION, f"alerts without a labelled event (n={len(groups[False])})", 0.17),
    }
    with mpl.rc_context(STYLE):
        figure = _figure(8.0, 1.4 + 0.36 * len(metrics))
        axes = figure.subplots()
        handles = []
        for key, (color, label, offset) in styles.items():
            if not groups[key]:
                continue
            for position, metric in enumerate(metrics):
                stats = summary(groups[key], metric)
                if stats is None:
                    continue
                q1, median, q3 = stats
                axes.hlines(position + offset, q1, q3, color=color, linewidth=1.6)
                axes.scatter(
                    [median],
                    [position + offset],
                    s=36,
                    color=color,
                    edgecolors=SURFACE,
                    linewidths=1.0,
                    zorder=3,
                )
            handles.append(
                Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    markeredgecolor=SURFACE,
                    label=label,
                )
            )
        axes.set_yticks(range(len(metrics)), metrics)
        axes.invert_yaxis()
        axes.set_xlim(left=0.0)
        axes.grid(axis="y", visible=False)
        axes.tick_params(axis="y", length=0)
        axes.set_xlabel(
            "|robust z| at the alert peak, in noise units (dot = median, line = middle 50%)"
        )
        axes.set_title(f"Feature deviations at alert peaks ({detector})")
        axes.legend(handles=handles, loc="lower right")
    return figure


BENCHMARK_PANELS: tuple[tuple[str, str, bool], ...] = (
    ("event_recall", "event recall", True),
    ("event_precision", "event precision", True),
    ("false_alerts_per_host_day", "false alerts per host-day\n(test period)", False),
    (
        "clean_false_alerts_per_host_day",
        "false alerts per host-day\n(clean twin, test period)",
        False,
    ),
    ("pr_auc", "PR-AUC (point level)", True),
)


def detector_comparison_figure(
    values: pd.DataFrame, *, detectors: Sequence[str], title: str
) -> Figure:
    """Per-seed results (dots), mean (large dot) and +-1 standard deviation for each detector.

    ``values`` has columns ``detector``, ``metric``, ``value`` with one row per seed.
    """
    with mpl.rc_context(STYLE):
        figure = _figure(2.6 * len(BENCHMARK_PANELS), 1.5 + 0.45 * len(detectors))
        axes = figure.subplots(1, len(BENCHMARK_PANELS), sharey=True)
        positions = {name: index for index, name in enumerate(detectors)}
        for panel, (metric, label, bounded) in zip(axes, BENCHMARK_PANELS, strict=True):
            subset = values[values["metric"] == metric].dropna(subset=["value"])
            for detector, rows in subset.groupby("detector", sort=False):
                y = positions[str(detector)]
                points = rows["value"].to_numpy(dtype=np.float64)
                panel.scatter(points, np.full(len(points), y), s=14, color=DATA, alpha=0.35)
                mean = float(points.mean())
                if len(points) > 1:
                    spread = float(points.std(ddof=1))
                    panel.hlines(y, mean - spread, mean + spread, color=DATA, linewidth=1.6)
                panel.scatter(
                    [mean], [y], s=44, color=DATA, edgecolors=SURFACE, linewidths=1.0, zorder=3
                )
            panel.set_title(label, fontsize=9)
            panel.grid(axis="y", visible=False)
            panel.tick_params(axis="y", length=0)
            if bounded:
                panel.set_xlim(-0.03, 1.03)
            else:
                panel.set_xlim(left=0.0)
        axes[0].set_yticks(range(len(detectors)), list(detectors))
        axes[0].set_ylim(len(detectors) - 0.5, -0.5)
        figure.suptitle(title, x=0.01, ha="left", fontsize=11)
        figure.supxlabel(
            "Small dots: one seed each; large dot: mean; line: mean ± 1 standard deviation.",
            fontsize=8,
            color=INK_SECONDARY,
        )
    return figure


# One marker shape per operating point: identity without colour or colliding text labels.
OPERATING_POINT_MARKERS: dict[str, str] = {
    "train_quantile": "s",
    "validation_quantile": "^",
    "clean_quantile": "D",
    "target_fpr": "o",
    "fixed": "P",
}


def operating_points_figure(means: pd.DataFrame, *, detectors: Sequence[str], title: str) -> Figure:
    """Mean event recall against mean false alerts per host-day for every operating point.

    ``means`` has columns ``detector``, ``operating_point``, ``event_recall`` and
    ``false_alerts_per_host_day``. Operating points are told apart by marker shape (one legend
    for all panels); each panel has its own x range because detectors differ widely.
    """
    columns = min(len(detectors), 2)
    rows = math.ceil(len(detectors) / columns)
    with mpl.rc_context(STYLE):
        figure = _figure(5.0 * columns, 1.2 + 3.2 * rows)
        grid = np.atleast_1d(figure.subplots(rows, columns, sharey=True, squeeze=False))
        panels = list(grid.ravel())
        seen: dict[str, Line2D] = {}
        for panel, detector in zip(panels, detectors, strict=False):
            subset = means[means["detector"] == detector]
            for row in subset.to_dict("records"):
                point = str(row["operating_point"])
                marker = OPERATING_POINT_MARKERS.get(point, "o")
                panel.scatter(
                    [float(row["false_alerts_per_host_day"])],
                    [float(row["event_recall"])],
                    s=52,
                    marker=marker,
                    color=DATA,
                    edgecolors=SURFACE,
                    linewidths=1.0,
                    zorder=3,
                )
                seen.setdefault(
                    point,
                    Line2D(
                        [],
                        [],
                        marker=marker,
                        linestyle="",
                        markersize=7,
                        markerfacecolor=DATA,
                        markeredgecolor=SURFACE,
                        label=point,
                    ),
                )
            panel.set_title(detector, fontsize=9)
            panel.set_ylim(-0.03, 1.05)
            panel.set_xlim(left=0.0)
            panel.margins(x=0.1)
        for panel in panels[len(detectors) :]:
            panel.set_visible(False)
        ordered = [seen[p] for p in OPERATING_POINT_MARKERS if p in seen]
        # The legend doubles as the title: a separate suptitle would collide with it.
        legend = figure.legend(
            handles=ordered,
            loc="outside upper center",
            ncols=len(ordered),
            title=title,
            alignment="left",
        )
        legend.get_title().set_fontsize(11)
        figure.supxlabel(
            "mean false alerts per host-day (test period; each panel has its own scale)",
            fontsize=9,
        )
        figure.supylabel("mean event recall (test period)", fontsize=9)
    return figure
