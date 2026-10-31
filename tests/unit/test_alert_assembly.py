"""Explainable alerts: consistency with detections, wording, severity and data-quality warnings."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from network_anomaly.explanations.alerts import (
    assemble_alerts,
    describe_metric,
    format_value,
    summarise,
)
from network_anomaly.explanations.playbook import CONFIRMATION_STEP, PLAYBOOK, suggested_checks
from network_anomaly.ingestion.validation import MetricsFrame, ValidatedMetrics, load_metrics
from network_anomaly.pipelines import DetectionConfig, DetectionResult, run_detection
from network_anomaly.reporting.export import read_alerts, write_alerts
from network_anomaly.schemas import Alert, MetricContribution, MetricFamily, ValidationReport
from network_anomaly.synthetic.generator import GeneratorConfig, generate_dataset, write_dataset

FIVE_MINUTES = pd.Timedelta("5min")
# Explanations describe co-occurrence; they must never assert a cause or name an attack.
FORBIDDEN = re.compile(
    r"\b(caus(e|ed|es|ing)|because|root cause|due to|attack\w*|intrusion|ddos|compromis\w*|"
    r"malicious|breach\w*)\b",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def validated(tmp_path_factory: pytest.TempPathFactory) -> ValidatedMetrics:
    path = tmp_path_factory.mktemp("data") / "demo.csv"
    write_dataset(generate_dataset(GeneratorConfig(seed=21, hosts=3, days=21)), path)
    return load_metrics(path)


def _detect(validated: ValidatedMetrics, detector: str, **fields: Any) -> DetectionResult:
    config = DetectionConfig.model_validate({"detector": {"type": detector}, **fields})
    return run_detection(validated, config, FIVE_MINUTES)


@pytest.fixture(scope="module", params=["robust_zscore", "isolation_forest"])
def result(request: pytest.FixtureRequest, validated: ValidatedMetrics) -> DetectionResult:
    return _detect(validated, request.param)


def _texts(alert: Alert) -> list[str]:
    return [
        alert.summary,
        alert.severity_reason,
        *alert.suggested_checks,
        *(warning.message for warning in alert.data_quality),
    ]


def test_alerts_are_consistent_with_the_detections(result: DetectionResult) -> None:
    alerts = result.alerts()
    assert alerts
    metrics = set(result.features.metrics)
    for alert in alerts:
        assert alert.start <= alert.fired_at < alert.end
        assert alert.start <= alert.peak_at < alert.end
        assert alert.score >= alert.threshold
        assert alert.detector == result.bundle.detector.name
        assert alert.segment in ("train", "validation", "test")
        assert alert.reference_exceedance is not None
        assert 0.0 <= alert.reference_exceedance <= 1.0
        assert set(alert.peak_deviations) == metrics
        assert 1 <= len(alert.contributions) <= 3
        values = [c.contribution for c in alert.contributions]
        assert values == sorted(values, reverse=True)
        assert alert.suggested_checks[-1] == CONFIRMATION_STEP


def test_alert_texts_never_assert_causes_or_attacks(result: DetectionResult) -> None:
    texts = [text for alert in result.alerts() for text in _texts(alert)]
    texts += [step for steps in PLAYBOOK.values() for step in steps]
    offending = [text for text in texts if FORBIDDEN.search(text)]
    assert offending == []


def test_rarer_scores_have_smaller_reference_exceedance(result: DetectionResult) -> None:
    alerts = sorted(result.alerts(), key=lambda a: a.score)
    exceedances = [a.reference_exceedance for a in alerts]
    assert exceedances == sorted(exceedances, reverse=True)


def test_severity_ignores_the_score_scale(result: DetectionResult) -> None:
    bundle = result.bundle

    def severities(scores: np.ndarray) -> list[str]:
        alerts = assemble_alerts(
            result.features,
            scores,
            result.flagged,
            result.segments,
            detector=bundle.detector,
            thresholds=bundle.thresholds,
            feature_medians=bundle.features.feature_medians,
            frequency=FIVE_MINUTES,
            config=bundle.config.alerting,
        )
        return [a.severity for a in alerts]

    assert severities(result.scores) == severities(result.scores * 10.0)


def test_in_sample_alerts_are_marked(result: DetectionResult) -> None:
    for alert in result.alerts():
        codes = {warning.code for warning in alert.data_quality}
        assert ("in_sample" in codes) == (alert.segment in ("train", "validation"))


def test_alerts_round_trip_through_jsonl(result: DetectionResult, tmp_path: Path) -> None:
    alerts = result.alerts()
    path = write_alerts(alerts, tmp_path / "alerts.jsonl")
    assert len(path.read_text().splitlines()) == len(alerts)
    assert read_alerts(path) == alerts


def test_unseen_hosts_gaps_masks_and_missing_metrics_are_reported(
    make_metrics: Callable[..., MetricsFrame],
) -> None:
    base = make_metrics(seed=8, days=4, gap_fraction=0.0, jitter_seconds=0)
    data = base.data
    newcomer = data[(data["host_id"] == "h0") & (data["timestamp"] >= "2026-01-08T00:00:00Z")]
    newcomer = newcomer.assign(host_id="h9").reset_index(drop=True)
    spike = (newcomer["timestamp"] >= "2026-01-08T12:00:00Z") & (
        newcomer["timestamp"] < "2026-01-08T12:30:00Z"
    )
    newcomer.loc[spike, "cpu_usage_percent"] = 99.0
    newcomer.loc[spike, "http_5xx_rate"] = 0.4
    newcomer.loc[spike, "requests_per_second"] = np.nan  # masked as invalid during validation
    newcomer = newcomer.drop(index=newcomer.index[spike][2])  # a telemetry gap inside the spike
    combined = pd.concat([data, newcomer], ignore_index=True)
    combined = combined.sort_values(["host_id", "timestamp"]).reset_index(drop=True)
    masked = combined["requests_per_second"].isna()
    mask = pd.DataFrame(
        {m: masked.to_numpy() if m == "requests_per_second" else False for m in base.metrics}
    )
    frame = MetricsFrame(combined, "host_id", base.metrics, invalid_mask=mask)
    report = ValidationReport(
        source="test", schema_name="metrics", rows=len(combined), columns=tuple(combined.columns)
    )
    result = _detect(ValidatedMetrics(frame, None, report), "robust_zscore")
    (alert,) = [a for a in result.alerts() if a.entity == "h9"]
    codes = {warning.code: warning for warning in alert.data_quality}
    assert {"unseen_entity", "telemetry_gap", "masked_cells", "missing_metrics"} <= set(codes)
    assert codes["missing_metrics"].metrics == ("requests_per_second",)
    assert alert.threshold_scope == "global"
    assert alert.severity == "critical"  # 5xx rate is a user-facing metric


def _item(
    metric: str,
    observed: float,
    baseline: float,
    same_time: float,
    *,
    robust: float,
    seasonal: float | None = None,
) -> MetricContribution:
    return MetricContribution(
        metric=metric,
        family=MetricFamily.ERRORS,
        unit="fraction",
        observed=observed,
        baseline=baseline,
        seasonal_baseline=same_time,
        deviation_percent=None,
        robust_z=robust,
        seasonal_z=seasonal,
        contribution=1.0,
    )


def test_values_are_formatted_in_their_units() -> None:
    assert format_value(0.0412, "fraction") == "4.12%"
    assert format_value(87.25, "%") == "87.2%"
    assert format_value(1234.4, "ms") == "1,234 ms"
    assert format_value(2_500_000, "bytes/s") == "2.5 MB/s"
    assert format_value(None, "ms") == "n/a"


def test_a_long_incident_is_described_against_the_same_time_baseline() -> None:
    # An hour into a 5xx surge the recent baseline has absorbed it: against the last hour the
    # rate looks like a drop, against the same time on previous days it is a large rise.
    item = _item("http_5xx_rate", 0.056, 0.153, 0.0014, robust=-50.0, seasonal=50.0)
    text = describe_metric(item)
    assert "vs 0.14% typical at this time of day" in text
    assert "+5.46 pp" in text


def test_only_metrics_that_deviated_are_mentioned_as_co_occurring() -> None:
    lead = _item("http_5xx_rate", 0.16, 0.0014, 0.0014, robust=40.0, seasonal=40.0)
    moved = _item("connection_error_rate", 0.12, 0.0016, 0.0016, robust=30.0)
    flat = _item("http_4xx_rate", 0.02, 0.02, 0.02, robust=0.4)
    start = pd.Timestamp("2026-01-30T14:25:00Z")
    text = summarise("web-02", start, start + pd.Timedelta("1h"), [lead, moved, flat])
    assert text.startswith("web-02: http_5xx_rate 16.00% vs 0.14%")
    assert "from 2026-01-30 14:25 to 15:25 UTC" in text
    assert "co-occurred with connection_error_rate" in text
    assert "http_4xx_rate" not in text


def test_suggested_checks_cover_each_family_once_and_end_with_confirmation() -> None:
    checks = suggested_checks(["requests_per_second", "network_rx_bytes_per_second", "rps_2"])
    assert checks[:2] == PLAYBOOK[MetricFamily.TRAFFIC][:2]
    assert checks[2] == PLAYBOOK[MetricFamily.NETWORK][0]
    assert checks[3] == PLAYBOOK[MetricFamily.GENERIC][0]
    assert checks[-1] == CONFIRMATION_STEP
    assert len(set(checks)) == len(checks)
