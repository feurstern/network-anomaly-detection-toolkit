# Network Anomaly Toolkit

Reproducible, explainable anomaly detection for infrastructure metrics and network flows - for
developers, system administrators, DevOps/SRE engineers and hosting providers who run web servers,
VPS platforms and network services.

It finds abnormal traffic, sudden request spikes, unusual combinations of CPU, memory, network,
disk and HTTP metrics, and signs of service degradation; turns them into alerts that say what
changed, against which baseline and what to check; and measures detectors honestly on synthetic
and public labelled data.

> **Anomalies are signals for investigation, not proof of an attack.** The toolkit flags behaviour
> that deviates from each host's own past. It does not classify attacks (DDoS, intrusion,
> compromise) and does not determine root causes.

**Status:** alpha (0.1.0). Python 3.12+, CPU only, no cloud services, no network access except
explicit dataset downloads.

| Question | Anomaly detection (this toolkit) | Attack classification | Root-cause analysis |
|---|---|---|---|
| Asks | Is this unusual for this host? | Which attack is this? | Why did it happen? |
| Needs | The host's own history | Labelled attack examples, signatures | Dependency maps, logs, traces, change records |
| Output | Score, alert, deviating metrics, checks | Attack class | A cause and a fix |

## Installation

Requires [uv](https://docs.astral.sh/uv/), which installs the pinned Python version itself.

```bash
uv sync                    # core install
uv sync --extra torch      # optional: CPU-only PyTorch for the autoencoder detector
uv run network-anomaly --help
```

## Quickstart

```bash
# 1. Deterministic synthetic metrics with ground truth (6 hosts, 35 days, 5-minute samples)
uv run network-anomaly generate --output data/synthetic/demo.csv --seed 42 --scenario traffic_spike

# 2. Check the file against the documented schema (exit code 3 if invalid)
uv run network-anomaly validate --input data/synthetic/demo.csv

# 3. Fit on the first 50 % of the timeline, calibrate on the next 20 %, score and explain everything
uv run network-anomaly detect --input data/synthetic/demo.csv --detector isolation_forest \
  --output reports/detections.csv

# 4. Point- and event-level metrics on the held-out last 30 %, plus a Markdown report and figures
uv run network-anomaly evaluate --input data/synthetic/demo.csv --predictions reports/detections.csv \
  --output-dir reports/evaluation
```

Step 3 prints a summary and the most severe test-period alerts:

```
Detector isolation_forest v1; training ends 2026-01-22T12:00:00+00:00, validation ends 2026-01-29T12:00:00+00:00
Threshold clean_quantile: global 0.6396, per-host for 2 host(s)
  train      rows   25,124  scored   24,890  flagged     92
  validation rows   10,024  scored   10,000  flagged     57
  test       rows   17,790  scored   17,742  flagged    100
Alerts: 33 (10 train, 9 validation, 14 test); 23 critical, 10 high
Most severe alerts in the test period:
  [critical] web-06: http_5xx_rate 15.13% vs 0.13% typical recently (+15.01 pp, z +50.0) from 2026-02-01 16:45 to 17:20 UTC; co-occurred with active_connections (5,106 connections vs 787 connections) and connection_error_rate (5.15% vs 0.13%).
  ...
Wrote reports/detections.csv, detections.meta.json and reports/detections.alerts.jsonl
```

Step 4 evaluates only the test period:

```
Point level: 17,742 of 17,790 rows scored, 2.2% labelled anomalous
  precision 0.940  recall 0.245  F1 0.388  FPR 0.0003  PR-AUC 0.699
Event level (alert after 2 flagged points, gaps <= 2; window -1/+3 intervals):
  events 21, detected 11 (recall 0.524); alerts 14, false 1 (precision 0.917); F1 0.667
  delay median 5.0 min, p90 10.0 min; false alerts 0.10/day (0.017 per host-day)
Wrote reports/evaluation/report.md, metrics.json and 4 figure(s): confusion_matrix.png, feature_deviations.png, pr_curve.png, timeseries.png
```

The same seed always produces the same file (SHA-256 `78e71b09…`) and the same results.

### An explained alert

Each line of `reports/detections.alerts.jsonl` is one alert (shortened here):

```json
{
  "alert_id": "web-06@20260201T164500Z",
  "entity": "web-06",
  "start": "2026-02-01T16:45:00Z", "fired_at": "2026-02-01T16:50:00Z", "end": "2026-02-01T17:20:00Z",
  "detector": "isolation_forest", "score": 0.757334, "threshold": 0.645279, "threshold_scope": "host",
  "reference_exceedance": 0.0,
  "severity": "critical",
  "severity_reason": "http_5xx_rate (user-facing) was at least 8 noise units from its baseline on 7 points",
  "explanation_method": "occlusion_keep_one",
  "contributions": [
    {"metric": "http_5xx_rate", "observed": 0.151329, "baseline": 0.001256, "seasonal_baseline": 0.001509,
     "robust_z": 50.0, "contribution": 0.065547},
    {"metric": "active_connections", "observed": 5106.0, "baseline": 787.499841, "robust_z": 14.071,
     "contribution": 0.059038}
  ],
  "summary": "web-06: http_5xx_rate 15.13% vs 0.13% typical recently (+15.01 pp, z +50.0) from 2026-02-01 16:45 to 17:20 UTC; co-occurred with active_connections (5,106 connections vs 787 connections) and connection_error_rate (5.15% vs 0.13%).",
  "suggested_checks": [
    "Group application and web-server logs for the interval by endpoint and status code.",
    "Check recent releases and configuration changes, and the health of dependencies (databases, caches, upstream APIs).",
    "..."
  ],
  "data_quality": []
}
```

How to read it: `score` is unitless (higher = more anomalous) and is **not a probability**;
`reference_exceedance` is the share of validation-period scores at least as high (an empirical
frequency); `severity` comes from how far metrics moved, never from the score; `contributions`
rank metrics for investigation and do not claim a cause; `data_quality` lists gaps, masked values,
missing baselines or unseen hosts that limit trust in the alert. Field-by-field reference:
[docs/data_schema.md](docs/data_schema.md#alerts-detectionsalertsjsonl).

### Going further

```bash
# Calibrate thresholds on a known-clean copy of the validation period (here: the clean twin)
uv run network-anomaly generate --output data/synthetic/demo.csv --seed 42 --scenario traffic_spike --with-clean-twin
uv run network-anomaly detect --input data/synthetic/demo.csv --detector robust_zscore \
  --clean-validation data/synthetic/demo.clean.csv --output reports/robust.csv
#   -> all 21 test-period spikes detected at 0.20 false alerts per host-day

# Save a fitted model and score new data with it
uv run network-anomaly detect --input data/synthetic/demo.csv --config configs/baseline.yaml \
  --output reports/fit.csv --save-model models/demo
uv run network-anomaly detect --input data/synthetic/demo.csv --model models/demo --output reports/scored.csv

# Compare all detectors on identical datasets (about 2.5 minutes; --skip-timing for quality only)
uv run network-anomaly benchmark --config configs/benchmark.yaml --output reports/benchmark/

# Public datasets: download pinned files (checksums verified) and convert them
uv run network-anomaly datasets list
uv run network-anomaly datasets download nab && uv run network-anomaly datasets prepare nab

# Network flows: aggregate per window into metrics, or score individual flows
uv run network-anomaly flows aggregate --input flows.csv --window 5min --output flow-metrics.csv
uv run network-anomaly flows detect --train normal-day.flows.csv --input today.flows.csv

# Prometheus: convert exported range queries (no connection is made)
uv run network-anomaly convert prometheus --metric requests_per_second=rps.json \
  --metric cpu_usage_percent=cpu.json --output metrics.csv
```

## Commands

| Command | What it does |
|---|---|
| `schema metrics\|flows` | Print the documented input schema: columns, units, valid ranges |
| `generate` | Deterministic synthetic datasets with labels, events, data-quality events, metadata and an optional clean twin; 11 scenarios, 3 presets, `--n-datasets` |
| `validate` | Strict validation with coded, line-numbered errors and fix hints; `--on-invalid mask` blanks invalid cells instead |
| `detect` | Fit on the training period, calibrate thresholds, score every row; write detections, provenance and explained alerts; `--save-model` / `--model` |
| `evaluate` | Point and event metrics on the test period (in-sample only with `--allow-in-sample`); `--output-dir` writes `report.md`, `metrics.json` and figures |
| `benchmark` | All detectors x presets x seeds on identical data; quality, false alerts on normal-only data, runtime and memory; `report.md`, CSV and JSON outputs |
| `datasets list\|download\|prepare` | NAB, SMD, CICIDS2017 (improved) and CIDDS-001: pinned downloads, verified checksums, conversion |
| `flows aggregate\|detect` | Flow records to window metrics; per-flow Isolation Forest scoring |
| `convert prometheus` | Prometheus `query_range` exports to the metrics schema |

Every command has `--help`. Exit codes: `0` ok · `1` internal error · `2` usage · `3` invalid data ·
`4` invalid configuration · `5` file, model or download problem · `6` leakage guard.

## Detectors

| Detector | Idea | Explanation |
|---|---|---|
| `static_threshold` | Fixed per-metric limits (classic alert rules); score = largest value / limit | Value / limit per metric |
| `robust_zscore` | Deviation from each host's rolling median in robust noise units (rolling MAD) | Exact per-metric terms |
| `isolation_forest` | One pooled model on host-normalised deviations | Grouped occlusion (drop-one, keep-one) |
| `one_class_svm` | RBF One-Class SVM on scaled deviations (trained on a capped, host-stratified sample) | Grouped occlusion |
| `autoencoder` (optional) | MLP reconstruction error; early stopping, saved history and best checkpoint | Exact per-metric reconstruction error |

Features are causal: each row is compared only with its own host's past (recent median, same time
on previous days, rate of change). Thresholds: `fixed`, `train_quantile`, `clean_quantile`
(default; optionally on a known-clean file) and `target_fpr` (needs labels), per host where there is
enough data. Example configurations are in [`configs/`](configs/).

## Benchmark results

Default configuration (moderate preset, 5 seeds, 20 hosts x 35 days, all scenarios, thresholds at a
0.5 % false-positive rate on validation; mean ± standard deviation over seeds):

| Detector | Event recall | Event precision | False alerts / host-day | PR-AUC | Explanation HitRate@100% |
|---|---|---|---|---|---|
| static_threshold | 0.39 ± 0.12 | 0.44 ± 0.18 | 0.14 ± 0.07 | 0.17 ± 0.08 | 0.63 ± 0.02 |
| robust_zscore | 0.89 ± 0.07 | 0.47 ± 0.06 | 0.23 ± 0.04 | 0.32 ± 0.10 | 0.94 ± 0.03 |
| isolation_forest | 0.42 ± 0.05 | 0.27 ± 0.06 | 0.26 ± 0.04 | 0.22 ± 0.06 | 0.91 ± 0.05 |
| one_class_svm | 0.57 ± 0.13 | 0.40 ± 0.07 | 0.20 ± 0.04 | 0.28 ± 0.13 | 0.93 ± 0.04 |

These numbers come from synthetic data generated by this project. **They do not demonstrate
production effectiveness**, no detector is declared best, and nothing was tuned on the test
period. Methodology, more results and what they cannot show: [docs/evaluation.md](docs/evaluation.md).

## Documentation

| Document | Contents |
|---|---|
| [architecture.md](docs/architecture.md) | Data flow, modules, leakage rules, saved models, the measured Rust boundary |
| [data_schema.md](docs/data_schema.md) | Input schemas, events, detections, alerts, validation codes, exit codes |
| [evaluation.md](docs/evaluation.md) | Splits, thresholds and their limits, point and event metrics, benchmark protocol and results |
| [datasets.md](docs/datasets.md) | The synthetic generator; NAB, SMD, CICIDS2017, CIDDS-001: licences, conversion, caveats |
| [deployment.md](docs/deployment.md) | Scheduled scoring with saved models, recalibration, sizing, security |
| [extending.md](docs/extending.md) | Adding a detector, a dataset adapter or a metric; integrating Rust kernels |
| [data_governance.md](docs/data_governance.md) | Sensitive data, pseudonymisation, retention, access control |

A Python API example is in [`examples/python_api.py`](examples/python_api.py).

## Data policy

Datasets, generated reports and trained models are never committed (see
[data/README.md](data/README.md)). Nothing is sent to external services. Flow addresses are
pseudonymised by default when public datasets are prepared.

## Development

```bash
uv sync --extra torch
uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License and author

Apache License 2.0 - see [LICENSE](LICENSE) and [NOTICE](NOTICE). Public datasets keep their own
licences and are not redistributed. Developed by [feurstern](https://github.com/feurstern).
