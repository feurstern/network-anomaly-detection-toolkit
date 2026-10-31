# Architecture

The toolkit is a Python package (`src/network_anomaly`) with a thin command-line interface on
top. Every command calls library functions, so everything the CLI does can also be done from
Python (see [`examples/python_api.py`](../examples/python_api.py)).

## Data flow

```
                    ┌──────────────── generate (synthetic, labelled, with a clean twin)
                    │  ┌───────────── datasets prepare (NAB, SMD, flow datasets)
                    │  │  ┌────────── flows aggregate (flow records -> window metrics)
                    │  │  │  ┌─────── convert prometheus (exported range queries)
                    ▼  ▼  ▼  ▼
CSV ──▶ ingestion: strict string read ──▶ coded validation ──▶ MetricsFrame  (+ LabelFrame, report)
                                                                   │
            preprocessing: epoch-anchored time grid, explicit gap rows, invalid-cell mask
                                                                   │
            features: FeatureBuilder.fit(train rows) / transform(all rows) ──▶ FeatureFrame
                      causal per-host z_local, z_season, roc; context; quality flags
                                                                   │
            detectors: fit(train) ──▶ score(all rows)   higher = more anomalous, NaN = unscorable
                                                                   │
            scoring: thresholds fitted on validation (or train, or a known-clean copy)
                                                                   │
            explanations: alert intervals ──▶ attribution, severity, wording, checks, warnings
                                                                   │
            reporting: detections.csv + .meta.json + .alerts.jsonl
                                                                   │
            evaluation (labelled data, test segment) ──▶ metrics.json, report.md, figures
            benchmarks (presets x seeds x detectors) ──▶ manifest, tasks.jsonl, summary, report
```

## Modules

| Path | Responsibility |
|---|---|
| `cli.py` | Typer application: parsing, progress output and exit codes only |
| `config.py` | Strict Pydantic models (`extra="forbid"`, frozen), YAML loading with key-path errors, interval parsing |
| `schemas.py` | Metric registry (unit, range, modelling space, direction, criticality), flow fields, records (`Event`, `Alert`, validation report) |
| `errors.py`, `logging_setup.py` | Error hierarchy with exit codes; text or JSON-lines logging |
| `ingestion/readers.py`, `ingestion/validation.py` | Read every cell as a string, parse explicitly, report coded issues with line numbers |
| `ingestion/events.py` | Events files (ground truth) |
| `ingestion/datasets/` | Public datasets: pinned files, verified download, converters |
| `ingestion/prometheus.py` | Prometheus range-query exports to the metrics table |
| `synthetic/` | Coupled web-service model, anomaly scenarios, presets, paired clean/anomalous generation |
| `preprocessing.py` | Regular per-entity grid; gaps stay gaps |
| `features/rolling.py`, `features/builder.py` | Grouped causal kernels; `FeatureBuilder`, `FeatureFrame`, `FeatureState` |
| `flows.py` | Window aggregation of flows into metrics; per-flow features and scoring |
| `detectors/` | `Detector` contract, statistical detectors, Isolation Forest, One-Class SVM, optional autoencoder |
| `scoring/thresholds.py`, `scoring/severity.py` | Operating points; severity rules on deviations |
| `explanations/` | Alert intervals, attribution, wording, playbook |
| `evaluation/` | Splits, point and event metrics, explanation hit rates, evaluation runner |
| `pipelines.py` | Fit, calibrate and score in one place; saved model bundles |
| `reporting/` | Detections CSV, alerts JSONL, Markdown reports, headless figures |
| `benchmarks/` | Benchmark configuration, runner, environment capture, report |

## Rules that keep results honest

* **No look-ahead.** Rolling windows use `closed="left"`: a point is never part of its own baseline,
  and every row only sees its own host's past. Tests corrupt everything after a random cut time
  and require all features before it to stay bit-identical; a builder patched to look ahead must
  fail them.
* **No identifiers or labels as features.** `FeatureFrame` refuses columns named after an entity,
  a timestamp or a label. Labels are split off at load time and only rejoin during evaluation.
* **Fitting only on training rows.** Detectors refuse frames not tagged `train`
  (`LeakageError`, exit code 6); thresholds refuse test rows; `evaluate` refuses train and
  validation rows unless `--allow-in-sample` is given, and then says the result is in-sample.
* **One set of cut times for all hosts**, aligned to the sampling grid, so a pooled model never
  learns one host's future while being evaluated on another host's past.
* **Nothing is filled in.** Missing intervals are explicit gap rows, never forward-filled or
  interpolated. Invalid cells either fail validation or, with `--on-invalid mask`, become missing
  values that are reported and that alerts warn about.
* **Scores are not probabilities.** Every output says so; thresholds are cut-offs on one
  detector's scale; severity comes from deviation rules, never from scores.

## Time grid and features

Timestamps are rounded to an epoch-anchored grid (`Timestamp.round(freq)`), so the grid of a slice
equals the grid of the whole dataset. Per host and metric, in the metric's modelling space (log1p
for volumes, latencies and counts):

| Feature | Definition |
|---|---|
| `z_local` | (x - median of the previous hour) / s, where s = 1.4826 x median of past one-step residuals over a day |
| `z_season` | (x - median of the same slot on up to 7 previous days of the same day type) / s |
| `roc` | (x - previous x) / s |

Scale floors (the larger of a registry minimum and the host's 10th-percentile training scale)
prevent infinite z-scores; z is clipped to ±50. Hosts never seen in training use pooled floors
and the global threshold, and their alerts say so.

## Saved models

`detect --save-model DIR` writes:

```
DIR/bundle.json          configuration, feature statistics, thresholds, split, score reference
DIR/detector/manifest.json   detector kind, version, parameters, feature schema, library versions
DIR/detector/state.json | model.joblib | model.pt + state.json
```

`detect --model DIR` scores new data with exactly these statistics and thresholds. Library
version differences are logged when a model is loaded. `model.joblib` is a pickle: **load only
models you created** (pickles can execute code). PyTorch weights are loaded with
`weights_only=True`.

## Rust boundary (designed, measured, not built)

The specification allows Rust only where a measured bottleneck justifies it. The candidate is the
set of grouped causal kernels in `features/rolling.py`.

**Contract.** Inputs are a contiguous row-major `float64` matrix (rows x metrics) sorted by
(entity, grid time), with `NaN` meaning missing, plus `int64` offsets where each entity's rows
begin. Each kernel returns a matrix of the same shape, `NaN` where its window is not ready. The
Python implementation and its tests (naive reference implementations, causality and corruption
tests) define the exact semantics; a Rust module would be selected only if importable, behind an
environment variable, and must pass the same tests on identical inputs and outputs.

**Measurement** (`scripts/bench_features.py`, 20 hosts x 35 days = 196,040 grid rows x 12
metrics, Python 3.12, pandas 3.1, Intel Core 7 240H):

| Step | Median s | Rows / s |
|---|---|---|
| rolling median (local baseline) | 0.37 | 536,000 |
| residual scale | 0.65 | 300,000 |
| seasonal median (7 days) | 0.70 | 281,000 |
| full feature step (fit + transform) | 3.23 | 61,000 |
| conversion to and from the contract | 0.008 | 0.5 % of kernel time |

**Conclusion.** One host reporting every 5 minutes produces 0.003 rows per second; 100,000 hosts
produce about 330. The Python step processes a 35-day history of 20 hosts in about 3 seconds and
streams far faster than any realistic fleet produces rows, so Rust is not justified today. The
contract makes it cheap to revisit (conversion costs 0.5 %), for example for multi-year backfills
over very large fleets. Re-run the script on your own hardware before deciding.
