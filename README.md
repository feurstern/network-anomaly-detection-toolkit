# Network Anomaly Toolkit

Reproducible, explainable anomaly detection for infrastructure metrics and network flows — for
developers, system administrators, DevOps/SRE engineers and hosting providers.

> **Status: alpha, under staged development.** This README documents only the commands that exist
> today: the vertical slice *generate → validate → detect → evaluate*. Explainable alert export,
> plots, benchmarks and full documentation are being added stage by stage.

**Anomalies are signals for investigation, not proof of an attack.** The toolkit flags behaviour
that deviates from a host's own recent and seasonal baseline. It does not classify attacks (DDoS,
intrusion, compromise) and does not perform root-cause analysis.

## Installation

Requires [uv](https://docs.astral.sh/uv/); uv installs the pinned Python (3.12) for you.

```bash
uv sync
uv run network-anomaly --help
```

## Quickstart

```bash
# 1. Deterministic synthetic metrics with ground-truth labels (6 hosts, 35 days, 5-minute samples)
uv run network-anomaly generate --output data/synthetic/demo.csv --seed 42 --scenario traffic_spike

# 2. Check the file against the documented schema (exit code 3 if invalid)
uv run network-anomaly validate --input data/synthetic/demo.csv

# 3. Fit on the first 50 % of the timeline, calibrate on the next 20 %, score everything
uv run network-anomaly detect --input data/synthetic/demo.csv --detector isolation_forest \
  --output reports/detections.csv

# 4. Point- and event-level metrics on the held-out test period (the last 30 %)
uv run network-anomaly evaluate --input data/synthetic/demo.csv --predictions reports/detections.csv
```

## Commands

| Command | What it does |
|---|---|
| `schema metrics\|flows` | Print the documented input schema: columns, units, valid ranges |
| `generate` | Paired synthetic datasets (with and without injected anomalies) plus events, data-quality events and metadata. 11 scenarios, `easy`/`moderate`/`difficult` presets, `--n-datasets` for independent seeds |
| `validate` | Strict schema validation with line-numbered errors and fix hints; `--on-invalid mask` blanks bad cells instead of rejecting the file |
| `detect` | Fit a detector and thresholds without touching the test period; write scores, flags and segments, plus a `.meta.json` provenance file. `--save-model` / `--model` to reuse a fitted model |
| `evaluate` | Precision/recall/F1/FPR/PR-AUC per point, and event recall/precision, detection delay and false alerts per host-day. Refuses in-sample rows unless `--allow-in-sample` |

Detectors: `static_threshold` (fixed limits), `robust_zscore` (rolling median / MAD per host),
`isolation_forest`, `one_class_svm`. Threshold strategies: `fixed`, `train_quantile`,
`clean_quantile` (default, needs no labels), `target_fpr` (needs labels). Example configurations
live in [`configs/`](configs/).

Scores are unitless and **not probabilities**; higher always means more anomalous.

Exit codes: `0` ok · `1` internal error · `2` usage · `3` invalid data · `4` invalid
configuration · `5` file or model problem · `6` leakage guard.

## Development

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest -q
```

## Data policy

Datasets, generated reports and trained models are never committed. See [data/README.md](data/README.md).

## License and author

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Developed by [feurstern](https://github.com/feurstern).
