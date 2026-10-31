# Data schemas

All files are UTF-8 CSV (or JSON / JSON Lines where stated) with a header row. Timestamps are
ISO 8601 **with a UTC offset** (`2026-01-05T14:30:00Z` or `2026-01-05T16:30:00+02:00`); naive
timestamps are rejected unless `--assume-utc` is given. `network-anomaly schema metrics|flows`
prints the same tables from the code.

## Metrics (input)

A *wide* table: one row per (timestamp, entity), one column per metric.

* **Required:** `timestamp`, the entity column (`host_id` by default; `--entity-column service`
  for services) and at least one metric.
* **Optional:** any subset of the canonical metrics below. Unknown numeric columns are accepted as
  generic metrics (unit unknown, no range check, both directions) with a warning.
* **Labels (evaluation only):** `is_anomaly` (0/1 on every row), `anomaly_event_id`,
  `anomaly_scenario`, `anomaly_severity`. They are split off when the file is loaded and never
  reach the features.
* **Sampling:** one interval (inferred from the median spacing and snapped to a common value such
  as 5 min, or given with `--freq`). Rows are placed on an epoch-aligned grid; a series that is
  regular but offset from the clock (e.g. at :02, :07, ...) is accepted and reported as info.

Units matter: rates are **fractions** (0.05, not 5), percentages are 0-100, volumes are per
second. `Modelled as` is the space features are computed in; `Direction` is which deviations count
(errors, latency and saturation only upwards; traffic both ways); `Critical` metrics are
user-facing symptoms that can raise alert severity to critical.

| Metric | Unit | Valid range | Modelled as | Direction | Critical | Description |
|---|---|---|---|---|---|---|
| `requests_per_second` | requests/s | >= 0 | log1p | both | no | Requests served per second, averaged over the sampling interval. |
| `http_4xx_rate` | fraction | [0, 1] | linear | up | no | Fraction of requests answered with a 4xx status code. |
| `http_5xx_rate` | fraction | [0, 1] | linear | up | yes | Fraction of requests answered with a 5xx status code. |
| `response_time_p50_ms` | ms | >= 0 | log1p | up | no | Median response time over the sampling interval. |
| `response_time_p95_ms` | ms | >= 0 | log1p | up | yes | 95th-percentile response time over the sampling interval. |
| `cpu_usage_percent` | % | [0, 100] | linear | up | no | CPU utilisation across all cores. |
| `memory_usage_percent` | % | [0, 100] | linear | up | no | Memory in use as a share of total memory. |
| `disk_io_bytes_per_second` | bytes/s | >= 0 | log1p | up | no | Disk read plus write throughput. |
| `network_rx_bytes_per_second` | bytes/s | >= 0 | log1p | both | no | Bytes received per second on the service interfaces. |
| `network_tx_bytes_per_second` | bytes/s | >= 0 | log1p | both | no | Bytes sent per second on the service interfaces. |
| `active_connections` | connections | >= 0 | log1p | up | no | Open client connections at sampling time. |
| `connection_error_rate` | fraction | [0, 1] | linear | up | yes | Fraction of connection attempts that failed (refused, reset or timed out). |
| `flows_per_second` | flows/s | >= 0 | log1p | both | no | Flow records starting in the window, per second. |
| `flow_packets_per_second` | packets/s | >= 0 | log1p | both | no | Packets of the window's flows, per second. |
| `flow_bytes_per_second` | bytes/s | >= 0 | log1p | both | no | Bytes of the window's flows, per second. |
| `unique_sources` | sources | >= 0 | log1p | up | no | Distinct source identifiers in the window. |
| `unique_destination_ports` | ports | >= 0 | log1p | up | no | Distinct destination ports in the window. |
| `syn_only_ratio` | fraction | [0, 1] | linear | up | no | Share of TCP flows with SYN but no ACK (connection attempts never completed). |
| `rst_ratio` | fraction | [0, 1] | linear | up | no | Share of TCP flows with a RST packet (connections refused or aborted). |

The last seven are produced by `network-anomaly flows aggregate` from flow records.

## Network flows (input)

One row per flow record. Required: `timestamp` (flow start) and at least one of `packets`,
`bytes`, `packets_fwd`, `packets_bwd`, `bytes_fwd`, `bytes_bwd`. Unknown columns are ignored with a
warning. Sensitive columns are reported by validation and are never model features.

| Field | Kind | Unit | Required | Sensitive | Description |
|---|---|---|---|---|---|
| `timestamp` | timestamp | - | yes | no | Flow start time, ISO 8601 with a UTC offset. |
| `src_addr` | identifier | - | no | yes | Source address or pseudonymised source identifier. |
| `dst_addr` | identifier | - | no | yes | Destination address or pseudonymised destination identifier. |
| `src_port` | port | - | no | no | Source port (0-65535). |
| `dst_port` | port | - | no | no | Destination port (0-65535). |
| `protocol` | categorical | - | no | no | Transport protocol name or IANA number (e.g. tcp, udp, 6, 17). |
| `duration_ms` | duration | ms | no | no | Time between the first and last packet of the flow. |
| `packets`, `bytes` | count | packets, bytes | no | no | Totals in both directions. |
| `packets_fwd`, `packets_bwd`, `bytes_fwd`, `bytes_bwd` | count | packets, bytes | no | no | Per direction. |
| `syn_count`, `ack_count`, `fin_count`, `rst_count`, `psh_count`, `urg_count` | count | packets | no | no | Packets with each TCP flag. |
| `conn_state` | categorical | - | no | no | Connection state or termination reason reported by the exporter. |
| `label` | label | - | no | no | Original dataset label (e.g. BENIGN or an attack class); evaluation only. |
| `is_anomaly` | label | - | no | no | Binary ground truth (1 = labelled anomalous); evaluation only. |

## Ground-truth events (`<name>.events.csv`)

`event_id, host_id, scenario, severity, start, end, detectable_from, affected_metrics,
peak_effect_sigma, params`. An event covers `[start, end)` on one entity. `affected_metrics`
(`;`-separated) lists the metrics the anomaly changed by more than two noise standard deviations
(synthetic data) or the dimensions engineers attributed it to (SMD); evaluation uses it to score
explanations. `severity` is the injected magnitude tier (low/medium/high), not alert severity.
`evaluate` reads `<input>.events.csv` when it exists; otherwise events are rebuilt from labels.

## Detections (`detect` output)

`detections.csv` - one row per input row, joinable to the input on (timestamp, entity):

| Column | Meaning |
|---|---|
| `timestamp`, entity column | As in the input |
| `segment` | `train`, `validation` or `test`; train and validation rows are in-sample |
| `status` | `scored`, `warmup` (baseline not ready yet) or `no_data` |
| `score` | Unitless, higher = more anomalous, **not a probability**; empty when not scored |
| `threshold`, `threshold_scope` | The cut-off for this entity and whether it is per-host or global |
| `is_flagged` | 1 when `score >= threshold` |
| metrics | The input values, for convenience |

`detections.meta.json` records provenance: toolkit version, input SHA-256, detector and its
configuration, feature settings, entity column, split cut times, fitted thresholds, alerting
settings and row counts per segment. `evaluate` needs it next to the CSV.

## Alerts (`detections.alerts.jsonl`)

One JSON object per line, one per alert (a group of flagged points on one entity that fired after
`min_points` flagged points, allowing gaps of up to `merge_gap` intervals).

| Field | Meaning |
|---|---|
| `alert_id`, `entity_column`, `entity` | Identity |
| `start`, `fired_at`, `end`, `peak_at`, `duration_minutes`, `flagged_points` | Timing; `fired_at` is when an operator would be paged; `end` is exclusive |
| `segment` | Segment of `fired_at` (in-sample alerts also carry a data-quality warning) |
| `detector`, `detector_version` | Which model produced it |
| `score`, `threshold`, `threshold_scope`, `threshold_strategy` | Peak score and the cut-off it crossed |
| `reference_exceedance` | Share of validation-period scores at least as high as `score`: an empirical frequency, not a probability |
| `severity`, `severity_reason` | `low` / `medium` / `high` / `critical` from deviation rules (see below), with the rule that fired |
| `explanation_method` | `metric_terms` (statistical detectors), `reconstruction_error` (autoencoder), `occlusion_drop_one` / `occlusion_keep_one` (Isolation Forest, One-Class SVM) |
| `contributions` | Top metrics: observed value, recent and same-time baselines, % change, robust z, seasonal z, contribution |
| `metric_ranking` | Every metric with a contribution, highest first |
| `peak_deviations` | Robust z of every metric at the peak |
| `summary` | One sentence: what deviated against which baseline, when, and what moved with it |
| `suggested_checks` | Investigation steps for the metric families involved, ending with a confirmation step |
| `data_quality` | Warnings that limit trust: `telemetry_gap`, `masked_cells`, `missing_metrics`, `seasonal_baseline_unavailable`, `unseen_entity`, `in_sample` |

**Severity rules** (first match wins; deviations are the larger of the recent and same-time
robust z in the metric's meaningful direction):

| Severity | Rule |
|---|---|
| critical | a critical metric (5xx rate, p95 latency, connection errors) >= 8 on at least 3 points |
| high | any metric >= 6, or at least 3 metrics >= 4 at the same point |
| medium | any metric >= 4 |
| low | otherwise |

Contributions rank metrics for investigation; they are not additive (except the autoencoder's),
not probabilities and not causal. Summaries describe co-occurrence ("co-occurred with") and never
assert a cause or an attack; a test enforces this wording.

## Validation issue codes

`validate` reports each problem with a code, severity, column, example values, line numbers and a
fix hint. Errors make the file invalid (exit code 3); with `--on-invalid mask`, invalid metric
cells become missing values instead and are reported as warnings.

| Severity | Codes |
|---|---|
| error | `missing_column`, `missing_timestamp`, `bad_timestamp`, `naive_timestamp` (without `--assume-utc`), `missing_entity`, `no_metrics`, `non_numeric`, `non_finite`, `out_of_range`, `missing_label`, `invalid_label`, `conflicting_duplicates`, `grid_collision`, `missing_volume_field` (flows), `invalid_value` (flows) |
| warning | `unknown_metric`, `unknown_column` (flows), `ignored_column`, `empty_column`, `duplicate_rows` (exact copies, dropped), `percent_scale_suspect`, `irregular_timestamps`, `mixed_frequencies`, `constant_metric`, `latency_percentiles_inverted`, `sensitive_columns` (flows) |
| info | `missing_intervals`, `missing_cells`, `grid_phase_offset`, `frequency_unknown`, `duplicate_flows` |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Internal error (or the output pipe was closed early, e.g. `| head`) |
| 2 | Usage error (unknown option, missing file) |
| 3 | Input data failed validation |
| 4 | Invalid configuration |
| 5 | Missing or unreadable file, model or download |
| 6 | Leakage guard: fitting or evaluating on rows that are not allowed |
