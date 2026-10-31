# Deployment

The toolkit analyses files offline: it reads CSV exports and writes CSV, JSON Lines, Markdown and
PNG files. It does not run a server, collect metrics or send anything anywhere. This page shows
how to run it on a schedule against your own monitoring data, and what to watch for.

## Shape of a deployment

```
monitoring system ──export──▶ metrics CSV ──validate──▶ detect --model ──▶ alerts.jsonl ──▶ your tooling
       (Prometheus, CloudWatch, flow exporter)                                (tickets, chat, dashboards)
```

1. **Export** metrics in the [metrics schema](data_schema.md): one row per host per interval, UTC
   timestamps, rates as fractions. Prometheus range queries can be converted with
   `network-anomaly convert prometheus`; flow records with `network-anomaly flows aggregate`.
2. **Fit once** on a representative history (several weeks, so weekly patterns are included):

   ```bash
   uv run network-anomaly detect --input history.csv --detector robust_zscore \
     --output reports/fit/detections.csv --save-model models/web-v1
   ```

   If you have a period you have checked and know to be normal, pass it with
   `--clean-validation vetted.csv`: thresholds fitted on a contaminated window hide incidents
   (see [evaluation.md](evaluation.md)).
3. **Score new data** with the frozen model, for example every 15 minutes from cron or a CI job:

   ```bash
   uv run network-anomaly detect --input last-48h.csv --model models/web-v1 \
     --output reports/live/detections.csv --alerts reports/live/alerts.jsonl
   ```

   Give each run enough history for the baselines: at least a day before the period you care
   about (the residual scale uses a day; the same-time baseline uses up to a week). Rows without a
   ready baseline are written with status `warmup` and are never flagged.
4. **Consume alerts.** Each line of `alerts.jsonl` is a self-contained JSON object with a summary,
   severity, the metrics involved and suggested checks. Forward new alerts (by `alert_id`) to your
   ticketing or chat system; keep the scores in a time-series store if you want to graph them.

## Keeping a model healthy

* **Recalibrate** after releases, migrations, capacity changes or seasonal shifts: refit on recent
  weeks and compare alert volume before switching. Thresholds assume the future looks like the
  calibration period.
* **Watch the detector itself:** daily alert counts per host, the share of `warmup` and `no_data`
  rows, and alerts with `unseen_entity` or `missing_metrics` warnings. A sudden change usually
  means a data problem, not an incident.
* **New hosts** are scored with pooled noise floors and the global threshold until the model is
  refitted with their history; their alerts say so.
* **Measure before you trust.** If you have incident records, label a few weeks and run
  `evaluate` on the test period; measure your own false alerts per host-day.

## Choosing a detector

| Need | Start with |
|---|---|
| Few hosts, little history, explanations engineers can verify | `robust_zscore` |
| Hard limits that must always alert (SLOs) | `static_threshold` with your limits in a config file |
| Unusual combinations of metrics that are each within range | `isolation_forest` or `one_class_svm` |
| Large histories, experimentation | `autoencoder` (`uv sync --extra torch`) |

Running two detectors side by side (a static limit for hard failures plus a learned detector) is a
common pattern; use `benchmark` on synthetic data and `evaluate` on your own labels to compare.

## Sizing

Measured on one core of an Intel Core 7 240H (see [evaluation.md](evaluation.md)): feature
building processes about 60,000 rows per second; scoring adds 7 µs per row for Isolation Forest
and 12 µs for One-Class SVM; the robust detector is negligible. 1,000 hosts at 5-minute resolution
produce 288,000 rows per day: about 5 seconds of feature building per day of data. Memory grows
with the rows processed per run (roughly 1.8 KB per row for the feature step with 12 metrics);
split very large histories by host group.

## Security and privacy

* The toolkit makes **no network connections** except `datasets download`, which fetches public
  datasets over HTTPS and verifies pinned checksums.
* **Saved models can contain pickles** (`model.joblib`); loading a pickle can execute code. Only
  load models you created, and protect the model directory like code. PyTorch weights are loaded
  with `weights_only=True`.
* Detections, alerts and reports contain host names and timestamps from your data; treat them as
  operational data with the same access rules as your monitoring system. See
  [data_governance.md](data_governance.md).
* Run scheduled jobs under a dedicated account with read access to the exports and write access
  to the output directory only.
