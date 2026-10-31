# Evaluation methodology

This page explains how `evaluate` and `benchmark` measure detectors, why each rule exists, and
what the numbers can and cannot tell you.

## Chronological splits

Every dataset is cut into three consecutive periods, with **one set of cut times for all hosts**,
aligned to the sampling grid:

| Segment | Default share | Used for |
|---|---|---|
| train | first 50 % | fitting feature statistics and the detector |
| validation | next 20 % | fitting thresholds (most strategies) |
| test | last 30 % | evaluation only |

Use `--train-until` / `--validation-until` for explicit cut times. Rolling features of validation
and test rows look back into earlier periods; that is the past, which production systems have
too, not leakage. Training data is **unlabelled and contains anomalies**, as it would in practice;
nothing is removed from it.

`evaluate` refuses train and validation rows (exit code 6) unless `--allow-in-sample` is given,
and then labels the result in-sample. An event that starts in one segment and continues into the
next belongs to the segment it started in: its later rows are excluded from point metrics, and
alerts on it count as neither true nor false.

## Thresholds and their limits

A threshold is a cut-off on one detector's unitless score scale. It is never a probability.

| Strategy | Fitted on | Needs | Limitation |
|---|---|---|---|
| `fixed` | nothing | a value you choose | Meaningful only for scores with a natural scale (static limits: 1.0) |
| `train_quantile` | training scores | nothing | Training data contains anomalies, and model detectors score their own training rows optimistically |
| `clean_quantile` | validation scores | a window believed normal | Anomalies inside the window raise the threshold and hide later incidents (the default when no labels exist) |
| `clean_quantile` + `--clean-validation FILE` | a known-clean copy of the validation window | a vetted normal file | You need such a file (the synthetic generator writes one with `--with-clean-twin`) |
| `target_fpr` | validation rows labelled normal | labels | Labels are rarely available; the false-positive rate is only met if the test period resembles validation |

With `per_host`, hosts with enough validation rows (at least 10 / (1 - q) of them, so the quantile
rests on at least ten exceedances) get their own threshold; others, and hosts never seen, use the
global one. All strategies assume the future resembles the calibration period: after a release,
a migration or seasonal change, recalibrate.

## Point-level metrics

Every scored interval is one prediction at the deployed threshold: precision, recall, F1,
false-positive rate and the confusion matrix. **Unscored rows (warm-up, missing data) count as not
flagged**, so anomalies a detector could not see count as misses. PR-AUC (average precision) is
threshold-free; unscored rows rank below every scored row instead of being dropped. It is
undefined when only one class is present.

**Point-adjusted metrics are not computed.** They mark a whole labelled event as detected after a
single hit, which rewards long events and makes even random scores look strong (Kim et al., AAAI
2022).

## Event-level metrics

Operators experience incidents, not rows, so each labelled event is evaluated once.

* **Alerts.** Flagged points of one entity separated by at most `merge_gap` unflagged or missing
  intervals form one alert (default 2). An alert needs `min_points` flagged points (default 2);
  it **fires** on its `min_points`-th point and is active until one interval after its last.
* **Matching.** An event is detected if an alert of the same entity is active inside its window,
  from `early_tolerance` intervals before its start (default 1, for clock jitter) to
  `late_tolerance` after its end (default 3, for detectors that react slightly late).
* **One alert, several events.** It detects all of them (reported as merged detections).
* **Several alerts, one event.** The event counts once (reported as fragmentation).
* **False alerts** fire inside the evaluated window and touch no event window.
* **Event precision** = detected events / (detected events + false alerts), so fragmenting one
  incident into many alerts cannot inflate it.
* **Detection delay** = first moment an alert is active in the window minus the event start, never
  negative; median and 90th percentile. For gradual events it is also given from
  `detectable_from`, the first time the injected effect exceeded two noise standard deviations.
* **False alerts per host-day** divides false alerts by the host-days of scored, event-free time.

## Explanation quality

Events can list the metrics an anomaly actually affected (the generator records metrics changed by
more than two noise standard deviations; SMD lists engineer-annotated dimensions). For each
detected event, the first alert's metric ranking is compared with that list:

* **HitRate@P%** (Su et al., KDD 2019): share of the affected metrics found among the top
  floor(P% x |affected|) ranked metrics; P = 100 % and 150 %.
* **Top-ranked metric affected:** how often the highest-ranked metric is in the list.

A high hit rate means explanations point where an engineer would look, not that they identify a
cause.

## Benchmark protocol

`network-anomaly benchmark --config configs/benchmark.yaml` runs every detector on the same
datasets with the same rules:

1. One synthetic dataset per (preset, seed), generated with its **clean twin** (identical noise,
   data-quality problems and level shifts, no anomalies).
2. Per (dataset, detector), in a fresh process limited to one BLAS thread: fit on the training
   segment; score the dataset and its twin; fit every operating point without touching the test
   segment; evaluate the test segment; count false alerts on the twin's test segment (where every
   alert is false), including how many fall into short data-quality problems or follow
   legitimate level shifts.
3. Timing: sequential fresh processes on the first dataset, a warm-up run and the median of
   repeated runs, feature building timed separately from fitting and scoring; peak resident
   memory read from the kernel.
4. Aggregation: mean and sample standard deviation (ddof = 1) over seeds.

The **headline operating point is fixed in the configuration before the run** (`target_fpr`, 0.5 %
on validation). No detector, setting or threshold is chosen by looking at test results, and no
winner is declared.

### Results (default configuration)

Moderate preset, 5 seeds, 20 hosts x 35 days at 5-minute resolution (about 200,000 rows per
dataset), all 11 scenarios, `target_fpr` = 0.5 %. Mean ± standard deviation over seeds; quality
numbers are deterministic and reproduce exactly from a clean checkout.

| Detector | Event recall | Event precision | False alerts / host-day | Clean-twin false alerts / host-day | PR-AUC | Explanation HitRate@100% |
|---|---|---|---|---|---|---|
| static_threshold | 0.39 ± 0.12 | 0.44 ± 0.18 | 0.14 ± 0.07 | 0.14 ± 0.07 | 0.17 ± 0.08 | 0.63 ± 0.02 |
| robust_zscore | 0.89 ± 0.07 | 0.47 ± 0.06 | 0.23 ± 0.04 | 0.24 ± 0.05 | 0.32 ± 0.10 | 0.94 ± 0.03 |
| isolation_forest | 0.42 ± 0.05 | 0.27 ± 0.06 | 0.26 ± 0.04 | 0.24 ± 0.03 | 0.22 ± 0.06 | 0.91 ± 0.05 |
| one_class_svm | 0.57 ± 0.13 | 0.40 ± 0.07 | 0.20 ± 0.04 | 0.13 ± 0.03 | 0.28 ± 0.13 | 0.93 ± 0.04 |

Observations, all specific to this synthetic setting:

* The robust baseline detects most events. Most injected scenarios move one or a few metrics far
  from their own baseline, which is exactly what it measures; scenarios that change correlations
  without large single-metric deviations would favour the multivariate models.
* Known-clean calibration (`clean_quantile` on the twin) raises recall for every detector (robust
  0.91, One-Class SVM 0.76) at the cost of more false alerts, confirming that anomalies in an
  unvetted validation window inflate label-free thresholds.
* Short data-quality problems (gaps, dropouts, noise bursts, glitches) cover 0.8 % of host-time but
  account for 23-29 % of the robust and One-Class SVM false alerts on normal-only data;
  legitimate level shifts cause very few.

Runtime on one machine (Intel Core 7 240H, one thread per process), first dataset:

| Step | Rows | Median s | Rows / s |
|---|---|---|---|
| feature building (fit + transform) | 195,566 | 3.2 | 61,000 |
| robust_zscore fit / score | 94,678 / 194,564 | 0.03 / 0.03 | 3.7 M / 6.9 M |
| isolation_forest fit / score | 94,678 / 194,564 | 0.29 / 1.32 | 323,000 / 147,000 |
| one_class_svm fit / score | 94,678 / 194,564 | 0.18 / 2.30 | 514,000 / 85,000 |

The full default benchmark (20 accuracy tasks plus timing) takes about 2.5 minutes on 16 logical
cores; `--skip-timing` takes about 1 minute.

## What these results cannot show

* **Synthetic data is not production.** The generator is this project's own model of a web
  service, with anomaly types its authors chose. Real systems have other failure modes, noisier
  and incomplete labels, drift and dependencies between hosts. **Synthetic benchmark performance
  does not demonstrate production effectiveness.**
* **Rankings are conditional** on the preset, scenario mix, feature windows and threshold strategy;
  run `configs/benchmark_full.yaml` (all presets, 10 seeds) before generalising.
* **Few seeds.** Five seeds give rough variability estimates; differences within one standard
  deviation are noise.
* **Labels from public datasets have known problems** (see [datasets.md](datasets.md)): NAB
  windows are wide, SMD's training half is unlabelled, flow datasets label internet background
  traffic as normal or unknown.
* **An anomaly is not an attack.** Detection measures deviation from a host's own past; deciding
  whether that is an incident, a release or an attack needs independent evidence.

## Observations on public data

From the adapters' validation runs (detector defaults, label-free thresholds):

* **SMD machine-1-1**, Isolation Forest: 4 of 8 incidents detected; explanation HitRate@100 %
  0.44 against the engineers' interpretation labels, well below synthetic levels.
* **CIDDS-001**, external server, four weeks aggregated to 5-minute windows: attack windows are
  clearly visible (flows/s up to 18x, destination ports 15x, half of TCP flows without
  handshake) and rank in the top 1 % of test scores, but **0 of 16 attacks crossed the threshold**:
  the internet-facing server receives constant, unlabelled scanning, so the "normal" validation
  window contains similar bursts and the label-free threshold rises above them.
* **CICIDS2017** (improved), per-flow scoring trained on benign Monday: PR-AUC 0.075 for Tuesday's
  FTP and SSH brute force (base rate 2.2 %), with no recall at a 1 % false-positive rate; each
  brute-force flow looks like an ordinary login.

These are the kind of limits a real deployment must expect: unsupervised detection finds
deviations, and both thresholds and expectations need calibrating against your own traffic.
