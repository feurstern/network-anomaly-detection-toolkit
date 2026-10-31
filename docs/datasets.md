# Datasets

No dataset is stored in the repository. Synthetic data is regenerated from a seed; public datasets
are downloaded by each user from the original publisher (`network-anomaly datasets download`),
verified against pinned checksums and converted locally (`network-anomaly datasets prepare`).
See [data_governance.md](data_governance.md) for handling rules.

## Synthetic generator

```bash
uv run network-anomaly generate --output data/synthetic/demo.csv --seed 42 \
  --preset moderate --scenario traffic_spike --scenario http_5xx_surge --with-clean-twin
```

**Model.** Each host is a small web service whose metrics are coupled the way real ones are:

| Quantity | Model |
|---|---|
| demand | host size x daily two-hump profile x weekend factor x legitimate level shifts x autocorrelated noise (Student-t in the difficult preset) |
| CPU | utilisation of the host's capacity plus background load |
| latency p50 / p95 | M/M/1 queueing delay that rises steeply near saturation, plus a swap penalty under memory pressure |
| connections | Little's law (arrival rate x time in system), capped; connection errors rise near the cap |
| network, disk | proportional to demand; nightly backup windows are normal behaviour |
| memory | lags towards a connection-dependent level; leaks and restarts |
| 4xx / 5xx rates | binomial error fractions; 5xx rises with overload |

**Scenarios** change the hidden drivers, and the symptoms propagate through the coupling (a traffic
spike also raises CPU, latency and connections): `traffic_spike`, `traffic_ramp` (gradual),
`traffic_burst`, `http_4xx_surge`, `http_5xx_surge`, `latency_degradation`, `cpu_saturation`,
`memory_pressure`, `connection_surge`, `throughput_spike`, `correlated_degradation`.

**Labels.** Every dataset is generated twice from the same random draws, with and without
injections. A row is labelled anomalous while any metric differs from the clean copy by more than
one noise standard deviation; each event records its start and end, scenario, magnitude tier, the
metrics changed by more than two standard deviations and when that first happened
(`detectable_from`). `--with-clean-twin` also writes the clean copy, usable as a known-normal
calibration file (`detect --clean-validation`).

**Data-quality problems** are applied identically to both copies and are never labelled: missing
cells, telemetry gaps, a metric dropping out, timestamp jitter, noise bursts and single-value
glitches (`<name>.dq_events.csv`).

**Presets** control magnitudes, noise and heavy tails, data-quality rates, level shifts, the share
of gradual scenarios and hosts that only appear after training:

| Preset | Noise | Magnitude mix (low / medium / high) | Level shifts per host | Late hosts | Data-quality problems |
|---|---|---|---|---|---|
| easy | low | 10 / 30 / 60 % | 0 | 0 % | none |
| moderate | medium | 30 / 40 / 30 % | 1 | 10 % | some |
| difficult | high, heavy-tailed | 55 / 35 / 10 % | 2 | 20 % | frequent |

**Reproducibility.** Random streams derive from `SeedSequence(seed, spawn_key=(dataset, stream,
host))`, so the same seed gives byte-identical files and adding a host does not change the others.
`<name>.metadata.json` records the seed, the resolved configuration, units, library versions and
the SHA-256 of every file.

## Public datasets

```bash
uv run network-anomaly datasets list
uv run network-anomaly datasets download nab           # into data/raw/nab/
uv run network-anomaly datasets prepare nab            # into data/processed/nab/
```

`download` is the only command that uses the network. It fetches pinned files over HTTPS and keeps
a file only if its size and checksum match the pinned value; `prepare` prints what it assumed and
the next commands to run.

### NAB - Numenta Anomaly Benchmark (metrics)

* **Source:** <https://github.com/numenta/NAB>, pinned to commit `ea702d75`, which is
  **MIT-licensed** (releases up to v1.1 are AGPL-3.0). 22 files, 2.6 MB.
* **Used:** the 17 `realAWSCloudwatch` series, `ec2_request_latency_system_failure` and
  `cpu_utilization_asg_misconfiguration`, with `labels/combined_windows.json`.
* **Conversion:** one file per series (`host_id` = series name), timestamps taken as UTC.
  CloudWatch `NetworkIn`, `DiskWriteBytes` and ELB `RequestCount` are totals per 5-minute period
  and are divided by 300. Disk *write* bytes go into `disk_io_bytes_per_second`; the latency series
  (an average) into `response_time_p50_ms`; `grok_asg_anomaly` stays a generic metric. Three series
  repeat 11 timestamps (a daylight-saving bug); the first of each is kept and the count reported.
* **Caveats:** units are inferred; each label window covers about 10 % of a series, so use event
  metrics, not point precision; the series are short (about two weeks).
* **Cite:** Lavin and Ahmad, "Evaluating Real-time Anomaly Detection Algorithms - the Numenta
  Anomaly Benchmark", IEEE ICMLA 2015.

### SMD - Server Machine Dataset (metrics, explanations)

* **Source:** <https://github.com/NetManAIOps/OmniAnomaly>, pinned commit `7fb0e0ac`, **MIT**.
  28 machines x 38 metrics, about 17 MB per machine; `--machines machine-1-1` (default),
  a comma-separated list or `all` (486 MB).
* **Conversion:** no timestamps exist; rows are placed on a 1-minute grid from 2026-01-01T00:00Z,
  training half then test half. Dimensions become generic metrics `dim_01` .. `dim_38`. Events are
  runs of labelled test rows, with the interpretation-label dimensions as `affected_metrics`, so
  `evaluate` reports explanation hit rates. The prepare step prints a `detect` command whose
  explicit split keeps the unlabelled training half out of evaluation.
* **Caveats:** the training half is unlabelled (written as 0 only to satisfy the schema); values
  are pre-scaled and anonymised; a few interpretation spans are shorter than their label runs.
* **Cite:** Su et al., "Robust Anomaly Detection for Multivariate Time Series through Stochastic
  Recurrent Neural Network", KDD 2019.

### CICIDS2017, improved labels (flows)

* **Source:** DistriNet, KU Leuven: <https://intrusion-detection.distrinet-research.be/CNS2022/>.
  One 344 MB archive (five days, 1.15 GB of CSV). **No explicit licence; citation requested** -
  `download` requires `--accept-terms`.
* **Conversion:** per day, streamed from the archive; UTC timestamps (microseconds), durations in
  milliseconds, flag counts, `label`; `is_anomaly` = 1 for attack flows, but **not** for
  `... - Attempted` flows (attacks that did not get through) unless `--attempted-as-anomalous`.
  Addresses are pseudonymised by default (see below).
* **Use:** per-flow scoring, training on Monday (benign only):
  `network-anomaly flows detect --train monday.flows.csv --input tuesday.flows.csv`.
* **Caveats:** a lab network with scripted attacks; despite the corrections, some flows remain
  mislabelled. Avoid the older 2021 archive, whose timestamps are shifted by two hours.
* **Cite:** Engelen, Rimmer and Joosen, "Troubleshooting an Intrusion Detection Dataset: the
  CICIDS2017 Case Study", IEEE SPW 2021.

### CIDDS-001 (flows)

* **Source:** Coburg University of Applied Sciences:
  <https://www.hs-coburg.de/forschung/forschungsprojekte-oeffentlich/informationstechnologie/cidds-coburg-intrusion-detection-data-sets.html>.
  One 403 MB archive. **No explicit licence; citation requested** (`--accept-terms`).
* **Conversion:** the external server's four weeks by default (`--part internal` for the 4 GB
  OpenStack part). Local German timestamps are converted to UTC (the report's "missing hour" on
  2017-03-26 is the daylight-saving jump). Byte counts such as `2.1 M` are expanded; TCP flags are
  recorded per flow as a set, so the `*_count` fields hold 1 if a flag occurred at least once;
  nfdump writes ICMP type and code into the port fields, so ICMP flows get no ports. Flows of class
  `attacker` or `victim` are anomalous; `suspicious` and `unknown` internet traffic is kept but
  not labelled. The attack logs become an events file for the aggregated entity `all`.
* **Use:** `network-anomaly flows aggregate` over the four weeks, then `detect` and
  `evaluate --events cidds-001-external.events.csv`.
* **Caveats:** an emulated small-business network; the exposed server receives unlabelled
  internet scanning that resembles the attacks (see [evaluation.md](evaluation.md)).
* **Cite:** Ring, Wunderlich, Grüdl, Landes and Hotho, "Flow-based benchmark data sets for
  intrusion detection", ECCWS 2017.

### Addresses in flow datasets

`prepare --ip-policy` decides what happens to `src_addr` and `dst_addr`:

| Policy | Effect |
|---|---|
| `hash` (default) | each address becomes `h-` + 16 hex characters of HMAC-SHA256 under a secret key from `NETWORK_ANOMALY_ANON_KEY` (at least 16 characters); the same address always maps to the same pseudonym under the same key |
| `drop` | the address columns are removed; per-window `unique_sources` is then unavailable |
| `keep` | addresses stay as published |

Addresses are never model features; they are only counted per window.

### Considered but not supported

| Dataset | Why not |
|---|---|
| Exathlon | CC BY-NC-SA and 3.2 GB zipped |
| Yahoo S5 | Non-commercial licence agreement required |
| CSE-CIC-IDS2018 | 12-hour timestamps without AM/PM |
| UNSW-NB15, UQ NF-v3 | Academic use only, manual download |
| CTU-13 | Real IP addresses; background flows unlabelled |
| NASA-HTTP (1995) | Unlabelled; terms ask for no analysis beyond general traffic patterns (a possible future adapter) |
