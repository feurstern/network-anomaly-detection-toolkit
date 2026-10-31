# Data governance

Infrastructure metrics and especially network-flow records can identify people, customers and
systems. Treat every input and every output of this toolkit as potentially sensitive.

## What the toolkit does and does not do

* **No transmission.** Nothing is sent to external services. The only network access is
  `network-anomaly datasets download`, which fetches public datasets from their publishers over
  HTTPS; it sends only the file request.
* **No accounts.** No cloud service, API key or GPU is needed.
* **No identifiers as features.** Entity names, addresses, timestamps and labels never reach a
  model; flow addresses are only counted per window.
* **No silent conversion.** Malformed values fail validation (or are masked and reported with
  `--on-invalid mask`); they are never turned into plausible numbers.

## Minimise what you collect

* Export only the metrics you analyse, for the hosts and period you need.
* For flow data, decide at conversion time whether you need addresses at all:
  `--ip-policy drop` removes them; `--ip-policy hash` (the default for public flow datasets)
  replaces them with keyed pseudonyms. `flows detect` never copies addresses into its output.
* Prefer aggregated flow metrics (`flows aggregate`) over raw flow records wherever window-level
  detection is enough.

## Pseudonymisation guidance

`hash` uses HMAC-SHA256 with a secret key, so the same address always maps to the same pseudonym
(counts and joins still work) while the mapping cannot be recomputed without the key.

* Generate a long random key and keep it out of the repository and out of output directories:

  ```bash
  export NETWORK_ANOMALY_ANON_KEY="$(openssl rand -hex 32)"
  ```

* Store the key in your secret manager; anyone with the key and a candidate address can test
  whether it appears in the data. Use different keys for datasets that must not be linkable.
* Pseudonymised data is still personal data under most privacy laws: it can be re-identified with
  the key or by correlating behaviour. Apply the same access rules as to the original.
* Rotating the key breaks continuity of pseudonyms; re-prepare the datasets you need to compare.
* Timestamps, ports and volumes can also identify activity. Remove what you do not need before
  sharing files.

## Storage and retention

* **Never commit data, models or reports.** `.gitignore` excludes `data/raw/*`,
  `data/processed/*`, `data/synthetic/*`, `reports/`, `models/` and model files; keep it that way.
  See [`data/README.md`](../data/README.md).
* Keep raw exports only until the processed files you need exist; apply your organisation's
  retention policy to everything derived from production systems, including detections, alerts
  and benchmark run folders.
* Synthetic data is reproducible from its seed and configuration; regenerate it instead of
  archiving it.
* Saved models contain statistics of the training data (per-host noise levels, score
  distributions) and, for model detectors, fitted parameters; store them with the data they came
  from.

## Access control

* Restrict raw data, processed data, models and reports to the people and service accounts that
  need them; outputs contain host names and incident timelines.
* Run scheduled jobs under a dedicated account with read access to inputs and write access to its
  output directory only.
* Only load saved models you created: `model.joblib` files are pickles, and loading a pickle can
  execute code.

## Public datasets

Only datasets that may be downloaded for research are supported, and none is redistributed with
the toolkit. Their licences and citation requests are listed by `network-anomaly datasets list`
and in [datasets.md](datasets.md); `--accept-terms` records your agreement where the publisher asks
for it. CICIDS2017 and CIDDS-001 contain IP addresses; they are pseudonymised by default when
prepared.

## Interpreting results responsibly

An anomaly is a deviation from a host's own past, not proof of an attack, intrusion or compromise.
Alerts describe what changed and what moved with it; deciding what happened requires independent
evidence (logs, traces, change records, the people who run the system).
