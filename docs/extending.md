# Extending the toolkit

## Add a detector

A detector turns a `FeatureFrame` into one score per row. The base class
(`network_anomaly.detectors.base.Detector`) enforces the contract: `fit` only accepts frames tagged
`train`, scores are NaN for unscorable rows, higher always means more anomalous, and models save
to a directory with a manifest.

1. **Configuration.** A strict Pydantic model with a literal `type`:

   ```python
   class MedianAbsoluteConfig(StrictModel):
       type: Literal["median_absolute"] = "median_absolute"
       feature_kinds: tuple[ModelFeatureKind, ...] = ("z_local",)
   ```

2. **Detector.** Implement four methods; `self.matrix(ff, rows)` returns the fitted feature
   columns:

   ```python
   class MedianAbsoluteDetector(Detector):
       name = "median_absolute"
       config: MedianAbsoluteConfig

       def __init__(self, config: MedianAbsoluteConfig | None = None) -> None:
           super().__init__(config or MedianAbsoluteConfig())
           self.feature_kinds = tuple(self.config.feature_kinds)

       def _fit(self, ff, rows): ...  # learn from training rows only
       def _score(self, ff, rows):
           return np.abs(self.matrix(ff, rows)).mean(axis=1)

       def _state(self, directory): ...  # write fitted state
       def _restore(self, directory): ...  # read it back
   ```

3. **Explanations.** Either return exact per-metric terms from `metric_terms` (the score is their
   maximum or sum; set `term_method`), or implement `score_features(matrix)` so alerts can explain
   the detector by grouped occlusion. Without either, alerts cannot rank metrics.
4. **Register** the config in the `DetectorConfig` union and the detector in `DETECTOR_TYPES`
   (`detectors/__init__.py`), and add the name to `DetectorName` in `cli.py` (a test checks that
   the two lists match).
5. **Test.** The contract suite in `tests/unit/test_detectors.py` runs every registered detector:
   shapes, NaN handling, score direction on an injected outlier, determinism, save and load,
   refusal to fit on non-training data. Add tests for anything specific to your model.
6. **Benchmark** it by adding `- type: median_absolute` to a benchmark configuration.

Optional heavy dependencies belong in an extra (see the `torch` extra) and must be imported lazily
with an actionable error, so the core install keeps working.

## Add a dataset adapter

Adapters live in `ingestion/datasets/` and produce the canonical schemas.

1. **Pin the files.** Every `RemoteFile` needs an HTTPS URL, its size and a SHA-256 (or the git blob
   id for files at a pinned commit). Download each file once, verify it, and record the checksum in
   `pins.py`. Never point at a moving `main` branch or "latest" URL.
2. **Describe the dataset** in `DATASETS` (`ingestion/datasets/__init__.py`): licence, source,
   size, intended use, caveats, and `terms` if the publisher asks for anything before use (this
   makes `--accept-terms` mandatory).
3. **Write `prepare`.** Read from `data/raw/<name>/`, write canonical CSVs (UTC timestamps, units
   of the registry, labels as `is_anomaly` plus an events file) to `data/processed/<name>/`, and
   return a `Prepared` with every assumption you had to make in `notes` and the next commands in
   `commands`. Fail with a clear error for unexpected columns; never coerce silently. Stream large
   files in chunks. Apply the address policy to any identifier column.
4. **Test offline** with tiny fixtures in the exact published format (see
   `tests/unit/test_datasets.py`); tests must never download anything.
5. **Document** the dataset, its licence and caveats in [datasets.md](datasets.md).

## Add a metric

Add a `MetricSpec` to `CANONICAL_METRICS` in `schemas.py`: unit, valid range, modelling space
(`log_scale` for volumes, latencies and counts), meaningful direction, whether it is a user-facing
symptom (`critical`) and a minimum robust scale. Validation, features, explanations, reports and
`network-anomaly schema` pick it up. If it belongs to a new family, add investigation steps to
`explanations/playbook.py`; the wording test rejects causal or attack language.

## Integrate Rust kernels

The boundary, its data contract and the measurements that currently argue against it are in
[architecture.md](architecture.md#rust-boundary-designed-measured-not-built). If measurements on
your workload change that conclusion:

1. Build a separate extension module (PyO3 + maturin) implementing the kernels of
   `features/rolling.py` over the contract: a contiguous `float64` matrix, `int64` entity offsets,
   `NaN` for missing values and for windows that are not ready.
2. Select it only when it is importable and enabled by an environment variable; the Python
   implementation remains the reference and the default.
3. Run the existing kernel tests against both implementations on identical inputs (naive
   reference, causality, corruption and host-independence tests) and require identical outputs.
4. Compare with `scripts/bench_features.py` on the same data and machine, including conversion
   overhead, before claiming any speed-up.
