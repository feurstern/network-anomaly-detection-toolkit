# Contributing

Thank you for helping. This project values correctness, reproducibility and honest evaluation over
the number of features.

## Set up

```bash
uv sync --extra torch          # everything, including the optional autoencoder
uv run network-anomaly --help
```

`uv sync` without the extra is the core install; the autoencoder tests are then skipped.

## Before every change is proposed

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest -q
```

All four must pass. The suite takes about a minute; `pytest -m "not integration"` runs the unit
tests only.

## Ground rules

* **No leakage.** Anything fitted (features, detectors, thresholds) sees training rows only, or the
  segment its strategy documents; never test rows. Add a test when you touch fitting code.
* **No silent conversion.** Invalid input fails with a coded, actionable error or is masked and
  reported; it is never turned into a plausible value.
* **Honest wording.** Scores are not probabilities; anomalies are not attacks; explanations
  describe co-occurrence, not causes. A test rejects causal and attack language in alert texts.
* **Determinism.** Seeds are explicit and recorded; rerunning a command must give identical
  quality metrics. Use `numpy.random.default_rng`, never global random state.
* **No data in the repository.** Never commit datasets, generated reports, models or anything
  derived from real traffic; tests use small synthetic fixtures (documentation IP ranges such as
  192.0.2.0/24 for addresses) and never touch the network.
* **Types and style.** Type hints everywhere (`mypy` is strict), Ruff formatting, docstrings that
  state the contract, comments only where the code cannot say why.

## Adding detectors, datasets or metrics

Follow [docs/extending.md](docs/extending.md): the detector contract suite and the dataset
adapter tests show what is expected. New dependencies need a clear reason; heavy or optional ones
belong in an extra and must be imported lazily.

## Reporting problems

Include the command, the toolkit version (`network-anomaly --version`), the full error message and,
if possible, a small synthetic file that reproduces it. Never attach real traffic or logs.
