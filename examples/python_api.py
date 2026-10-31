"""Use the toolkit from Python: generate data, detect, explain alerts, evaluate.

Everything the CLI does is available as library calls. Run:

    uv run python examples/python_api.py
"""

import tempfile
from pathlib import Path

import pandas as pd

from network_anomaly.evaluation.runner import evaluate_table, table_from_result
from network_anomaly.ingestion.events import read_events
from network_anomaly.ingestion.validation import load_metrics
from network_anomaly.pipelines import DetectionConfig, run_detection
from network_anomaly.synthetic.generator import GeneratorConfig, generate_dataset, write_dataset

FIVE_MINUTES = pd.Timedelta("5min")


def main(hosts: int = 4, days: float = 21, seed: int = 7) -> None:
    with tempfile.TemporaryDirectory() as directory:
        # 1. A labelled synthetic dataset plus its clean twin (same data, no anomalies).
        path = Path(directory) / "demo.csv"
        config = GeneratorConfig(seed=seed, hosts=hosts, days=days)
        write_dataset(generate_dataset(config), path, with_clean_twin=True)

        # 2. Strict validation; labels are split off and never reach the features.
        validated = load_metrics(path)
        clean = load_metrics(path.with_name("demo.clean.csv"))

        # 3. Fit on the first half, calibrate on a known-clean copy of the next 20 %, score all.
        detection = DetectionConfig.model_validate({"detector": {"type": "robust_zscore"}})
        result = run_detection(validated, detection, FIVE_MINUTES, clean=clean)

        # 4. Explained alerts for the held-out test period.
        alerts = [alert for alert in result.alerts() if alert.segment == "test"]
        print(f"{len(alerts)} alerts in the test period; the first three:")
        for alert in alerts[:3]:
            print(f"  [{alert.severity}] {alert.summary}")

        # 5. Point- and event-level metrics on the test segment only.
        report = evaluate_table(
            table_from_result(result, validated.labels),
            read_events(path.with_name("demo.events.csv")),
            detector=result.bundle.detector.name,
            split=result.bundle.split.as_dict(),
            frequency=FIVE_MINUTES,
        )
        print(report.summary())


if __name__ == "__main__":
    main()
