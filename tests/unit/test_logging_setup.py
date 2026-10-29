import io
import json
import logging

from network_anomaly.logging_setup import PACKAGE_LOGGER, configure_logging


def test_json_lines_include_extra_fields() -> None:
    stream = io.StringIO()
    configure_logging("INFO", "json", stream=stream)
    logging.getLogger("network_anomaly.ingestion").info(
        "rows loaded", extra={"rows": 42, "path": "demo.csv"}
    )
    record = json.loads(stream.getvalue())
    assert record["message"] == "rows loaded"
    assert record["level"] == "INFO"
    assert record["logger"] == "network_anomaly.ingestion"
    assert record["rows"] == 42
    assert record["path"] == "demo.csv"
    assert record["ts"].endswith("+00:00")


def test_text_format_filters_by_level_and_renders_extras() -> None:
    stream = io.StringIO()
    configure_logging("WARNING", "text", stream=stream)
    log = logging.getLogger("network_anomaly.detectors")
    log.info("hidden")
    log.warning("threshold fallback", extra={"host": "web-1"})
    output = stream.getvalue()
    assert "hidden" not in output
    assert "WARNING network_anomaly.detectors: threshold fallback host=web-1" in output


def test_reconfiguring_replaces_the_previous_handler() -> None:
    configure_logging("INFO", "text", stream=io.StringIO())
    configure_logging("DEBUG", "json", stream=io.StringIO())
    logger = logging.getLogger(PACKAGE_LOGGER)
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG
