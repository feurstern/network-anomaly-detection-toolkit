"""Structured logging for library code and the CLI.

Library modules log through ``logging.getLogger(__name__)`` and never install handlers themselves;
the CLI calls :func:`configure_logging` once. Fields passed via ``extra={...}`` are rendered as
``key=value`` pairs in text mode and as JSON members in JSON mode, so logs stay machine-readable
without an extra logging framework.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Literal, TextIO

LogFormat = Literal["text", "json"]

PACKAGE_LOGGER = "network_anomaly"

_RESERVED_ATTRIBUTES = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
}


def _extra_fields(record: logging.LogRecord) -> dict[str, object]:
    """Return the user-supplied ``extra`` fields attached to a log record."""
    return {
        key: value
        for key, value in vars(record).items()
        if key not in _RESERVED_ATTRIBUTES and not key.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Render records as ``LEVEL logger: message key=value ...`` for terminals."""

    def format(self, record: logging.LogRecord) -> str:
        line = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        extras = _extra_fields(record)
        if extras:
            line += " " + " ".join(f"{key}={value}" for key, value in extras.items())
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(
    level: str = "INFO", fmt: LogFormat = "text", stream: TextIO | None = None
) -> None:
    """Send this package's logs to ``stream`` (default: stderr) in the requested format.

    Only the package logger is configured, so logging set up by an embedding application is left
    untouched. Calling the function again replaces the previous handler.
    """
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
