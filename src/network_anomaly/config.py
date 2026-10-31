"""Validated configuration models and YAML loading.

Configuration and record models derive from :class:`StrictModel`: unknown keys are rejected and
instances are immutable, so a typo in a YAML file fails loudly with the offending key path instead
of being silently ignored.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from network_anomaly.errors import ConfigError

_MAX_SHOWN_INPUT = 80
_FREQUENCY_HINT = "Use a duration such as 30s, 1min, 5min or 1h."


_COMMON_INTERVALS_SECONDS: tuple[int, ...] = (
    *(1, 2, 5, 10, 15, 20, 30),
    *(60, 120, 300, 600, 900, 1200, 1800),
    *(3600, 7200, 10800, 21600, 43200, 86400),
)


def snap_interval(seconds: float, tolerance: float = 0.1) -> pd.Timedelta:
    """Round an observed median spacing to the nearest common sampling interval.

    Clock jitter and gaps move the median spacing slightly (e.g. 301 s for 5-minute data); a
    value within ``tolerance`` of a common interval snaps to it; others are kept to the second.
    """
    nearest = min(_COMMON_INTERVALS_SECONDS, key=lambda common: abs(seconds - common) / common)
    if abs(seconds - nearest) / nearest <= tolerance:
        return pd.Timedelta(seconds=nearest)
    return pd.Timedelta(seconds=round(seconds))


def parse_frequency(text: str) -> pd.Timedelta:
    """Parse a sampling interval such as ``"5min"`` into a positive :class:`pandas.Timedelta`."""
    try:
        frequency = pd.Timedelta(text)
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"invalid sampling interval {text!r}", hint=_FREQUENCY_HINT) from exc
    if frequency <= pd.Timedelta(0):
        raise ConfigError(f"sampling interval must be positive, got {text!r}", hint=_FREQUENCY_HINT)
    return frequency


def interval_text(interval: pd.Timedelta) -> str:
    """Short text for an interval, the inverse of :func:`parse_frequency`: 5min, 1h, 1D, 30s."""
    seconds = interval.total_seconds()
    for unit, size in (("D", 86_400), ("h", 3_600), ("min", 60), ("s", 1)):
        if seconds >= size and seconds % size == 0:
            return f"{int(seconds // size)}{unit}"
    return str(interval)


class StrictModel(BaseModel):
    """Immutable Pydantic model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def format_validation_error(error: ValidationError, source: str) -> str:
    """Turn a Pydantic error into one actionable line per problem, keyed by dotted path."""
    lines = [f"{source}: invalid configuration ({error.error_count()} problem(s))"]
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        if item["type"] == "extra_forbidden":
            lines.append(f"  - {location}: unknown option (check the spelling against the docs)")
            continue
        line = f"  - {location}: {item['msg']}"
        if item["type"] != "missing":
            shown = repr(item.get("input"))
            if len(shown) > _MAX_SHOWN_INPUT:
                shown = shown[: _MAX_SHOWN_INPUT - 3] + "..."
            line += f" (got {shown})"
        lines.append(line)
    return "\n".join(lines)


def parse_config[ModelT: BaseModel](
    data: Any, model: type[ModelT], *, source: str = "<config>"
) -> ModelT:
    """Validate an already-parsed mapping against ``model`` or raise :class:`ConfigError`."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, source)) from exc


def load_yaml_config[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    """Read a YAML file and validate it against ``model``.

    Raises:
        ConfigError: if the file is missing or unreadable, is not valid YAML, is not a mapping, or
            fails validation. Messages name the file, and the line or key path where possible.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise ConfigError(f"{path}: invalid YAML{where}: {problem}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path}: the top level must be a mapping of option names to values, "
            f"not {type(data).__name__}"
        )
    return parse_config(data, model, source=str(path))
