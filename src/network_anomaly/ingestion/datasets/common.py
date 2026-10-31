"""Shared pieces of the public-dataset adapters."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from network_anomaly.schemas import TIMESTAMP_COLUMN


@dataclass
class Prepared:
    """What ``prepare`` produced, what it had to assume, and how to use the result."""

    dataset: str
    files: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Prepared {self.dataset}: {len(self.files)} file(s)"]
        lines += [f"  {path}" for path in self.files[:10]]
        if len(self.files) > 10:
            lines.append(f"  ... and {len(self.files) - 10} more")
        lines += [f"note: {note}" for note in self.notes]
        if self.commands:
            lines.append("Next steps:")
            lines += [f"  {command}" for command in self.commands]
        return "\n".join(lines)


def iso_utc(values: pd.Series) -> pd.Series:
    """ISO 8601 UTC strings; sub-second digits only when present."""
    utc = values.dt.tz_convert("UTC")
    fractional = bool((utc.dropna().dt.microsecond != 0).any())
    return utc.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ" if fractional else "%Y-%m-%dT%H:%M:%SZ")


def write_table(frame: pd.DataFrame, path: Path, *, append: bool = False) -> Path:
    """Write a canonical CSV with ISO 8601 UTC timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out[TIMESTAMP_COLUMN] = iso_utc(out[TIMESTAMP_COLUMN])
    out.to_csv(
        path,
        index=False,
        mode="a" if append else "w",
        header=not append,
        lineterminator="\n",
        na_rep="",
    )
    return path
