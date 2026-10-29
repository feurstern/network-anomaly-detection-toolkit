"""Strict CSV reading: every cell is read as text and parsed explicitly during validation.

Reading everything as strings means pandas never guesses types, so malformed values are reported by
validation (with file line numbers) instead of being silently converted to NaN or to objects.
"""

import csv
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from network_anomaly.errors import ArtifactError, DataValidationError

logger = logging.getLogger(__name__)

# Cell contents that mean "no observation". Anything else must parse exactly.
MISSING_TOKENS: tuple[str, ...] = ("", "NA", "N/A", "NaN", "nan", "null", "NULL", "None")

_MAX_REPORTED_LINES = 5


@dataclass(frozen=True)
class RawTable:
    """CSV contents as whitespace-stripped strings, plus provenance for error messages.

    ``line_numbers[i]`` is the 1-based line in the source file holding data row ``i``.
    """

    source: str
    frame: pd.DataFrame
    line_numbers: npt.NDArray[np.int64]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self.frame.columns)


def _scan_records(path: Path) -> tuple[list[str], npt.NDArray[np.int64]]:
    """Return the header and the line number of every data record, rejecting ragged rows.

    pandas silently pads rows that have too few fields, so field counts are checked here with the
    standard ``csv`` module (which also yields exact line numbers). Blank lines are skipped, as
    pandas does.
    """
    record_lines: list[int] = []
    ragged: list[int] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise DataValidationError(f"{path}: the file is empty")
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                ragged.append(reader.line_num)
            record_lines.append(reader.line_num)
    if ragged:
        shown = ", ".join(str(line) for line in ragged[:_MAX_REPORTED_LINES])
        raise DataValidationError(
            f"{path}: {len(ragged)} row(s) do not have exactly {len(header)} fields "
            f"(lines {shown}{', ...' if len(ragged) > _MAX_REPORTED_LINES else ''})",
            hint="Every row must have exactly as many comma-separated fields as the header.",
        )
    return [name.strip() for name in header], np.asarray(record_lines, dtype=np.int64)


def read_csv_strings(path: Path) -> RawTable:
    """Read a CSV file with a header row, keeping every cell as a whitespace-stripped string.

    Raises:
        ArtifactError: if the file does not exist or cannot be read.
        DataValidationError: for structural problems that make the table unreadable (no header,
            empty or duplicate column names, ragged rows, non-UTF-8 text, no data rows).
    """
    if not path.is_file():
        raise ArtifactError(f"input file not found: {path}")
    try:
        header, record_lines = _scan_records(path)
        duplicated = sorted(name for name, count in Counter(header).items() if count > 1)
        if duplicated:
            raise DataValidationError(
                f"{path}: duplicate column names in the header: {', '.join(duplicated)}",
                hint="Rename or drop the repeated columns.",
            )
        if any(name == "" for name in header):
            raise DataValidationError(f"{path}: the header contains an empty column name")
        if len(record_lines) == 0:
            raise DataValidationError(f"{path}: the file has a header but no data rows")
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError as exc:
        raise DataValidationError(f"{path}: not UTF-8 encoded text ({exc.reason})") from exc
    except (pd.errors.ParserError, csv.Error) as exc:
        raise DataValidationError(f"{path}: malformed CSV: {exc}") from exc
    except OSError as exc:
        raise ArtifactError(f"cannot read {path}: {exc}") from exc

    if len(frame) != len(record_lines):
        raise DataValidationError(
            f"{path}: could not align {len(frame)} parsed rows with {len(record_lines)} records",
            hint="Check for unbalanced quotes or embedded line breaks inside fields.",
        )
    frame.columns = pd.Index(header)
    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    logger.debug("read csv", extra={"path": str(path), "rows": len(frame), "columns": len(header)})
    return RawTable(source=str(path), frame=frame, line_numbers=record_lines)
