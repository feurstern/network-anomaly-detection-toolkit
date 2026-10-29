"""Exception hierarchy shared by the library and the CLI.

Every failure that can reach a user is a :class:`ToolkitError` subclass with a stable process exit
code and, where possible, a hint explaining how to fix the problem. Anything else is a bug and maps
to :attr:`ExitCode.INTERNAL`.
"""

from enum import IntEnum
from typing import ClassVar


class ExitCode(IntEnum):
    """Process exit codes returned by the ``network-anomaly`` CLI."""

    OK = 0
    INTERNAL = 1
    USAGE = 2
    DATA_VALIDATION = 3
    CONFIG = 4
    ARTIFACT = 5
    LEAKAGE = 6


class ToolkitError(Exception):
    """Base class for expected, user-facing failures."""

    exit_code: ClassVar[ExitCode] = ExitCode.INTERNAL

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return self.message if self.hint is None else f"{self.message}\nhint: {self.hint}"


class ConfigError(ToolkitError):
    """A configuration file or command-line option is invalid."""

    exit_code = ExitCode.CONFIG


class DataValidationError(ToolkitError):
    """Input data violates the documented schema."""

    exit_code = ExitCode.DATA_VALIDATION


class ArtifactError(ToolkitError):
    """A file or saved model could not be read or written, or is incompatible with this version."""

    exit_code = ExitCode.ARTIFACT


class LeakageError(ToolkitError):
    """An operation would let labels, test-period or future data influence fitting."""

    exit_code = ExitCode.LEAKAGE


class NotFittedError(ToolkitError):
    """A component was used before ``fit`` was called."""

    exit_code = ExitCode.INTERNAL
