"""Reproducible, explainable anomaly detection for infrastructure metrics and network flows.

Anomaly scores produced by this package are signals for investigation. They are not calibrated
probabilities and they are not evidence of an attack.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("network-anomaly-toolkit")
# Only reachable when the package is imported from a source tree that was never installed.
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
