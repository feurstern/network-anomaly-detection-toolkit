"""What a benchmark ran on - recorded so that results can be compared and reproduced.

Everything is read locally (``/proc``, ``/sys``, package metadata, ``git``); nothing is sent
anywhere. Peak memory is the process's peak resident set size: on Linux ``VmHWM`` from
``/proc/self/status``, reset before a measurement through ``/proc/self/clear_refs``; elsewhere the
lifetime ``ru_maxrss``. The method used is recorded with every number.
"""

import hashlib
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from threadpoolctl import threadpool_info

PACKAGES = (
    "network-anomaly-toolkit",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "pydantic",
    "joblib",
    "threadpoolctl",
)


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _field(text: str | None, key: str) -> str | None:
    for line in (text or "").splitlines():
        name, _, value = line.partition(":")
        if name.strip() == key:
            return value.strip()
    return None


def cpu_model() -> str:
    return _field(_read("/proc/cpuinfo"), "model name") or platform.processor() or "unknown"


def total_memory_bytes() -> int | None:
    value = _field(_read("/proc/meminfo"), "MemTotal")
    return int(value.split()[0]) * 1024 if value else None


def cpu_governor() -> str | None:
    text = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return text.strip() if text else None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def project_root() -> Path | None:
    """The source checkout this package runs from, if any (it holds ``uv.lock``)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "uv.lock").is_file():
            return parent
    return None


def git_state(root: Path) -> dict[str, Any] | None:
    """Commit and dirty flag of the checkout, or None outside a git repository."""
    git = shutil.which("git")
    if git is None:
        return None

    def run(*args: str) -> str | None:
        completed = subprocess.run(
            [git, "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return None
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict[str, Any]:
    """Hardware, operating system, interpreter, packages, thread pools and source revision."""
    root = project_root()
    lock = root / "uv.lock" if root is not None else None
    return {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": {
            "model": cpu_model(),
            "logical_cores": os.cpu_count(),
            "governor": cpu_governor(),
        },
        "memory_bytes": total_memory_bytes(),
        "python": {"version": platform.python_version(), "implementation": sys.implementation.name},
        "packages": package_versions(),
        "threadpools": [
            {k: info.get(k) for k in ("user_api", "internal_api", "num_threads", "version")}
            for info in threadpool_info()
        ],
        "uv_lock_sha256": _sha256(lock) if lock is not None else None,
        "git": git_state(root) if root is not None else None,
    }


def reset_peak_rss() -> bool:
    """Reset the kernel's peak-RSS counter for this process (Linux); False if unavailable."""
    try:
        Path("/proc/self/clear_refs").write_text("5", encoding="ascii")
    except OSError:
        return False
    return True


def current_rss_bytes() -> int | None:
    value = _field(_read("/proc/self/status"), "VmRSS")
    return int(value.split()[0]) * 1024 if value else None


def peak_rss_bytes() -> tuple[int, str]:
    """Peak resident memory of this process and the method used to read it."""
    value = _field(_read("/proc/self/status"), "VmHWM")
    if value:
        return int(value.split()[0]) * 1024, "VmHWM"
    try:
        import resource  # POSIX only
    except ImportError:
        return 0, "unavailable"
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in kilobytes on Linux and in bytes on macOS.
    return (peak if sys.platform == "darwin" else peak * 1024), "ru_maxrss"
