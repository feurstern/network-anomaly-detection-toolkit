"""Verified downloads of pinned public files.

This is the only part of the toolkit that uses the network, and only when the user runs
``network-anomaly datasets download``. It sends nothing but the HTTPS request for the file.
Every file is checked against its pinned SHA-256 (or git blob id) before it is kept; a partial or
altered download is deleted, never used.
"""

import hashlib
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from network_anomaly import __version__
from network_anomaly.errors import ArtifactError

logger = logging.getLogger(__name__)

CHUNK = 1 << 20
TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class RemoteFile:
    """A file to fetch: where from, where to (relative to the dataset directory), how to check."""

    url: str
    target: str
    size: int
    sha256: str | None = None
    git_sha1: str | None = None

    def __post_init__(self) -> None:
        if not self.url.startswith("https://"):
            raise ValueError(f"only HTTPS downloads are allowed: {self.url}")
        if self.sha256 is None and self.git_sha1 is None:
            raise ValueError(f"{self.target}: a pinned checksum is required")


class _Digest:
    """SHA-256 or git blob SHA-1 (``sha1("blob <size>\\0" + content)``), fed chunk by chunk."""

    def __init__(self, remote: RemoteFile) -> None:
        self.remote = remote
        if remote.sha256 is not None:
            self.hasher, self.expected = hashlib.sha256(), remote.sha256
        else:
            self.hasher = hashlib.sha1(usedforsecurity=False)
            self.hasher.update(f"blob {remote.size}\0".encode())
            self.expected = str(remote.git_sha1)
        self.size = 0

    def update(self, chunk: bytes) -> None:
        self.size += len(chunk)
        self.hasher.update(chunk)

    def matches(self) -> bool:
        return self.size == self.remote.size and self.hasher.hexdigest() == self.expected


def verify(path: Path, remote: RemoteFile) -> bool:
    """Whether ``path`` holds exactly the pinned content."""
    if not path.is_file() or path.stat().st_size != remote.size:
        return False
    digest = _Digest(remote)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.matches()


def fetch(
    remote: RemoteFile, directory: Path, progress: Callable[[str], None] | None = None
) -> Path:
    """Download ``remote`` into ``directory`` unless a verified copy is already there."""
    target = directory / remote.target
    if verify(target, remote):
        if progress:
            progress(f"  {remote.target}: already downloaded and verified")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = _Digest(remote)
    request = urllib.request.Request(
        remote.url, headers={"User-Agent": f"network-anomaly-toolkit/{__version__}"}
    )
    if progress:
        size = (
            f"{remote.size / 2**20:,.1f} MiB"
            if remote.size >= 2**20
            else f"{remote.size / 2**10:,.0f} KiB"
        )
        progress(f"  {remote.target}: downloading {size}")
    try:
        with (
            urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response,
            partial.open("wb") as handle,
        ):
            for chunk in iter(lambda: response.read(CHUNK), b""):
                digest.update(chunk)
                handle.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise ArtifactError(
            f"download of {remote.url} failed: {exc}",
            hint="Check the network connection and retry; verified files are kept.",
        ) from exc
    if not digest.matches():
        partial.unlink(missing_ok=True)
        raise ArtifactError(
            f"{remote.url} does not match its pinned checksum ({digest.size:,} bytes received, "
            f"{remote.size:,} expected)",
            hint="The source may have changed; do not use the file. Please report the issue.",
        )
    partial.replace(target)
    logger.info("downloaded", extra={"file": remote.target, "bytes": remote.size})
    return target
