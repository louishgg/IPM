"""Filesystem and reporting primitives shared by every acquisition workflow."""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator

from .contracts import utc_timestamp

try:
    import fcntl
except ImportError:  # pragma: no cover - supported deployment is macOS/Linux.
    fcntl = None


class AcquisitionLockedError(RuntimeError):
    """Another process owns a dataset's acquisition checkpoint."""


def _safe_segment(value: str, *, label: str) -> str:
    segment = str(value).strip().lower().replace("_", "-")
    if not segment or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-." for char in segment):
        raise ValueError(f"{label} contains unsafe path characters: {value!r}")
    if segment in {".", ".."}:
        raise ValueError(f"{label} is not a valid path segment: {value!r}")
    return segment


@dataclass(frozen=True, slots=True)
class AcquisitionRuntimePaths:
    """All regenerable acquisition state, rooted outside durable raw data."""

    project_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())

    @property
    def root(self) -> Path:
        return self.project_root / "runtime" / "acquisition"

    @property
    def yfinance_cache(self) -> Path:
        return self.root / "yfinance"

    @property
    def locks(self) -> Path:
        return self.root / "locks"

    @property
    def sector_history(self) -> Path:
        return self.root / "sector-history"

    @property
    def sector_history_snapshots(self) -> Path:
        return self.sector_history / "snapshots"

    @property
    def sector_history_statuses(self) -> Path:
        return self.sector_history / "statuses"

    @property
    def sector_history_publications(self) -> Path:
        return self.sector_history / "publications"

    @property
    def sector_history_pending_publish(self) -> Path:
        return self.sector_history / "pending_publish.json"

    def lock_path(self, scope: str, dataset: str) -> Path:
        parts = (
            _safe_segment(scope, label="scope"),
            _safe_segment(dataset, label="dataset"),
        )
        return self.locks / ("-".join(parts) + ".lock")

class AcquisitionReporter:
    """Write timestamped acquisition progress to the active terminal."""

    def __init__(
        self,
        *,
        terminal: IO[str] | None = None,
        clock=None,
    ) -> None:
        self.terminal = terminal if terminal is not None else sys.stderr
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()

    def __call__(self, message: str) -> None:
        stamp = utc_timestamp(self.clock())
        text = " ".join(str(message).splitlines()).strip()
        line = f"{stamp} {text}"
        with self._lock:
            self.terminal.write(line + "\n")
            self.terminal.flush()


@contextmanager
def acquisition_lock(path: Path) -> Iterator[None]:
    """Hold one non-blocking advisory lock for a dataset checkpoint."""

    if fcntl is None:
        raise RuntimeError(
            "Acquisition locking requires fcntl; the supported runtime is macOS/Linux."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AcquisitionLockedError(
                f"Another acquisition process owns the checkpoint lock: {path}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


__all__ = [
    "AcquisitionLockedError",
    "AcquisitionReporter",
    "AcquisitionRuntimePaths",
    "acquisition_lock",
]
