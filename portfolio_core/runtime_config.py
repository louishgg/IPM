"""Deterministic process-wide runtime setup shared by both domains."""

from __future__ import annotations

import os
import tempfile
import warnings


_MATPLOTLIB_CACHE_SUBDIRECTORY = "matplotlib"
_IGNORED_WARNING_PATTERN = r".*Timestamp\.utcnow is deprecated.*"


def apply_runtime_settings() -> None:
    """Apply deterministic plotting and warning settings."""
    mpl_config_dir = os.path.join(
        tempfile.gettempdir(),
        _MATPLOTLIB_CACHE_SUBDIRECTORY,
    )
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
    warnings.filterwarnings(
        "ignore",
        message=_IGNORED_WARNING_PATTERN,
        category=Warning,
    )


__all__ = ["apply_runtime_settings"]
