"""Validated client runtimes used by provider adapters."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Callable


REQUIRED_YFINANCE_VERSION = "1.5.1"
REQUIRED_CURL_CFFI_VERSION = "0.15.0"


@dataclass(frozen=True, slots=True)
class ClientInfo:
    name: str
    version: str


def validate_yfinance_runtime(
    *,
    yfinance_module=None,
    package_version: Callable[[str], str] = importlib_metadata.version,
) -> ClientInfo:
    """Fail before any provider call when the Conda transport is stale."""

    if yfinance_module is None:
        import yfinance as yfinance_module

    actual_yfinance = str(getattr(yfinance_module, "__version__", "")).strip() or "unknown"
    try:
        actual_curl_cffi = str(package_version("curl-cffi")).strip()
    except importlib_metadata.PackageNotFoundError:
        actual_curl_cffi = "not-installed"
    if (
        actual_yfinance != REQUIRED_YFINANCE_VERSION
        or actual_curl_cffi != REQUIRED_CURL_CFFI_VERSION
    ):
        raise RuntimeError(
            "Yahoo acquisition requires yfinance "
            f"{REQUIRED_YFINANCE_VERSION} and curl-cffi "
            f"{REQUIRED_CURL_CFFI_VERSION} in the "
            "Conda environment `finance`; found "
            f"yfinance={actual_yfinance}, curl-cffi={actual_curl_cffi}. "
            "Update the environment with `conda env update -n finance -f "
            "environment.yml` before retrying."
        )
    return ClientInfo(
        name="yfinance",
        version=actual_yfinance,
    )


def prepare_yfinance(
    cache_dir: Path,
    *,
    yfinance_module=None,
    package_version: Callable[[str], str] = importlib_metadata.version,
) -> ClientInfo:
    """Validate versions, then configure the single project runtime cache."""

    if yfinance_module is None:
        import yfinance as yfinance_module

    info = validate_yfinance_runtime(
        yfinance_module=yfinance_module,
        package_version=package_version,
    )
    setter = getattr(yfinance_module, "set_tz_cache_location", None)
    if not callable(setter):
        raise RuntimeError("The installed yfinance has no public cache-location API")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    setter(str(cache_dir))
    return info


__all__ = [
    "ClientInfo",
    "REQUIRED_CURL_CFFI_VERSION",
    "REQUIRED_YFINANCE_VERSION",
    "prepare_yfinance",
    "validate_yfinance_runtime",
]
