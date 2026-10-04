"""Coordinate deterministic, offline live preparation stages."""

from __future__ import annotations

from .config import DEFAULT_CONFIG
from .analysis_data import validate_prepared_sector_contract
from .preparation_builders import prepare_brinson_data, prepare_core_data
from .preparation_artifacts import (
    validate_preparation_manifest,
    write_preparation_manifest,
)


def run_preparation(stage: str, *, check: bool = False) -> None:
    """Build or validate one live preparation stage without network access."""
    if stage not in {"core", "brinson", "all"}:
        raise ValueError(f"Unsupported preparation stage: {stage!r}")

    include_brinson = stage in {"brinson", "all"}
    if check:
        validate_preparation_manifest(
            DEFAULT_CONFIG.paths,
            include_brinson=include_brinson,
            check_raw=True,
        )
        validate_prepared_sector_contract(DEFAULT_CONFIG.paths)
        return

    if stage in {"core", "all"}:
        prepare_core_data(DEFAULT_CONFIG)
        write_preparation_manifest(
            DEFAULT_CONFIG.paths,
            include_brinson=False,
        )

    validate_preparation_manifest(
        DEFAULT_CONFIG.paths,
        include_brinson=False,
        check_raw=True,
    )
    validate_prepared_sector_contract(DEFAULT_CONFIG.paths)

    if stage == "core":
        return

    prepare_brinson_data(DEFAULT_CONFIG)
    write_preparation_manifest(
        DEFAULT_CONFIG.paths,
        include_brinson=True,
    )
    validate_preparation_manifest(
        DEFAULT_CONFIG.paths,
        include_brinson=True,
        check_raw=True,
    )


__all__ = ["run_preparation"]
