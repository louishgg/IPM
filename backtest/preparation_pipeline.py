"""Coordinate deterministic, offline backtest preparation stages."""

from .shares_preparation import (
    load_prepared_shares,
    prepare_shares_dataset,
)
from .config import DEFAULT_CONFIG
from .preparation_builders import (
    prepare_core_data,
    validate_prepared_data,
)
from .preparation_artifacts import write_preparation_manifest


def run_preparation(stage: str, *, check: bool = False) -> None:
    """Build or validate one preparation stage without network access."""
    if stage not in {"core", "brinson", "all"}:
        raise ValueError(f"Unsupported preparation stage: {stage!r}")

    if check:
        validate_prepared_data(
            DEFAULT_CONFIG.paths,
            DEFAULT_CONFIG.market,
            DEFAULT_CONFIG.benchmark,
            include_brinson=stage in {"brinson", "all"},
            check_raw=True,
        )
        if stage in {"brinson", "all"}:
            load_prepared_shares(
                DEFAULT_CONFIG.paths.shares,
                DEFAULT_CONFIG.brinson,
            )
        return

    if stage in {"core", "all"}:
        prepare_core_data(
            DEFAULT_CONFIG.market,
            DEFAULT_CONFIG.paths,
            DEFAULT_CONFIG.benchmark,
        )
        write_preparation_manifest(
            DEFAULT_CONFIG.paths,
            include_brinson=False,
        )

    backtest_data = validate_prepared_data(
        DEFAULT_CONFIG.paths,
        DEFAULT_CONFIG.market,
        DEFAULT_CONFIG.benchmark,
        include_brinson=False,
        check_raw=True,
    )

    if stage in {"brinson", "all"}:
        prepare_shares_dataset(backtest_data, DEFAULT_CONFIG)
        write_preparation_manifest(
            DEFAULT_CONFIG.paths,
            include_brinson=True,
        )
        validate_prepared_data(
            DEFAULT_CONFIG.paths,
            DEFAULT_CONFIG.market,
            DEFAULT_CONFIG.benchmark,
            include_brinson=True,
            check_raw=True,
        )
        load_prepared_shares(
            DEFAULT_CONFIG.paths.shares,
            DEFAULT_CONFIG.brinson,
        )


__all__ = ["run_preparation"]
