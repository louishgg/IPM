"""Live registration facade for the centralized acquisition CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from data_acquisition.contracts import CommandRequest

from .acquisition_planning import (
    BENCHMARK_DATASET,
    PRICE_DATASET,
    SHARES_DATASET,
)
from .acquisition_execution import execute


def acquire_live_all(command_request: CommandRequest) -> int:
    """Run the documented live acquisition sequence and stop on failure."""
    for dataset in (PRICE_DATASET, BENCHMARK_DATASET, "sectors", SHARES_DATASET):
        ticker_selection = (
            command_request.tickers_file if dataset == SHARES_DATASET else None
        )
        result = execute(
            replace(
                command_request,
                dataset=dataset,
                tickers_file=ticker_selection,
            )
        )
        if result:
            return result
    return 0


def acquisition_handlers() -> dict[str, Callable[[CommandRequest], int]]:
    """Return every fixed live handler for the root acquisition CLI."""
    return {
        PRICE_DATASET: execute,
        BENCHMARK_DATASET: execute,
        "sectors": execute,
        SHARES_DATASET: execute,
        "all": acquire_live_all,
    }


__all__ = ["acquire_live_all", "acquisition_handlers"]
