"""Backtest registration facade for the centralized acquisition CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from data_acquisition.contracts import CommandRequest
from data_acquisition.sector_acquisition_execution import execute_sector_acquisition
from data_acquisition.sector_acquisition_planning import dry_run_sector_acquisition

from .acquisition_planning import (
    backtest_sector_requirements,
)
from .acquisition_execution import (
    acquire_backtest_benchmark,
    acquire_backtest_prices,
    acquire_backtest_shares,
)


def acquire_backtest_sectors(request: CommandRequest) -> int:
    if request.dry_run:
        dry_run_sector_acquisition(request, backtest_sector_requirements)
        return 0
    return execute_sector_acquisition(request, backtest_sector_requirements)


def acquire_backtest_all(request: CommandRequest) -> int:
    """Run every backtest acquisition handler in the documented order."""
    handlers = (
        (acquire_backtest_prices, replace(request, tickers_file=None)),
        (acquire_backtest_benchmark, replace(request, tickers_file=None)),
        (acquire_backtest_sectors, replace(request, tickers_file=None)),
        (acquire_backtest_shares, request),
    )
    for handler, handler_request in handlers:
        result = handler(handler_request)
        if result:
            return result
    return 0


def acquisition_handlers() -> dict[str, Callable[[CommandRequest], int]]:
    """Return every fixed backtest handler for the root acquisition CLI."""
    return {
        "prices": acquire_backtest_prices,
        "benchmark": acquire_backtest_benchmark,
        "sectors": acquire_backtest_sectors,
        "shares": acquire_backtest_shares,
        "all": acquire_backtest_all,
    }


__all__ = [
    "acquire_backtest_all",
    "acquire_backtest_sectors",
    "acquisition_handlers",
]
