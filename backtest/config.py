"""Immutable typed configuration for the IPM backtest workflow."""

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from portfolio_core.accounting_config import PortfolioAccountingConfig
from portfolio_core.dates import month_end_index
from portfolio_core.strategies import Strategy

from .paths import DEFAULT_PATHS, BacktestPaths


BACKTEST_WARMUP_MONTHS = 12
BACKTEST_BASELINE_END_DATE = date(2024, 1, 31)


@dataclass(frozen=True, slots=True)
class BacktestMarketConfig:
    """Market-data analysis window."""

    start_date: date = date(2014, 1, 31)
    end_date: date = date(2026, 1, 31)

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("data start_date cannot follow end_date")


@dataclass(frozen=True, slots=True)
class BacktestEvaluationConfig:
    """Development, expanding-validation, and test-window boundaries."""

    signal_warmup_start_date: date = date(2014, 1, 31)
    signal_warmup_end_date: date = date(2014, 12, 31)
    initial_research_start_date: date = date(2015, 1, 31)
    initial_research_end_date: date = date(2017, 12, 31)
    validation_start_date: date = date(2018, 1, 31)
    validation_end_date: date = date(2023, 12, 31)
    test_start_date: date = date(2024, 1, 31)

    def __post_init__(self) -> None:
        if not (
            self.signal_warmup_start_date
            <= self.signal_warmup_end_date
            < self.initial_research_start_date
            <= self.initial_research_end_date
            < self.validation_start_date
            <= self.validation_end_date
            < self.test_start_date
        ):
            raise ValueError(
                "evaluation windows must be ordered as warm-up, initial "
                "research, validation, and test window"
            )


@dataclass(frozen=True, slots=True)
class BacktestBenchmarkConfig:
    """Required coverage window for the Yahoo-derived benchmark artifact."""

    coverage_start_date: date = date(2014, 1, 1)
    coverage_end_date: date = date(2026, 1, 31)

    def __post_init__(self) -> None:
        if self.coverage_start_date > self.coverage_end_date:
            raise ValueError("benchmark coverage start cannot follow its end")

    @property
    def required_dates(self) -> pd.DatetimeIndex:
        """Inclusive month ends required from the benchmark artifact."""
        start = pd.Timestamp(self.coverage_start_date) + pd.offsets.MonthEnd(0)
        end = pd.Timestamp(self.coverage_end_date) + pd.offsets.MonthEnd(0)
        return month_end_index(start, end)


@dataclass(frozen=True, slots=True)
class BacktestBrinsonConfig:
    """Brinson attribution window and Yahoo acquisition lookback."""

    start_date: date = date(2024, 1, 31)
    end_date: date = date(2026, 1, 31)
    yahoo_lookback_months: int = 12

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("Brinson start_date cannot follow end_date")
        if self.yahoo_lookback_months <= 0:
            raise ValueError("yahoo_lookback_months must be positive")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Immutable shared configuration with optional analysis strategy."""

    paths: BacktestPaths = field(default_factory=lambda: DEFAULT_PATHS)
    market: BacktestMarketConfig = field(default_factory=BacktestMarketConfig)
    benchmark: BacktestBenchmarkConfig = field(
        default_factory=BacktestBenchmarkConfig
    )
    brinson: BacktestBrinsonConfig = field(default_factory=BacktestBrinsonConfig)
    strategy: Strategy | None = None
    accounting: PortfolioAccountingConfig = field(
        default_factory=PortfolioAccountingConfig
    )
    evaluation: BacktestEvaluationConfig = field(
        default_factory=BacktestEvaluationConfig
    )

    def __post_init__(self) -> None:
        if self.strategy is not None and not isinstance(self.strategy, Strategy):
            raise TypeError(
                "backtest strategy must implement the canonical Strategy contract"
            )
        if self.strategy is not None and (
            self.paths.strategy_id != self.strategy.strategy_id
        ):
            raise ValueError(
                "Backtest output strategy ID must match the configured strategy"
            )
        if not (
            self.market.start_date
            <= self.evaluation.signal_warmup_start_date
            <= self.evaluation.validation_end_date
            <= self.market.end_date
        ):
            raise ValueError("development windows must lie inside the data window")
        if not (
            self.market.start_date
            <= self.evaluation.test_start_date
            <= self.market.end_date
        ):
            raise ValueError("test split must lie inside the data window")
        if not (
            self.market.start_date
            <= self.brinson.start_date
            <= self.brinson.end_date
            <= self.market.end_date
        ):
            raise ValueError("Brinson window must lie inside the data window")
        if self.benchmark.coverage_start_date > self.market.start_date:
            raise ValueError("benchmark coverage must start no later than market data")
        if self.benchmark.coverage_end_date < self.market.end_date:
            raise ValueError("benchmark coverage must include the market-data end")


DEFAULT_CONFIG = BacktestConfig()
