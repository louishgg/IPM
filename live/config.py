"""Immutable configuration for the historical live strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from portfolio_core.accounting_config import PortfolioAccountingConfig
from portfolio_core.strategies import Strategy

from .paths import DEFAULT_PATHS, LivePaths


@dataclass(frozen=True, slots=True)
class DecisionPeriod:
    """One causal signal, sizing, execution, and valuation interval."""

    rebalance_id: str
    membership_effective_date: date
    signal_cutoff: date
    sizing_date: date
    execution_date: date
    valuation_end: date
    sizing_field: str = "Close"
    execution_field: str = "Open"
    valuation_field: str = "Open"

    def __post_init__(self) -> None:
        if not self.rebalance_id.strip():
            raise ValueError("rebalance_id cannot be empty")
        for name, value in (
            ("sizing_field", self.sizing_field),
            ("execution_field", self.execution_field),
            ("valuation_field", self.valuation_field),
        ):
            if value not in {"Open", "Close"}:
                raise ValueError(f"{name} must be Open or Close")
        if not (
            self.membership_effective_date
            <= self.signal_cutoff
            <= self.sizing_date
            <= self.execution_date
            < self.valuation_end
        ):
            raise ValueError(
                "Decision dates must satisfy membership <= signal <= sizing "
                "<= execution < valuation"
            )
        # Signals use the completed close. A same-date open cannot use them.
        field_order = {"Open": 0, "Close": 1}
        if not (
            (self.signal_cutoff, 1)
            <= (self.sizing_date, field_order[self.sizing_field])
            <= (self.execution_date, field_order[self.execution_field])
        ):
            raise ValueError("Decision Open/Close fields violate causal ordering")


def _default_schedule() -> tuple[DecisionPeriod, ...]:
    return (
        DecisionPeriod(
            rebalance_id="R1",
            membership_effective_date=date(2026, 2, 9),
            signal_cutoff=date(2026, 2, 27),
            sizing_date=date(2026, 2, 27),
            execution_date=date(2026, 3, 2),
            valuation_end=date(2026, 4, 1),
        ),
        DecisionPeriod(
            rebalance_id="R2",
            membership_effective_date=date(2026, 3, 23),
            signal_cutoff=date(2026, 3, 31),
            sizing_date=date(2026, 3, 31),
            execution_date=date(2026, 4, 1),
            valuation_end=date(2026, 5, 1),
        ),
        DecisionPeriod(
            rebalance_id="R3",
            membership_effective_date=date(2026, 4, 9),
            signal_cutoff=date(2026, 4, 30),
            sizing_date=date(2026, 4, 30),
            execution_date=date(2026, 5, 1),
            valuation_end=date(2026, 5, 6),
            valuation_field="Close",
        ),
    )


@dataclass(frozen=True, slots=True)
class LiveMarketConfig:
    """Active historical Yahoo price window and competition dates."""

    competition_start: date = date(2026, 2, 13)
    competition_end: date = date(2026, 5, 6)
    download_start: date = date(2023, 1, 1)
    download_end: date = date(2026, 5, 7)  # Yahoo's end bound is exclusive.
    # Larger revisions require review; a vanished cached observation always does.
    price_replacement_review_threshold: float = 0.01

    def __post_init__(self) -> None:
        if self.competition_start > self.competition_end:
            raise ValueError("competition_start cannot follow competition_end")
        if self.download_start > self.competition_start:
            raise ValueError("price history must begin before the competition")
        if self.download_end <= self.competition_end:
            raise ValueError("Yahoo's exclusive end must cover competition_end")
        if not 0 <= self.price_replacement_review_threshold < 1:
            raise ValueError("price replacement review threshold must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class LiveBenchmarkConfig:
    """Yahoo S&P 500 Total Return acquisition window."""

    ticker: str = "^SP500TR"
    download_start: date = date(2026, 2, 12)
    download_end: date = date(2026, 5, 7)  # exclusive

    def __post_init__(self) -> None:
        if not self.ticker.strip():
            raise ValueError("benchmark ticker cannot be empty")
        if self.download_start >= self.download_end:
            raise ValueError("benchmark start must precede its exclusive end")


@dataclass(frozen=True, slots=True)
class LiveBrinsonConfig:
    """Yahoo shares acquisition lookback for live attribution."""

    yahoo_lookback_months: int = 12

    def __post_init__(self) -> None:
        if self.yahoo_lookback_months <= 0:
            raise ValueError("yahoo_lookback_months must be positive")


@dataclass(frozen=True, slots=True)
class LiveConfig:
    """Immutable shared configuration with optional analysis strategy."""

    paths: LivePaths = field(default_factory=lambda: DEFAULT_PATHS)
    market: LiveMarketConfig = field(default_factory=LiveMarketConfig)
    benchmark: LiveBenchmarkConfig = field(default_factory=LiveBenchmarkConfig)
    brinson: LiveBrinsonConfig = field(default_factory=LiveBrinsonConfig)
    strategy: Strategy | None = None
    accounting: PortfolioAccountingConfig = field(
        default_factory=PortfolioAccountingConfig
    )
    schedule: tuple[DecisionPeriod, ...] = field(default_factory=_default_schedule)

    def __post_init__(self) -> None:
        if not self.schedule:
            raise ValueError("schedule cannot be empty")
        ids = [period.rebalance_id for period in self.schedule]
        if len(ids) != len(set(ids)):
            raise ValueError("schedule rebalance IDs must be unique")
        if (
            tuple(sorted(self.schedule, key=lambda item: item.execution_date))
            != self.schedule
        ):
            raise ValueError("schedule must be ordered by execution date")
        if self.schedule[0].sizing_date < self.market.competition_start:
            raise ValueError("first sizing cannot precede competition_start")
        if self.schedule[-1].valuation_end != self.market.competition_end:
            raise ValueError("last valuation must equal competition_end")
        for previous, current in zip(self.schedule, self.schedule[1:]):
            if (previous.valuation_end, previous.valuation_field) != (
                current.execution_date, current.execution_field
            ):
                raise ValueError("Adjacent live periods must share a valuation/execution boundary")
            field_order = {"Open": 0, "Close": 1}
            if (current.sizing_date, field_order[current.sizing_field]) < (
                previous.execution_date, field_order[previous.execution_field]
            ):
                raise ValueError("Sizing cannot precede the previous execution checkpoint")
        if self.strategy is not None and not isinstance(self.strategy, Strategy):
            raise TypeError(
                "live strategy must implement the canonical Strategy contract"
            )
        if self.strategy is not None and (
            self.paths.strategy_id != self.strategy.strategy_id
        ):
            raise ValueError(
                "Live output strategy ID must match the configured strategy"
            )
DEFAULT_CONFIG = LiveConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "DecisionPeriod",
    "LiveBenchmarkConfig",
    "LiveBrinsonConfig",
    "LiveConfig",
    "LiveMarketConfig",
]
