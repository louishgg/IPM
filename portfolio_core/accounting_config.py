"""Portfolio-accounting configuration and strategy-independent spread estimates."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite

import numpy as np
import pandas as pd


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class TransactionCostConfig:
    """Own-liquidity one-way half-spread assumptions.

    The estimate deliberately uses only the security's own absolute liquidity.
    It is therefore invariant to the strategy, candidate universe, and the
    arbitrary per-share scale of a split-adjusted price series.
    """

    base_bps: float = 1.0
    liquidity_bps: float = 7.0
    min_bps: float = 1.0
    max_bps: float = 15.0
    liquidity_anchor_usd: float = 10_000_000.0

    def __post_init__(self) -> None:
        for name, value in (
            ("base_bps", self.base_bps),
            ("liquidity_bps", self.liquidity_bps),
            ("min_bps", self.min_bps),
            ("max_bps", self.max_bps),
            ("liquidity_anchor_usd", self.liquidity_anchor_usd),
        ):
            _require_finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.min_bps > self.max_bps:
            raise ValueError("min_bps cannot exceed max_bps")
        if not self.min_bps <= self.base_bps <= self.max_bps:
            raise ValueError("base_bps must lie between min_bps and max_bps")
        if self.liquidity_anchor_usd <= 0.0:
            raise ValueError("liquidity_anchor_usd must be positive")


@dataclass(frozen=True, slots=True)
class PortfolioAccountingConfig:
    """Single source of portfolio-accounting rules."""

    initial_capital: float = 1_000_000.0
    fee_per_trade: float = 2.0
    cash_interest_rate: float = 0.02
    loan_interest_rate: float = 0.08
    day_count_days: int = 365
    whole_share_orders: bool = True
    minimum_positions: int = 20
    minimum_long_positions: int = 10
    minimum_short_positions: int = 10
    maximum_gross_exposure: float = 2.0
    transaction_costs: TransactionCostConfig = field(
        default_factory=TransactionCostConfig
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("initial_capital", self.initial_capital),
            ("fee_per_trade", self.fee_per_trade),
            ("cash_interest_rate", self.cash_interest_rate),
            ("loan_interest_rate", self.loan_interest_rate),
            ("maximum_gross_exposure", self.maximum_gross_exposure),
        ):
            _require_finite(name, value)
        if self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        if self.fee_per_trade < 0.0:
            raise ValueError("fee_per_trade cannot be negative")
        if self.cash_interest_rate < 0.0 or self.loan_interest_rate < 0.0:
            raise ValueError("interest rates cannot be negative")
        if (
            not isinstance(self.day_count_days, int)
            or isinstance(self.day_count_days, bool)
            or self.day_count_days <= 0
        ):
            raise ValueError("day_count_days must be a positive integer")
        if not isinstance(self.whole_share_orders, bool):
            raise TypeError("whole_share_orders must be boolean")
        for name, value in (
            ("minimum_positions", self.minimum_positions),
            ("minimum_long_positions", self.minimum_long_positions),
            ("minimum_short_positions", self.minimum_short_positions),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.maximum_gross_exposure <= 0.0:
            raise ValueError("maximum_gross_exposure must be positive")


DEFAULT_ACCOUNTING_CONFIG = PortfolioAccountingConfig()

SPREAD_SENSITIVITY_MULTIPLIERS = {
    "low": 0.5,
    "base": 1.0,
    "high": 1.5,
}


def spread_sensitivity_configs(
    base: PortfolioAccountingConfig,
) -> dict[str, PortfolioAccountingConfig]:
    """Return low/base/high cases relative to the supplied base configuration."""

    cases: dict[str, PortfolioAccountingConfig] = {}
    for name, multiplier in SPREAD_SENSITIVITY_MULTIPLIERS.items():
        if multiplier == 1.0:
            cases[name] = base
            continue
        cases[name] = replace(
            base,
            transaction_costs=replace(
                base.transaction_costs,
                liquidity_bps=(
                    base.transaction_costs.liquidity_bps * float(multiplier)
                ),
            ),
        )
    return cases


def make_tc_rate_from_dollar_volume(
    dvol_slice: pd.Series,
    *,
    config: TransactionCostConfig | None = None,
) -> pd.Series:
    """Return the own-liquidity one-way half-spread rate for each security."""

    selected = config or DEFAULT_ACCOUNTING_CONFIG.transaction_costs
    dvol = dvol_slice.copy()
    dvol.index = dvol.index.astype(str)
    if dvol.index.has_duplicates:
        raise ValueError("Dollar-volume inputs contain duplicate asset IDs")
    dvol = dvol.sort_index().replace([np.inf, -np.inf], np.nan)
    dvol = dvol.where(dvol > 0, np.nan)
    scale = np.sqrt(selected.liquidity_anchor_usd / dvol)
    spread_bps = selected.base_bps + selected.liquidity_bps * scale
    spread_bps = spread_bps.clip(
        lower=selected.min_bps,
        upper=selected.max_bps,
    ).fillna(selected.max_bps)
    return (spread_bps / 10_000.0).astype(float)


__all__ = [
    "DEFAULT_ACCOUNTING_CONFIG",
    "PortfolioAccountingConfig",
    "SPREAD_SENSITIVITY_MULTIPLIERS",
    "TransactionCostConfig",
    "make_tc_rate_from_dollar_volume",
    "spread_sensitivity_configs",
]
