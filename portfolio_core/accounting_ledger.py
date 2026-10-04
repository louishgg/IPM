"""Immutable, deterministic portfolio-accounting ledger."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Collection

import numpy as np
import pandas as pd

from .accounting_config import (
    DEFAULT_ACCOUNTING_CONFIG,
    PortfolioAccountingConfig,
    make_tc_rate_from_dollar_volume,
)
from .rebalance_planner import round_half_away_from_zero


_TOLERANCE = 1e-10


class InvalidRebalanceError(ValueError):
    """The requested strategy target intrinsically violates a hard rule."""


class InfeasibleRebalanceError(RuntimeError):
    """Execution effects prevent a requested target from remaining compliant."""


def _normalized_series(
    values: pd.Series | None,
    *,
    label: str,
    nonnegative: bool = False,
    drop_zeros: bool = False,
) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    result = values.copy()
    result.index = result.index.astype(str)
    if result.index.has_duplicates:
        raise ValueError(f"{label} contains duplicate asset IDs")
    if (result.index.str.strip() == "").any():
        raise ValueError(f"{label} contains a blank asset ID")
    result = pd.to_numeric(result, errors="raise").astype(float)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must be finite")
    if nonnegative and result.lt(-_TOLERANCE).any():
        raise ValueError(f"{label} cannot be negative")
    if nonnegative:
        result = result.clip(lower=0.0)
    if drop_zeros:
        result = result.loc[~np.isclose(result, 0.0, atol=_TOLERANCE, rtol=0.0)]
    return result.sort_index()


def _items(series: pd.Series) -> tuple[tuple[str, float], ...]:
    return tuple((str(asset_id), float(value)) for asset_id, value in series.items())


def _raw_items(series: pd.Series) -> tuple[tuple[object, object], ...]:
    """Preserve factory inputs for the dataclass's single validation pass."""

    return tuple(series.items())


def _series(items: tuple[tuple[str, float], ...]) -> pd.Series:
    if not items:
        return pd.Series(dtype=float)
    keys = [str(key) for key, _value in items]
    if len(keys) != len(set(keys)):
        raise ValueError("Immutable accounting items contain duplicate asset IDs")
    return pd.Series(dict(items), dtype=float).sort_index()


def _normalized_date(value: object | None, *, label: str) -> pd.Timestamp | None:
    if value is None:
        return None
    date = pd.Timestamp(value)
    if pd.isna(date) or date.tz is not None:
        raise ValueError(f"{label} must be finite and timezone-naive")
    return date


@dataclass(frozen=True, slots=True, init=False)
class LedgerState:
    """Immutable brokerage-account state between dated events."""

    position_items: tuple[tuple[str, float], ...]
    cash: float
    restricted_items: tuple[tuple[str, float], ...]
    state_date: pd.Timestamp | None

    @classmethod
    def _from_series(
        cls,
        shares: pd.Series,
        cash: float,
        restricted_short_proceeds: pd.Series,
        state_date: object | None,
    ) -> "LedgerState":
        """Build a validated state for trusted accounting transitions only."""

        if not isfinite(float(cash)):
            raise ValueError("cash must be finite")
        date = _normalized_date(state_date, label="state_date")
        positions = _normalized_series(
            shares,
            label="positions",
            drop_zeros=True,
        )
        restricted = _normalized_series(
            restricted_short_proceeds,
            label="restricted short proceeds",
            nonnegative=True,
            drop_zeros=True,
        )
        invalid_restrictions = sorted(
            asset_id
            for asset_id in restricted.index
            if float(positions.get(asset_id, 0.0)) >= 0.0
        )
        if invalid_restrictions:
            raise ValueError(
                "Restricted proceeds require an open short position: "
                f"{invalid_restrictions}"
            )
        state = object.__new__(cls)
        object.__setattr__(state, "position_items", _items(positions))
        object.__setattr__(state, "cash", float(cash))
        object.__setattr__(state, "restricted_items", _items(restricted))
        object.__setattr__(state, "state_date", date)
        return state

    @classmethod
    def initial(
        cls,
        config: PortfolioAccountingConfig = DEFAULT_ACCOUNTING_CONFIG,
        *,
        state_date: object | None = None,
    ) -> "LedgerState":
        return cls._from_series(
            pd.Series(dtype=float),
            float(config.initial_capital),
            pd.Series(dtype=float),
            state_date,
        )

    @property
    def shares(self) -> pd.Series:
        return _series(self.position_items)

    @property
    def restricted_short_proceeds(self) -> pd.Series:
        return _series(self.restricted_items)

    @property
    def restricted_total(self) -> float:
        return float(sum(value for _, value in self.restricted_items))

    @property
    def free_cash(self) -> float:
        return float(self.cash - self.restricted_total)

    @property
    def loan(self) -> float:
        return float(max(-self.free_cash, 0.0))


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Derived account values at one reference-price vector."""

    date: pd.Timestamp | None
    position_value_items: tuple[tuple[str, float], ...]
    cash: float
    restricted_short_proceeds: float
    free_cash: float
    loan: float
    equity: float
    gross_exposure: float
    maximum_position_weight: float
    position_count: int
    long_count: int
    short_count: int

    @property
    def position_values(self) -> pd.Series:
        return _series(self.position_value_items)


@dataclass(frozen=True, slots=True)
class LedgerRebalancePlan:
    """Immutable ledger input after strategy-owned turnover filtering."""

    target_items: tuple[tuple[str, float], ...]
    strategy_weight_items: tuple[tuple[str, float], ...]
    price_items: tuple[tuple[str, float], ...]
    liquidity_items: tuple[tuple[str, float], ...]
    eligible_asset_ids: tuple[str, ...]
    execution_date: pd.Timestamp

    def __post_init__(self) -> None:
        targets = _normalized_series(
            _series(self.target_items),
            label="target shares",
        )
        weights = _normalized_series(
            _series(self.strategy_weight_items),
            label="strategy target weights",
        )
        prices = _normalized_series(
            _series(self.price_items),
            label="reference prices",
        )
        liquidity = _series(self.liquidity_items)
        liquidity.index = liquidity.index.astype(str)
        if (liquidity.index.str.strip() == "").any():
            raise ValueError("own liquidity contains a blank asset ID")
        liquidity = pd.to_numeric(liquidity, errors="raise").astype(float)
        liquidity = liquidity.replace([np.inf, -np.inf], np.nan).sort_index()
        eligible = tuple(
            sorted({str(value).strip() for value in self.eligible_asset_ids})
        )
        if any(not value for value in eligible):
            raise ValueError("Eligible universe contains a blank asset ID")
        date = _normalized_date(self.execution_date, label="execution_date")
        assert date is not None
        object.__setattr__(self, "target_items", _items(targets))
        object.__setattr__(self, "strategy_weight_items", _items(weights))
        object.__setattr__(self, "price_items", _items(prices))
        object.__setattr__(self, "liquidity_items", _items(liquidity))
        object.__setattr__(self, "eligible_asset_ids", eligible)
        object.__setattr__(self, "execution_date", date)

    @classmethod
    def from_series(
        cls,
        target_shares: pd.Series,
        strategy_target_weights: pd.Series,
        reference_prices: pd.Series,
        own_liquidity_usd: pd.Series,
        eligible_asset_ids: Collection[str],
        execution_date: object,
    ) -> "LedgerRebalancePlan":
        return cls(
            _raw_items(target_shares),
            _raw_items(strategy_target_weights),
            _raw_items(reference_prices),
            _raw_items(own_liquidity_usd),
            tuple(eligible_asset_ids),
            execution_date,
        )

    @property
    def target_shares(self) -> pd.Series:
        return _series(self.target_items)

    @property
    def strategy_target_weights(self) -> pd.Series:
        return _series(self.strategy_weight_items)

    @property
    def reference_prices(self) -> pd.Series:
        return _series(self.price_items)

    @property
    def own_liquidity_usd(self) -> pd.Series:
        return _series(self.liquidity_items)


@dataclass(frozen=True, slots=True)
class TradeLeg:
    """One brokerage order leg, including each side of a sign flip."""

    asset_id: str
    leg_type: str
    current_shares: float
    target_shares: float
    trade_shares: float
    effective_execution_price: float
    fixed_fee: float
    spread_cost: float
    cash_effect: float
    restricted_proceeds_change: float


@dataclass(frozen=True, slots=True)
class TradeExecution:
    """Per-security audit for one executed rebalance transition."""

    asset_id: str
    applied_rule: str
    current_shares: float
    requested_target_shares: float
    applied_target_shares: float
    trade_shares: float
    reference_price: float
    effective_execution_price: float
    reference_notional: float
    order_count: int
    fixed_fee: float
    spread_rate: float
    spread_cost: float
    cash_effect: float
    restricted_proceeds_change: float
    legs: tuple[TradeLeg, ...]


@dataclass(frozen=True, slots=True)
class RebalanceResult:
    """Applied ledger transition and its complete reconciliation data."""

    state: LedgerState
    before: AccountSnapshot
    after: AccountSnapshot
    executions: tuple[TradeExecution, ...]
    requested_items: tuple[tuple[str, float], ...]
    applied_items: tuple[tuple[str, float], ...]
    feasibility_scale: float
    adjustment_reason: str
    order_count: int
    fixed_fees: float
    spread_cost: float
    turnover: float

    @property
    def requested_shares(self) -> pd.Series:
        return _series(self.requested_items)

    @property
    def applied_shares(self) -> pd.Series:
        return _series(self.applied_items)


def account_snapshot(
    state: LedgerState,
    reference_prices: pd.Series,
    *,
    date: object | None = None,
) -> AccountSnapshot:
    shares = state.shares
    prices = _normalized_series(reference_prices, label="reference prices")
    missing = sorted(set(shares.index) - set(prices.index))
    if missing:
        raise ValueError(f"Missing reference prices for open positions: {missing}")
    held_prices = prices.reindex(shares.index)
    if held_prices.le(0.0).any():
        invalid = held_prices.index[held_prices.le(0.0)].tolist()
        raise ValueError(f"Reference prices must be positive: {invalid}")
    position_values = shares * held_prices
    equity = float(state.cash + position_values.sum())
    gross_notional = float(position_values.abs().sum())
    gross = gross_notional / equity if equity > 0.0 else float("inf")
    maximum = (
        float(position_values.abs().max() / equity)
        if equity > 0.0 and not position_values.empty
        else (0.0 if position_values.empty else float("inf"))
    )
    snapshot_date = (
        _normalized_date(date, label="snapshot date")
        if date is not None
        else state.state_date
    )
    return AccountSnapshot(
        date=snapshot_date,
        position_value_items=_items(position_values),
        cash=float(state.cash),
        restricted_short_proceeds=state.restricted_total,
        free_cash=state.free_cash,
        loan=state.loan,
        equity=equity,
        gross_exposure=float(gross),
        maximum_position_weight=float(maximum),
        position_count=int(shares.ne(0.0).sum()),
        long_count=int(shares.gt(0.0).sum()),
        short_count=int(shares.lt(0.0).sum()),
    )


def project_financing_amounts(
    cash: float,
    restricted_short_proceeds: float,
    start: object,
    end: object,
    config: PortfolioAccountingConfig = DEFAULT_ACCOUNTING_CONFIG,
) -> tuple[float, float, float, float]:
    """Return the unrounded financing transition and cash/loan split."""

    start_date = _normalized_date(start, label="interest start")
    end_date = _normalized_date(end, label="interest end")
    assert start_date is not None and end_date is not None
    if end_date < start_date:
        raise ValueError("interest end cannot precede interest start")
    days = int((end_date - start_date).days)
    free_cash = float(cash - restricted_short_proceeds)
    annual_rate = (
        float(config.cash_interest_rate)
        if free_cash > 0.0
        else float(config.loan_interest_rate)
    )
    interest = float(
        free_cash
        * (
            (1.0 + annual_rate / float(config.day_count_days)) ** days
            - 1.0
        )
    )
    cash_credit = interest if free_cash > 0.0 else 0.0
    loan_charge = -interest if free_cash < 0.0 else 0.0
    return (
        float(cash + interest),
        interest,
        float(cash_credit),
        float(loan_charge),
    )


def _trade_rule(current: float, target: float) -> str:
    if current == 0.0:
        return "enter_long" if target > 0.0 else "enter_short"
    if target == 0.0:
        return "exit_long" if current > 0.0 else "exit_short"
    if current < 0.0 < target:
        return "flip_short_to_long"
    if current > 0.0 > target:
        return "flip_long_to_short"
    if current > 0.0:
        return "increase_long" if target > current else "decrease_long"
    return "increase_short" if target < current else "decrease_short"


def _order_count(current: float, target: float) -> int:
    if np.isclose(current, target, atol=_TOLERANCE, rtol=0.0):
        return 0
    if current != 0.0 and target != 0.0 and np.sign(current) != np.sign(target):
        return 2
    return 1


def _leg_type(current: float, target: float) -> str:
    if current > 0.0 and target == 0.0:
        return "close_long"
    if current < 0.0 and target == 0.0:
        return "cover_short"
    if current == 0.0 and target > 0.0:
        return "open_long"
    if current == 0.0 and target < 0.0:
        return "open_short"
    if current > 0.0 and target > 0.0:
        return "increase_long" if target > current else "decrease_long"
    if current < 0.0 and target < 0.0:
        return "increase_short" if target < current else "decrease_short"
    raise RuntimeError("Invalid unsplit trade leg")


def _validate_whole_share_transition(
    current: float,
    target: float,
    *,
    asset_id: str,
) -> None:
    current_is_whole = np.isclose(current, round(current), atol=_TOLERANCE, rtol=0.0)
    target_is_whole = np.isclose(target, round(target), atol=_TOLERANCE, rtol=0.0)
    trade_is_whole = np.isclose(
        target - current,
        round(target - current),
        atol=_TOLERANCE,
        rtol=0.0,
    )
    if current_is_whole and not target_is_whole:
        raise InvalidRebalanceError(
            f"Voluntary target for {asset_id} is not a whole share"
        )
    if not current_is_whole and target != 0.0 and not trade_is_whole:
        raise InvalidRebalanceError(
            f"Voluntary trade for fractional corporate-action position "
            f"{asset_id} is not a whole share"
        )


def _validate_intrinsic_plan(
    state: LedgerState,
    plan: LedgerRebalancePlan,
    config: PortfolioAccountingConfig,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    current = state.shares
    target = plan.target_shares
    weights = plan.strategy_target_weights
    assets = current.index.union(target.index).union(weights.index).sort_values()
    current = current.reindex(assets).fillna(0.0)
    target = target.reindex(assets).fillna(0.0)
    weights = weights.reindex(assets).fillna(0.0)
    prices = plan.reference_prices.reindex(assets)
    invalid_prices = prices.isna() | ~np.isfinite(prices) | prices.le(0.0)
    if invalid_prices.any():
        raise InvalidRebalanceError(
            "Missing or invalid execution prices for "
            f"{prices.index[invalid_prices].tolist()}"
        )

    requested_weight_positions = weights.loc[weights.ne(0.0)]
    eligible = set(plan.eligible_asset_ids)
    outside = sorted(set(requested_weight_positions.index) - eligible)
    if outside:
        raise InvalidRebalanceError(
            f"Strategy target contains assets outside the eligible universe: {outside}"
        )
    gross = float(requested_weight_positions.abs().sum())
    if gross > config.maximum_gross_exposure + _TOLERANCE:
        raise InvalidRebalanceError(
            f"Requested gross exposure {gross:.12g} exceeds "
            f"{config.maximum_gross_exposure:.12g}"
        )
    target_nonzero = target.loc[target.ne(0.0)]
    retained_outside = sorted(set(target_nonzero.index) - eligible)
    if retained_outside:
        raise InvalidRebalanceError(
            "Post-threshold target retains ineligible assets: "
            f"{retained_outside}"
        )
    requested_counts = (
        len(requested_weight_positions),
        int(requested_weight_positions.gt(0.0).sum()),
        int(requested_weight_positions.lt(0.0).sum()),
    )
    required = (
        config.minimum_positions,
        config.minimum_long_positions,
        config.minimum_short_positions,
    )
    if any(
        actual < minimum
        for actual, minimum in zip(requested_counts, required, strict=True)
    ):
        raise InvalidRebalanceError(
            "Requested strategy target violates required position counts: "
            f"positions/longs/shorts={requested_counts}, required={required}"
        )
    sign_mismatch = sorted(
        asset_id
        for asset_id in assets
        if weights.loc[asset_id] != 0.0
        and target.loc[asset_id] != 0.0
        and np.sign(weights.loc[asset_id]) != np.sign(target.loc[asset_id])
    )
    if sign_mismatch:
        raise InvalidRebalanceError(
            f"Target-share signs disagree with strategy weights: {sign_mismatch}"
        )

    for asset_id in assets:
        before = float(current.loc[asset_id])
        after = float(target.loc[asset_id])
        if config.whole_share_orders:
            _validate_whole_share_transition(before, after, asset_id=str(asset_id))
    return current, target, weights, prices


def _restricted_after_trade(
    current: float,
    target: float,
    current_restricted: float,
    bid_price: float,
) -> float:
    if current < 0.0 and target < 0.0:
        if abs(target) > abs(current):
            return float(current_restricted + (abs(target) - abs(current)) * bid_price)
        if abs(target) < abs(current):
            retained_fraction = abs(target) / abs(current)
            return float(current_restricted * retained_fraction)
        return float(current_restricted)
    if current < 0.0 <= target:
        return 0.0
    if current >= 0.0 > target:
        return float(abs(target) * bid_price)
    return 0.0


def _execute_exact(
    state: LedgerState,
    target: pd.Series,
    requested_target: pd.Series,
    prices: pd.Series,
    liquidity: pd.Series,
    execution_date: pd.Timestamp,
    config: PortfolioAccountingConfig,
    *,
    apply_fees: bool,
    apply_spread: bool,
) -> tuple[LedgerState, tuple[TradeExecution, ...]]:
    current = state.shares
    assets = current.index.union(target.index).union(prices.index).sort_values()
    current = current.reindex(assets).fillna(0.0)
    target = target.reindex(assets).fillna(0.0)
    requested_target = requested_target.reindex(assets).fillna(0.0)
    prices = prices.reindex(assets)
    liquidity = liquidity.reindex(assets)
    rates = make_tc_rate_from_dollar_volume(
        liquidity,
        config=config.transaction_costs,
    ).reindex(assets)
    restricted_before = state.restricted_short_proceeds.reindex(assets).fillna(0.0)
    restricted_after = restricted_before.copy()
    cash = float(state.cash)
    executions: list[TradeExecution] = []

    for asset_id in assets:
        before = float(current.loc[asset_id])
        after = float(target.loc[asset_id])
        trade = float(after - before)
        if np.isclose(trade, 0.0, atol=_TOLERANCE, rtol=0.0):
            continue
        reference_price = float(prices.loc[asset_id])
        spread_rate = float(rates.loc[asset_id])
        applied_rate = spread_rate if apply_spread else 0.0
        bid_price = reference_price * (1.0 - applied_rate)
        transitions = (
            ((before, 0.0), (0.0, after))
            if before != 0.0
            and after != 0.0
            and np.sign(before) != np.sign(after)
            else ((before, after),)
        )
        restricted_cursor = float(restricted_before.loc[asset_id])
        legs: list[TradeLeg] = []
        for leg_before, leg_after in transitions:
            leg_trade = float(leg_after - leg_before)
            leg_effective_price = reference_price * (
                1.0 + applied_rate
                if leg_trade > 0.0
                else 1.0 - applied_rate
            )
            leg_fee = float(config.fee_per_trade) if apply_fees else 0.0
            leg_spread_cost = float(
                abs(leg_trade) * reference_price * applied_rate
            )
            leg_cash_effect = float(
                -leg_trade * leg_effective_price - leg_fee
            )
            new_restricted = _restricted_after_trade(
                float(leg_before),
                float(leg_after),
                restricted_cursor,
                bid_price,
            )
            legs.append(
                TradeLeg(
                    asset_id=str(asset_id),
                    leg_type=_leg_type(float(leg_before), float(leg_after)),
                    current_shares=float(leg_before),
                    target_shares=float(leg_after),
                    trade_shares=leg_trade,
                    effective_execution_price=float(leg_effective_price),
                    fixed_fee=leg_fee,
                    spread_cost=leg_spread_cost,
                    cash_effect=leg_cash_effect,
                    restricted_proceeds_change=float(
                        new_restricted - restricted_cursor
                    ),
                )
            )
            cash += leg_cash_effect
            restricted_cursor = new_restricted
        orders = len(legs)
        if orders != _order_count(before, after):
            raise RuntimeError("Trade-leg order count does not reconcile")
        effective_price = legs[0].effective_execution_price
        fee = float(sum(leg.fixed_fee for leg in legs))
        spread_cost = float(sum(leg.spread_cost for leg in legs))
        cash_effect = float(sum(leg.cash_effect for leg in legs))
        new_restricted = restricted_cursor
        restricted_after.loc[asset_id] = new_restricted
        executions.append(
            TradeExecution(
                asset_id=str(asset_id),
                applied_rule=_trade_rule(before, after),
                current_shares=before,
                requested_target_shares=float(requested_target.loc[asset_id]),
                applied_target_shares=after,
                trade_shares=trade,
                reference_price=reference_price,
                effective_execution_price=float(effective_price),
                reference_notional=float(trade * reference_price),
                order_count=orders,
                fixed_fee=fee,
                spread_rate=spread_rate,
                spread_cost=spread_cost,
                cash_effect=cash_effect,
                restricted_proceeds_change=float(
                    new_restricted - float(restricted_before.loc[asset_id])
                ),
                legs=tuple(legs),
            )
        )
    target = target.loc[~np.isclose(target, 0.0, atol=_TOLERANCE, rtol=0.0)]
    restricted_after = restricted_after.reindex(target.index).fillna(0.0)
    restricted_after = restricted_after.loc[target.lt(0.0)]
    next_state = LedgerState._from_series(
        target,
        cash,
        restricted_after,
        execution_date,
    )
    return next_state, tuple(executions)


def execute_forced_liquidation(
    state: LedgerState,
    *,
    asset_id: str,
    reference_price: float,
    own_liquidity_usd: float,
    execution_date: object,
    config: PortfolioAccountingConfig = DEFAULT_ACCOUNTING_CONFIG,
    apply_fees: bool = True,
    apply_spread: bool = True,
) -> tuple[LedgerState, TradeExecution]:
    """Close one complete signed position through the canonical trade engine."""

    if not isinstance(apply_fees, bool) or not isinstance(apply_spread, bool):
        raise TypeError("cost switches must be boolean")
    asset = str(asset_id).strip()
    if not asset:
        raise ValueError("Forced-liquidation asset ID cannot be empty")
    date = _normalized_date(execution_date, label="execution_date")
    assert date is not None
    if state.state_date != date:
        raise ValueError(
            "Ledger state must be advanced to the forced-liquidation date"
        )
    price = float(reference_price)
    liquidity = float(own_liquidity_usd)
    if not isfinite(price) or price <= 0.0:
        raise ValueError("Forced-liquidation reference price must be positive")
    if not isfinite(liquidity) or liquidity <= 0.0:
        raise ValueError("Forced-liquidation liquidity must be positive")
    positions = state.shares
    current = float(positions.get(asset, 0.0))
    if np.isclose(current, 0.0, atol=_TOLERANCE, rtol=0.0):
        raise ValueError(
            f"Forced liquidation {asset} has no open position to consume"
        )

    target = positions.copy()
    target.loc[asset] = 0.0
    next_state, executions = _execute_exact(
        state,
        target,
        target,
        pd.Series({asset: price}, dtype=float),
        pd.Series({asset: liquidity}, dtype=float),
        date,
        config,
        apply_fees=apply_fees,
        apply_spread=apply_spread,
    )
    if len(executions) != 1 or executions[0].asset_id != asset:
        raise RuntimeError("Forced liquidation did not produce exactly one trade")
    execution = replace(
        executions[0],
        applied_rule="mandatory_off_universe_liquidation",
    )
    return next_state, execution


def _snapshot_is_compliant(
    snapshot: AccountSnapshot,
    config: PortfolioAccountingConfig,
) -> bool:
    return bool(
        np.isfinite(snapshot.equity)
        and snapshot.equity > 0.0
        and snapshot.position_count >= config.minimum_positions
        and snapshot.long_count >= config.minimum_long_positions
        and snapshot.short_count >= config.minimum_short_positions
        and snapshot.gross_exposure <= config.maximum_gross_exposure + _TOLERANCE
    )


def _scaled_target_shares(
    requested: pd.Series,
    current: pd.Series,
    scale: float,
    *,
    whole_share_orders: bool,
) -> pd.Series:
    """Scale requested holdings, rounding halves away from zero for whole-share orders.

    Existing fractions are preserved through integer trades unless the rounded
    target is a full exit.
    """
    ideal = requested.astype(float) * float(scale)
    if not whole_share_orders:
        return ideal
    scaled = round_half_away_from_zero(ideal).astype(float)
    current = current.reindex(scaled.index).fillna(0.0)
    fractional_current = ~np.isclose(
        current,
        np.round(current),
        atol=_TOLERANCE,
        rtol=0.0,
    )
    retained = fractional_current & scaled.ne(0.0)
    if retained.any():
        reachable_trade = round_half_away_from_zero(
            (ideal - current).loc[retained]
        ).astype(float)
        scaled.loc[retained] = current.loc[retained] + reachable_trade
    return scaled


def _minimum_count_scale(
    requested: pd.Series,
    config: PortfolioAccountingConfig,
) -> float:
    """Return the half-share rounding bound for total, long and short count minima.

    Too few requested names gives infinity; execution still checks the resulting
    basket, including existing fractions.
    """
    nonzero = requested.loc[requested.ne(0.0)]

    def threshold(values: pd.Series, required: int) -> float:
        if required == 0:
            return 0.0
        if len(values) < required:
            return float("inf")
        # The kth-smallest threshold keeps at least k targets nonzero when rounded.
        thresholds = sorted(0.5 / abs(float(value)) for value in values)
        return thresholds[required - 1]

    return max(
        threshold(nonzero, config.minimum_positions),
        threshold(nonzero.loc[nonzero.gt(0.0)], config.minimum_long_positions),
        threshold(nonzero.loc[nonzero.lt(0.0)], config.minimum_short_positions),
    )


def execute_rebalance(
    state: LedgerState,
    plan: LedgerRebalancePlan,
    config: PortfolioAccountingConfig = DEFAULT_ACCOUNTING_CONFIG,
    *,
    apply_fees: bool = True,
    apply_spread: bool = True,
) -> RebalanceResult:
    """Execute an order-independent rebalance and return a new ledger and trade audit.

    Requested holdings share one scale before rounding; dated state must already
    be at execution_date. InvalidRebalanceError marks invalid targets, while
    InfeasibleRebalanceError means the execution search found no compliant basket.
    """

    if not isinstance(apply_fees, bool) or not isinstance(apply_spread, bool):
        raise TypeError("cost switches must be boolean")
    if state.state_date is not None and state.state_date != plan.execution_date:
        raise ValueError(
            "Ledger state must be advanced to the execution date before trading"
        )
    current, requested, _weights, prices = _validate_intrinsic_plan(
        state,
        plan,
        config,
    )
    liquidity = plan.own_liquidity_usd.reindex(prices.index)
    before = account_snapshot(state, prices, date=plan.execution_date)
    if before.equity <= 0.0:
        raise InvalidRebalanceError("Cannot rebalance an account with nonpositive equity")

    # All other execution inputs are fixed within this rebalance.
    basket_results: dict[
        tuple[tuple[str, float], ...],
        tuple[pd.Series, LedgerState, tuple[TradeExecution, ...], AccountSnapshot],
    ] = {}

    def attempt(scale: float) -> tuple[pd.Series, LedgerState, tuple[TradeExecution, ...], AccountSnapshot]:
        target = _scaled_target_shares(
            requested,
            current,
            scale,
            whole_share_orders=config.whole_share_orders,
        )
        basket_key = _items(target)
        cached = basket_results.get(basket_key)
        if cached is not None:
            return cached
        candidate_state, executions = _execute_exact(
            state,
            target,
            requested,
            prices,
            liquidity,
            plan.execution_date,
            config,
            apply_fees=apply_fees,
            apply_spread=apply_spread,
        )
        snapshot = account_snapshot(
            candidate_state,
            prices,
            date=plan.execution_date,
        )
        result = target, candidate_state, executions, snapshot
        basket_results[basket_key] = result
        return result

    applied, next_state, executions, after = attempt(1.0)
    scale = 1.0
    reason = ""
    if not _snapshot_is_compliant(after, config):
        lower = _minimum_count_scale(requested, config)
        if not np.isfinite(lower) or lower > 1.0 + _TOLERANCE:
            raise InfeasibleRebalanceError(
                "No scaled whole-share basket can preserve required position counts"
            )
        lower = min(1.0, float(np.nextafter(lower, 1.0)))
        low_target, low_state, low_executions, low_snapshot = attempt(lower)
        if not _snapshot_is_compliant(low_snapshot, config):
            raise InfeasibleRebalanceError(
                "No common scale factor produces a compliant rounded basket"
            )
        low = lower
        high = 1.0
        best = (low_target, low_state, low_executions, low_snapshot)
        # Bisect the common-scale interval for post-cost compliance, searching
        # rounded versions of this basket rather than all integer portfolios.
        for _ in range(80):
            midpoint = (low + high) / 2.0
            # Further iterations cannot change either bound at this precision.
            if midpoint == low or midpoint == high:
                break
            candidate = attempt(midpoint)
            if _snapshot_is_compliant(candidate[3], config):
                low = midpoint
                best = candidate
            else:
                high = midpoint
        scale = low
        applied, next_state, executions, after = best
        reason = "execution_feasibility_scaling"

    fees = float(sum(item.fixed_fee for item in executions))
    spread = float(sum(item.spread_cost for item in executions))
    order_count = int(sum(item.order_count for item in executions))
    reference_turnover = float(
        sum(abs(item.reference_notional) for item in executions) / before.equity
    )
    expected_equity = before.equity - fees - spread
    if not np.isclose(after.equity, expected_equity, atol=1e-7, rtol=0.0):
        raise RuntimeError(
            "Ledger rebalance does not reconcile pre/post equity to fees and spread"
        )
    restricted_delta = sum(item.restricted_proceeds_change for item in executions)
    if not np.isclose(
        after.restricted_short_proceeds,
        before.restricted_short_proceeds + restricted_delta,
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError("Restricted short proceeds do not reconcile")
    return RebalanceResult(
        state=next_state,
        before=before,
        after=after,
        executions=executions,
        requested_items=_items(requested),
        applied_items=_items(applied),
        feasibility_scale=float(scale),
        adjustment_reason=reason,
        order_count=order_count,
        fixed_fees=fees,
        spread_cost=spread,
        turnover=reference_turnover,
    )


__all__ = [
    "AccountSnapshot",
    "InfeasibleRebalanceError",
    "InvalidRebalanceError",
    "LedgerRebalancePlan",
    "LedgerState",
    "RebalanceResult",
    "TradeExecution",
    "TradeLeg",
    "account_snapshot",
    "execute_rebalance",
    "execute_forced_liquidation",
    "project_financing_amounts",
]
