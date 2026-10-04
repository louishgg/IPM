"""Strategy-owned turnover filtering before portfolio-ledger execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd


def _series_from_items(items: tuple[tuple[str, float], ...]) -> pd.Series:
    return pd.Series(dict(items), dtype=float).sort_index()


def _normalized_series(values: pd.Series, *, label: str) -> pd.Series:
    result = values.copy()
    result.index = result.index.astype(str)
    if result.index.has_duplicates:
        raise ValueError(f"{label} contains duplicate asset IDs")
    result = pd.to_numeric(result, errors="raise").astype(float)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must be finite")
    return result.sort_index()


def round_half_away_from_zero(values: pd.Series) -> pd.Series:
    """Round signed values to the nearest integer, with halves away from zero."""

    normalized = _normalized_series(values, label="share targets")
    rounded = np.sign(normalized) * np.floor(np.abs(normalized) + 0.5)
    return pd.Series(rounded.astype("int64"), index=normalized.index, dtype="int64")


@dataclass(frozen=True, slots=True)
class RebalanceTargets:
    """Immutable result of translating strategy weights into share targets."""

    applied_items: tuple[tuple[str, float], ...]
    target_weight_items: tuple[tuple[str, float], ...]
    current_weight_items: tuple[tuple[str, float], ...]
    weight_drift_items: tuple[tuple[str, float], ...]

    @property
    def applied_shares(self) -> pd.Series:
        return _series_from_items(self.applied_items)

    @property
    def target_weights(self) -> pd.Series:
        return _series_from_items(self.target_weight_items)

    @property
    def current_weights(self) -> pd.Series:
        return _series_from_items(self.current_weight_items)

    @property
    def weight_drift(self) -> pd.Series:
        return _series_from_items(self.weight_drift_items)


def plan_rebalance_targets(
    current_shares: pd.Series,
    target_weights: pd.Series,
    sizing_prices: pd.Series,
    sizing_equity: float,
    *,
    turnover_threshold: float,
    apply_turnover_threshold: bool = True,
    whole_share_orders: bool = True,
) -> RebalanceTargets:
    """Create final share targets without applying any brokerage accounting.

    The selected strategy owns ``turnover_threshold``. Entries, complete exits,
    and sign flips always trade. The threshold can suppress only a same-side
    resize. A zero threshold therefore suppresses no nonzero resize.
    """

    if not isfinite(float(sizing_equity)) or sizing_equity <= 0.0:
        raise ValueError("sizing_equity must be finite and positive")
    if not isfinite(float(turnover_threshold)) or turnover_threshold < 0.0:
        raise ValueError("turnover_threshold must be finite and nonnegative")
    if not isinstance(apply_turnover_threshold, bool):
        raise TypeError("apply_turnover_threshold must be boolean")
    if not isinstance(whole_share_orders, bool):
        raise TypeError("whole_share_orders must be boolean")

    current = _normalized_series(current_shares, label="current shares")
    weights = _normalized_series(target_weights, label="target weights")
    prices = _normalized_series(sizing_prices, label="sizing prices")
    assets = current.index.union(weights.index).union(prices.index).sort_values()
    current = current.reindex(assets).fillna(0.0)
    weights = weights.reindex(assets).fillna(0.0)
    prices = prices.reindex(assets)
    invalid_prices = prices.isna() | prices.le(0.0)
    if invalid_prices.any():
        raise ValueError(
            "Sizing prices must be finite and positive for every target/current "
            f"asset: {prices.index[invalid_prices].tolist()}"
        )

    desired = weights * float(sizing_equity) / prices
    if whole_share_orders:
        requested = round_half_away_from_zero(desired).reindex(assets).astype(float)
        fractional_current = ~np.isclose(
            current,
            np.round(current),
            atol=1e-10,
            rtol=0.0,
        )
        retained_fractional = fractional_current & requested.ne(0.0)
        if retained_fractional.any():
            # Integer orders preserve fractions delivered by corporate actions;
            # full exits may close the entire fractional position.
            reachable_trade = round_half_away_from_zero(
                (desired - current).loc[retained_fractional]
            ).astype(float)
            requested.loc[retained_fractional] = (
                current.loc[retained_fractional] + reachable_trade
            )
    else:
        requested = desired.astype(float)
    current_values = current * prices
    current_weights = current_values / float(sizing_equity)
    drift = (weights - current_weights).abs()

    current_nonzero = current.ne(0.0)
    requested_nonzero = requested.ne(0.0)
    entry = ~current_nonzero & requested_nonzero
    exit_position = current_nonzero & ~requested_nonzero
    flip = (
        current_nonzero
        & requested_nonzero
        & np.sign(current).ne(np.sign(requested))
    )
    same_side_resize = (
        current_nonzero
        & requested_nonzero
        & np.sign(current).eq(np.sign(requested))
        & requested.ne(current)
    )
    if apply_turnover_threshold:
        execute = entry | exit_position | flip | (
            same_side_resize & drift.gt(float(turnover_threshold))
        )
    else:
        execute = requested.ne(current)
    applied = requested.where(execute, current).astype(float)

    def _items(series: pd.Series) -> tuple[tuple[str, float], ...]:
        return tuple((str(key), float(value)) for key, value in series.items())

    return RebalanceTargets(
        applied_items=_items(applied),
        target_weight_items=_items(weights),
        current_weight_items=_items(current_weights),
        weight_drift_items=_items(drift),
    )


__all__ = [
    "RebalanceTargets",
    "plan_rebalance_targets",
    "round_half_away_from_zero",
]
