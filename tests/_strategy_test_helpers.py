"""Shared deterministic fixtures for canonical strategy tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from portfolio_core.strategies import StrategyDecisionContext
from portfolio_core.strategies.momentum import MomentumStrategy
from portfolio_core.strategies.research_parameters import (
    BufferParameters, ExposureParameters, ResearchParameters, SignalParameters,
    SizingParameters, StockSelection, VolatilityProfile, WholeSectorSelection,
)


RESEARCH_STRATEGY_IDS = (
    "momentum", "reversal", "monthly_trend", "low_volatility", "sector_momentum",
)


def research_test_parameters(strategy_id, **changes) -> ResearchParameters:
    """One complete packet per family, without expanding a search grid."""
    signals = {
        "momentum": SignalParameters("momentum", 11, 1, None, None),
        "reversal": SignalParameters("reversal", 1, 0, None, None),
        "monthly_trend": SignalParameters("monthly_trend", None, None, None, 10),
        "low_volatility": SignalParameters(
            "low_volatility", None, None, VolatilityProfile(60, 24), None,
        ),
        "sector_momentum": SignalParameters("sector_momentum", 1, 0, None, None),
    }
    parameters = momentum_test_parameters(
        signal=signals[strategy_id],
        selection=(WholeSectorSelection(3, 3) if strategy_id == "sector_momentum"
                   else StockSelection(10, 10)),
    )
    return replace(parameters, **changes)


def momentum_test_parameters(**changes) -> ResearchParameters:
    """Complete test packet for the former plain 11/1 momentum mechanics."""
    parameters = ResearchParameters(
        SignalParameters("momentum", 11, 1, None, None),
        StockSelection(10, 10),
        SizingParameters("equal", None),
        BufferParameters(False, None),
        ExposureParameters(1.0, 0.5),
        False, 0.0,
    )
    return replace(parameters, **changes)


def momentum_test_strategy(**changes) -> MomentumStrategy:
    return MomentumStrategy(momentum_test_parameters(**changes))


def research_contract_parameters() -> ResearchParameters:
    """Complete packet shared by research schema and configuration tests."""
    return ResearchParameters(
        SignalParameters("momentum", 11, 1, None, None),
        StockSelection(10, 20),
        SizingParameters("equal", None),
        BufferParameters(False, None),
        ExposureParameters(1.5, 0.5),
        False,
        0.01,
    )


def monthly_history(
    *,
    asset_count: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], dict[str, str]]:
    assets = tuple(f"A{index:02d}" for index in range(asset_count))
    dates = pd.date_range("2020-01-31", periods=13, freq="ME")
    close = pd.DataFrame(100.0, index=dates, columns=assets)
    momentum = np.linspace(0.60, -0.60, asset_count)
    close.iloc[-2] = 100.0 * (1.0 + momentum)
    close.iloc[-1] = close.iloc[-2] * np.linspace(0.75, 1.25, asset_count)
    volume = pd.DataFrame(
        np.broadcast_to(
            np.linspace(900_000.0, 1_400_000.0, asset_count),
            close.shape,
        ),
        index=dates,
        columns=assets,
    )
    sectors = {
        asset_id: str(10 + 5 * (index % 6))
        for index, asset_id in enumerate(assets)
    }
    return close, volume, assets, sectors


def strategy_context(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    assets: tuple[str, ...],
    sectors: dict[str, str],
    *,
    previous: pd.Series | None = None,
    signal_source_max_date: pd.Timestamp | None = None,
) -> StrategyDecisionContext:
    return StrategyDecisionContext(
        close_history=close,
        volume_history=volume,
        candidate_asset_ids=assets,
        sector_code_by_asset_id=sectors,
        previous_target_weights=(
            pd.Series(dtype=float) if previous is None else previous
        ),
        signal_cutoff=pd.Timestamp(close.index[-1]),
        signal_source_max_date=(
            pd.Timestamp(close.index[-1])
            if signal_source_max_date is None
            else signal_source_max_date
        ),
    )


def decision_context(data, start, previous, *, selected=None, sector_returns=None):
    """Synthetic backtest context using information at the signal cutoff only."""
    from backtest.data_loading import active_pit_asset_ids
    from portfolio_core.sector_assignments import sector_rows_asof
    from portfolio_core.strategies.research_state import SelectedSectors

    sectors = sector_rows_asof(data.sector_assignments, start)
    return StrategyDecisionContext(
        data.data_close.loc[:start], data.data_volume.loc[:start],
        tuple(sorted(active_pit_asset_ids(data, start))),
        sectors.GICS_Sector_Code.astype(str).to_dict(), previous, start, start,
        previous_selected_sectors=selected or SelectedSectors(),
        sector_returns=sector_returns.loc[:start] if sector_returns is not None else pd.DataFrame(),
    )
