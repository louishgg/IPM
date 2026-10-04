"""Causal research calculations over shared prepared live history."""

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from live.analysis import run_strategy_analysis
from live.config import DEFAULT_CONFIG
from live.research_history import (
    build_live_research_requirements,
    interval_holding_returns,
    prepare_live_research_history,
)
from portfolio_core.strategies.momentum import MomentumStrategy
from portfolio_core.strategies import build_registered_strategy
from portfolio_core.strategies.sector_momentum import SectorMomentumStrategy
from portfolio_core.strategies.research_parameters import (
    ExposureParameters, SignalParameters, SizingParameters, VolatilityProfile,
    WholeSectorSelection,
)
from _strategy_test_helpers import momentum_test_parameters, research_test_parameters
from test_live_analysis import _synthetic_inputs, _successor_bundle


def _historical_inputs():
    inputs = _synthetic_inputs()
    source_dates = inputs.market_monthly.groupby("Month").Observation_Date.max().to_dict()
    earliest = inputs.membership.Effective_Date.min()
    first_members = inputs.membership.loc[
        inputs.membership.Effective_Date.eq(earliest)
    ].copy()
    first_members["Effective_Date"] = pd.Timestamp("2024-01-31")
    membership = pd.concat([first_members, inputs.membership], ignore_index=True)
    base = inputs.sector_assignments.drop_duplicates("Asset_ID").copy()
    historic = []
    existing_dates = set(inputs.sector_assignments.As_Of_Date)
    for date in source_dates.values():
        warmup_month = (inputs.evaluation_periods.Period_Start.min().to_period("M") - 1).start_time
        if date < warmup_month:
            date += pd.offsets.MonthEnd(0)
        if date >= pd.Timestamp("2024-01-31") and date not in existing_dates:
            rows = base.copy()
            rows["As_Of_Date"] = date
            historic.append(rows)
    assignments = pd.concat([inputs.sector_assignments, *historic], ignore_index=True)
    return replace(inputs, membership=membership, sector_assignments=assignments)


@pytest.mark.parametrize("signal,signal_returns", [
    (SignalParameters("momentum", 11, 1, None, None), 12),
    (SignalParameters("momentum", 36, 1, None, None), 37),
    (SignalParameters("reversal", 1, 0, None, None), 1),
    (SignalParameters("reversal", 36, 1, None, None), 37),
    (SignalParameters("monthly_trend", None, None, None, 10), 9),
    (SignalParameters("monthly_trend", None, None, None, 38), 37),
    (SignalParameters("low_volatility", None, None, VolatilityProfile(60, 24), None), 24),
    (SignalParameters("low_volatility", None, None, VolatilityProfile(36, 36), None), 36),
    (SignalParameters("sector_momentum", 12, 1, None, None), 13),
    (SignalParameters("sector_momentum", 36, 1, None, None), 37),
], ids=["momentum", "long-momentum", "reversal", "long-reversal", "sma", "long-sma",
        "selection-60-24", "selection-36-36", "sector", "long-sector"])
@pytest.mark.parametrize("sizing,sizing_returns", [
    (SizingParameters("equal", None), 0),
    (SizingParameters("inverse_volatility", VolatilityProfile(60, 24)), 24),
    (SizingParameters("inverse_volatility", VolatilityProfile(36, 36)), 36),
], ids=["equal", "sizing-60-24", "sizing-36-36"])
def test_history_requirements_keep_signal_and_sizing_bounds_independent(
    signal, signal_returns, sizing, sizing_returns,
):
    parameters = research_test_parameters(signal.family, signal=signal, sizing=sizing)
    strategy = build_registered_strategy(signal.family, parameters.payload())
    schedule = pd.DataFrame({"Signal_Cutoff": pd.to_datetime(["2026-01-30", "2026-02-27"])})
    requirements = build_live_research_requirements(strategy, schedule)
    first = pd.Timestamp("2026-01-31")
    assert requirements.last_signal == pd.Timestamp("2026-02-28")
    assert requirements.earliest_price_label == first - pd.offsets.MonthEnd(max(signal_returns, sizing_returns))
    assert requirements.earliest_sector_label == (
        first - pd.offsets.MonthEnd(signal_returns) if signal.family == "sector_momentum" else None
    )


def test_fixed_exposure_live_report_runs_without_reference_warmup():
    inputs = _historical_inputs()
    strategy = MomentumStrategy(momentum_test_parameters(exposure=ExposureParameters(1.5, 0.5)))
    history = prepare_live_research_history(inputs, strategy)
    config = replace(DEFAULT_CONFIG, strategy=strategy,
                     paths=DEFAULT_CONFIG.paths.for_strategy("momentum"))
    result = run_strategy_analysis(inputs, config, research_history=history)
    assert result.decisions.Strategy_ID.eq("momentum").all()
    assert result.nav.End_NAV.notna().all()
    assert result.research_diagnostics.Target_Gross.eq(1.5).all()
    assert not any("Reference" in c for c in result.research_diagnostics)
    residual = result.sector_residuals.groupby("Rebalance_ID").Applied_Net_Dollars.sum()
    invested = result.nav.loc[result.nav.Period_Type.eq("invested")].set_index("Rebalance_ID")
    pd.testing.assert_series_equal(residual.reindex(invested.index),
        invested.Post_Trade_Signed_Market_Value, check_names=False, rtol=0, atol=1e-8)


def test_missing_historical_membership_fails_before_sector_output():
    inputs = _synthetic_inputs()
    strategy = SectorMomentumStrategy(momentum_test_parameters(
        signal=SignalParameters("sector_momentum", 12, 1, None, None),
        selection=WholeSectorSelection(3, 3)))
    with pytest.raises(ValueError, match="Historical membership missing"):
        prepare_live_research_history(inputs, strategy)


def test_sector_members_are_not_filtered_using_next_month_availability():
    inputs = _historical_inputs()
    strategy = SectorMomentumStrategy(momentum_test_parameters(
        signal=SignalParameters("sector_momentum", 1, 0, None, None),
        selection=WholeSectorSelection(3, 3)))
    end = pd.Timestamp(inputs.schedule.Signal_Cutoff.min()) + pd.offsets.MonthEnd(0)
    member = inputs.membership.Asset_ID.iloc[0]
    monthly = inputs.market_monthly.loc[~(inputs.market_monthly.Asset_ID.eq(member)
                                        & inputs.market_monthly.Month.eq(end))]
    with pytest.raises(ValueError, match=f"Unvalueable starting constituent {member}"):
        prepare_live_research_history(replace(inputs, market_monthly=monthly), strategy)


def test_research_constructs_monthly_matrices_once_per_decision(monkeypatch):
    from live import research_history
    inputs = _historical_inputs()
    strategy = MomentumStrategy(momentum_test_parameters())
    calls = []
    original = research_history.monthly_matrices
    def matrices(monthly, *, cutoff):
        calls.append(pd.Timestamp(cutoff))
        return original(monthly, cutoff=cutoff)
    monkeypatch.setattr(research_history, "monthly_matrices", matrices)
    history = prepare_live_research_history(inputs, strategy)
    assert calls == []
    config = replace(DEFAULT_CONFIG, strategy=strategy, paths=DEFAULT_CONFIG.paths.for_strategy("momentum"))
    result = run_strategy_analysis(inputs, config, research_history=history)
    assert calls == list(pd.to_datetime(inputs.schedule.Signal_Cutoff))
    expected = inputs.schedule.set_index("Rebalance_ID").Signal_Cutoff
    actual = result.decisions.groupby("Rebalance_ID").Signal_Source_Max_Date.first()
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_preparation_validates_every_cutoff_without_building_matrices(monkeypatch):
    from live import research_history
    inputs = _historical_inputs()
    first = pd.Timestamp(inputs.schedule.Signal_Cutoff.min())
    monthly = inputs.market_monthly.copy()
    row = monthly.index[monthly.Month.eq(first + pd.offsets.MonthEnd(0)) & monthly.Close.notna()][0]
    monthly.loc[row, "Available_Date"] = first + pd.Timedelta(days=1)
    def unexpected_matrices(*args, **kwargs):
        pytest.fail("Preparation only needs validated monthly rows")
    monkeypatch.setattr(research_history, "monthly_matrices", unexpected_matrices)
    with pytest.raises(ValueError, match="future evidence"):
        prepare_live_research_history(replace(inputs, market_monthly=monthly),
            MomentumStrategy(momentum_test_parameters()))


@pytest.mark.parametrize("future_asset", ["OLD", "NEW"], ids=["predecessor", "successor"])
def test_event_returns_reject_future_dividend_evidence(future_asset):
    start, end = pd.to_datetime(["2025-01-31", "2025-02-28"])
    actions = _successor_bundle([{
        "event_id": "EXCHANGE", "effective_date": "2025-02-15",
        "event_type": "stock_exchange", "continuity_class": "predecessor_extinguished",
        "from_asset_id": "OLD", "to_asset_id": "NEW", "leg_type": "stock", "quantity": 1,
    }])
    actions.events["Effective_Date"] = pd.to_datetime(actions.events.Effective_Date)
    monthly = pd.DataFrame({
        "Month": [start, end], "Asset_ID": ["OLD", "NEW"],
        "Observation_Date": [start, end], "Available_Date": [start, end],
        "Close": [100., 120.], "Nominal_Close": [100., 120.],
    })
    dividends = pd.DataFrame({
        "Asset_ID": ["OLD", "NEW"], "Factor": [.99, .98],
        "Ex_Date": pd.to_datetime(["2025-02-10", "2025-02-20"]),
        "Available_Date": pd.to_datetime(["2025-02-10", "2025-02-20"]),
    })
    inputs = SimpleNamespace(market_monthly=monthly, monthly_dividends=dividends,
                             corporate_actions=actions)
    assert interval_holding_returns(inputs, ["OLD"], start, end, as_of=end).OLD == pytest.approx(
        120 / .98 / (100 * .99) - 1,
    )
    dividends.loc[dividends.Asset_ID.eq(future_asset), "Available_Date"] = end + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match=f"future dividend evidence for {future_asset}"):
        interval_holding_returns(inputs, ["OLD"], start, end, as_of=end)
