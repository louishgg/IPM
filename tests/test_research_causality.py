"""Future-data mutations cross the production boundaries with unsliced inputs."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.engine import _BacktestAuditCollector, _run_backtest_impl
from backtest.sector_returns import build_sector_returns
from live.monthly_history import aggregate_monthly_market
from live.research_history import LiveResearchHistory
from portfolio_core.sector_assignments import sector_rows_asof
from portfolio_core.strategies import build_registered_strategy
from portfolio_core.strategies.research_parameters import SizingParameters, VolatilityProfile
from portfolio_core.strategies.research_state import SelectedSectors
from _strategy_test_helpers import RESEARCH_STRATEGY_IDS, research_test_parameters
from test_engine_offline import make_synthetic_backtest_data


@pytest.fixture(scope="module")
def causal_inputs():
    data, dates = make_synthetic_backtest_data()
    cutoff = dates[-2]
    history = build_sector_returns(data, dates[-1])
    original = replace(data, sector_return_history=history)
    prices, volume = data.data_close.copy(), data.data_volume.copy()
    prices.loc[prices.index > cutoff] *= np.linspace(0.8, 1.2, len(prices.columns))
    volume.loc[volume.index > cutoff] *= 1000
    returns = history.returns.copy()
    returns.loc[returns.index > cutoff] = np.linspace(-0.9, 9, len(returns.columns))
    changed = replace(original, data_close=prices, data_volume=volume,
                      sector_return_history=replace(history, returns=returns))
    return original, changed, cutoff, dates[-2:]


@pytest.mark.parametrize("strategy_id", RESEARCH_STRATEGY_IDS)
@pytest.mark.parametrize("domain", ["backtest", "live"])
def test_research_decisions_ignore_future_inputs(causal_inputs, strategy_id, domain):
    original, changed, cutoff, dates = causal_inputs
    parameters = research_test_parameters(
        strategy_id, sizing=SizingParameters("inverse_volatility", VolatilityProfile(60, 24)),
    )
    strategy = build_registered_strategy(strategy_id, parameters.payload())

    def decide(data):
        if domain == "backtest":
            audit = _BacktestAuditCollector(strategy)
            _run_backtest_impl(data, dates, strategy, strategy_audit=audit)
            rows = pd.DataFrame(audit.decision_records)
            assert not rows.empty and rows.Decision_Complete.all()
            # Future valuation may change NAV; the cutoff decision must not change.
            return rows
        daily = pd.concat({"Close": data.data_close.stack(future_stack=True),
                           "Volume": data.data_volume.stack(future_stack=True)}, axis=1)
        daily = daily.rename_axis(["Date", "Asset_ID"]).reset_index()
        monthly = aggregate_monthly_market(daily.assign(Price_Source="yahoo"), through=dates[-1])
        history = LiveResearchHistory(data.sector_return_history.returns)
        context = history.context(
            SimpleNamespace(market_monthly=monthly), cutoff, data.data_close.columns,
            sector_rows_asof(data.sector_assignments, cutoff).GICS_Sector_Code.to_dict(),
            pd.Series(dtype=float), SelectedSectors(),
        )
        assert context.signal_source_max_date == cutoff
        assert context.close_history.index.max() == cutoff
        assert context.sector_returns.index.max() == cutoff
        decision = strategy.decide(context)
        assert decision.is_complete
        return decision

    baseline, mutated = decide(original), decide(changed)
    if domain == "backtest":
        pd.testing.assert_frame_equal(baseline, mutated)
    else:
        pd.testing.assert_frame_equal(baseline.signal_audit, mutated.signal_audit)
        pd.testing.assert_series_equal(baseline.raw_target_weights, mutated.raw_target_weights)
        assert baseline.original_long_asset_ids == mutated.original_long_asset_ids
        assert baseline.original_short_asset_ids == mutated.original_short_asset_ids
