"""Reversal direction, causal execution/reference and complete overlay coverage."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from portfolio_core.strategies.reversal import ReversalStrategy
from portfolio_core.strategies.research_parameters import SignalParameters
from backtest.research_grid import expand_grid, read_grid
from _strategy_test_helpers import monthly_history, strategy_context
import _research_test_helpers as shared


def packet(**changes):
    return shared.packet(signal=SignalParameters('reversal', 1, 0, None, None), **changes)


def test_endpoints_direction_ties_missing_and_execution_refill():
    close, volume, assets, sectors = monthly_history(asset_count=60)
    close.iloc[-2] = 100
    close.iloc[-1] = np.linspace(70, 130, 60)
    close.iloc[-1, 1] = close.iloc[-1, 0]  # Stable Asset_ID tie break.
    close.iloc[-2, 2] = np.nan
    context = strategy_context(close, volume, assets, sectors)
    strategy = ReversalStrategy(packet())
    d = strategy.decide(context)
    returns = (close.iloc[-1] / close.iloc[-2] - 1).dropna()
    expected_long = tuple(sorted(returns.index, key=lambda a: (returns[a], a)))
    expected_short = tuple(sorted(returns.index, key=lambda a: (-returns[a], a)))
    assert d.original_long_asset_ids == expected_long[:10]
    assert d.original_short_asset_ids == expected_short[:10]
    np.testing.assert_allclose(d.signal_audit.loc[returns.index, 'Strategy_Score'], -returns)
    np.testing.assert_allclose(d.signal_audit.loc[returns.index, 'Prior_Month_Return'], returns)
    eligible = set(assets) - {expected_long[0], expected_short[0]}
    execution = strategy.finalize_for_execution(d, eligible)
    assert execution.final_long_asset_ids == expected_long[1:11]
    assert execution.final_short_asset_ids == expected_short[1:11]


@pytest.mark.parametrize('formation,skip', [(2, 1), (4, 2)])
def test_custom_reversal_window_has_matching_scores_and_audit_label(formation, skip):
    close, volume, assets, sectors = monthly_history(asset_count=60)
    parameters = shared.packet(signal=SignalParameters('reversal', formation, skip, None, None))
    strategy = ReversalStrategy(parameters)
    decision = strategy.decide(strategy_context(close, volume, assets, sectors))
    expected = (close.shift(skip) / close.shift(formation + skip) - 1).iloc[-1]
    assert 'Prior_Month_Return' not in decision.signal_audit
    np.testing.assert_allclose(decision.signal_audit.Formation_Return, expected)
    np.testing.assert_allclose(decision.signal_audit.Strategy_Score, -expected)


def test_complete_reversal_grid_and_signal_contract():
    grid = read_grid("backtest/grids/reversal_grid.json")
    rows, rejected, duplicates = expand_grid(grid)
    assert (len(rows), len(rejected), len(duplicates)) == (1152, 576, 0)
    for row in rows:
        p = json.loads(row['Parameters_JSON'])
        assert p['signal']['family'] == 'reversal'
        assert (p['signal']['formation_months'], p['signal']['skip_months']) == (1, 0)
    for f, s in ((0, 0), (1, -1)):
        with pytest.raises(ValueError):
            SignalParameters('reversal', f, s, None, None)
    with pytest.raises(ValueError):
        ReversalStrategy(shared.packet())


@pytest.mark.parametrize('check', [
    shared.assert_sizing_contract,
    shared.assert_neutral_execution_contract,
])
def test_shared_execution_contracts_for_reversal(check):
    check(ReversalStrategy, packet())


def test_reversal_search_serial_parallel_resume(tmp_path):
    shared.assert_search_serial_parallel_resume(tmp_path, shared.tiny_grid('reversal'))


def test_monthly_reassessment_can_reverse_entire_book_without_cohorts():
    close, volume, assets, sectors = monthly_history(asset_count=40)
    close.iloc[-2] = 100
    close.iloc[-1] = np.linspace(80, 120, len(assets))
    strategy = ReversalStrategy(packet())
    initial = strategy.decide(strategy_context(close, volume, assets, sectors))
    next_date = close.index[-1] + pd.offsets.MonthEnd(1)
    close.loc[next_date] = close.iloc[-1] * np.linspace(1.2, 0.8, len(assets))
    volume.loc[next_date] = 1e6
    context = replace(strategy_context(close, volume, assets, sectors),
                      previous_target_weights=initial.raw_target_weights)
    following = strategy.decide(context)
    assert following.original_long_asset_ids == initial.original_short_asset_ids
    assert following.original_short_asset_ids == initial.original_long_asset_ids
    assert len(following.raw_target_weights) == 20
