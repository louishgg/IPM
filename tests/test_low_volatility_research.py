"""Linked low-volatility selection, independent sizing and causal readiness."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from portfolio_core.strategies.low_volatility import LowVolatilityStrategy
from portfolio_core.strategies.portfolio_construction import stock_volatility
from portfolio_core.strategies.research_parameters import SignalParameters, VolatilityProfile, SizingParameters
from backtest.research_grid import expand_grid, read_grid, readiness_bounds
from _strategy_test_helpers import decision_context
from _strategy_test_helpers import strategy_context
import _research_test_helpers as shared


def packet(profile=(60, 24), **changes):
    return shared.packet(signal=SignalParameters('low_volatility', None, None, VolatilityProfile(*profile), None), **changes)


def context(prices):
    return strategy_context(prices, prices * 0 + 1e6, tuple(prices.columns), {a: '10' for a in prices})


@pytest.mark.parametrize('profile', [(36, 36), (60, 24)])
def test_minima_exact_window_direction_missing_and_execution(profile):
    data = shared.extended_dataset()
    w, m = profile
    prices = data.data_close.iloc[:m]
    strategy = LowVolatilityStrategy(packet(profile))
    assert not strategy.decide(context(prices)).is_complete
    prices = data.data_close.iloc[:m+1]
    d = strategy.decide(context(prices))
    assert d.is_complete  # 60/24 does not require 60 returns.
    expected = prices.pct_change(fill_method=None).iloc[-w:].std(ddof=1)
    np.testing.assert_allclose(d.signal_audit.Selection_Volatility, expected)
    longs = tuple(sorted(expected.index, key=lambda a: (expected[a], a)))
    shorts = tuple(sorted(expected.index, key=lambda a: (-expected[a], a)))
    assert d.original_long_asset_ids == longs[:10]
    assert d.original_short_asset_ids == shorts[:10]
    execution = strategy.finalize_for_execution(d, set(prices.columns)-{longs[0], shorts[0]})
    assert execution.final_long_asset_ids == longs[1:11]
    assert execution.final_short_asset_ids == shorts[1:11]
    missing = prices.copy()
    missing.iloc[-2, 0] = np.nan
    md = strategy.decide(context(missing))
    assert not md.signal_audit.iloc[0].Strategy_Eligible
    assert md.signal_audit.iloc[0].Strategy_Exclusion_Reason == 'insufficient_signal_history'
    # Calendar gaps are not compressed into adjacent-month returns.
    assert not strategy.decide(context(prices.drop(prices.index[-2]))).is_complete
    history = data.data_close.iloc[:80].copy()
    tail = history.pct_change(fill_method=None).iloc[-w:]
    np.testing.assert_allclose(stock_volatility(history, VolatilityProfile(*profile)).iloc[-1], tail.std(ddof=1))
    history.iloc[:10] *= 3
    np.testing.assert_allclose(stock_volatility(history, VolatilityProfile(*profile)).iloc[-1], tail.std(ddof=1))


@pytest.mark.parametrize('selection', [(36, 36), (60, 24)])
@pytest.mark.parametrize('sizing', [(36, 36), (60, 24)])
def test_selection_and_sizing_are_independent(selection, sizing):
    prices = shared.extended_dataset().data_close.iloc[:70].copy()
    prices.iloc[-1, 0] = np.nan
    prices.iloc[-1, 1] = np.inf
    prices.iloc[:, 2] = 100  # Zero is a valid selection signal.
    equal = LowVolatilityStrategy(packet(selection)).decide(context(prices))
    assert equal.signal_audit.iloc[2].Strategy_Eligible
    assert equal.signal_audit.iloc[2].Selection_Volatility == 0
    d = LowVolatilityStrategy(packet(selection, sizing=SizingParameters('inverse_volatility', VolatilityProfile(*sizing)))).decide(context(prices))
    assert not d.signal_audit.iloc[2].Strategy_Eligible
    assert d.signal_audit.iloc[2].Strategy_Exclusion_Reason == 'unavailable_sizing_volatility'
    for field, profile in [('Selection_Volatility', selection), ('Sizing_Volatility', sizing)]:
        clean = prices.where(np.isfinite(prices)).pct_change(fill_method=None).iloc[-profile[0]:]
        expected = clean.std(ddof=1).where(clean.count() >= profile[1])
        np.testing.assert_allclose(d.signal_audit[field], expected, atol=1e-14, equal_nan=True)
    assert d.is_complete
    for ids, side in [(d.original_long_asset_ids, 1), (d.original_short_asset_ids, -1)]:
        inverse = 1/d.signal_audit.loc[list(ids), 'Sizing_Volatility']
        np.testing.assert_allclose(d.raw_target_weights.loc[list(ids)], side*.5*inverse/inverse.sum())


def test_zero_ties_missing_and_nonfinite_inputs():
    prices = shared.extended_dataset().data_close.iloc[:50]*0+100
    d = LowVolatilityStrategy(packet()).decide(context(prices))
    assert d.is_complete and len(set(d.original_long_asset_ids+d.original_short_asset_ids)) == 20
    assert not LowVolatilityStrategy(packet(sizing=SizingParameters('inverse_volatility', VolatilityProfile(60,24)))).decide(context(prices)).is_complete
    prices.iloc[-30:] = np.nan
    assert not LowVolatilityStrategy(packet()).decide(context(prices)).is_complete
    prices.iloc[-30:] = np.inf
    assert not LowVolatilityStrategy(packet()).decide(context(prices)).is_complete


def test_fixed_exposure_and_readiness():
    data = shared.extended_dataset()
    cutoff = pd.Timestamp('2017-01-31')
    strategy = LowVolatilityStrategy(packet())
    changed = data.data_close.copy()
    changed.loc[changed.index > cutoff] *= 4
    later = replace(data, data_close=changed)
    a = strategy.decide(decision_context(data, cutoff, pd.Series(dtype=float)))
    assert a.raw_target_weights.abs().sum() == pytest.approx(1)
    bounds = readiness_bounds(packet(), data)
    assert bounds['First_Construction_Lower_Bound'] == '2016-01-31'
    assert not bounds['History_Unready']
    assert bounds == readiness_bounds(packet(), later)



def test_complete_grid_profiles_and_rejections():
    grid = read_grid("backtest/grids/low_volatility_grid.json")
    rows, rejected, duplicates = expand_grid(grid)
    assert (len(rows),len(rejected),len(duplicates)) == (1152,576,0)
    pairs = set()
    for r in rows:
        p = json.loads(r['Parameters_JSON'])
        sel = p['signal']['selection_volatility']['window']
        size = p['sizing']['volatility']
        pairs.add((sel, None if size is None else size['window']))
    assert pairs == {(s,z) for s in (60,) for z in (None,60)}
    with pytest.raises(ValueError):
        packet((24,36))
    with pytest.raises(ValueError):
        LowVolatilityStrategy(shared.packet())


def test_search_serial_parallel_resume(tmp_path):
    grid = shared.tiny_grid('low_volatility')
    shared.assert_search_serial_parallel_resume(tmp_path, grid)
    shared.assert_unready_branch(tmp_path / 'unready', grid)


def test_preflight_separates_linked_selection_profiles(monkeypatch,tmp_path):
    import backtest.research_search as search
    grid = shared.tiny_grid()
    grid.update(strategy_id='low_volatility', signal_profiles=[
        dict(selection_volatility=dict(window=window, minimum_observations=minimum))
        for window, minimum in ((36, 36), (60, 24))
    ])
    seen=[]
    original=search.readiness_bounds
    def bounds(p, data, *, evaluation_config):
        assert evaluation_config is search.DEFAULT_CONFIG.evaluation
        result = original(p, data, evaluation_config=evaluation_config)
        seen.append((p.signal.selection_volatility, result['First_Construction_Lower_Bound']))
        return result
    monkeypatch.setattr(search,'readiness_bounds',bounds)
    search.run_research_search(shared.extended_dataset(),grid,tmp_path,preflight_only=True)
    assert seen == [
        (VolatilityProfile(36, 36), '2017-01-31'),
        (VolatilityProfile(60, 24), '2016-01-31'),
    ]
