"""Approved SMA windows, complete two-sided rankings and shared execution."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from portfolio_core.strategies.monthly_trend import MonthlyTrendStrategy
from portfolio_core.strategies.research_parameters import SignalParameters
from backtest.research_grid import expand_grid, read_grid, readiness_bounds
from _strategy_test_helpers import monthly_history, strategy_context
import _research_test_helpers as shared


def packet(window=10, **changes):
    return shared.packet(signal=SignalParameters('monthly_trend', None, None, None, window), **changes)


@pytest.mark.parametrize('window', [3, 6, 9, 10, 12])
@pytest.mark.parametrize('direction', [-1, 1])
def test_windows_same_sign_books_missing_observations_and_execution(window, direction):
    close, volume, assets, sectors = monthly_history(asset_count=60)
    close.iloc[:] = 100
    close.iloc[-1] = 100 + direction * np.linspace(1, 30, 60)
    close.iloc[-1, 1] = close.iloc[-1, 0]
    close.iloc[-2, 2] = np.nan  # Interior gap excludes this asset.
    close.iloc[-window-1, 3] = np.nan  # Outside the window does not.
    strategy = MonthlyTrendStrategy(packet(window))
    d = strategy.decide(strategy_context(close, volume, assets, sectors))
    raw = close.iloc[-1] / close.iloc[-window:].mean() - 1
    raw = raw.where(close.iloc[-window:].count().eq(window)).dropna()
    longs = tuple(sorted(raw.index, key=lambda a: (-raw[a], a)))
    shorts = tuple(sorted(raw.index, key=lambda a: (raw[a], a)))
    assert (raw * direction > 0).all()
    assert d.original_long_asset_ids == longs[:10]
    assert d.original_short_asset_ids == shorts[:10]
    assert (d.raw_target_weights > 0).sum() == (d.raw_target_weights < 0).sum() == 10
    np.testing.assert_allclose(d.signal_audit.loc[raw.index, 'Price_To_SMA'], raw)
    assert not d.signal_audit.loc[assets[2], 'Strategy_Eligible']
    assert d.signal_audit.loc[assets[2], 'Strategy_Exclusion_Reason'] == 'insufficient_signal_history'
    execution = strategy.finalize_for_execution(d, set(assets) - {longs[0], shorts[0]})
    assert execution.final_long_asset_ids == longs[1:11]
    assert execution.final_short_asset_ids == shorts[1:11]


def test_missing_calendar_month_short_history_nonfinite_and_incomplete():
    close, volume, assets, sectors = monthly_history(asset_count=40)
    strategy = MonthlyTrendStrategy(packet(3))
    for prices in (close.iloc[-2:], close.drop(close.index[-2])):
        d = strategy.decide(strategy_context(prices, volume.reindex(prices.index), assets, sectors))
        assert not d.is_complete and d.raw_target_weights.empty
    close.iloc[-2, :25] = np.inf
    d = strategy.decide(strategy_context(close, volume, assets, sectors))
    assert not d.is_complete
    assert d.signal_audit.Strategy_Eligible.sum() == 15


def test_readiness_is_independent_of_future_prices():
    data = shared.extended_dataset()
    cutoff = pd.Timestamp('2016-01-31')
    changed = data.data_close.copy()
    changed.loc[changed.index > cutoff] *= np.linspace(.1, 5, len(changed.columns))
    later = replace(data, data_close=changed)
    assert readiness_bounds(packet(), data) == readiness_bounds(packet(), later)



def test_complete_grid_and_invalid_sign_gate_or_windows():
    grid = read_grid("backtest/grids/monthly_trend_grid.json")
    rows, rejected, duplicates = expand_grid(grid)
    assert (len(rows), len(rejected), len(duplicates)) == (5760, 2880, 0)
    assert {json.loads(r['Parameters_JSON'])['signal']['moving_average_months'] for r in rows} == {3, 6, 9, 10, 12}
    for window in (1, 0, -1):
        with pytest.raises(ValueError):
            packet(window)
    bad = read_grid("backtest/grids/monthly_trend_grid.json")
    bad['signal_profiles'] = [dict(moving_average_months=10, sign_gate=True)]
    rows, rejected, _ = expand_grid(bad)
    assert not rows and rejected
    with pytest.raises(ValueError):
        MonthlyTrendStrategy(shared.packet())


@pytest.mark.parametrize('check', [
    shared.assert_sizing_contract,
    shared.assert_neutral_execution_contract,
])
def test_shared_execution_contracts(check):
    check(MonthlyTrendStrategy, packet())


def test_search_serial_parallel_resume(tmp_path):
    grid = shared.tiny_grid('monthly_trend')
    grid['signal_profiles'] = [dict(moving_average_months=10)]
    shared.assert_search_serial_parallel_resume(tmp_path, grid)


def test_preflight_caches_each_signal_window_separately(monkeypatch, tmp_path):
    import backtest.research_search as search
    grid = shared.tiny_grid()
    grid.update(strategy_id='monthly_trend', signal_profiles=[dict(moving_average_months=w) for w in (3, 12)])
    seen = []
    original = search.readiness_bounds
    def bounds(p, data, *, evaluation_config):
        seen.append(p.signal.moving_average_months)
        assert evaluation_config is search.DEFAULT_CONFIG.evaluation
        return original(p, data, evaluation_config=evaluation_config)
    monkeypatch.setattr(search, 'readiness_bounds', bounds)
    search.run_research_search(shared.extended_dataset(), grid, tmp_path, preflight_only=True)
    assert seen == [3, 12]


def test_monthly_reassessment_reverses_books_and_flat_ties_keep_both_sides():
    close, volume, assets, sectors = monthly_history(asset_count=40)
    close.iloc[:] = 100
    strategy = MonthlyTrendStrategy(packet(3))
    flat = strategy.decide(strategy_context(close, volume, assets, sectors))
    assert flat.is_complete
    assert len(set(flat.original_long_asset_ids + flat.original_short_asset_ids)) == 20
    close.iloc[-1] = np.linspace(90, 110, len(assets))
    first = strategy.decide(strategy_context(close, volume, assets, sectors))
    next_date = close.index[-1] + pd.offsets.MonthEnd(1)
    close.loc[next_date] = np.linspace(120, 80, len(assets))
    volume.loc[next_date] = 1e6
    following = strategy.decide(replace(strategy_context(close, volume, assets, sectors),
                                       previous_target_weights=first.raw_target_weights))
    assert set(following.original_long_asset_ids) == set(first.original_short_asset_ids)
    assert set(following.original_short_asset_ids) == set(first.original_long_asset_ids)
