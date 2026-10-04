"""Dated sector signals, complete baskets, state, lifecycle and research coverage."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from backtest.sector_returns import build_sector_returns, interval_holding_returns
from _strategy_test_helpers import decision_context
from backtest.research_grid import expand_grid, readiness_bounds, read_grid
from backtest.engine import _run_backtest_impl, _BacktestAuditCollector
from portfolio_core.accounting_ledger import InfeasibleRebalanceError
from portfolio_core.strategies.sector_momentum import SectorMomentumStrategy, sector_scores
from portfolio_core.strategies.research_parameters import (
    SignalParameters, WholeSectorSelection, SizingParameters, VolatilityProfile,
    BufferParameters,
)
from portfolio_core.strategies.research_state import SelectedSectors
import _research_test_helpers as shared


def packet(f=1, s=0, **kwargs):
    return shared.packet(signal=SignalParameters('sector_momentum', f, s, None, None),
                         selection=kwargs.pop('selection', WholeSectorSelection(1,1)), **kwargs)


def dataset():
    data = shared.extended_dataset()
    assignments = data.sector_assignments.copy()
    assets = list(data.data_close.columns)
    # Two full baskets, each safely above the floor; classification metadata stays valid.
    mapping = {a: ('10' if i < len(assets)//2 else '15') for i,a in enumerate(assets)}
    assignments['GICS_Sector_Code'] = assignments.Asset_ID.map(mapping)
    assignments['Sector'] = assignments.GICS_Sector_Code.map({'10':'Energy','15':'Materials'})
    return replace(data, sector_assignments=assignments)


def context(data, cutoff, previous=None, selected=None, returns=None):
    returns = build_sector_returns(data, cutoff).returns if returns is None else returns
    return decision_context(data, pd.Timestamp(cutoff),
                            pd.Series(dtype=float) if previous is None else previous,
                            selected=selected, sector_returns=returns)


@pytest.mark.parametrize('f,s,first', [(1,0,'2015-02-28'),(1,1,'2015-03-31'),
    (6,0,'2015-07-31'),(6,1,'2015-08-31'),(12,0,'2016-01-31'),(12,1,'2016-02-29')])
def test_signal_windows_and_readiness(f,s,first):
    data = dataset()
    history = build_sector_returns(data, '2017-12-31')
    data = replace(data, sector_return_history=history)
    strategy = SectorMomentumStrategy(packet(f,s))
    cutoff = pd.Timestamp(first)
    assert not strategy.decide(context(data, cutoff-pd.offsets.MonthEnd(1), returns=history.returns)).is_complete
    d = strategy.decide(context(data, cutoff, returns=history.returns))
    assert d.is_complete
    end = cutoff-pd.offsets.MonthEnd(s)
    expected = (1+history.returns.reindex(pd.date_range(end=end, periods=f, freq='ME'))).prod()-1
    np.testing.assert_allclose(d.signal_audit.Momentum, d.signal_audit.Construction_Sector.map(expected))
    assert readiness_bounds(packet(f,s), data)['First_Signal_Lower_Bound'] == first
    gap = history.returns.copy()
    gap.loc[end] = np.nan
    assert not strategy.decide(context(data, cutoff, returns=gap)).is_complete


def test_dated_membership_classification_and_exclusions():
    data = dataset()
    assets = list(data.data_close.columns)
    a = assets[0]
    start, finish = pd.Timestamp('2015-03-31'), pd.Timestamp('2015-04-30')
    assignments = data.sector_assignments.copy()
    moved = assignments.Asset_ID.eq(a) & assignments.As_Of_Date.ge(start)
    assignments.loc[moved, 'GICS_Sector_Code'] = '60'
    assignments.loc[moved, 'Sector'] = 'Real Estate'
    prices = data.data_close.copy()
    prices.loc[finish, assets[1]] = np.nan
    membership = data.pit_matrix.copy()
    membership.loc[start, assets[2]] = False
    data = replace(data, data_close=prices, pit_matrix=membership, sector_assignments=assignments)
    history = build_sector_returns(data, '2015-05-31')
    assert history.returns.loc[:start,'60'].isna().all()
    assert history.returns.at[finish,'60'] == pytest.approx(prices.at[finish,a]/prices.at[start,a]-1)
    rows = history.constituents.query('Start == @start')
    assert assets[2] not in set(rows.Asset_ID)
    assert not rows.set_index('Asset_ID').at[assets[1], 'Included']
    assert rows.set_index('Asset_ID').at[assets[1], 'Exclusion_Reason']
    for _, group in rows[rows.Included].groupby('GICS_Sector_Code'):
        assert group.Basket_Weight.sum() == pytest.approx(1)
    assert history.returns.index.min() == pd.Timestamp('2015-02-28')
    assert not sector_scores(history.returns, finish, 6, 0).notna()['60']
    broken = replace(data, sector_assignments=assignments[~(assignments.Asset_ID.eq(a)&assignments.As_Of_Date.eq(start))])
    with pytest.raises(ValueError, match='Missing exact-date'):
        build_sector_returns(broken, finish)


@pytest.mark.parametrize('sizing', [SizingParameters('equal',None),
    SizingParameters('inverse_volatility',VolatilityProfile(60,24)),
    SizingParameters('inverse_volatility',VolatilityProfile(36,36))])
def test_whole_baskets_sizing_and_execution(sizing):
    data = dataset()
    strategy = SectorMomentumStrategy(packet(sizing=sizing))
    c = context(data, '2018-01-31')
    d = strategy.decide(c)
    assert d.is_complete
    assert len(d.raw_target_weights) == len(c.candidate_asset_ids)
    for side, ids in ((1,d.original_long_asset_ids),(-1,d.original_short_asset_ids)):
        weights = d.raw_target_weights.loc[list(ids)]
        assert weights.sum() == pytest.approx(side*.5)
        base = pd.Series(1.,index=ids) if sizing.method == 'equal' else 1/d.signal_audit.loc[list(ids),'Sizing_Volatility']
        np.testing.assert_allclose(weights, side*.5*base/base.sum())
    eligible = set(c.candidate_asset_ids)-{d.original_long_asset_ids[0]}
    executed = strategy.finalize_for_execution(d, eligible)
    assert set(executed.final_target_weights.index) == eligible
    assert executed.final_target_weights[executed.final_target_weights > 0].sum() == pytest.approx(.5)
    with pytest.raises(InfeasibleRebalanceError):
        strategy.finalize_for_execution(d, set(d.original_long_asset_ids[:9])|set(d.original_short_asset_ids))


def test_sector_buffer_ties_floor_and_incomplete_hold():
    data = dataset()
    c = context(data, '2017-01-31')
    returns = c.sector_returns*0
    prior = SelectedSectors(('15',),('10',))
    strategy = SectorMomentumStrategy(packet(buffer=BufferParameters(True,1.5)))
    d = strategy.decide(replace(c, sector_returns=returns, previous_selected_sectors=prior))
    assert d.sector_basket.selected == prior  # k=1 exit cutoff=1: long '15' loses priority; short '10' retained first.
    assert d.signal_audit.Buffered_Short.any()
    weights = d.raw_target_weights
    assert strategy.validate_prior(weights, c, prior).equals(weights)
    with pytest.raises(InfeasibleRebalanceError, match='whole-sector'):
        strategy.validate_prior(weights.iloc[1:], c, prior)
    # A new constituent/classification invalidates a partial prior basket.
    codes = dict(c.sector_code_by_asset_id)
    codes[d.original_long_asset_ids[0]] = '10'
    with pytest.raises(InfeasibleRebalanceError):
        strategy.validate_prior(weights, replace(c,sector_code_by_asset_id=codes), prior)
    small = tuple(sorted((*d.original_long_asset_ids[:9], *d.original_short_asset_ids)))
    incomplete = strategy.decide(replace(c,candidate_asset_ids=small))
    assert not incomplete.is_complete and incomplete.raw_target_weights.empty


def test_lifecycle_valuation_is_shared_and_unvalueable_is_error():
    from test_portfolio_lifecycle import _events
    data = dataset()
    events, legs, sources = _events()
    events['Effective_Date'] = pd.to_datetime(['2015-02-11','2015-02-21'])
    parent, child = data.data_close.columns[:2]
    legs['From_Asset_ID'] = legs.From_Asset_ID.replace({'PARENT':parent})
    legs['To_Asset_ID'] = legs.To_Asset_ID.replace({'CHILD':child})
    deliveries = pd.DataFrame([dict(Event_ID=events.Event_ID.iloc[0], Asset_ID=child,
        From_Asset_IDs=parent, Execution_Date=pd.Timestamp('2015-02-11'), Reference_Close=50., Volume=1e6)])
    data = replace(data, security_events=events, security_event_legs=legs,
                   security_event_sources=sources, event_delivery_executions=deliveries)
    history = build_sector_returns(data, '2015-02-28')
    row = history.constituents.set_index('Asset_ID').loc[parent]
    assert row.Included
    assert row.Holding_Return == pytest.approx(225/data.data_close.at[pd.Timestamp('2015-01-31'),parent]-1)
    with pytest.raises(RuntimeError,match='Unvalueable'):
        interval_holding_returns(data, ['absent'], pd.Timestamp('2015-01-31'), pd.Timestamp('2015-02-28'))




def tiny_grid():
    grid = read_grid("backtest/grids/sector_momentum_grid.json")
    for key,value in list(grid.items()):
        if isinstance(value,list): grid[key]=value[:1]
    # The synthetic dataset has two sectors; keep its custom mechanics grid explicit.
    grid.update(n_long_sectors=[1], n_short_sectors=[1])
    return grid


def test_grid_incompatibilities_and_rounding_deduplication():
    grid = read_grid('backtest/grids/sector_momentum_grid.json')
    assert grid['n_long_sectors'] == grid['n_short_sectors'] == [3]
    rows,rejected,duplicates = expand_grid(grid)
    assert len(rows)==1296 and len(rejected)==1296 and not duplicates
    assert all('neutral' in r['Reason'].lower() for r in rejected)
    # Smaller custom books still exercise rounding-equivalent buffer deduplication.
    small = tiny_grid()
    small['buffer_profiles'] = grid['buffer_profiles']
    rows,rejected,duplicates = expand_grid(small)
    assert len(rows)==2 and not rejected and len(duplicates)==1
    grid = tiny_grid()
    grid['n_long']=[10]
    with pytest.raises(ValueError,match='fixed stock'):
        expand_grid(grid)


def test_search_regression_resume_and_history(tmp_path):
    from hashlib import sha256
    from backtest.research_search import run_research_search
    grid = tiny_grid()
    grid['sizing_profiles']=[dict(method='inverse_volatility',volatility=dict(window=60,minimum_observations=24)),
        dict(method='inverse_volatility',volatility=dict(window=120,minimum_observations=120))]
    result = run_research_search(dataset(),grid,tmp_path,workers=1)
    assert result['ranked_candidates']==1
    assert run_research_search(dataset(),grid,tmp_path,workers=2)['resumed_candidates']==2
    frame = pd.read_csv(tmp_path/'candidates.csv.gz')
    assert set(frame.Status)=={'ranked','unranked_history'}
    manifest = json.loads((tmp_path/'sector_return_manifest.json').read_text())
    assert manifest['end']=='2023-12-31'
    assert set(manifest['artifacts']) == {
        'sector_returns.csv.gz', 'sector_return_constituents.csv.gz',
        'sector_return_summary.csv.gz',
    }
    for name, digest in manifest['artifacts'].items():
        assert sha256((tmp_path/name).read_bytes()).hexdigest() == digest
        assert not pd.read_csv(tmp_path/name).empty
    assert not list(tmp_path.rglob('*.csv'))


def test_multi_sector_budgets_buffers_and_zero_sizing():
    data = dataset()
    c = context(data, '2018-01-31')
    codes = {a: str(10+5*(i//10)) for i,a in enumerate(c.candidate_asset_ids)}
    # Six baskets: 10 assets each. No stock-ranking truncation within sectors.
    dates = c.sector_returns.index
    returns = pd.DataFrame({code:float(i)/100 for i,code in enumerate(sorted(set(codes.values())))}, index=dates)
    strategy = SectorMomentumStrategy(packet(selection=WholeSectorSelection(2,3), buffer=BufferParameters(True,1.5)))
    prior = SelectedSectors(('25','30'),('10','15','20'))
    c = replace(c,sector_code_by_asset_id=codes,sector_returns=returns,previous_selected_sectors=prior)
    d = strategy.decide(c)
    assert d.sector_basket.selected == SelectedSectors(('30','25'),('10','15','20'))
    group = d.raw_target_weights.groupby(pd.Series(codes)).sum()
    np.testing.assert_allclose(group.loc[['25','30']], [.25,.25])
    np.testing.assert_allclose(group.loc[['10','15','20']], [-1/6]*3)
    assert len(d.original_long_asset_ids)==20 and len(d.original_short_asset_ids)==30
    assert d.signal_audit.Buffered_Long.sum()==20
    prices = c.close_history.copy()
    prices.loc[:, list(codes)[:10]] = 100.
    inverse = SectorMomentumStrategy(packet(selection=WholeSectorSelection(2,3),
        sizing=SizingParameters('inverse_volatility',VolatilityProfile(60,24))))
    outcome = inverse.decide(replace(c,close_history=prices))
    assert outcome.is_complete
    assert '10' not in outcome.sector_basket.eligible_members
    assert outcome.signal_audit.iloc[:10].Strategy_Exclusion_Reason.eq('unavailable_sizing_volatility').all()


def test_incomplete_sector_signal_after_activation_values_hold_and_rejects_partial_basket():
    data = dataset()
    history = build_sector_returns(data,'2016-12-31')
    cutoff = pd.Timestamp('2016-08-31')
    history.returns.loc[cutoff] = np.nan
    data = replace(data,sector_return_history=history)
    strategy = SectorMomentumStrategy(packet())
    audit = _BacktestAuditCollector(strategy)
    dates = pd.date_range('2015-01-31','2016-12-31',freq='ME')
    nav = _run_backtest_impl(data,dates,strategy,strategy_audit=audit)
    assert any(r.get('Used_Hold_Logic') for r in audit.research_records)
    assert nav.loc[cutoff] != nav.loc[cutoff+pd.offsets.MonthEnd(1)]
    # A current member moving into a new sector prevents the held book from
    # remaining complete; no silent subset or frozen NAV is permitted.
    assignments = data.sector_assignments.copy()
    a = data.data_close.columns[0]
    mask = assignments.Asset_ID.eq(a)&assignments.As_Of_Date.eq(cutoff)
    assignments.loc[mask,'GICS_Sector_Code']='60'
    assignments.loc[mask,'Sector']='Real Estate'
    with pytest.raises(InfeasibleRebalanceError, match='whole-sector'):
        _run_backtest_impl(replace(data,sector_assignments=assignments),dates,strategy)


def test_rounding_cannot_drop_one_required_sector_constituent():
    data = dataset()
    prices=data.data_close.copy()
    prices.iloc[:,0]*=1e9  # Same return/rank; no whole voluntary share is affordable.
    data=replace(data,data_close=prices)
    with pytest.raises(InfeasibleRebalanceError):
        _run_backtest_impl(data,pd.date_range('2015-01-31','2015-04-30',freq='ME'),SectorMomentumStrategy(packet()))


def test_prior_basket_hold_accepts_lifecycle_weight_drift():
    data=dataset()
    strategy=SectorMomentumStrategy(packet())
    c=context(data,'2017-01-31')
    d=strategy.decide(c)
    weights=d.raw_target_weights.copy()
    weights.iloc[0]*=1.2
    # The engine updates prior target weights from marked positions following
    # lifecycle events. Hold fallback carries that exposure; it is no new sizing.
    assert strategy.validate_prior(weights,c,d.sector_basket.selected).equals(weights)
    with pytest.raises(InfeasibleRebalanceError):
        strategy.validate_prior(weights.iloc[1:],c,d.sector_basket.selected)
