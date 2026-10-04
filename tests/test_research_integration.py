"""Development boundary and interrupted-search integration regressions."""
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from backtest import data_loading as loading
from backtest import research_search as search
from _research_test_helpers import extended_dataset, tiny_grid


def test_monthly_prefix_never_parses_later_numerical_observations(tmp_path):
    path = tmp_path / 'prices.csv'
    path.write_text('Date,Asset_ID,Price_Close,Volume\n2023-12-31,A,100,10\n'
                    '2024-01-31,A,not-a-price,not-a-volume\n')
    frame = pd.read_csv(loading._monthly_prefix(path, '2023-12-31'))
    assert frame.Price_Close.tolist() == [100]
    assert frame.Volume.tolist() == [10]


def test_development_loader_bounds_all_valuation_inputs(monkeypatch, tmp_path):
    cutoff = pd.Timestamp('2023-12-31')
    dates = pd.to_datetime(['2023-11-30', '2023-12-31', '2024-01-31'])
    prices = pd.DataFrame({'A': [100., 101.]}, index=dates[:2])
    paths = SimpleNamespace(price_sources=SimpleNamespace(yahoo_close_csv=tmp_path/'absent'),
                            project_root=tmp_path)
    monkeypatch.setattr(loading, 'validate_preparation_manifest', lambda **kw: None)
    def price_loader(config, paths, **kw):
        assert pd.Timestamp(config.end_date) == cutoff
        assert kw == {'development_only': True}
        return prices, prices
    monkeypatch.setattr(loading, 'load_price_data', price_loader)
    monkeypatch.setattr(loading, 'load_pit_universe', price_loader)
    monkeypatch.setattr(loading, 'load_asset_metadata', lambda paths: pd.DataFrame())
    monkeypatch.setattr(loading, 'load_sector_assignments', lambda paths:
                        pd.DataFrame({'As_Of_Date': dates}))
    events = pd.DataFrame({'Event_ID': ['early', 'late'], 'Effective_Date': dates[1:]})
    links = pd.DataFrame({'Event_ID': ['early', 'late']})
    monkeypatch.setattr(loading, 'load_security_event_data', lambda paths: (events, links, links))
    mappings = pd.DataFrame(columns=['Scope', 'Provider', 'Review_Status'])
    monkeypatch.setattr(loading, 'load_security_identity_bundle', lambda path:
                        SimpleNamespace(provider_mappings=mappings))
    monkeypatch.setattr(loading, 'load_price_basis', lambda paths: None)
    def derive(**kw):
        assert kw['events'].Event_ID.tolist() == ['early']
        assert kw['legs'].Event_ID.tolist() == ['early']
        assert kw['observations'].empty
        return pd.DataFrame()
    monkeypatch.setattr(loading, 'derive_event_delivery_executions', derive)
    monkeypatch.setattr(loading, 'assemble_backtest_dataset', lambda **kw: kw)
    result = loading.load_backtest_data(paths=paths, development_only=True)
    assert result['sector_assignments'].As_Of_Date.tolist() == [dates[0]]
    assert result['security_event_sources'].Event_ID.tolist() == ['early']
    with pytest.raises(ValueError, match='Brinson'):
        loading.load_backtest_data(paths=paths, development_only=True, include_brinson=True)


def test_interrupted_search_resumes_completed_checkpoint(monkeypatch, tmp_path):
    data, grid = extended_dataset(), tiny_grid()
    grid['gross'] = [1., 1.5]
    original = search._evaluate
    def interrupt_second(row):
        if row['Candidate_Number'] == 2:
            raise RuntimeError('simulated interruption')
        return original(row)
    monkeypatch.setattr(search, '_evaluate', interrupt_second)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        search.run_research_search(data, grid, tmp_path)
    checkpoints = list((tmp_path/'candidates').glob('*/checkpoint.json'))
    assert len(checkpoints) == 1
    before = checkpoints[0].read_bytes()
    monkeypatch.setattr(search, '_evaluate', original)
    result = search.run_research_search(data, grid, tmp_path)
    assert result['mode'] == 'full'
    assert result['selected_candidate_numbers'] == [1, 2]
    assert result['completed_candidates'] == 2
    assert result['resumed_candidates'] == 1
    assert checkpoints[0].read_bytes() == before
    timings = json.loads((tmp_path/'timing.json').read_text())
    assert all(row['evaluation_seconds'] > 0 for row in timings['candidates'])
    assert timings['resumed_candidates'] == 1
