"""JSON owns choices; schema and execution retain domain/compatibility checks."""

from dataclasses import replace
import json

import pytest

from backtest.research_grid import expand_grid, read_grid
import backtest.search as strategy_search
from portfolio_core.strategies import configuration, registry
from portfolio_core.strategies.research_parameters import (
    BufferParameters, ExposureParameters, SignalParameters, VolatilityProfile,
)
from _strategy_test_helpers import research_contract_parameters


def test_json_edits_control_registered_defaults_and_default_search(tmp_path, monkeypatch):
    defaults = configuration.read_json_object(configuration.DEFAULTS_PATH)
    defaults["momentum"]["signal"]["formation_months"] = 8
    defaults["momentum"]["turnover_threshold"] = 0.02
    path = tmp_path / "defaults.json"
    path.write_text(json.dumps(defaults))
    grid = read_grid("backtest/grids/momentum_grid.json")
    grid["signal_profiles"] = [dict(formation_months=4, skip_months=2)]
    (tmp_path / "momentum_grid.json").write_text(json.dumps(grid))
    monkeypatch.setattr(configuration, "DEFAULTS_PATH", path)
    monkeypatch.setattr(strategy_search, "GRIDS_DIRECTORY", tmp_path)
    selected = registry.get_strategy_definition("momentum")
    assert selected.build().parameters.signal.formation_months == 8
    assert selected.build().parameters.turnover_threshold == 0.02
    received = []
    monkeypatch.setattr(strategy_search, "load_backtest_data", lambda *a, **k: object())
    from backtest import research_search
    monkeypatch.setattr(research_search, "run_research_search", lambda data, grid, *a, **k: received.append(grid))
    strategy_search.main(["--strategy", "momentum", "--preflight-only"])
    assert received == [grid]
    path.write_text("{}")
    assert selected.strategy_id == "momentum"
    with pytest.raises(ValueError):
        selected.build()


@pytest.mark.parametrize('change', [
    lambda d: d['selection'].pop('n_long'),
    lambda d: d.update(unknown=1),
    lambda d: d['selection'].update(n_long=True),
    lambda d: d['exposure'].update(gross=True),
    lambda d: d['buffer'].update(enabled=1),
    lambda d: d.update(turnover_threshold=-1),
])
def test_invalid_json_defaults_fail_without_python_fallback(tmp_path, monkeypatch, change):
    values = configuration.parameter_defaults('momentum')
    change(values)
    path = tmp_path / 'defaults.json'
    path.write_text(json.dumps({'momentum': values}))
    monkeypatch.setattr(configuration, 'DEFAULTS_PATH', path)
    with pytest.raises(ValueError):
        registry.build_registered_strategy('momentum')
    with pytest.raises(ValueError):
        registry.build_registered_strategy('momentum', values)


def test_nested_momentum_defaults_share_explicit_packet_validation(tmp_path, monkeypatch):
    values = configuration.parameter_defaults("momentum")
    values["signal"]["formation_months"] = 8
    path = tmp_path / "defaults.json"
    path.write_text(json.dumps({"momentum": values}))
    monkeypatch.setattr(configuration, "DEFAULTS_PATH", path)
    default = registry.build_registered_strategy("momentum")
    assert default.parameters.signal.formation_months == 8
    assert default.parameters == registry.build_registered_strategy("momentum", values).parameters
    values["signal"]["family"] = "reversal"
    path.write_text(json.dumps({"momentum": values}))
    for supplied in (None, values):
        with pytest.raises(ValueError, match="requires signal.family='momentum'"):
            registry.build_registered_strategy("momentum", supplied)


@pytest.mark.parametrize('text', ['[]', '{', '{"gross": [1], "gross": [2]}'])
def test_grid_reader_rejects_malformed_or_duplicate_json(tmp_path, text):
    path = tmp_path / 'invalid.json'
    path.write_text(text)
    with pytest.raises(ValueError):
        read_grid(path)


@pytest.mark.parametrize('family,signal', [
    ('momentum', dict(formation_months=4, skip_months=2)),
    ('reversal', dict(formation_months=2, skip_months=1)),
    ('monthly_trend', dict(moving_average_months=7)),
    ('low_volatility', dict(selection_volatility=dict(window=20, minimum_observations=10))),
    ('sector_momentum', dict(formation_months=4, skip_months=2)),
])
def test_custom_json_choices_need_no_python_allowlist(tmp_path, family, signal):
    grid = read_grid(f'backtest/grids/{family}_grid.json')
    grid = {key: value[:1] if isinstance(value, list) else value for key, value in grid.items()}
    grid.update(
        signal_profiles=[signal],
        sizing_profiles=[dict(method='inverse_volatility', volatility=dict(window=18, minimum_observations=12))],
        buffer_profiles=[dict(enabled=True, exit_multiplier=1.4)],
        turnover_threshold=[0.02], sector_neutral=[False], long_share=[0.55], gross=[1.2],
    )
    if family == 'sector_momentum':
        grid.update(n_long_sectors=[4], n_short_sectors=[4])
    else:
        grid.update(n_long=[12], n_short=[11])
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps(grid))
    rows, rejected, duplicates = expand_grid(read_grid(path))
    assert len(rows) == 1 and not rejected and not duplicates
    value = json.loads(rows[0]['Parameters_JSON'])
    assert value['signal']['family'] == family
    assert value['exposure'] == dict(gross=1.2, long_share=0.55)
    assert value['turnover_threshold'] == 0.02


@pytest.mark.parametrize('make', [
    lambda: VolatilityProfile(1, 1),
    lambda: VolatilityProfile(12, 13),
    lambda: VolatilityProfile(12, True),
    lambda: SignalParameters('momentum', 0, 1, None, None),
    lambda: SignalParameters('reversal', 1, -1, None, None),
    lambda: SignalParameters('monthly_trend', None, None, None, 1),
    lambda: BufferParameters(True, 0.9),
    lambda: BufferParameters(True, float('inf')),
    lambda: BufferParameters(True, True),
    lambda: ExposureParameters(-1, 0.5),
    lambda: ExposureParameters(True, 0.5),
    lambda: ExposureParameters(1, 1),
    lambda: ExposureParameters(float("inf"), 0.5),
    lambda: ExposureParameters(1, float("nan")),
    lambda: ExposureParameters(1, 0),
    lambda: replace(research_contract_parameters(), turnover_threshold=-0.01),
])
def test_domain_constraints_still_reject_invalid_values(make):
    with pytest.raises(ValueError):
        make()


def test_obsolete_research_packets_and_grid_axes_require_migration(tmp_path):
    from portfolio_core.strategies.research_parameters import ResearchParameters
    old = research_contract_parameters().payload()
    old["schema_version"] = "1.0.0"
    with pytest.raises(ValueError, match="migration.*fixed exposure"):
        ResearchParameters.from_payload(old)
    old["schema_version"] = "2.0.0"
    old["exposure"]["mode"] = "fixed"
    with pytest.raises(ValueError, match="migration"):
        ResearchParameters.from_payload(old)
    grid = read_grid("backtest/grids/momentum_grid.json")
    grid["exposure_profiles"] = [{"mode": "volatility_targeted"}]
    path = tmp_path / "obsolete.json"
    path.write_text(json.dumps(grid))
    for reader in (lambda: read_grid(path), lambda: expand_grid(grid)):
        with pytest.raises(ValueError, match="migration"):
            reader()


@pytest.mark.parametrize("family", ["momentum", "reversal", "monthly_trend", "low_volatility", "sector_momentum"])
def test_supplied_profiles_have_fixed_exposure_and_only_60_24(family):
    grid = read_grid(f"backtest/grids/{family}_grid.json")
    assert grid["schema_version"] == "2.0.0"
    assert "exposure_profiles" not in grid
    assert grid["gross"] == [1., 1.5, 2.]
    for profile in grid["sizing_profiles"]:
        if profile["volatility"] is not None:
            assert profile["volatility"] == dict(window=60, minimum_observations=24)
    if family == "low_volatility":
        assert grid["signal_profiles"] == [dict(selection_volatility=dict(window=60, minimum_observations=24))]


def test_custom_json_fixed_exposure_executes_and_resumes_one_synthetic_candidate(tmp_path):
    from backtest.research_search import run_research_search
    from _research_test_helpers import extended_dataset, tiny_grid

    grid = tiny_grid()
    grid.update(
        signal_profiles=[dict(formation_months=4, skip_months=2)],
        n_long=[12], n_short=[11], long_share=[0.55], gross=[1.2],
        turnover_threshold=[0.02],
        sizing_profiles=[dict(method='inverse_volatility', volatility=dict(window=18, minimum_observations=12))],
        buffer_profiles=[dict(enabled=True, exit_multiplier=1.4)],
    )
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps(grid))
    data = extended_dataset()
    output = tmp_path / 'search'
    first = run_research_search(data, read_grid(path), output)
    assert first['ranked_candidates'] == first['completed_candidates'] == 1
    checkpoints = list((output / 'candidates').glob('*/checkpoint.json'))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0].read_bytes()
    resumed = run_research_search(data, read_grid(path), output)
    assert resumed['resumed_candidates'] == 1
    assert checkpoints[0].read_bytes() == checkpoint
