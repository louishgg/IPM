"""Complete strategy packets, lazy defaults and complete search-grid boundaries."""

import json

import pytest

from backtest.research_grid import expand_grid, read_grid
from portfolio_core.strategies import (
    available_strategy_ids,
    build_registered_strategy,
    build_strategy_from_file,
    get_strategy_definition,
    load_saved_strategy_parameters,
    save_strategy_parameters,
)
from portfolio_core.strategies import configuration
from _strategy_test_helpers import research_test_parameters


def _research_packet(strategy_id):
    return research_test_parameters(strategy_id).payload()


def test_registry_lists_five_implementations_without_reading_defaults(monkeypatch):
    monkeypatch.setattr(configuration, "DEFAULTS_PATH", configuration.DEFAULTS_PATH.with_name("absent.json"))
    assert available_strategy_ids() == (
        "low_volatility", "momentum",
        "monthly_trend", "reversal", "sector_momentum",
    )
    for strategy_id in available_strategy_ids():
        assert get_strategy_definition(strategy_id).strategy_id == strategy_id
    with pytest.raises(FileNotFoundError):
        build_registered_strategy("momentum")
    with pytest.raises(ValueError, match="complete --params-file"):
        build_registered_strategy("reversal")


@pytest.mark.parametrize("strategy_id", available_strategy_ids())
def test_complete_raw_and_saved_packets_round_trip(tmp_path, strategy_id):
    strategy = (
        build_registered_strategy(strategy_id)
        if get_strategy_definition(strategy_id).default_available
        else build_registered_strategy(strategy_id, _research_packet(strategy_id))
    )
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(strategy.parameters.payload()))
    assert build_strategy_from_file(strategy_id, raw).parameters == strategy.parameters
    saved = tmp_path / "saved.json"
    save_strategy_parameters(strategy, saved)
    assert build_strategy_from_file(strategy_id, saved).parameters == strategy.parameters
    assert load_saved_strategy_parameters(strategy_id, saved)["strategy_id"] == strategy_id


def test_incomplete_and_malformed_nested_packets_fail(tmp_path):
    packet = _research_packet("momentum")
    packet["sizing"]["volatility"] = {"window": 36}
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="parameters.sizing.volatility.*minimum_observations"):
        build_strategy_from_file("momentum", path)
    packet = _research_packet("momentum")
    packet["selection"].pop("n_long")
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="parameters.selection.*n_long"):
        build_strategy_from_file("momentum", path)


def test_momentum_grid_has_expected_candidate_count():
    candidates, _, _ = expand_grid(read_grid("backtest/grids/momentum_grid.json"))
    assert len(candidates) == 5760
