"""Strategy registry metadata and strict parameter-file contracts."""

import json

import pytest

from portfolio_core.strategies import (
    available_strategy_ids, build_registered_strategy, build_strategy_from_file,
    get_strategy_definition, save_strategy_parameters,
    load_parameter_packet,
)
from _strategy_test_helpers import momentum_test_parameters


def test_registry_has_five_strategies_and_only_momentum_defaults():
    assert set(available_strategy_ids()) == {
        "momentum", "reversal", "monthly_trend", "low_volatility", "sector_momentum",
    }
    assert {name for name in available_strategy_ids()
            if get_strategy_definition(name).default_available} == {"momentum"}


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_registered_strategy("unknown_strategy")


def test_saved_wrapper_rejects_mismatched_strategy_and_version(tmp_path):
    strategy = build_registered_strategy("momentum", momentum_test_parameters().payload())
    path = tmp_path / "strategy_parameters.json"
    save_strategy_parameters(strategy, path)
    with pytest.raises(ValueError, match="declares strategy"):
        build_strategy_from_file("reversal", path)
    wrapper = json.loads(path.read_text())
    wrapper["strategy_version"] = "0.0.0"
    path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError, match="version"):
        build_strategy_from_file("momentum", path)


def test_duplicate_json_keys_and_partial_packets_fail(tmp_path):
    path = tmp_path / "packet.json"
    path.write_text('{"turnover_threshold": 9, "turnover_threshold": 12}')
    with pytest.raises(ValueError, match="duplicate"):
        build_strategy_from_file("momentum", path)
    packet = momentum_test_parameters().payload()
    packet.pop("selection")
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="complete"):
        build_strategy_from_file("momentum", path)


@pytest.mark.parametrize("text", ("[]", "{", '{"sizing": {"method": "equal", "method": "inverse_volatility"}}'))
def test_parameter_packet_reader_retains_contextual_json_failures(tmp_path, text):
    path = tmp_path / "parameters.json"
    path.write_text(text)
    with pytest.raises(ValueError) as error:
        load_parameter_packet(path)
    assert "Invalid strategy parameter file" in str(error.value)
    assert str(path) in str(error.value)


def test_parameter_packet_reader_preserves_none_and_missing_file(tmp_path):
    assert load_parameter_packet(None) is None
    path = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError, match="Missing strategy parameter file"):
        load_parameter_packet(path)
