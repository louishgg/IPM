"""Dispatch contracts for the clean-break command-line modules."""

from __future__ import annotations

import pytest
import json

import backtest.analyze as backtest_analyze
import backtest.prepare as backtest_prepare
import backtest.search as backtest_search
import backtest.research_search as research_search
import data_acquisition.acquire as acquire
import efficient_frontier.analyze as frontier_analyze
import live.analyze as live_analyze
import live.prepare as live_prepare
from portfolio_core.strategies import available_strategy_ids
from _strategy_test_helpers import research_test_parameters


@pytest.mark.parametrize(
    "module",
    (
        acquire,
        backtest_prepare,
        live_prepare,
        backtest_analyze,
        backtest_search,
        live_analyze,
        frontier_analyze,
    ),
)
def test_cli_help_exits_without_dispatch(module):
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--help"])
    assert exit_info.value.code == 0


@pytest.mark.parametrize(
    ("module", "arguments"),
    (
        (acquire, ["invalid", "prices"]),
        (backtest_prepare, ["invalid"]),
        (live_prepare, ["invalid"]),
        (backtest_analyze, ["--invalid"]),
        (backtest_analyze, ["strategy", "--strategy", "unknown_strategy"]),
        (backtest_search, ["--strategy", "unknown_strategy"]),
        (backtest_search, ["--strategy", "momentum", "--unknown-option"]),
        (live_analyze, ["invalid"]),
        (live_analyze, ["strategy", "--strategy", "unknown_strategy"]),
        (frontier_analyze, ["--invalid"]),
    ),
)
def test_cli_invalid_arguments_fail_before_dispatch(module, arguments, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid arguments reached data loading or dispatch")

    if module is backtest_search:
        monkeypatch.setattr(module, "load_backtest_data", forbidden)
        monkeypatch.setattr(research_search, "run_research_search", forbidden)
    elif module in (backtest_analyze, live_analyze):
        monkeypatch.setattr(module, "build_strategy_from_file", forbidden)
        monkeypatch.setattr(module, "run_analysis", forbidden)

    with pytest.raises(SystemExit) as exit_info:
        module.main(arguments)
    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    ("module", "arguments", "expected"),
    (
        (backtest_prepare, ["all", "--check"], ("all", True)),
        (live_prepare, ["core"], ("core", False)),
    ),
)
def test_preparation_cli_dispatch(module, arguments, expected, monkeypatch):
    calls = []
    monkeypatch.setattr(
        module,
        "run_preparation",
        lambda stage, *, check=False: calls.append((stage, check)),
    )
    module.main(arguments)
    assert calls == [expected]


@pytest.mark.parametrize("module", (backtest_analyze, live_analyze))
def test_analysis_cli_dispatch(module, monkeypatch):
    calls = []
    monkeypatch.setattr(
        module,
        "run_analysis",
        lambda stage, *, config: calls.append((stage, config)),
    )
    module.main([
        "strategy",
        "--strategy",
        "momentum",
    ])
    assert calls[0][0] == "strategy"
    assert calls[0][1].strategy.strategy_id == "momentum"
    assert calls[0][1].paths.strategy_id == "momentum"


@pytest.mark.parametrize("module", (backtest_analyze, live_analyze))
def test_analysis_cli_requires_an_explicit_strategy(module):
    with pytest.raises(SystemExit) as exit_info:
        module.main(["strategy"])
    assert exit_info.value.code == 2


@pytest.mark.parametrize("module", (backtest_analyze, live_analyze))
def test_brinson_cli_rejects_parameter_overrides(module, tmp_path):
    path = tmp_path / "params.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        module.main([
            "brinson",
            "--strategy",
            "momentum",
            "--params-file",
            str(path),
        ])
    assert exit_info.value.code == 2


@pytest.mark.parametrize("module", (backtest_analyze, live_analyze))
@pytest.mark.parametrize("strategy_id", available_strategy_ids())
def test_brinson_cli_uses_saved_identity_without_resolving_strategy(module, strategy_id, monkeypatch):
    monkeypatch.setattr(module, "build_strategy_from_file", lambda *args: pytest.fail("strategy resolved"))
    calls = []
    monkeypatch.setattr(module, "run_analysis", lambda stage, *, config: calls.append((stage, config)))
    module.main(["brinson", "--strategy", strategy_id])
    stage, config = calls[0]
    assert stage == "brinson" and config.strategy is None
    assert config.paths.strategy_id == strategy_id


@pytest.mark.parametrize("module", (backtest_analyze, live_analyze))
@pytest.mark.parametrize("strategy_id", available_strategy_ids())
def test_analysis_cli_resolves_parameter_file_for_selected_strategy(
    module,
    strategy_id,
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "params.json"
    packet = research_test_parameters(strategy_id).payload()
    path.write_text(json.dumps(packet), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        module,
        "run_analysis",
        lambda stage, *, config: calls.append((stage, config)),
    )

    module.main([
        "strategy",
        "--strategy",
        strategy_id,
        "--params-file",
        str(path),
    ])

    assert calls[0][1].strategy.strategy_id == strategy_id
    assert calls[0][1].paths.strategy_id == strategy_id


@pytest.mark.parametrize("module", (backtest_analyze, live_analyze))
def test_unpromoted_research_cli_requires_complete_parameter_file(module):
    with pytest.raises(ValueError, match="complete --params-file"):
        module.main(["strategy", "--strategy", "reversal"])


def test_frontier_analysis_cli_dispatch(monkeypatch):
    from pathlib import Path
    from efficient_frontier.contracts import DiagnosticSettings
    defaults = DiagnosticSettings()
    help_text = " ".join(frontier_analyze.build_parser().format_help().split())
    assert f"(default: {defaults.lookback}; minimum: 12)" in help_text
    assert f"Default: {defaults.floor_multiplier} times original absolute weight" in help_text
    assert f"Default: {defaults.cap_multiplier} times original absolute weight" in help_text
    assert f"(default: {defaults.brti})" in help_text
    calls = []
    monkeypatch.setattr(frontier_analyze, "run_analysis", lambda *args: calls.append(args) or 7)
    assert frontier_analyze.main(["--run-dir", "/saved/momentum"]) == 7
    assert calls == [(Path("/saved/momentum"), DiagnosticSettings(), None)]
    assert frontier_analyze.main(["--run-dir", "/saved/momentum", "--lookback", "24", "--brti", "4",
                                  "--floor-multiplier", "0.75", "--cap-multiplier", "1.5", "--output-root", "/reports"]) == 7
    assert calls[-1] == (Path("/saved/momentum"), DiagnosticSettings(24, 0.75, 1.5, 4), Path("/reports"))


def test_frontier_requires_explicit_saved_run():
    with pytest.raises(SystemExit) as exc:
        frontier_analyze.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("workers", (None, 1, 4, 6, 12))
def test_search_cli_dispatches_configurable_worker_count(monkeypatch, tmp_path, workers):
    data = object()
    calls = []
    def load(*args, development_only):
        assert development_only is True
        return data
    monkeypatch.setattr(backtest_search, "load_backtest_data", load)
    def search(received_data, grid, output, **kwargs):
        assert received_data is data
        assert grid["strategy_id"] == "momentum"
        assert output == tmp_path
        assert kwargs["preflight_only"] is True
        calls.append(kwargs["workers"])
    monkeypatch.setattr(research_search, "run_research_search", search)
    arguments = ["--strategy", "momentum", "--output-dir", str(tmp_path), "--preflight-only"]
    if workers is not None:
        arguments.extend(["--workers", str(workers)])
    backtest_search.main(arguments)
    assert calls == [6 if workers is None else workers]


@pytest.mark.parametrize("workers", ("0", "-1", "1.5", "many"))
def test_search_cli_rejects_invalid_worker_counts_before_loading(monkeypatch, workers):
    monkeypatch.setattr(
        backtest_search, "load_backtest_data",
        lambda *args: pytest.fail("invalid workers must fail before data loading"),
    )
    with pytest.raises(SystemExit) as exit_info:
        backtest_search.main(["--strategy", "momentum", "--workers", workers])
    assert exit_info.value.code == 2


def test_search_cli_propagates_research_interrupt(monkeypatch):
    def load(*args, development_only):
        assert development_only is True
        return object()
    monkeypatch.setattr(backtest_search, "load_backtest_data", load)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(research_search, "run_research_search", interrupted)
    with pytest.raises(KeyboardInterrupt):
        backtest_search.main(["--strategy", "momentum", "--workers", "4"])


def test_search_cli_propagates_calculation_failure(monkeypatch):
    monkeypatch.setattr(backtest_search, "load_backtest_data", lambda *a, **k: object())
    def fail(*args, **kwargs):
        raise RuntimeError("candidate 2: calculation defect")
    monkeypatch.setattr(research_search, "run_research_search", fail)
    with pytest.raises(RuntimeError, match="candidate 2: calculation defect"):
        backtest_search.main(["--strategy", "momentum", "--workers", "1"])
