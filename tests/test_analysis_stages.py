"""Mirrored backtest/live analysis-stage orchestration contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

import backtest.analysis as backtest_analysis
import live.analysis as live_analysis
import live.brinson_attribution as live_brinson
from backtest.config import DEFAULT_CONFIG as BACKTEST_DEFAULT
from backtest.paths import BacktestPaths
from live.config import DEFAULT_CONFIG as LIVE_DEFAULT
from live.paths import LivePaths
from portfolio_core.strategies import (
    build_registered_strategy,
    save_strategy_parameters,
)
from portfolio_core.strategies.configuration import parameter_defaults


def _backtest_config(tmp_path=None):
    strategy = build_registered_strategy("momentum")
    base_paths = (
        BACKTEST_DEFAULT.paths
        if tmp_path is None
        else BacktestPaths(tmp_path / "backtest")
    )
    return replace(
        BACKTEST_DEFAULT,
        strategy=strategy,
        paths=base_paths.for_strategy(strategy.strategy_id),
    )


def _live_config(tmp_path=None):
    strategy = build_registered_strategy("momentum")
    base_paths = (
        LIVE_DEFAULT.paths
        if tmp_path is None
        else LivePaths(tmp_path / "live")
    )
    return replace(
        LIVE_DEFAULT,
        strategy=strategy,
        paths=base_paths.for_strategy(strategy.strategy_id),
    )


def test_backtest_all_calculates_strategy_and_brinson_before_writing(
    monkeypatch,
):
    config = _backtest_config()
    calls = []
    backtest_data = object()
    shares = object()
    benchmark = pd.Series(dtype=float)
    holdings = pd.DataFrame({"Asset_ID": ["A"]})
    artifacts = SimpleNamespace(result=SimpleNamespace(test_holdings_gross=holdings))
    attribution = {"result": True}

    monkeypatch.setattr(
        backtest_analysis,
        "apply_runtime_settings",
        lambda: calls.append("runtime"),
    )
    monkeypatch.setattr(
        backtest_analysis,
        "load_backtest_data",
        lambda *args, **kwargs: calls.append(
            ("load_data", kwargs["include_brinson"])
        )
        or backtest_data,
    )
    monkeypatch.setattr(
        backtest_analysis,
        "load_prepared_shares",
        lambda paths, cfg: calls.append("load_shares") or shares,
    )

    def validate_shares(data, loaded, cfg):
        assert data is backtest_data
        assert loaded is shares
        assert cfg is config.brinson
        calls.append("validate_shares")

    monkeypatch.setattr(backtest_analysis, "validate_share_coverage", validate_shares)
    monkeypatch.setattr(
        backtest_analysis,
        "load_benchmark",
        lambda *args: calls.append("load_benchmark") or benchmark,
    )
    monkeypatch.setattr(
        backtest_analysis,
        "calculate_strategy_artifacts",
        lambda data, bench, cfg: calls.append("calculate_strategy") or artifacts,
    )

    def calculate_brinson(data, *, holdings_df, shares_monthly, paths):
        assert data is backtest_data
        assert holdings_df is holdings
        assert shares_monthly is shares
        assert paths is config.paths
        calls.append("calculate_brinson")
        return attribution

    monkeypatch.setattr(
        backtest_analysis,
        "run_brinson_attribution",
        calculate_brinson,
    )
    monkeypatch.setattr(
        backtest_analysis,
        "save_strategy_artifacts",
        lambda result, cfg: calls.append("save_strategy"),
    )
    monkeypatch.setattr(
        backtest_analysis,
        "save_brinson_result",
        lambda result, paths: calls.append("save_brinson"),
    )

    actual = backtest_analysis.run_analysis("all", config=config)

    assert actual == (artifacts, attribution)
    assert calls == [
        "runtime",
        ("load_data", True),
        "load_shares",
        "validate_shares",
        "load_benchmark",
        "calculate_strategy",
        "calculate_brinson",
        "save_strategy",
        "save_brinson",
    ]


@pytest.mark.parametrize("stage", ["brinson", "all"])
def test_backtest_share_coverage_failure_prevents_calculation_and_writes(monkeypatch, tmp_path, stage):
    config = _backtest_config(tmp_path)
    save_strategy_parameters(config.strategy, config.paths.strategy_parameters_json)
    calls = []
    data, shares = object(), pd.DataFrame({"Asset_ID": ["A"]})
    monkeypatch.setattr(backtest_analysis, "apply_runtime_settings", lambda: None)
    monkeypatch.setattr(backtest_analysis, "load_backtest_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(backtest_analysis, "load_prepared_shares", lambda *args: shares)

    def reject_coverage(received_data, received_shares, brinson):
        assert received_data is data and received_shares is shares and brinson is config.brinson
        calls.append("validate")
        raise RuntimeError("missing shares coverage")

    def forbidden(*args, **kwargs):
        pytest.fail("Calculation or persistence preceded coverage validation")

    monkeypatch.setattr(backtest_analysis, "validate_share_coverage", reject_coverage)
    for name in ("calculate_strategy_artifacts", "run_brinson_attribution",
                 "save_strategy_artifacts", "save_brinson_result"):
        monkeypatch.setattr(backtest_analysis, name, forbidden)
    with pytest.raises(RuntimeError, match="^missing shares coverage$"):
        backtest_analysis.run_analysis(stage, config=config)
    assert calls == ["validate"]


def test_backtest_brinson_loads_saved_holdings_without_running_strategy(
    monkeypatch,
    tmp_path,
):
    config = _backtest_config(tmp_path)
    save_strategy_parameters(config.strategy, config.paths.strategy_parameters_json)
    backtest_data = object()
    shares = object()
    captured = {}

    monkeypatch.setattr(backtest_analysis, "apply_runtime_settings", lambda: None)
    monkeypatch.setattr(
        backtest_analysis,
        "load_backtest_data",
        lambda *args, **kwargs: backtest_data,
    )
    monkeypatch.setattr(
        backtest_analysis,
        "load_prepared_shares",
        lambda *args: shares,
    )

    def validate_shares(data, loaded, cfg):
        assert data is backtest_data and loaded is shares and cfg is config.brinson
        captured["validated"] = True

    monkeypatch.setattr(backtest_analysis, "validate_share_coverage", validate_shares)
    monkeypatch.setattr(
        backtest_analysis,
        "calculate_strategy_artifacts",
        lambda *args: (_ for _ in ()).throw(AssertionError("strategy ran")),
    )

    def calculate(data, *, shares_monthly, paths):
        assert captured["validated"]
        captured.update(shares_monthly=shares_monthly, paths=paths)
        return {"result": True}

    monkeypatch.setattr(backtest_analysis, "run_brinson_attribution", calculate)
    monkeypatch.setattr(backtest_analysis, "save_brinson_result", lambda *args: None)

    backtest_analysis.run_analysis("brinson", config=config)

    assert captured["shares_monthly"] is shares
    assert "holdings_df" not in captured


@pytest.mark.parametrize(
    ("stage", "expected_include_brinson"),
    (("strategy", False), ("brinson", True), ("all", True)),
)
def test_backtest_analysis_loads_the_requested_manifest_once(
    monkeypatch,
    tmp_path,
    stage,
    expected_include_brinson,
):
    config = _backtest_config(tmp_path)
    save_strategy_parameters(config.strategy, config.paths.strategy_parameters_json)
    manifest_scopes = []
    backtest_data = object()
    shares = object()
    artifacts = SimpleNamespace(result=SimpleNamespace(test_holdings_gross=pd.DataFrame()))

    monkeypatch.setattr(backtest_analysis, "apply_runtime_settings", lambda: None)

    def load_data(*args, **kwargs):
        manifest_scopes.append(
            (
                kwargs.get("include_brinson", False),
                kwargs.get("check_raw", False),
            )
        )
        return backtest_data

    monkeypatch.setattr(backtest_analysis, "load_backtest_data", load_data)
    monkeypatch.setattr(
        backtest_analysis,
        "load_benchmark",
        lambda *args: pd.Series(dtype=float),
    )
    monkeypatch.setattr(
        backtest_analysis,
        "calculate_strategy_artifacts",
        lambda *args: artifacts,
    )
    monkeypatch.setattr(
        backtest_analysis,
        "load_prepared_shares",
        lambda *args: shares,
    )
    monkeypatch.setattr(backtest_analysis, "validate_share_coverage", lambda *args: None)
    monkeypatch.setattr(
        backtest_analysis,
        "run_brinson_attribution",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        backtest_analysis,
        "remove_saved_brinson_result",
        lambda *args: None,
    )
    monkeypatch.setattr(
        backtest_analysis,
        "save_strategy_artifacts",
        lambda *args: None,
    )
    monkeypatch.setattr(
        backtest_analysis,
        "save_brinson_result",
        lambda *args: None,
    )

    backtest_analysis.run_analysis(stage, config=config)

    assert manifest_scopes == [(expected_include_brinson, False)]


def test_live_all_passes_the_same_in_memory_strategy_result_to_brinson(
    monkeypatch,
):
    config = _live_config()
    calls = []
    strategy_result = object()
    attribution_result = object()

    monkeypatch.setattr(live_analysis, "apply_runtime_settings", lambda: None)
    monkeypatch.setattr(
        live_analysis,
        "run_strategy",
        lambda cfg: calls.append("calculate_strategy") or strategy_result,
    )

    def run_brinson(cfg, *, strategy):
        assert strategy is strategy_result
        calls.append("calculate_brinson")
        return attribution_result

    monkeypatch.setattr(live_brinson, "run_brinson", run_brinson)
    monkeypatch.setattr(
        live_analysis,
        "save_strategy_result",
        lambda result, paths: calls.append("save_strategy"),
    )
    monkeypatch.setattr(
        live_brinson,
        "save_brinson_result",
        lambda result, paths: calls.append("save_brinson"),
    )
    actual = live_analysis.run_analysis("all", config=config)

    assert actual == (strategy_result, attribution_result)
    assert calls == [
        "calculate_strategy",
        "calculate_brinson",
        "save_strategy",
        "save_brinson",
    ]


def test_live_brinson_stage_calculates_before_writing(monkeypatch, tmp_path):
    config = _live_config(tmp_path)
    save_strategy_parameters(config.strategy, config.paths.strategy_parameters_json)
    calls = []
    attribution_result = object()

    monkeypatch.setattr(live_analysis, "apply_runtime_settings", lambda: None)
    monkeypatch.setattr(
        live_brinson,
        "run_brinson",
        lambda cfg: calls.append("calculate_brinson") or attribution_result,
    )
    monkeypatch.setattr(
        live_brinson,
        "save_brinson_result",
        lambda result, paths: calls.append("save_brinson"),
    )

    assert (
        live_analysis.run_analysis("brinson", config=config)
        is attribution_result
    )
    assert calls == ["calculate_brinson", "save_brinson"]


def test_live_brinson_stage_reads_saved_holdings_and_nav_without_strategy(monkeypatch, tmp_path):
    config = _live_config(tmp_path)
    save_strategy_parameters(config.strategy, config.paths.strategy_parameters_json)
    holdings = pd.DataFrame({"Asset_ID": ["A"], "Weight": [.5]})
    nav = pd.DataFrame({"Start_NAV": [1_000_000.], "End_NAV": [1_100_000.]})
    config.paths.strategy_holdings_csv.parent.mkdir(parents=True, exist_ok=True)
    holdings.to_csv(config.paths.strategy_holdings_csv, index=False)
    nav.to_csv(config.paths.strategy_nav_csv, index=False)
    inputs, shares, expected = object(), pd.DataFrame(), object()
    monkeypatch.setattr(live_analysis, "apply_runtime_settings", lambda: None)
    monkeypatch.setattr(live_analysis, "run_strategy", lambda *args: pytest.fail("strategy ran"))
    monkeypatch.setattr(live_brinson, "load_analysis_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(live_brinson, "load_prepared_shares", lambda *args: shares)

    def calculate(received_inputs, received_shares, saved_holdings, saved_nav):
        assert received_inputs is inputs and received_shares is shares
        pd.testing.assert_frame_equal(saved_holdings, holdings)
        pd.testing.assert_frame_equal(saved_nav, nav)
        return expected

    written = []
    monkeypatch.setattr(live_brinson, "run_live_brinson", calculate)
    monkeypatch.setattr(live_brinson, "save_brinson_result", lambda result, paths: written.append(result))
    assert live_analysis.run_analysis("brinson", config=config) is expected
    assert written == [expected]


def _write_stale_attribution(paths):
    for path in paths.attribution.result_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n", encoding="utf-8")
    unknown = paths.attribution.tables_dir / "user_notes.txt"
    unknown.write_text("keep\n", encoding="utf-8")
    return unknown


@pytest.mark.parametrize("domain", ("backtest", "live"))
def test_strategy_stage_removes_only_stale_brinson_after_calculation(
    monkeypatch,
    tmp_path,
    domain,
):
    old_strategy = build_registered_strategy(
        "momentum", {**parameter_defaults("momentum"), "signal": {
            **parameter_defaults("momentum")["signal"], "formation_months": 6,
        }}
    )
    new_strategy = build_registered_strategy(
        "momentum", {**parameter_defaults("momentum"), "signal": {
            **parameter_defaults("momentum")["signal"], "formation_months": 12,
        }}
    )
    if domain == "backtest":
        config = replace(_backtest_config(tmp_path), strategy=new_strategy)
        other = BacktestPaths(tmp_path / "backtest").for_strategy("reversal")
    else:
        config = replace(_live_config(tmp_path), strategy=new_strategy)
        other = LivePaths(tmp_path / "live").for_strategy("reversal")

    save_strategy_parameters(old_strategy, config.paths.strategy_parameters_json)
    unknown = _write_stale_attribution(config.paths)
    other_file = other.attribution.result_files[0]
    other_file.parent.mkdir(parents=True, exist_ok=True)
    other_file.write_text("other strategy\n", encoding="utf-8")
    calls = []

    def save(result, cfg):
        assert all(not path.exists() for path in cfg.paths.attribution.result_files)
        assert unknown.exists()
        assert other_file.exists()
        if domain == "backtest":
            assert config.paths.brinson_holdings_gross_csv.exists()
        save_strategy_parameters(cfg.strategy, cfg.paths.strategy_parameters_json)
        calls.append("save")

    if domain == "backtest":
        config.paths.brinson_holdings_gross_csv.write_text(
            "old holdings\n", encoding="utf-8"
        )
        monkeypatch.setattr(backtest_analysis, "apply_runtime_settings", lambda: None)
        monkeypatch.setattr(
            backtest_analysis,
            "load_backtest_data",
            lambda *args: object(),
        )
        monkeypatch.setattr(
            backtest_analysis,
            "load_benchmark",
            lambda *args: pd.Series(dtype=float),
        )
        monkeypatch.setattr(
            backtest_analysis,
            "calculate_strategy_artifacts",
            lambda *args: calls.append("calculate") or object(),
        )
        monkeypatch.setattr(backtest_analysis, "save_strategy_artifacts", save)
        backtest_analysis.run_analysis("strategy", config=config)
    else:
        monkeypatch.setattr(live_analysis, "apply_runtime_settings", lambda: None)
        monkeypatch.setattr(
            live_analysis,
            "run_strategy",
            lambda cfg: calls.append("calculate") or object(),
        )
        monkeypatch.setattr(live_analysis, "save_strategy_result", save)
        live_analysis.run_analysis("strategy", config=config)

    assert calls == ["calculate", "save"]
    saved = config.paths.strategy_parameters_json.read_text(encoding="utf-8")
    assert '"formation_months": 12' in saved


@pytest.mark.parametrize("domain", ("backtest", "live"))
def test_strategy_calculation_failure_preserves_previous_attribution(
    monkeypatch,
    tmp_path,
    domain,
):
    if domain == "backtest":
        config = _backtest_config(tmp_path)
        monkeypatch.setattr(
            backtest_analysis,
            "apply_runtime_settings",
            lambda: None,
        )
        monkeypatch.setattr(
            backtest_analysis,
            "load_backtest_data",
            lambda *args: object(),
        )
        monkeypatch.setattr(
            backtest_analysis,
            "load_benchmark",
            lambda *args: pd.Series(dtype=float),
        )
        monkeypatch.setattr(
            backtest_analysis,
            "calculate_strategy_artifacts",
            lambda *args: (_ for _ in ()).throw(RuntimeError("calculation defect")),
        )
        run = lambda: backtest_analysis.run_analysis("strategy", config=config)
    else:
        config = _live_config(tmp_path)
        monkeypatch.setattr(live_analysis, "apply_runtime_settings", lambda: None)
        monkeypatch.setattr(
            live_analysis,
            "run_strategy",
            lambda cfg: (_ for _ in ()).throw(RuntimeError("calculation defect")),
        )
        run = lambda: live_analysis.run_analysis("strategy", config=config)

    _write_stale_attribution(config.paths)

    with pytest.raises(RuntimeError, match="calculation defect"):
        run()

    assert all(path.exists() for path in config.paths.attribution.result_files)


@pytest.mark.parametrize(
    ("runner", "config"),
    (
        (backtest_analysis.run_analysis, BACKTEST_DEFAULT),
        (live_analysis.run_analysis, LIVE_DEFAULT),
    ),
)
def test_analysis_rejects_strategy_neutral_configuration(runner, config):
    with pytest.raises(ValueError, match="explicitly resolved strategy"):
        runner("strategy", config=config)
