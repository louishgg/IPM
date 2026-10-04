"""Explicit synthetic inputs and reusable research strategy contract checks."""

from dataclasses import replace

import numpy as np
import pandas as pd

from backtest.engine import _BacktestAuditCollector, _run_backtest_impl
from backtest.research_grid import read_grid
from portfolio_core.strategies.research_parameters import (
    BufferParameters, ExposureParameters, SizingParameters, VolatilityProfile,
)
from _strategy_test_helpers import momentum_test_parameters as packet, strategy_context
from test_engine_offline import make_synthetic_backtest_data


def tiny_grid(strategy_id="momentum"):
    grid = read_grid(f"backtest/grids/{strategy_id}_grid.json")
    for k, v in list(grid.items()):
        if isinstance(v, list):
            grid[k] = [v[0]]
    return grid


def extended_dataset():
    from _sector_test_helpers import _synthetic_sector_assignments

    data, _ = make_synthetic_backtest_data()
    dates = pd.date_range("2014-01-31", "2023-12-31", freq="ME")
    assets = list(data.data_close.columns)
    t = np.arange(len(dates))[:, None]
    a = np.arange(len(assets))[None, :]
    prices = pd.DataFrame(
        100
        * np.cumprod(
            1 + 0.005 + 0.0001 * a + 0.02 * np.sin((t + 1) * (a + 3) / 37), axis=0
        ),
        index=dates,
        columns=assets,
    )
    volumes = pd.DataFrame(1e6, index=dates, columns=assets)
    return replace(
        data,
        data_close=prices,
        data_volume=volumes,
        rolling_dollar_vol=(prices * volumes).rolling(3, min_periods=1).median(),
        pit_matrix=pd.DataFrame(True, index=dates, columns=assets),
        sector_assignments=_synthetic_sector_assignments(dates, assets),
        valid_trading_days=dates,
    )


def assert_sizing_contract(strategy_type, parameters):
    data, _ = make_synthetic_backtest_data()
    context = strategy_context(
        data.data_close,
        data.data_volume,
        tuple(data.data_close.columns),
        {a: "10" for a in data.data_close},
    )
    p = replace(parameters, sizing=SizingParameters("inverse_volatility", VolatilityProfile(60, 24)))
    decision = strategy_type(p).decide(context)
    expected = data.data_close.pct_change(fill_method=None).std(ddof=1)
    np.testing.assert_allclose(decision.signal_audit.Sizing_Volatility, expected)
    assert decision.is_complete
    assert (
        not strategy_type(
            replace(
                p,
                sizing=SizingParameters(
                    "inverse_volatility", VolatilityProfile(36, 36)
                ),
            )
        )
        .decide(context)
        .is_complete
    )
    constant = data.data_close.copy()
    constant.loc[:, :] = 100
    assert (
        not strategy_type(p)
        .decide(replace(context, close_history=constant))
        .is_complete
    )


def assert_neutral_execution_contract(strategy_type, parameters):
    data, dates = make_synthetic_backtest_data()
    strategy = strategy_type(
        replace(parameters,
            sector_neutral=True,
            buffer=BufferParameters(True, 1.5),
            turnover_threshold=0.05,
            exposure=ExposureParameters(2, 0.5),
        )
    )
    audit = _BacktestAuditCollector(strategy)
    _, diag = _run_backtest_impl(
        data, dates[12:18], strategy, return_diagnostics=True, strategy_audit=audit
    )
    records = pd.DataFrame(audit.sector_records)
    assert not records.empty
    assert np.allclose(records.Target_Net_Dollars, 0, atol=1e-7)
    assert np.allclose(
        records.Applied_Net_Dollars, records.Mechanical_Net_Dollars, atol=1e-7
    )
    execution = pd.DataFrame(audit.research_records).query("Status == 'executed'")
    assert execution.Post_Trade_Gross.le(2 + 1e-10).all()
    assert execution.Constraint_Override.eq("neutrality_suppression_override").any()
    assert diag.Long_Count.eq(10).all() and diag.Short_Count.eq(10).all()


def assert_search_serial_parallel_resume(tmp_path, grid):
    from backtest.research_search import run_research_search

    data = extended_dataset()
    grid = dict(grid)
    grid["gross"] = [1, 1.5]
    first = run_research_search(data, grid, tmp_path / "serial", workers=1)
    assert first["ranked_candidates"] == 2
    artifacts = {name: (tmp_path / "serial" / name).read_bytes()
                 for name in ("candidates.csv.gz", "fold_metrics.csv.gz")}
    resumed = run_research_search(data, grid, tmp_path / "serial", workers=2)
    assert resumed["resumed_candidates"] == 2
    second = run_research_search(data, grid, tmp_path / "parallel", workers=2)
    assert second["ranked_candidates"] == 2
    for name in ("candidates.csv.gz", "fold_metrics.csv.gz"):
        assert artifacts[name] == (tmp_path / "serial" / name).read_bytes()
        assert artifacts[name] == (tmp_path / "parallel" / name).read_bytes()
    candidates = pd.read_csv(tmp_path / "serial" / "candidates.csv.gz")
    assert candidates.Validation_Return_Count.eq(72).all()
    return_files = list((tmp_path / "serial" / "candidates").glob("*/validation_returns.csv.gz"))
    assert len(return_files) == 2
    for path in return_files:
        returns = pd.read_csv(path, index_col=0, parse_dates=True)
        assert len(returns) == 72 and returns.index[-1] == pd.Timestamp("2023-12-31")
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    serial_csvs = {p.relative_to(serial) for p in serial.rglob("*.csv.gz")}
    parallel_csvs = {p.relative_to(parallel) for p in parallel.rglob("*.csv.gz")}
    assert serial_csvs == parallel_csvs
    for relative in serial_csvs:
        assert (serial / relative).read_bytes() == (parallel / relative).read_bytes()
    assert not list(tmp_path.rglob("*.csv"))


def assert_unready_branch(tmp_path, grid):
    from backtest.research_search import run_research_search

    grid = dict(grid)
    grid["sizing_profiles"] = [dict(method="inverse_volatility",
        volatility=dict(window=120, minimum_observations=120))]
    result = run_research_search(extended_dataset(), grid, tmp_path, workers=1)
    assert result["history_bound_unready"] == 1 and result["ranked_candidates"] == 0
    assert pd.read_csv(tmp_path / "candidates.csv.gz").Status.tolist() == [
        "unranked_history"
    ]
