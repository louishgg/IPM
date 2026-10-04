"""Explicit evaluation and expanding-validation search contracts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import backtest.analysis as analysis
import backtest.evaluation as evaluation
from backtest.config import DEFAULT_CONFIG
from portfolio_core.price_basis import price_basis_spec
from portfolio_core.ranking import add_deterministic_ranks
from _strategy_test_helpers import momentum_test_strategy


def _nav(dates) -> pd.Series:
    index = pd.DatetimeIndex(dates)
    return pd.Series(
        [1_000_000.0 + 10_000.0 * offset for offset in range(len(index))],
        index=index,
    )


def _diagnostics(dates) -> pd.DataFrame:
    index = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {
            "Fees": np.full(len(index), 2.0),
            "Spread_Cost": np.full(len(index), 3.0),
            "Num_Trades": np.ones(len(index), dtype=int),
            "Turnover": np.full(len(index), 0.1),
        },
        index=index,
    )


@pytest.mark.parametrize("return_diagnostics", (False, True))
@pytest.mark.parametrize("return_holdings", (False, True))
def test_evaluate_strategy_preserves_engine_shape(
    monkeypatch,
    return_diagnostics,
    return_holdings,
):
    dates = pd.date_range("2021-01-31", periods=2, freq="ME")
    nav = _nav(dates)
    outputs = [nav]
    if return_diagnostics:
        outputs.append(_diagnostics(dates[1:]))
    if return_holdings:
        outputs.append(pd.DataFrame({"Asset_ID": ["AAA.O"]}))
    expected = tuple(outputs) if len(outputs) > 1 else nav
    selected = momentum_test_strategy(turnover_threshold=0.037)
    calls = []

    def fake_run(*args, **kwargs):
        assert args[2] is selected
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(evaluation, "_run_backtest_impl", fake_run)

    actual = evaluation.evaluate_strategy(
        object(),
        dates,
        selected,
        include_costs=False,
        return_diagnostics=return_diagnostics,
        return_holdings=return_holdings,
    )

    assert actual is expected
    assert calls[0]["return_diagnostics"] is return_diagnostics
    assert calls[0]["return_holdings"] is return_holdings
    assert calls[0]["apply_fees"] is False
    assert calls[0]["apply_spread"] is False


def test_strategy_artifacts_reuse_test_window_gross_holdings_without_selection(
    monkeypatch,
):
    strategy = momentum_test_strategy()
    calls = []
    holdings_requests = []
    holdings = pd.DataFrame({"Asset_ID": ["AAA.O"]})

    def fake_evaluate(
        backtest_data,
        dates,
        received_strategy,
        *,
        include_costs,
        return_diagnostics,
        **kwargs,
    ):
        dates = pd.DatetimeIndex(dates)
        calls.append((dates, received_strategy, include_costs, return_diagnostics))
        nav = _nav(dates)
        if kwargs.get("return_holdings", False):
            holdings_requests.append((dates, include_costs))
            return nav, holdings
        if return_diagnostics:
            return nav, _diagnostics(dates[1:])
        return nav

    monkeypatch.setattr(evaluation, "evaluate_strategy", fake_evaluate)
    monkeypatch.setattr(
        analysis, "calculate_spread_sensitivity", lambda *args: pd.DataFrame()
    )
    monkeypatch.setattr(
        analysis, "calculate_performance_summary",
        lambda *args: pd.DataFrame({"Series": ["Test_Window_Gross"]}),
    )
    dates = pd.date_range("2015-01-31", "2026-01-31", freq="ME")

    artifacts = analysis.calculate_strategy_artifacts(
        SimpleNamespace(
            valid_trading_days=dates,
            price_basis=price_basis_spec("backtest"),
        ),
        pd.Series(dtype=float),
        replace(
            DEFAULT_CONFIG,
            strategy=strategy,
            paths=DEFAULT_CONFIG.paths.for_strategy(strategy.strategy_id),
        ),
    )
    result = artifacts.result

    assert result.strategy is strategy
    assert result.test_holdings_gross is holdings
    assert len(holdings_requests) == 1
    assert holdings_requests[0][0].equals(result.test_dates)
    assert holdings_requests[0][1] is False
    pd.testing.assert_series_equal(result.test_results_gross, _nav(result.test_dates))
    assert result.development_dates.max() == pd.Timestamp("2023-12-31")
    assert result.test_dates.min() == pd.Timestamp("2024-01-31")
    assert len(result.development_results.pct_change().dropna()) == 107
    assert len(result.test_results.pct_change().dropna()) == 24
    assert len(result.full_results.pct_change().dropna()) == 132
    assert len(calls) == 6
    assert all(received is strategy for _, received, _, _ in calls)
    assert [include_costs for _, _, include_costs, _ in calls] == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize("order", ([0, 1, 2, 3], [2, 0, 3, 1]))
def test_shared_ranking_breaks_metric_ties_by_parameters(order):
    frame = pd.DataFrame(
        {
            "Parameters_JSON": ['{"lookback":9}', '{"lookback":3}', '{"lookback":6}', '{"lookback":1}'],
            "Validation_PctReturn": [12.0, 12.0, 5.0, 20.0],
            "Validation_Sharpe": [1.0, 1.0, 1.5, 0.5],
        },
        index=[40, 10, 30, 20],
    ).iloc[order]
    expected = pd.DataFrame(
        {"PctReturn_Rank": [3, 2, 4, 1], "Sharpe_Rank": [3, 2, 1, 4]},
        index=[40, 10, 30, 20],
    ).reindex(frame.index)

    ranked = add_deterministic_ranks(
        frame,
        (("Validation_PctReturn", "PctReturn_Rank"), ("Validation_Sharpe", "Sharpe_Rank")),
        tie_breaker="Parameters_JSON",
    )

    pd.testing.assert_frame_equal(ranked[expected.columns], expected)
    pd.testing.assert_frame_equal(ranked[frame.columns], frame)
