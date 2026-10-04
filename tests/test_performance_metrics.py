"""Independent arithmetic examples for the shared performance definitions."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.benchmark import calculate_advanced_metrics, calculate_performance_summary
from backtest.evaluation import calculate_return_metrics
from portfolio_core.accounting_config import DEFAULT_ACCOUNTING_CONFIG
from portfolio_core.performance_metrics import annualized_ratio, calculate_risk_metrics
from portfolio_core.price_basis import price_basis_spec
from portfolio_core.simulation_assumptions import build_simulation_assumptions


def test_hand_calculated_sharpe_and_monthly_grid_adapter():
    # Excess returns 0, .01, .02 have mean .01 and sample deviation .01.
    returns = pd.Series([0., .01, .02])
    assert annualized_ratio(returns, 12) == pytest.approx(np.sqrt(12))
    result = calculate_return_metrics(returns, cash_interest_rate=0)
    assert result["Sharpe"] == pytest.approx(np.sqrt(12))
    assert result["PctReturn"] == pytest.approx(3.02)
    assert result["Maximum_Drawdown_Pct"] == 0


def test_known_capm_and_active_return_statistics():
    x = pd.Series([-.02, .01, .03, -.01, .04])
    rf = pd.Series([.0001, .0003, .0001, .0001, .0001])
    portfolio = rf + .001 + 1.5 * x
    result = calculate_risk_metrics(portfolio, rf + x, rf, periods_per_year=252)
    assert result["Beta"] == pytest.approx(1.5)
    assert result["CAPM_Alpha_OLS_Ann"] == pytest.approx(.252)
    assert result["Treynor"] == pytest.approx((.001 + 1.5 * x.mean()) * 252 / 1.5)
    active = .001 + .5 * x
    assert result["Tracking_Error_Ann"] == pytest.approx(active.std(ddof=1) * np.sqrt(252))
    assert result["Information_Ratio"] == pytest.approx(active.mean() / active.std(ddof=1) * np.sqrt(252))
    assert result["Portfolio_Ann_Vol"] == pytest.approx(portfolio.std(ddof=1) * np.sqrt(252))


def test_benchmark_identity_and_undefined_cases():
    b = pd.Series([.01, -.02, .03, .02])
    rf = pd.Series(0., index=b.index)
    same = calculate_risk_metrics(b, b, rf, periods_per_year=252)
    assert same["Beta"] == pytest.approx(1)
    assert same["CAPM_Alpha_OLS_Ann"] == pytest.approx(0, abs=1e-14)
    assert same["Tracking_Error_Ann"] == 0
    assert np.isnan(same["Information_Ratio"])
    cash = calculate_risk_metrics(rf, b, rf, periods_per_year=252, tolerance=1e-12)
    assert np.isnan(cash["Sharpe"]) and np.isnan(cash["Treynor"])
    constant = calculate_risk_metrics(b, rf, rf, periods_per_year=252)
    assert np.isnan(constant["Beta"]) and np.isnan(constant["CAPM_Alpha_OLS_Ann"])
    short = calculate_risk_metrics(b[:2], b[:2], rf[:2], periods_per_year=252)
    assert np.isnan(short["Beta"])
    assert np.isnan(annualized_ratio(b[:1], 252))
    near_zero_beta = calculate_risk_metrics(.01 + 1e-14 * b, b, rf, periods_per_year=252, tolerance=1e-12)
    assert np.isnan(near_zero_beta["Treynor"])


def test_backtest_preserves_zero_variance_and_regression_diagnostics():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    constant = pd.Series(100., index=dates)
    zero = calculate_advanced_metrics(constant, constant, cash_interest_rate=0)
    assert zero["Sharpe"] == zero["Information_Ratio"] == 0
    assert np.isnan(zero["Beta"])
    # Independent OLS residual and t statistic, rather than a second helper call.
    p = np.array([.02, -.01, .04, .015])
    b = np.array([.01, -.02, .03, .01])
    values = lambda r: pd.Series(np.r_[100., 100. * np.cumprod(1 + r)], index=dates)
    actual = calculate_advanced_metrics(values(p), values(b), cash_interest_rate=0)
    beta = np.cov(p, b, ddof=1)[0, 1] / np.var(b, ddof=1)
    assert actual["Beta"] == pytest.approx(beta)
    assert actual["CAPM_Alpha_OLS_Ann"] == pytest.approx((p.mean() - beta * b.mean()) * 12)
    assert 0 <= actual["Alpha_P_Value"] <= 1
    assert 0 <= actual["Beta_P_Value"] <= 1


@pytest.mark.parametrize("bad", [pd.Series([1., np.nan]), pd.Series([1., np.inf]), pd.Series([1., 2.], index=[0, 0])])
def test_invalid_returns_are_rejected(bad):
    with pytest.raises(ValueError):
        annualized_ratio(bad, 252)


def test_alignment_and_frequency_are_explicit():
    values = pd.Series([.01, .02, .03])
    with pytest.raises(ValueError, match="aligned"):
        calculate_risk_metrics(values, values.set_axis([1, 2, 3]), values, periods_per_year=12)
    with pytest.raises(ValueError, match="positive"):
        annualized_ratio(values, 0)


def test_backtest_performance_uses_stable_period_ids_and_current_labels():
    dates = pd.date_range("2020-01-31", periods=14, freq="ME")
    nav = pd.Series(
        [100.0 + value for value in range(len(dates))],
        index=dates,
    )
    benchmark = pd.Series(
        [100.0 + 0.5 * value for value in range(len(dates))],
        index=dates,
    )
    accounting = replace(
        DEFAULT_ACCOUNTING_CONFIG,
        cash_interest_rate=0.03,
    )
    price_basis = price_basis_spec("backtest")
    results = SimpleNamespace(
        development_dates=dates,
        test_dates=dates,
        development_results=nav,
        test_results=nav,
        full_results=nav,
        development_results_gross=nav,
        test_results_gross=nav,
        full_results_gross=nav,
        accounting_config=accounting,
        price_basis=price_basis,
        simulation_fingerprint=str(
            build_simulation_assumptions(
                accounting,
                price_basis,
            )["simulation_fingerprint"]
        ),
    )

    summary = calculate_performance_summary(results, benchmark)
    metrics = summary.loc[summary["Cost_Treatment"].isin(["Net", "Gross"])]

    assert set(metrics["Period_ID"]) == {"development", "test_window", "full"}
    assert metrics["Period"].tolist() == [
        "Development Period (2020-2021) - Net of Costs",
        "Test Window (2020-2021) - Net of Costs",
        "Full Backtest Window (2020-2021) - Net of Costs",
        "Development Period (2020-2021) - Gross of Costs",
        "Test Window (2020-2021) - Gross of Costs",
        "Full Backtest Window (2020-2021) - Gross of Costs",
    ]
    notes = summary.loc[summary["Period"].eq("Notes")].iloc[0]
    assert "3% risk-free rate" in notes["Sharpe"]
    assert "3% risk-free rate" in notes["Treynor"]
