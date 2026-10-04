"""Frequency-independent arithmetic risk statistics; no valuation or file I/O.

Callers own return construction and risk-free conversion. Geometric growth is
deliberately separate from the arithmetic excess/active means used here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _returns(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="raise").astype(float)
    if values.index.has_duplicates or not values.index.is_monotonic_increasing:
        raise ValueError("Return observations must have unique ordered indices")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Return observations must be finite and nonmissing")
    return values


def annualized_ratio(
    differential: pd.Series,
    periods_per_year: float,
    *,
    zero_variance_value: float = np.nan,
    tolerance: float = 0.0,
) -> float:
    """Mean differential return / sample deviation, scaled by sqrt(frequency).

The backtest adapter explicitly retains its historical zero-variance value of
zero. New live statistics use NaN for undefined ratios.
    """
    values = _returns(differential)
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    std = float(values.std(ddof=1))
    return (
        float(values.mean() / std * np.sqrt(periods_per_year))
        if len(values) >= 2 and std > tolerance else float(zero_variance_value)
    )


def calculate_risk_metrics(
    portfolio: pd.Series,
    benchmark: pd.Series,
    risk_free: pd.Series,
    *,
    periods_per_year: float,
    zero_variance_value: float = np.nan,
    tolerance: float = 0.0,
) -> dict[str, float]:
    """Standard Sharpe, active-return IR, and excess-return CAPM statistics.

The OLS calculation and classical two-sided t-test diagnostics retain the
backtest's arithmetic and minimum of three observations. No rows are dropped.
    """
    portfolio, benchmark, risk_free = map(_returns, (portfolio, benchmark, risk_free))
    if not (portfolio.index.equals(benchmark.index) and portfolio.index.equals(risk_free.index)):
        raise ValueError("Portfolio, benchmark and risk-free returns must be aligned")
    excess = portfolio - risk_free
    benchmark_excess = benchmark - risk_free
    active = portfolio - benchmark
    options = dict(zero_variance_value=zero_variance_value, tolerance=tolerance)
    sharpe = annualized_ratio(excess, periods_per_year, **options)
    information = annualized_ratio(active, periods_per_year, **options)
    alpha = beta = alpha_p = beta_p = np.nan
    x, y = benchmark_excess.to_numpy(), excess.to_numpy()
    n = len(x)
    if n >= 3:
        x_bar, y_bar = x.mean(), y.mean()
        x_dev, y_dev = x - x_bar, y - y_bar
        sxx, sxy = np.sum(x_dev ** 2), np.sum(x_dev * y_dev)
        if sxx > (n - 1) * tolerance ** 2:
            beta = sxy / sxx
            alpha = y_bar - beta * x_bar
            residual = y - (alpha + beta * x)
            dof = n - 2
            s2 = np.sum(residual ** 2) / dof
            se_beta = np.sqrt(s2 / sxx)
            se_alpha = np.sqrt(s2 * (1.0 / n + x_bar ** 2 / sxx))
            t_beta = beta / se_beta if np.isfinite(se_beta) and se_beta > 0 else np.nan
            t_alpha = alpha / se_alpha if np.isfinite(se_alpha) and se_alpha > 0 else np.nan
            beta_p = 2.0 * (1.0 - stats.t.cdf(np.abs(t_beta), dof)) if np.isfinite(t_beta) else np.nan
            alpha_p = 2.0 * (1.0 - stats.t.cdf(np.abs(t_alpha), dof)) if np.isfinite(t_alpha) else np.nan
    treynor = (
        excess.mean() * periods_per_year / beta
        if np.isfinite(beta) and abs(beta) > tolerance else np.nan
    )
    return {
        "Portfolio_Ann_Vol": float(portfolio.std(ddof=1) * np.sqrt(periods_per_year)),
        "Benchmark_Ann_Vol": float(benchmark.std(ddof=1) * np.sqrt(periods_per_year)),
        "Active_Return_Ann_Arith": float(active.mean() * periods_per_year),
        "Tracking_Error_Ann": float(active.std(ddof=1) * np.sqrt(periods_per_year)),
        "Sharpe": sharpe,
        "Beta": float(beta),
        "Treynor": float(treynor),
        "Information_Ratio": information,
        "CAPM_Alpha_OLS_Ann": float(alpha * periods_per_year),
        "Alpha_P_Value": float(alpha_p),
        "Beta_P_Value": float(beta_p),
    }
