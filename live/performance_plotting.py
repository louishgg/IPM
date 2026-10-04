"""Adapt authoritative daily net valuations to the shared performance charts."""

import pandas as pd

from portfolio_core.performance_plotting import render_performance_plots

from .paths import LivePaths


def save_performance_plots(daily_nav: pd.DataFrame, paths: LivePaths) -> None:
    """Keep all valuations for curves and complete close-to-close returns for bars."""
    eligible = daily_nav["Include_In_Risk_Metrics"]
    if not pd.api.types.is_bool_dtype(eligible) or eligible.isna().any():
        raise ValueError("Daily return eligibility must be nonmissing booleans")
    dates = pd.DatetimeIndex(pd.to_datetime(daily_nav["Date"], errors="raise"))
    levels = daily_nav[["NAV", "Benchmark_Level"]].copy()
    levels.columns = ["Strategy", "S&P 500 Total Return"]
    levels.index = dates
    returns = daily_nav.loc[eligible, ["Portfolio_Return", "Benchmark_Return"]].copy()
    returns.columns = levels.columns
    returns.index = dates[eligible.to_numpy(dtype=bool)]
    render_performance_plots(
        levels, returns,
        cumulative_path=paths.cumulative_performance_png,
        drawdown_path=paths.drawdown_png,
        returns_path=paths.daily_returns_png,
        window_label="Live Window", return_frequency="daily",
        window_start=dates.min(), window_end=dates.max(),
    )
