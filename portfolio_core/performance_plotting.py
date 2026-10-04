"""Shared net-performance charts for monthly backtests and daily live replay."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .io import save_figure
from .plot_style import (
    LINE_FIGSIZE, PERIOD_FIGSIZE, SERIES_COLORS, date_axis_ticks,
    style_date_axis, style_return_axes,
)


_SERIES = ("Strategy", "S&P 500 Total Return")


def _validated_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if tuple(frame.columns) != _SERIES:
        raise ValueError(f"{name} must contain aligned strategy and benchmark columns")
    dates = frame.index
    if (
        not isinstance(dates, pd.DatetimeIndex)
        or dates.hasnans
        or not dates.is_monotonic_increasing
    ):
        raise ValueError(f"{name} must have ordered, nonmissing dates")
    values = frame.astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{name} must contain finite, nonmissing values")
    return values


def render_performance_plots(
    levels: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    cumulative_path: Path,
    drawdown_path: Path,
    returns_path: Path,
    window_label: str,
    return_frequency: Literal["monthly", "daily"],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> None:
    """Render aligned levels and fractional returns; retain same-day valuations."""
    import matplotlib.pyplot as plt

    levels = _validated_frame(levels, "Performance levels")
    returns = _validated_frame(returns, "Period returns")
    if len(levels) < 2 or (levels <= 0.0).any().any():
        raise ValueError("Performance levels require at least two positive valuations")
    if returns.empty or returns.index.has_duplicates:
        raise ValueError("Period returns require unique, nonempty dates")
    if not returns.index.isin(levels.index).all():
        raise ValueError("Period return dates must belong to the valuation window")
    date_axis_ticks(window_start, window_end, return_frequency)
    if window_start != levels.index[0] or window_end != levels.index[-1]:
        raise ValueError("Performance window must match its opening and closing valuations")

    monthly = return_frequency == "monthly"
    date_format = "%Y-%m" if monthly else "%Y-%m-%d"
    growth = levels / levels.iloc[0]
    chart_specs = (
        ((growth - 1.0) * 100.0, cumulative_path,
         "Cumulative Performance", "Cumulative return (%)", 0.5),
        ((growth / growth.cummax() - 1.0) * 100.0, drawdown_path,
         "Drawdowns", "Drawdown (%)", 0.6),
    )
    for values, path, title, ylabel, zero_alpha in chart_specs:
        figure, ax = plt.subplots(figsize=LINE_FIGSIZE)
        for label, linewidth in zip(_SERIES, (2.0, 2.2)):
            ax.plot(
                values.index, values[label], label=label,
                color=SERIES_COLORS[label], linewidth=linewidth,
            )
        style_return_axes(
            ax, title=f"{window_label} {title}", ylabel=ylabel,
            xlabel="Month" if monthly else "Date", zero_alpha=zero_alpha,
        )
        style_date_axis(
            ax, start=window_start, end=window_end, frequency=return_frequency,
        )
        ax.legend()
        figure.tight_layout()
        save_figure(figure, path, atomic=True)

    figure, ax = plt.subplots(figsize=PERIOD_FIGSIZE)
    x_values = np.arange(len(returns))
    width = 0.375
    for label, offset in zip(_SERIES, (-width / 2.0, width / 2.0)):
        ax.bar(
            x_values + offset, returns[label].to_numpy() * 100.0,
            width=width, color=SERIES_COLORS[label], label=label,
        )
    style_return_axes(
        ax, title=f"{window_label} {return_frequency.title()} Returns",
        ylabel="Monthly return (%)" if monthly else "Daily close-to-close return (%)",
        xlabel="Month" if monthly else "Session close", grid_axis="y",
    )
    ticks = np.arange(0, len(returns), 1 if monthly else 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        returns.index[ticks].strftime(date_format), rotation=45, ha="right",
    )
    ax.legend()
    figure.tight_layout()
    save_figure(figure, returns_path, atomic=True)
