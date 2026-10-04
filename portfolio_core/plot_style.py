"""Shared colours, figure sizes, and axes styling for return charts."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd


SERIES_COLORS = {
    "Allocation": "#2F6B9A",
    "Pure selection": "#D9822B",
    "Interaction": "#8064A2",
    "Reconstructed benchmark": "#5C8D89",
    "S&P 500 Total Return": "#B79B5B",
    "Strategy": "#8B5E6B",
    "Active return": "#4B5563",
    "Bounded frontier": "#6F7198",
    "Relaxed frontier": "#7D8752",
}
LINE_FIGSIZE = (10, 5.5)
PERIOD_FIGSIZE = (12, 6)


def date_axis_ticks(
    start: pd.Timestamp, end: pd.Timestamp, frequency: Literal["monthly", "daily"],
) -> pd.DatetimeIndex:
    """Choose calendar ticks independently of observations and market holidays."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("Chart window requires valid, increasing start and end dates")
    if frequency == "daily":
        first = start + pd.Timedelta(days=(0 - start.weekday()) % 7)
        ticks = pd.date_range(first, end, freq="14D")
    elif frequency == "monthly":
        ticks = pd.date_range(start, end, freq=pd.offsets.MonthEnd(3))
    else:
        raise ValueError("Date-axis frequency must be monthly or daily")
    # A short window can finish before the first scheduled tick.
    return ticks if len(ticks) else pd.DatetimeIndex([start])


def style_date_axis(
    ax: Any, *, start: pd.Timestamp, end: pd.Timestamp,
    frequency: Literal["monthly", "daily"],
) -> None:
    """Give charts of the same window identical labels and calendar limits."""
    ticks = date_axis_ticks(start, end, frequency)
    ax.set_xticks(ticks)
    date_format = "%Y-%m" if frequency == "monthly" else "%Y-%m-%d"
    ax.set_xticklabels(ticks.strftime(date_format), rotation=30, ha="right")
    # Retain the usual five-percent line-chart padding, shared across renderers.
    padding = (pd.Timestamp(end) - pd.Timestamp(start)) * 0.05
    ax.set_xlim(pd.Timestamp(start) - padding, pd.Timestamp(end) + padding)


def style_return_axes(
    ax: Any,
    *,
    title: str,
    ylabel: str,
    xlabel: str,
    grid_axis: str = "both",
    zero_alpha: float = 0.6,
) -> None:
    """Apply the common zero reference, labels, and quiet grid locally."""
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=zero_alpha)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis=grid_axis, alpha=0.3)
