"""Parameterized benchmark and signed Brinson decomposition renderers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .brinson_attribution import STOCK_EFFECT_COLUMNS, add_benchmark_consistency_columns
from .io import save_figure
from .plot_style import SERIES_COLORS, date_axis_ticks, style_date_axis, style_return_axes


_EFFECT_LABELS = {
    "Allocation_Effect": "Allocation",
    "Selection_Effect": "Pure selection",
    "Interaction_Effect": "Interaction",
}


def _prepare_plot_frame(frame, value_columns, context):
    """Validate and normalize a plotting copy before any output is written."""
    missing = sorted({"Date", "Next_Date", *value_columns} - set(frame.columns))
    if missing:
        raise ValueError(f"Brinson {context} data is missing columns: {missing}")
    if frame.empty:
        raise RuntimeError(f"Brinson {context} requires nonempty plot data")
    result = frame.copy()
    for column in ("Date", "Next_Date"):
        result[column] = pd.to_datetime(result[column], errors="raise").dt.tz_localize(None)
        if result[column].isna().any():
            raise ValueError(f"Brinson {context} requires valid {column} values")
    result = result.sort_values("Next_Date", kind="stable").reset_index(drop=True)
    if (result["Date"] >= result["Next_Date"]).any():
        raise ValueError(f"Brinson {context} requires increasing period boundaries")
    if not np.array_equal(result["Date"].iloc[1:], result["Next_Date"].iloc[:-1]):
        raise ValueError(f"Brinson {context} requires contiguous periods")
    result[value_columns] = result[value_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(result[value_columns].to_numpy(dtype=float, na_value=np.nan)).all():
        raise RuntimeError(f"Brinson {context} requires finite plot values")
    return result


def prepare_benchmark_consistency_plot_frame(benchmark_audit: pd.DataFrame) -> pd.DataFrame:
    """Derive cumulative comparisons from finite, chronologically sorted returns."""
    result = add_benchmark_consistency_columns(_prepare_plot_frame(
        benchmark_audit, ["Reconstructed_Benchmark_Return", "SP500TR_Return"],
        "benchmark consistency",
    ))
    if not np.isfinite(result[["Cumulative_Reconstructed_Benchmark", "Cumulative_SP500TR"]]).all().all():
        raise RuntimeError("Brinson benchmark consistency requires finite cumulative values")
    return result


def prepare_active_decomposition_plot_frame(monthly_attribution: pd.DataFrame) -> pd.DataFrame:
    """Select the Combined book and require every plotted stock effect."""
    if "Side" not in monthly_attribution:
        raise ValueError("Brinson decomposition data is missing columns: ['Side']")
    combined = monthly_attribution.loc[monthly_attribution["Side"].eq("Combined")]
    if combined.empty:
        raise RuntimeError("Brinson decomposition plots require combined attribution rows")
    return _prepare_plot_frame(
        combined, [*(f"Scaled_{column}" for column in STOCK_EFFECT_COLUMNS), "Scaled_Active_Return"],
        "decomposition",
    )


def plot_signed_stacked_bars(
    ax: Any,
    x_values: np.ndarray,
    components: list[tuple[str, np.ndarray]],
) -> None:
    """Plot signed stacks with independent positive and negative baselines."""
    positive_bottom = np.zeros(len(x_values), dtype=float)
    negative_bottom = np.zeros(len(x_values), dtype=float)
    for label, values in components:
        values = np.asarray(values, dtype=float)
        bottoms = np.where(values >= 0.0, positive_bottom, negative_bottom)
        ax.bar(
            x_values,
            values,
            bottom=bottoms,
            label=label,
            color=SERIES_COLORS.get(label),
            width=0.75,
        )
        positive_bottom += np.where(values >= 0.0, values, 0.0)
        negative_bottom += np.where(values < 0.0, values, 0.0)


def _cumulative_dates(frame, window_start, window_end, date_frequency):
    """Prepend distinct opening dates without changing the interval data."""
    date_axis_ticks(window_start, window_end, date_frequency)
    first_start = frame["Date"].iloc[0]
    if window_start > first_start or window_end != frame["Next_Date"].iloc[-1]:
        raise ValueError("Cumulative chart window must cover every complete period")
    # A domain adapter may extend the stock-only curve across verified initial cash.
    opening = pd.DatetimeIndex([window_start, first_start]).unique()
    return opening.append(pd.DatetimeIndex(frame["Next_Date"])), len(opening)


def _period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start.year != end.year:
        return f"{start.day} {start:%b %Y}–{end.day} {end:%b %Y}"
    left = str(start.day) if start.month == end.month else f"{start.day} {start:%b}"
    return f"{left}–{end.day} {end:%b}"


def render_benchmark_consistency(
    plot_frame: pd.DataFrame,
    output_path: Path,
    *,
    reconstructed_label: str,
    title: str,
    xlabel: str,
    figsize: tuple[float, float],
    atomic: bool,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    date_frequency: Literal["monthly", "daily"],
) -> None:
    """Render two cumulative benchmark series with domain-owned labels."""
    import matplotlib.pyplot as plt

    dates, opening_count = _cumulative_dates(
        plot_frame, window_start, window_end, date_frequency,
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        dates,
        np.r_[np.zeros(opening_count), plot_frame["Cumulative_Reconstructed_Benchmark"] * 100.0],
        label=reconstructed_label,
        linewidth=2.2,
        color=SERIES_COLORS["Reconstructed benchmark"],
    )
    ax.plot(
        dates,
        np.r_[np.zeros(opening_count), plot_frame["Cumulative_SP500TR"] * 100.0],
        label="S&P 500 Total Return",
        linewidth=2.2,
        color=SERIES_COLORS["S&P 500 Total Return"],
    )
    style_return_axes(
        ax, title=title, ylabel="Cumulative return (%)", xlabel=xlabel,
        zero_alpha=0.5,
    )
    style_date_axis(ax, start=window_start, end=window_end, frequency=date_frequency)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, atomic=atomic)


def render_signed_decomposition(
    combined: pd.DataFrame,
    period_output_path: Path,
    cumulative_output_path: Path,
    *,
    period_title: str,
    cumulative_title: str,
    xlabel: str,
    period_figsize: tuple[float, float],
    cumulative_figsize: tuple[float, float],
    label_rotation: float,
    marker_size: float,
    atomic: bool,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    date_frequency: Literal["monthly", "daily"],
) -> None:
    """Render three signed BF effects and their arithmetic cumulative sums."""
    import matplotlib.pyplot as plt

    dates, opening_count = _cumulative_dates(
        combined, window_start, window_end, date_frequency,
    )
    labels = (
        [_period_label(row.Date, row.Next_Date) for row in combined.itertuples()]
        if date_frequency == "daily" else combined["Next_Date"].dt.strftime("%Y-%m")
    )
    x_values = np.arange(len(combined))
    components = [
        (_EFFECT_LABELS[column], combined[f"Scaled_{column}"].to_numpy(dtype=float) * 100.0)
        for column in STOCK_EFFECT_COLUMNS
    ]
    active = combined["Scaled_Active_Return"].to_numpy(dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=period_figsize)
    plot_signed_stacked_bars(
        ax,
        x_values,
        components,
    )
    (active_line,) = ax.plot(
        x_values,
        active,
        color=SERIES_COLORS["Active return"],
        linewidth=1.8,
        marker="o",
        markersize=marker_size,
        label="Active return",
    )
    style_return_axes(
        ax, title=period_title, ylabel="Contribution to return (pp)",
        xlabel="Holding period" if date_frequency == "daily" else xlabel, grid_axis="y",
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=label_rotation, ha="center" if label_rotation == 0 else "right")
    # Matplotlib collects lines before bars; keep the total last in the legend.
    ax.legend(handles=[*ax.containers, active_line])
    fig.tight_layout()
    save_figure(fig, period_output_path, atomic=atomic)

    fig, ax = plt.subplots(figsize=cumulative_figsize)
    for label, values in components:
        ax.plot(
            dates, np.r_[np.zeros(opening_count), np.cumsum(values)], label=label,
            linewidth=2.2, color=SERIES_COLORS[label],
        )
    ax.plot(
        dates,
        np.r_[np.zeros(opening_count), np.cumsum(active)],
        label="Active return",
        linewidth=2.0,
        color=SERIES_COLORS["Active return"],
    )
    style_return_axes(
        ax, title=cumulative_title,
        ylabel="Cumulative arithmetic contribution (pp)", xlabel=xlabel,
    )
    style_date_axis(ax, start=window_start, end=window_end, frequency=date_frequency)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, cumulative_output_path, atomic=atomic)


__all__ = [
    "prepare_benchmark_consistency_plot_frame",
    "prepare_active_decomposition_plot_frame",
    "plot_signed_stacked_bars",
    "render_benchmark_consistency",
    "render_signed_decomposition",
]
