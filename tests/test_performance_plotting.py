"""Shared figure semantics and monthly/daily observation boundaries."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from backtest import benchmark as backtest_plots
from backtest.paths import BacktestPaths
from live import performance_plotting as live_plots
from live.paths import LivePaths
from portfolio_core import performance_plotting as plots
from portfolio_core.plot_style import LINE_FIGSIZE, PERIOD_FIGSIZE, SERIES_COLORS
from tests.test_live_replay_calendar import canonical


@pytest.fixture
def captured(monkeypatch):
    figures = {}
    save = plots.save_figure

    def capture(figure, path, *, atomic):
        assert atomic is True
        figures[Path(path).stem] = figure
        save(figure, path, atomic=atomic)

    monkeypatch.setattr(plots, "save_figure", capture)
    return figures


def _levels():
    return pd.DataFrame(
        {"Strategy": [100., 110., 88., 121.],
         "S&P 500 Total Return": [200., 180., 198., 220.]},
        index=pd.to_datetime(["2026-02-13", "2026-02-17", "2026-02-18", "2026-02-20"]),
    )


def _render(levels, returns, tmp_path, frequency="daily"):
    plots.render_performance_plots(
        levels, returns,
        cumulative_path=tmp_path / "cumulative_performance.png",
        drawdown_path=tmp_path / "drawdown.png",
        returns_path=tmp_path / f"{frequency}_returns.png",
        window_label="Test Window" if frequency == "monthly" else "Live Window",
        return_frequency=frequency,
        window_start=levels.index.min(), window_end=levels.index.max(),
    )


@pytest.mark.parametrize("frequency", ["monthly", "daily"])
def test_values_style_grouped_bars_and_png_cleanup(captured, tmp_path, frequency):
    levels = _levels()
    if frequency == "monthly":
        levels.index = pd.date_range("2024-01-31", periods=4, freq=pd.offsets.MonthEnd())
    before = levels.copy(deep=True)
    open_figures = plt.get_fignums()
    _render(levels, levels.pct_change(fill_method=None).iloc[1:], tmp_path, frequency)

    expected = {
        "cumulative_performance": ([0, 10, -12, 21], [0, -10, -1, 10]),
        "drawdown": ([0, 0, -20, 0], [0, -10, -1, 0]),
    }
    for name, series in expected.items():
        figure = captured[name]
        assert tuple(figure.get_size_inches()) == LINE_FIGSIZE
        ax = figure.axes[0]
        for line, label, values in zip(ax.lines[:2], levels.columns, series):
            assert line.get_ydata() == pytest.approx(values)
            assert line.get_color() == SERIES_COLORS[label]
            assert pd.DatetimeIndex(line.get_xdata()).equals(levels.index)
        assert ax.get_ylabel().endswith("(%)")

    bars = captured[f"{frequency}_returns"]
    assert tuple(bars.get_size_inches()) == PERIOD_FIGSIZE
    ax = bars.axes[0]
    for i, (container, label, expected_values) in enumerate(zip(
        ax.containers, levels.columns, ([10., -20., 37.5], [-10., 10., 100 / 9]),
    )):
        assert [bar.get_height() for bar in container] == pytest.approx(expected_values)
        assert [bar.get_x() + bar.get_width() / 2 for bar in container] == pytest.approx(
            np.arange(3) + (-0.1875 if i == 0 else 0.1875),
        )
        assert all(bar.get_facecolor() == colors.to_rgba(SERIES_COLORS[label]) for bar in container)
    window = "Test Window" if frequency == "monthly" else "Live Window"
    assert ax.get_title() == f"{window} {frequency.title()} Returns"
    for figure in captured.values():
        ax = figure.axes[0]
        assert [item.get_text() for item in ax.get_legend().texts] == list(levels.columns)
        assert any(line.get_visible() for line in ax.get_ygridlines())
    for path in tmp_path.glob("*.png"):
        assert mpimg.imread(path).size > 0
    assert len(list(tmp_path.glob("*.png"))) == 3
    assert plt.get_fignums() == open_figures
    pd.testing.assert_frame_equal(levels, before)


def test_backtest_adapter_uses_independent_net_test_window(monkeypatch, tmp_path):
    dates = pd.date_range("2024-01-31", "2026-01-31", freq=pd.offsets.MonthEnd())
    net = pd.Series(100 * 1.01 ** np.arange(len(dates)), index=dates)
    result = SimpleNamespace(test_results=net, test_results_gross=net * 2, full_results=net * 3)
    benchmark = pd.Series(
        np.arange(27) + 200.,
        index=pd.date_range("2023-12-31", periods=27, freq=pd.offsets.MonthEnd()),
    )
    paths = BacktestPaths(tmp_path / "backtest").for_strategy("momentum")

    def capture(levels, returns, **kwargs):
        assert levels.index.equals(dates)
        assert len(returns) == 24
        pd.testing.assert_series_equal(levels.Strategy, net, check_names=False)
        pd.testing.assert_series_equal(
            levels["S&P 500 Total Return"], benchmark.loc[dates], check_names=False,
        )
        assert returns.Strategy.to_numpy() == pytest.approx(np.repeat(.01, 24))
        assert kwargs["return_frequency"] == "monthly"
        assert kwargs["returns_path"] == paths.monthly_returns_png

    monkeypatch.setattr(backtest_plots, "render_performance_plots", capture)
    backtest_plots.save_performance_plots(result, benchmark, paths)


@pytest.mark.parametrize("terminal_open", [False, True])
def test_live_curves_keep_inception_and_bars_exclude_partial_intervals(
    captured, tmp_path, terminal_open,
):
    daily = pd.DataFrame({
        "Date": ["2026-02-13", "2026-02-13", "2026-02-17", "2026-02-18", "2026-02-20"],
        "NAV": [100., 110., 88., 121., 120.],
        "Benchmark_Level": [200., 180., 198., 220., 240.],
        "Include_In_Risk_Metrics": [False, False, True, True, False],
    })
    daily["Portfolio_Return"] = daily.NAV.pct_change(fill_method=None)
    daily["Benchmark_Return"] = daily.Benchmark_Level.pct_change(fill_method=None)
    if not terminal_open:
        daily = daily.iloc[:-1].copy()
    before = daily.copy(deep=True)
    paths = LivePaths(tmp_path / "live").for_strategy("momentum")
    live_plots.save_performance_plots(daily, paths)
    lines = captured["cumulative_performance"].axes[0].lines
    assert lines[0].get_ydata() == pytest.approx([0., 10., -12., 21.] + ([20.] if terminal_open else []))
    assert len(lines[0].get_xdata()) == len(daily)
    bars = captured["daily_returns"].axes[0].containers
    assert [bar.get_height() for bar in bars[0]] == pytest.approx([-20., 37.5])
    assert [bar.get_height() for bar in bars[1]] == pytest.approx([10., 100 / 9])
    pd.testing.assert_frame_equal(daily, before)


def test_canonical_live_plots_keep_cash_period_and_56_complete_returns(
    canonical, captured, tmp_path,
):
    _, result = canonical
    paths = LivePaths(tmp_path / "live").for_strategy("momentum")
    live_plots.save_performance_plots(result.daily_nav, paths)
    ax = captured["daily_returns"].axes[0]
    assert [len(container) for container in ax.containers] == [56, 56]
    np.testing.assert_array_equal(ax.get_xticks(), np.arange(0, 56, 5))
    labels = [label.get_text() for label in ax.get_xticklabels()]
    assert (labels[0], labels[-1]) == ("2026-02-17", "2026-05-06")
    line = captured["cumulative_performance"].axes[0].lines[0]
    assert len(line.get_ydata()) == 58
    assert pd.Timestamp(line.get_xdata()[0]) == pd.Timestamp("2026-02-13")
    assert line.get_ydata()[-1] == pytest.approx(
        100 * (result.nav.End_NAV.iloc[-1] / 1_000_000 - 1),
    )


def test_daily_bar_labels_do_not_force_an_off_cadence_final_date(captured, tmp_path):
    dates = pd.bdate_range("2026-02-13", periods=13)
    levels = pd.DataFrame({label: np.arange(13) + 100. for label in _levels().columns}, index=dates)
    _render(levels, levels.pct_change().iloc[1:], tmp_path)
    ax = captured["daily_returns"].axes[0]
    np.testing.assert_array_equal(ax.get_xticks(), [0, 5, 10])
    assert [len(container) for container in ax.containers] == [12, 12]
    assert ax.get_xticklabels()[-1].get_text() == dates[-2].strftime("%Y-%m-%d")


@pytest.mark.parametrize("start,end,expected", [
    ("2026-02-13", "2026-03-06", ["2026-02-16", "2026-03-02"]),
    ("2026-02-16", "2026-03-02", ["2026-02-16", "2026-03-02"]),
    ("2026-02-13", "2026-02-14", ["2026-02-13"]),
])
def test_live_calendar_labels_cover_holidays_and_short_windows(start, end, expected):
    from portfolio_core.plot_style import date_axis_ticks

    ticks = date_axis_ticks(pd.Timestamp(start), pd.Timestamp(end), "daily")
    assert ticks.equals(pd.DatetimeIndex(expected))


@pytest.mark.parametrize("invalid", ["missing_level", "zero_level", "unordered", "missing_return", "duplicate_return", "outside_window"])
def test_invalid_data_fails_before_writing_any_figure(tmp_path, invalid):
    levels = _levels()
    returns = levels.pct_change(fill_method=None).iloc[1:]
    if invalid == "missing_level":
        levels.iloc[1, 1] = np.nan
    elif invalid == "zero_level":
        levels.iloc[1, 0] = 0.
    elif invalid == "unordered":
        levels = levels.iloc[::-1]
    elif invalid == "missing_return":
        returns.iloc[1, 0] = np.nan
    elif invalid == "duplicate_return":
        returns.index = returns.index[:1].repeat(len(returns))
    else:
        returns.index = returns.index + pd.Timedelta(days=100)
    with pytest.raises(ValueError):
        _render(levels, returns, tmp_path)
    assert not list(tmp_path.iterdir())


def test_missing_monthly_benchmark_is_not_silently_dropped(tmp_path):
    levels = _levels()
    paths = BacktestPaths(tmp_path / "backtest").for_strategy("momentum")
    with pytest.raises(ValueError, match="finite, nonmissing"):
        backtest_plots.save_performance_plots(
            SimpleNamespace(test_results=levels.Strategy),
            levels["S&P 500 Total Return"].drop(levels.index[1]), paths,
        )
    assert not paths.figures_dir.exists()


def test_save_failure_closes_figure_and_leaves_no_partial_png(monkeypatch, tmp_path):
    from matplotlib.figure import Figure

    def fail(*args, **kwargs):
        raise OSError("test save failure")

    monkeypatch.setattr(Figure, "savefig", fail)
    levels = _levels()
    before = plt.get_fignums()
    with pytest.raises(OSError, match="test save failure"):
        _render(levels, levels.pct_change(fill_method=None).iloc[1:], tmp_path)
    assert plt.get_fignums() == before
    assert not list(tmp_path.iterdir())
