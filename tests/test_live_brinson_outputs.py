"""Live Brinson API, output-path, and figure contract tests."""

from pathlib import Path
from dataclasses import fields, replace
from types import SimpleNamespace

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from live import brinson_attribution as live_brinson
from live.paths import LivePaths
from backtest import brinson_attribution as backtest_brinson
from backtest.paths import BacktestPaths
from portfolio_core import brinson_plotting as brinson_plots
from portfolio_core.brinson_attribution import (
    BrinsonResult, add_benchmark_consistency_columns, run_brinson_pipeline,
)
from portfolio_core.plot_style import SERIES_COLORS
from portfolio_core import performance_plotting as performance_plots
from tests.test_portfolio_brinson_attribution import _two_sector_inputs


def _brinson_result() -> live_brinson.LiveBrinsonResult:
    benchmark, audit, holdings = _two_sector_inputs()
    holdings["Stock_Return"] = [.04, .10, .04, .10]
    next_dates = {"Date": pd.Timestamp("2026-04-01"), "Next_Date": pd.Timestamp("2026-05-01")}
    benchmark = pd.concat([benchmark, benchmark.assign(
        **next_dates, Benchmark_Return=[.03, -.01],
    )], ignore_index=True)
    audit = add_benchmark_consistency_columns(pd.concat([audit, audit.assign(
        **next_dates, Reconstructed_Benchmark_Return=.006, SP500TR_Return=-.005,
    )], ignore_index=True))
    holdings = pd.concat([holdings, holdings.assign(
        **next_dates, Weight=[.36, .84, -.12, -.28], Stock_Return=[-.02, .01, -.02, .01],
    )], ignore_index=True)
    stock = run_brinson_pipeline(benchmark, audit, holdings)
    nav_rows, start = [], 1_000_000.
    for index, period in enumerate(audit.itertuples(index=False)):
        post_trade = start - 50.
        held = holdings.loc[holdings.Date.eq(period.Date)]
        end = post_trade * (1 + (held.Weight * held.Stock_Return).sum()) + 150.
        nav_rows.append({
            "Rebalance_ID": f"R{index + 1}", "Period_Type": "invested", "Period_Start": period.Date,
            "Period_End": period.Next_Date, "Start_Field": "Open", "End_Field": "Open",
            "Start_NAV": start, "Post_Trade_NAV": post_trade, "End_NAV": end,
            "Position_Count": len(held), "Interest": 150., "Cash_Interest_Credit": 200.,
            "Loan_Interest_Charge": 50., "Fixed_Fees": 20., "Spread_Cost": 30.,
            "Period_Return": end / start - 1., "Benchmark_Return": period.SP500TR_Return,
        })
        start = end
    return live_brinson.LiveBrinsonResult(
        **{item.name: getattr(stock, item.name) for item in fields(BrinsonResult)},
        account_reconciliation=live_brinson._account_reconciliation(pd.DataFrame(nav_rows), holdings, stock),
    )


def test_saved_brinson_outputs_preserve_paths_schemas_and_plot_semantics(monkeypatch, tmp_path):
    result = _brinson_result()
    paths = LivePaths(tmp_path / "live").for_strategy("momentum")
    captured: dict[str, object] = {}
    save_figure = brinson_plots.save_figure

    def capture(fig, path: Path, *, atomic: bool = True) -> None:
        assert atomic is True
        assert Path(path).parent == paths.attribution.figures_dir
        captured[Path(path).name] = fig
        save_figure(fig, path, atomic=atomic)

    monkeypatch.setattr(brinson_plots, "save_figure", capture)
    live_brinson.save_brinson_result(result, paths)
    attribution = paths.attribution
    assert all(path.is_file() for path in (
        attribution.benchmark_sector_csv, attribution.benchmark_audit_csv,
        attribution.sector_attribution_csv, attribution.monthly_attribution_csv,
        attribution.period_attribution_csv, attribution.account_reconciliation_csv,
    ))
    assert pd.read_csv(attribution.benchmark_audit_csv).columns.tolist() == result.benchmark_audit.columns.tolist()
    for path, expected in (
        (attribution.sector_attribution_csv, result.sector_attribution),
        (attribution.monthly_attribution_csv, result.period_attribution),
        (attribution.period_attribution_csv, result.total_attribution),
        (attribution.account_reconciliation_csv, result.account_reconciliation),
    ):
        saved = pd.read_csv(path)
        assert saved.columns.tolist() == expected.columns.tolist()
        assert saved.Attribution_Method.eq("Brinson-Fachler").all()
        numeric = expected.select_dtypes(include="number").columns
        np.testing.assert_allclose(saved[numeric], expected[numeric], atol=1e-14, rtol=0, equal_nan=True)
    saved = pd.read_csv(attribution.monthly_attribution_csv)
    combined = saved.loc[saved.Side.eq("Combined")]
    np.testing.assert_allclose(combined.Scaled_Interaction_Effect, [-.0032, .0056], atol=1e-14, rtol=0)
    account = pd.read_csv(attribution.account_reconciliation_csv)
    np.testing.assert_allclose(account.Interaction_Effect,
                               combined.Scaled_Interaction_Effect.to_numpy() * account.Post_Trade_NAV / account.Start_NAV,
                               atol=1e-14, rtol=0)
    np.testing.assert_allclose(account.Linked_Interaction_Effect,
                               account.Interaction_Effect * account.Link_Factor, atol=1e-14, rtol=0)
    for name in captured:
        assert mpimg.imread(attribution.figures_dir / name).size > 0

    try:
        assert set(captured) == {
            "active_decomposition_cumulative.png",
            "active_decomposition_period.png",
            "benchmark_consistency.png",
        }
        benchmark_figure = captured["benchmark_consistency.png"]
        benchmark_ax = benchmark_figure.axes[0]
        assert benchmark_ax.get_title() == "Live Benchmark Consistency Check"
        assert benchmark_ax.get_xlabel() == "Date"
        assert [text.get_text() for text in benchmark_ax.get_legend().texts] == [
            "Reconstructed live S&P 500 sector benchmark",
            "S&P 500 Total Return",
        ]
        assert list(benchmark_ax.lines[0].get_ydata()) == pytest.approx(
            [0., 4.4, 5.0264]
        )
        assert list(benchmark_ax.lines[1].get_ydata()) == pytest.approx(
            [0., 4., 3.48]
        )

        period_figure = captured["active_decomposition_period.png"]
        period_ax = period_figure.axes[0]
        assert period_ax.get_title() == "Live Brinson-Fachler Active Return Decomposition"
        assert period_ax.get_xlabel() == "Holding period"
        assert [label.get_text() for label in period_ax.get_xticklabels()] == [
            "2 Mar–1 Apr", "1 Apr–1 May",
        ]
        assert period_ax.get_ylabel() == "Contribution to return (pp)"
        assert [text.get_text() for text in period_ax.get_legend().texts] == [
            "Allocation",
            "Pure selection",
            "Interaction",
            "Active return",
        ]
        active_line = next(line for line in period_ax.lines if line.get_label() == "Active return")
        assert list(active_line.get_ydata()) == pytest.approx([1.6, -.4])
        assert active_line.get_color() == SERIES_COLORS["Active return"]

        cumulative_figure = captured["active_decomposition_cumulative.png"]
        cumulative_ax = cumulative_figure.axes[0]
        assert cumulative_ax.get_title() == (
            "Live Cumulative Brinson-Fachler Active Return Decomposition"
        )
        assert cumulative_ax.get_xlabel() == "Date"
        assert cumulative_ax.get_ylabel() == "Cumulative arithmetic contribution (pp)"
        assert [text.get_text() for text in cumulative_ax.get_legend().texts] == [
            "Allocation",
            "Pure selection",
            "Interaction",
            "Active return",
        ]
        lines = {line.get_label(): line for line in cumulative_ax.lines}
        for label, values in {"Allocation": [-.64, -.96], "Pure selection": [2.56, 1.92],
                              "Interaction": [-.32, .24], "Active return": [1.6, 1.2]}.items():
            assert list(lines[label].get_ydata()) == pytest.approx([0., *values])
            assert lines[label].get_color() == SERIES_COLORS[label]
    finally:
        for figure in captured.values():
            plt.close(figure)


def _with_initial_cash(result):
    """Extend the plotting fixture with a recorded cash interval and no stock row."""
    opening = pd.Timestamp("2026-02-13")
    first = result.benchmark_audit.Date.min()
    audit = result.benchmark_audit.iloc[[0]].copy()
    audit["Date"], audit["Next_Date"] = opening, first
    audit[["Reconstructed_Benchmark_Return", "SP500TR_Return"]] = 0.
    cash = result.account_reconciliation.iloc[[0]].copy()
    cash.loc[:, cash.select_dtypes(include="number").columns] = 0.
    cash["Period_Start"], cash["Period_End"], cash["Period_Type"] = opening, first, "cash"
    return replace(
        result,
        benchmark_audit=pd.concat([audit, result.benchmark_audit], ignore_index=True),
        account_reconciliation=pd.concat([cash, result.account_reconciliation], ignore_index=True),
    )


@pytest.mark.parametrize("cash", [False, True])
def test_live_lines_share_window_ticks_and_keep_opening_zeros_out_of_bars(monkeypatch, tmp_path, cash):
    result = _brinson_result()
    if cash:
        result = _with_initial_cash(result)
    originals = {item.name: getattr(result, item.name).copy(deep=True) for item in fields(result)}
    figures = {}

    def capture(figure, path, **kwargs):
        figures[Path(path).stem] = figure
        plt.close(figure)

    monkeypatch.setattr(brinson_plots, "save_figure", capture)
    monkeypatch.setattr(performance_plots, "save_figure", capture)
    paths = LivePaths(tmp_path / "live").for_strategy("momentum")
    live_brinson.save_brinson_result(result, paths)
    start, end = result.benchmark_audit.Date.min(), result.benchmark_audit.Next_Date.max()
    levels = pd.DataFrame({"Strategy": [100., 105.], "S&P 500 Total Return": [200., 202.]},
                          index=pd.DatetimeIndex([start, end]))
    performance_plots.render_performance_plots(
        levels, levels.pct_change().iloc[1:],
        cumulative_path=paths.cumulative_performance_png, drawdown_path=paths.drawdown_png,
        returns_path=paths.daily_returns_png, window_label="Live Window", return_frequency="daily",
        window_start=start, window_end=end,
    )
    expected_ticks = (["2026-02-16"] if cash else []) + [
        "2026-03-02", "2026-03-16", "2026-03-30", "2026-04-13", "2026-04-27",
    ]
    reference = figures["cumulative_performance"].axes[0]
    for name in ("cumulative_performance", "drawdown", "benchmark_consistency", "active_decomposition_cumulative"):
        ax = figures[name].axes[0]
        np.testing.assert_array_equal(ax.get_xticks(), reference.get_xticks())
        assert ax.get_xlim() == reference.get_xlim()
        assert [label.get_text() for label in ax.get_xticklabels()] == expected_ticks
        assert pd.Timestamp(ax.lines[0].get_xdata()[0]) == start
        assert pd.Timestamp(ax.lines[0].get_xdata()[-1]) == end
    combined = result.period_attribution.loc[result.period_attribution.Side.eq("Combined")]
    active = next(line for line in figures["active_decomposition_cumulative"].axes[0].lines
                  if line.get_label() == "Active return")
    prefix = 2 if cash else 1
    np.testing.assert_array_equal(active.get_ydata()[:prefix], np.zeros(prefix))
    np.testing.assert_allclose(active.get_ydata()[prefix:], 100 * combined.Scaled_Active_Return.cumsum())
    assert not pd.DatetimeIndex(active.get_xdata()).has_duplicates
    assert pd.Timestamp(active.get_xdata()[prefix - 1]) == combined.Date.min()
    assert [len(container) for container in figures["active_decomposition_period"].axes[0].containers] == [2, 2, 2]
    for name, before in originals.items():
        pd.testing.assert_frame_equal(getattr(result, name), before)


@pytest.mark.parametrize("period_count", [1, 2, 24])
def test_backtest_cumulative_plots_include_opening_and_quarterly_month_ends(monkeypatch, tmp_path, period_count):
    result = _brinson_result()
    dates = pd.date_range("2024-01-31", periods=period_count + 1, freq=pd.offsets.MonthEnd())
    first_period = result.period_attribution.loc[
        result.period_attribution.Date.eq(result.period_attribution.Date.min())
    ]
    result = replace(
        result,
        benchmark_audit=pd.concat([
            result.benchmark_audit.iloc[:1].assign(Date=start, Next_Date=end)
            for start, end in zip(dates[:-1], dates[1:])
        ], ignore_index=True),
        period_attribution=pd.concat([
            first_period.assign(Date=start, Next_Date=end)
            for start, end in zip(dates[:-1], dates[1:])
        ], ignore_index=True),
    )
    figures = {}

    def capture(figure, path, **kwargs):
        figures[Path(path).stem] = figure
        plt.close(figure)

    monkeypatch.setattr(brinson_plots, "save_figure", capture)
    monkeypatch.setattr(performance_plots, "save_figure", capture)
    paths = BacktestPaths(tmp_path).for_strategy("momentum")
    backtest_brinson.save_brinson_result(result, paths)
    levels = pd.DataFrame({"Strategy": np.arange(len(dates)) + 100.,
                           "S&P 500 Total Return": np.arange(len(dates)) + 200.}, index=dates)
    performance_plots.render_performance_plots(
        levels, levels.pct_change().iloc[1:],
        cumulative_path=paths.cumulative_performance_png, drawdown_path=paths.drawdown_png,
        returns_path=paths.monthly_returns_png, window_label="Test Window", return_frequency="monthly",
        window_start=dates[0], window_end=dates[-1],
    )
    reference = figures["cumulative_performance"].axes[0]
    expected_ticks = ["2024-01"] if period_count < 3 else [
        "2024-01", "2024-04", "2024-07", "2024-10", "2025-01",
        "2025-04", "2025-07", "2025-10", "2026-01",
    ]
    for name in ("benchmark_consistency", "active_decomposition_cumulative"):
        ax = figures[name].axes[0]
        assert [label.get_text() for label in ax.get_xticklabels()] == expected_ticks
        np.testing.assert_array_equal(ax.get_xticks(), reference.get_xticks())
        assert ax.get_xlim() == reference.get_xlim()
        for line in ax.lines:
            if line.get_label().startswith("_"):
                continue
            assert pd.DatetimeIndex(line.get_xdata()).equals(dates)
            assert line.get_ydata()[0] == 0.


@pytest.mark.parametrize("start,end,label", [
    ("2026-03-02", "2026-04-01", "2 Mar–1 Apr"),
    ("2026-04-01", "2026-05-01", "1 Apr–1 May"),
    ("2026-05-01", "2026-05-06", "1–6 May"),
    ("2025-12-31", "2026-01-31", "31 Dec 2025–31 Jan 2026"),
])
def test_attribution_period_labels_include_both_boundaries(start, end, label):
    assert brinson_plots._period_label(pd.Timestamp(start), pd.Timestamp(end)) == label


@pytest.mark.parametrize("fault", ["invalid_date", "missing_date", "wrong_boundary", "unrecorded_cash"])
def test_live_account_window_validation_precedes_output_writes(tmp_path, fault):
    result = _with_initial_cash(_brinson_result())
    account = result.account_reconciliation
    if fault == "unrecorded_cash":
        account.loc[0, "Period_Type"] = "invested"
    else:
        account["Period_Start"] = account["Period_Start"].astype(object)
        account.loc[0, "Period_Start"] = {
            "invalid_date": "not-a-date", "missing_date": pd.NaT,
            "wrong_boundary": pd.Timestamp("2026-02-14"),
        }[fault]
    paths = LivePaths(tmp_path).for_strategy("momentum")
    with pytest.raises(ValueError):
        live_brinson.save_brinson_result(result, paths)
    assert not paths.attribution.tables_dir.exists()
    assert not paths.attribution.figures_dir.exists()


def test_three_effect_bars_stack_positive_and_negative_values_independently():
    class RecordingAxes:
        def __init__(self):
            self.bars = {}

        def bar(self, x, values, *, bottom, label, color, width):
            self.bars[label] = (np.array(values), np.array(bottom), color)

    axes = RecordingAxes()
    brinson_plots.plot_signed_stacked_bars(axes, np.arange(2), [
        ("Allocation", np.array([1., -2.])),
        ("Pure selection", np.array([-3., 4.])),
        ("Interaction", np.array([5., -6.])),
    ])
    for label, expected in {"Allocation": [0., 0.], "Pure selection": [0., 0.],
                             "Interaction": [1., -2.]}.items():
        np.testing.assert_array_equal(axes.bars[label][1], expected)
        assert axes.bars[label][2] == SERIES_COLORS[label]


@pytest.mark.parametrize("domain", ["backtest", "live"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fault", [
    "empty", "missing_interaction", "invalid_date", "missing_date", "nonfinite_effect",
    "invalid_benchmark_date", "nonfinite_benchmark", "missing_benchmark_column",
    "invalid_start", "missing_start", "missing_start_column", "reversed_interval", "period_gap",
    "invalid_benchmark_start",
])
def test_invalid_plot_data_fails_before_creating_or_replacing_outputs(tmp_path, domain, existing, fault):
    paths_type, saver = (
        (BacktestPaths, backtest_brinson.save_brinson_result) if domain == "backtest"
        else (LivePaths, live_brinson.save_brinson_result)
    )
    paths = paths_type(tmp_path / domain).for_strategy("momentum")
    result = _brinson_result()
    if fault == "empty":
        result = replace(result, period_attribution=result.period_attribution.iloc[0:0])
    elif fault == "missing_interaction":
        result = replace(result, period_attribution=result.period_attribution.drop(columns="Scaled_Interaction_Effect"))
    elif fault == "missing_benchmark_column":
        result = replace(result, benchmark_audit=result.benchmark_audit.drop(columns="SP500TR_Return"))
    elif fault == "missing_start_column":
        result = replace(result, period_attribution=result.period_attribution.drop(columns="Date"))
    elif fault in {"invalid_start", "missing_start", "invalid_benchmark_start"}:
        frame = result.benchmark_audit if fault == "invalid_benchmark_start" else result.period_attribution
        frame["Date"] = frame["Date"].astype(object)
        frame.loc[0, "Date"] = pd.NaT if fault == "missing_start" else "not-a-date"
    elif fault in {"reversed_interval", "period_gap"}:
        combined_row = result.period_attribution.index[result.period_attribution.Side.eq("Combined")][-1]
        column = "Next_Date" if fault == "reversed_interval" else "Date"
        result.period_attribution.loc[combined_row, "Date"] = (
            result.period_attribution.loc[combined_row, column] + pd.Timedelta(days=1)
        )
    elif fault in {"invalid_date", "missing_date", "invalid_benchmark_date"}:
        frame = result.benchmark_audit if fault == "invalid_benchmark_date" else result.period_attribution
        frame["Next_Date"] = frame["Next_Date"].astype(object)
        frame.loc[0, "Next_Date"] = pd.NaT if fault == "missing_date" else "not-a-date"
    elif fault == "nonfinite_effect":
        result.period_attribution.loc[0, "Scaled_Interaction_Effect"] = np.inf
    else:
        result.benchmark_audit.loc[0, "SP500TR_Return"] = np.nan

    if existing:
        for path in paths.attribution.result_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"existing output")
    originals = {field.name: getattr(result, field.name).copy(deep=True) for field in fields(result)}
    with pytest.raises((ValueError, RuntimeError)):
        saver(result, paths)
    for name, original in originals.items():
        pd.testing.assert_frame_equal(getattr(result, name), original)
    if existing:
        assert all(path.read_bytes() == b"existing output" for path in paths.attribution.result_files)
    else:
        assert not paths.attribution.tables_dir.exists()
        assert not paths.attribution.figures_dir.exists()


@pytest.mark.parametrize("field,prepare", [
    ("benchmark_audit", brinson_plots.prepare_benchmark_consistency_plot_frame),
    ("period_attribution", brinson_plots.prepare_active_decomposition_plot_frame),
])
def test_plot_preparation_sorts_and_normalizes_a_copy(field, prepare):
    original = getattr(_brinson_result(), field).iloc[::-1].copy()
    original["Next_Date"] = original["Next_Date"].astype(str)
    original["Date"] = original["Date"].astype(str)
    before = original.copy(deep=True)
    prepared = prepare(original)
    assert prepared.Next_Date.is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(prepared.Next_Date)
    assert pd.api.types.is_datetime64_any_dtype(prepared.Date)
    pd.testing.assert_frame_equal(original, before)


def test_run_brinson_is_side_effect_free(monkeypatch, tmp_path):
    paths = LivePaths(tmp_path / "live").for_strategy("momentum")
    expected = _brinson_result()
    strategy = SimpleNamespace(holdings=pd.DataFrame(), nav=pd.DataFrame())
    config = SimpleNamespace(paths=paths)

    monkeypatch.setattr(live_brinson, "load_analysis_inputs", lambda _, **kwargs: object())
    monkeypatch.setattr(live_brinson, "load_prepared_shares", lambda _: pd.DataFrame())
    monkeypatch.setattr(
        live_brinson,
        "run_live_brinson",
        lambda inputs, shares, holdings, nav: expected,
    )
    monkeypatch.setattr(
        live_brinson,
        "save_brinson_result",
        lambda result, output_paths: pytest.fail("unexpected output write"),
    )

    actual = live_brinson.run_brinson(
        config,
        strategy=strategy,
    )

    assert actual is expected
    assert not paths.attribution.tables_dir.exists()
    assert not paths.attribution.figures_dir.exists()
