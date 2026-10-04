"""Daily marking is independent of decisions, persistence and engine reruns."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from live.analysis_data import save_analysis_result
from live.paths import LivePaths
from live.performance import DailyValuationData, build_daily_nav, performance_summary
from live.config import DEFAULT_CONFIG
from portfolio_core.accounting_ledger import LedgerState
from tests.test_live_replay_calendar import canonical


def data_for(inputs):
    return DailyValuationData.from_frames(
        inputs.market_daily, inputs.benchmark_daily, inputs.corporate_actions,
        start=inputs.evaluation_periods.Period_Start.iloc[0],
        end=inputs.evaluation_periods.Period_End.iloc[-1],
    )


@pytest.mark.parametrize(("quantity", "expected"), [
    (0.0, False), (1e-12, False), (1.0, True), (-1.0, True),
])
def test_position_event_uses_ledger_normalized_holdings(quantity, expected):
    start, end = pd.Timestamp("2026-02-10"), pd.Timestamp("2026-02-20")
    actions = SimpleNamespace(
        events=pd.DataFrame({"Event_ID": ["E"], "Effective_Date": [end]}),
        legs=pd.DataFrame({"Event_ID": ["E"], "From_Asset_ID": ["HELD"]}),
    )
    data = DailyValuationData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), actions)
    state = LedgerState._from_series(
        pd.Series({"HELD": quantity}), 1_000_000., pd.Series(dtype=float), start,
    )
    assert data.has_position_event(state, end) is expected


def test_full_daily_window_cash_weekends_and_execution_costs(canonical):
    inputs, result = canonical
    daily = result.daily_nav
    closes = daily.loc[daily.Observation.eq("session_close")].set_index("Date")
    assert len(closes) == 57 and daily.Include_In_Risk_Metrics.sum() == 56
    assert not daily.Include_In_Risk_Metrics.iloc[:2].any()
    assert not (closes.index.dayofweek >= 5).any()
    assert pd.Timestamp("2026-02-16") not in closes.index
    first_full = closes.loc["2026-02-17"]
    assert first_full.Risk_Free_Return == pytest.approx((1 + .02 / 365) ** 4 - 1)
    assert first_full.Portfolio_Return == pytest.approx(first_full.Risk_Free_Return)
    p = result.nav.loc[result.nav.Rebalance_ID.eq("R1")].iloc[0]
    holdings = result.holdings.loc[result.holdings.Rebalance_ID.eq("R1")].set_index("Asset_ID")
    prices = inputs.market_daily.loc[inputs.market_daily.Date.eq(p.Period_Start)].set_index("Asset_ID").Close
    expected = p.Post_Trade_Cash + (holdings.Shares * prices.reindex(holdings.index)).sum()
    assert closes.loc[p.Period_Start, "NAV"] == pytest.approx(expected, abs=1e-7)
    assert p.Start_NAV - p.Post_Trade_NAV == pytest.approx(p.Fixed_Fees + p.Spread_Cost)
    assert daily.NAV.iloc[-1] == pytest.approx(result.nav.End_NAV.iloc[-1], abs=1e-7)
    assert np.prod(1 + daily.Portfolio_Return.dropna()) == pytest.approx(daily.NAV.iloc[-1] / daily.NAV.iloc[0])
    excess = daily.loc[daily.Include_In_Risk_Metrics, "Portfolio_Return"] - daily.loc[daily.Include_In_Risk_Metrics, "Risk_Free_Return"]
    assert result.performance.iloc[0].Sharpe_Ratio == pytest.approx(excess.mean() / excess.std(ddof=1) * np.sqrt(252))


def test_capm_p_values_survive_live_csv_export(canonical, tmp_path):
    _, result = canonical
    risk = result.daily_nav.loc[result.daily_nav.Include_In_Risk_Metrics]
    regression = stats.linregress(
        risk.Benchmark_Return - risk.Risk_Free_Return,
        risk.Portfolio_Return - risk.Risk_Free_Return,
    )
    alpha_p = 2 * stats.t.sf(
        abs(regression.intercept / regression.intercept_stderr), len(risk) - 2,
    )
    paths = LivePaths(tmp_path / "live").for_strategy("momentum")
    save_analysis_result(result, paths)
    saved = pd.read_csv(paths.strategy_performance_csv).set_index("Series")
    strategy = saved.loc["Live_Strategy"]
    assert np.isfinite(alpha_p) and np.isfinite(regression.pvalue)
    assert strategy.Alpha_P_Value == pytest.approx(alpha_p, abs=1e-12)
    assert strategy.Beta_P_Value == pytest.approx(regression.pvalue, abs=1e-12)
    columns = ["Alpha_P_Value", "Beta_P_Value"]
    pd.testing.assert_frame_equal(
        saved[columns], result.performance.set_index("Series")[columns],
    )


def test_reconstruction_is_pure_and_uses_no_engine_or_csv(canonical, monkeypatch):
    inputs, result = canonical
    before = [frame.copy(deep=True) for frame in (result.nav, result.holdings, result.trades)]
    def forbidden(*args, **kwargs):
        raise AssertionError("Daily valuation must not read outputs or rerun decisions")
    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr("live.analysis.run_strategy_analysis", forbidden)
    reconstructed = build_daily_nav(*before, data_for(inputs), DEFAULT_CONFIG.accounting)
    pd.testing.assert_frame_equal(reconstructed, result.daily_nav)
    for actual, original in zip(before, (result.nav, result.holdings, result.trades)):
        pd.testing.assert_frame_equal(actual, original)
    metrics = performance_summary(result.nav, reconstructed, DEFAULT_CONFIG.accounting, result.performance.iloc[0].Label, inputs.price_basis, result.performance.iloc[0].Simulation_Fingerprint)
    pd.testing.assert_frame_equal(metrics, result.performance)


def test_missing_interior_quote_and_benchmark_session_fail(canonical):
    inputs, result = canonical
    data = data_for(inputs)
    asset = result.holdings.loc[result.holdings.Rebalance_ID.eq("R1"), "Asset_ID"].iloc[0]
    date = pd.Timestamp("2026-03-05")
    market, closes = data.market.copy(), data.closes.copy()
    market.loc[(date, asset), "Close"] = np.nan
    closes.loc[date, asset] = np.nan
    with pytest.raises(ValueError, match="Missing prepared Close.*2026-03-05"):
        build_daily_nav(result.nav, result.holdings, result.trades, replace(data, market=market, closes=closes), DEFAULT_CONFIG.accounting)
    with pytest.raises(ValueError, match="benchmark dates must match"):
        DailyValuationData.from_frames(inputs.market_daily, inputs.benchmark_daily.loc[inputs.benchmark_daily.Date.ne(date)], inputs.corporate_actions, start=pd.Timestamp("2026-02-13"), end=pd.Timestamp("2026-05-06"))


def test_tampered_trade_cash_is_not_silently_accepted(canonical):
    inputs, result = canonical
    trades = result.trades.copy()
    trades.loc[0, "Cash_Effect"] += 1
    with pytest.raises(ValueError, match="trade cash effect"):
        build_daily_nav(result.nav, result.holdings, trades, data_for(inputs), DEFAULT_CONFIG.accounting)


def test_tampered_benchmark_return_is_rejected(canonical):
    inputs, result = canonical
    nav = result.nav.copy()
    nav.loc[0, "Benchmark_Return"] += .01
    with pytest.raises(ValueError, match="saved benchmark return"):
        build_daily_nav(nav, result.holdings, result.trades, data_for(inputs), DEFAULT_CONFIG.accounting)


@pytest.mark.parametrize("cash,rate", [(1200., .02), (200., .08)])
def test_daily_financing_preserves_short_restrictions(canonical, cash, rate):
    inputs, _ = canonical
    state = LedgerState._from_series(
        pd.Series({"A000": -10.}), cash, pd.Series({"A000": 1000.}),
        pd.Timestamp("2026-02-13"),
    )
    marked = data_for(inputs).advance(state, pd.Timestamp("2026-02-17"), DEFAULT_CONFIG.accounting)
    expected = cash + (cash - 1000.) * ((1 + rate / 365) ** 4 - 1)
    assert marked.cash == pytest.approx(expected)
    assert marked.restricted_total == 1000.
    pd.testing.assert_series_equal(marked.shares, state.shares)
