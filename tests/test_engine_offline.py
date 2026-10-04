"""Deterministic in-memory regression test for the engine wrapper."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import backtest.engine as engine
from backtest.data_loading import BacktestDataset
from backtest.engine import (
    _BacktestAuditCollector,
    _run_backtest_impl,
    decision_audit_columns,
    run_backtest,
    trade_audit_columns,
)
from portfolio_core.sector_assignments import sector_rows_asof
from portfolio_core.strategies import (
    StrategyDecisionContext,
)
from _sector_test_helpers import (
    _synthetic_sector_assignments as _sector_assignments,
)
from _strategy_test_helpers import momentum_test_strategy


from portfolio_core.strategies.research_parameters import (
    BufferParameters, ExposureParameters, StockSelection,
)


_DEFAULT_STRATEGY = momentum_test_strategy(
    selection=StockSelection(20, 20), exposure=ExposureParameters(1.5, 0.7),
    turnover_threshold=0.015,
)


def make_synthetic_backtest_data() -> tuple[BacktestDataset, pd.DatetimeIndex]:
    dates = pd.date_range("2014-01-31", periods=30, freq="ME")
    tickers = [f"T{i:02d}" for i in range(60)]
    time = np.arange(len(dates), dtype=float)[:, None]
    asset = np.arange(len(tickers), dtype=float)[None, :]
    returns = (
        0.004
        + (asset - 29.5) * 0.00015
        + 0.008 * np.sin((time + 1.0) * (asset + 3.0) / 37.0)
    )
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    volumes = 1_000_000.0 + asset * 25_000.0 + time * 1_000.0

    close = pd.DataFrame(prices, index=dates, columns=tickers)
    volume = pd.DataFrame(
        np.broadcast_to(volumes, prices.shape),
        index=dates,
        columns=tickers,
    )
    pit_matrix = pd.DataFrame(True, index=dates, columns=tickers)
    backtest_data = BacktestDataset(
        data_close=close,
        data_volume=volume,
        pit_matrix=pit_matrix,
        sector_assignments=_sector_assignments(dates, tickers),
        valid_trading_days=dates,
        rolling_dollar_vol=(close * volume).rolling(
            window=3,
            min_periods=1,
        ).median(),
        asset_to_ticker={ticker: ticker for ticker in tickers},
    )
    return backtest_data, dates


def test_public_wrapper_matches_unified_ledger_synthetic_baseline():
    backtest_data, dates = make_synthetic_backtest_data()

    nav, diagnostics, holdings = run_backtest(
        backtest_data,
        dates,
        _DEFAULT_STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
    )

    assert len(nav) == 18
    assert len(diagnostics) == 17
    assert len(holdings) == 680
    assert float(nav.iloc[-1]) == pytest.approx(
        1_077_985.883077512,
        abs=1e-9,
    )
    assert list(diagnostics.columns) == [
        "NAV",
        "Num_Trades",
        "Fees",
        "Spread_Cost",
        "Turnover",
        "Turnover_Threshold",
        "Interest",
        "Cash_Interest_Credit",
        "Loan_Interest_Charge",
        "Cash",
        "Signed_Market_Value",
        "Gross_Market_Value",
        "Restricted_Short_Proceeds",
        "Free_Cash",
        "Loan",
        "Gross_Exposure",
        "Maximum_Position_Weight",
        "Position_Count",
        "Long_Count",
        "Short_Count",
        "Feasibility_Scale",
        "Feasibility_Adjustment",
        "Used_Hold_Logic",
        "Net_Exposure",
        "Post_Trade_Gross_Exposure",
        "Post_Trade_Net_Exposure",
        "Target_Gross",
        "Construction_Ready",
    ]
    assert list(holdings.columns) == [
        "Date",
        "Next_Date",
        "Asset_ID",
        "Ticker",
        "GICS_Sector_Code",
        "Sector",
        "Sector_As_Of_Date",
        "Sector_Source_Type",
        "Sector_Source_Reference",
        "Weight",
        "Stock_Return",
    ]
    assert np.allclose(
        diagnostics["NAV"],
        diagnostics["Cash"] + diagnostics["Signed_Market_Value"],
    )
    assert np.allclose(
        diagnostics["Free_Cash"],
        diagnostics["Cash"] - diagnostics["Restricted_Short_Proceeds"],
    )
    assert np.allclose(
        diagnostics["Loan"],
        (-diagnostics["Free_Cash"]).clip(lower=0.0),
    )
    assert np.allclose(
        diagnostics["Interest"],
        diagnostics["Cash_Interest_Credit"]
        - diagnostics["Loan_Interest_Charge"],
    )
    assert (diagnostics["Gross_Exposure"] <= 2.0 + 1e-10).all()
    assert holdings["Asset_ID"].equals(holdings["Ticker"])
    assert pd.to_datetime(holdings["Sector_As_Of_Date"]).equals(
        pd.to_datetime(holdings["Date"])
    )
    assert holdings["Sector_Source_Type"].eq("Wikipedia").all()


def test_disabled_diagnostics_do_not_accumulate_records_or_change_outputs(monkeypatch):
    backtest_data, dates = make_synthetic_backtest_data()
    expected_nav, diagnostics, expected_holdings = run_backtest(
        backtest_data, dates, _DEFAULT_STRATEGY,
        return_diagnostics=True, return_holdings=True,
    )
    assert not diagnostics.empty
    assemble_outputs = engine._assemble_backtest_outputs

    def assemble_without_diagnostics(state, **kwargs):
        assert state.diagnostics == []
        return assemble_outputs(state, **kwargs)

    monkeypatch.setattr(
        engine, "_assemble_backtest_outputs", assemble_without_diagnostics
    )
    audit = _BacktestAuditCollector(_DEFAULT_STRATEGY)
    nav, holdings = _run_backtest_impl(
        backtest_data, dates, _DEFAULT_STRATEGY,
        return_diagnostics=False, return_holdings=True,
        strategy_audit=audit, debug=True,
    )
    pd.testing.assert_series_equal(nav, expected_nav, check_exact=True)
    pd.testing.assert_frame_equal(holdings, expected_holdings, check_exact=True)
    assert not audit.decisions_frame().empty
    assert not audit.trades_frame().empty


def test_internal_strategy_audit_preserves_outputs_and_reconciles_trades():
    backtest_data, dates = make_synthetic_backtest_data()
    expected_nav, expected_diagnostics, expected_holdings = run_backtest(
        backtest_data,
        dates,
        _DEFAULT_STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
    )
    audit = _BacktestAuditCollector(_DEFAULT_STRATEGY)

    nav, diagnostics, holdings = _run_backtest_impl(
        backtest_data,
        dates,
        _DEFAULT_STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
        strategy_audit=audit,
    )

    pd.testing.assert_series_equal(nav, expected_nav)
    pd.testing.assert_frame_equal(diagnostics, expected_diagnostics)
    pd.testing.assert_frame_equal(holdings, expected_holdings)

    decisions = audit.decisions_frame()
    trades = audit.trades_frame()
    assert list(decisions.columns) == decision_audit_columns(_DEFAULT_STRATEGY)
    assert list(trades.columns) == trade_audit_columns(_DEFAULT_STRATEGY)
    assert not decisions.empty
    assert not trades.empty
    assert not decisions.duplicated(["Signal_Cutoff", "Asset_ID"]).any()
    assert not trades.duplicated(["Execution_Date", "Asset_ID"]).any()
    assert pd.to_datetime(decisions["Sector_As_Of_Date"]).equals(
        pd.to_datetime(decisions["Signal_Cutoff"])
    )
    assert pd.to_datetime(trades["Sector_As_Of_Date"]).equals(
        pd.to_datetime(trades["Execution_Date"])
    )
    for frame in (decisions, trades):
        assert frame[
            [
                "GICS_Sector_Code",
                "Sector",
                "Sector_Source_Type",
                "Sector_Source_Reference",
            ]
        ].ne("").all().all()
        assert frame["Sector_Source_Type"].eq("Wikipedia").all()

    assert trades["Trade_Notional"].ne(0.0).all()
    assert trades["Order_Count"].isin([1, 2]).all()
    np.testing.assert_allclose(
        trades["Target_Position_Value"] - trades["Current_Position_Value"],
        trades["Trade_Notional"],
        atol=1e-10,
        rtol=0.0,
    )

    audit_by_end = trades.groupby("Valuation_End", sort=True).agg(
        Num_Trades=("Order_Count", "sum"),
        Fees=("Fixed_Fee", "sum"),
        Spread_Cost=("Spread_Cost", "sum"),
        Turnover=("Turnover_Contribution", "sum"),
    )
    audit_by_end.index = pd.to_datetime(audit_by_end.index)
    expected = diagnostics.loc[
        audit_by_end.index,
        ["Num_Trades", "Fees", "Spread_Cost", "Turnover"],
    ]
    np.testing.assert_array_equal(
        audit_by_end["Num_Trades"].to_numpy(),
        expected["Num_Trades"].to_numpy(),
    )
    np.testing.assert_allclose(
        audit_by_end[["Fees", "Spread_Cost", "Turnover"]],
        expected[["Fees", "Spread_Cost", "Turnover"]],
        atol=1e-10,
        rtol=0.0,
    )


def test_forced_exit_uses_prior_sector_and_preserves_numerical_nav():
    backtest_data, dates = make_synthetic_backtest_data()
    _, _, baseline_holdings = run_backtest(
        backtest_data,
        dates,
        _DEFAULT_STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
    )
    forced_asset = str(
        baseline_holdings.loc[
            baseline_holdings["Date"].eq(dates[19]),
            "Asset_ID",
        ].iloc[0]
    )
    pit = backtest_data.pit_matrix.copy()
    pit.loc[dates[20]:, forced_asset] = False
    with_exit = replace(backtest_data, pit_matrix=pit)

    active_rows = []
    for date, rows in with_exit.sector_assignments.groupby(
        "As_Of_Date", sort=False
    ):
        active = set(pit.loc[pd.Timestamp(date)].loc[lambda row: row].index)
        active_rows.append(rows.loc[rows["Asset_ID"].isin(active)])
    active_only = replace(
        with_exit,
        sector_assignments=pd.concat(active_rows, ignore_index=True),
    )

    expected = run_backtest(with_exit, dates, _DEFAULT_STRATEGY)
    audit = _BacktestAuditCollector(_DEFAULT_STRATEGY)
    actual = _run_backtest_impl(
        active_only,
        dates,
        _DEFAULT_STRATEGY,
        strategy_audit=audit,
    )

    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    forced = audit.trades_frame().loc[
        lambda frame: frame["Asset_ID"].eq(forced_asset)
        & frame["Execution_Date"].eq(dates[20])
    ]
    assert len(forced) == 1
    assert forced.iloc[0]["Applied_Rule"] in {"exit_long", "exit_short"}
    assert forced.iloc[0]["Target_Position_Value"] == 0.0
    assert forced.iloc[0]["Sector_As_Of_Date"] == dates[19]
    assert pd.isna(forced.iloc[0]["Strategy_Score"])
    assert forced.iloc[0][list(_DEFAULT_STRATEGY.signal_columns)].isna().all()


def test_current_strategy_applies_buffer_and_preserves_context_inputs():
    assets = [f"A{i:02d}" for i in range(30)]
    dates = pd.date_range("2020-01-31", periods=13, freq="ME")
    expected_momentum = pd.Series(
        np.linspace(0.5, -0.5, 30), index=assets
    )
    close = pd.DataFrame(100.0, index=dates, columns=assets)
    close.loc[dates[-2]] = 100.0 * (1.0 + expected_momentum)
    close.loc[dates[-1]] = close.loc[dates[-2]]
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=assets)
    previous_weights = pd.Series({"A10": 0.05, "A19": -0.05})
    sectors = {asset: "20" for asset in assets}
    original_inputs = (
        close.copy(),
        volume.copy(),
        previous_weights.copy(),
    )

    strategy = momentum_test_strategy(
        exposure=ExposureParameters(1.0, 0.6), buffer=BufferParameters(True, 1.2),
    )
    decision = strategy.decide(
        StrategyDecisionContext(
            close_history=close,
            volume_history=volume,
            candidate_asset_ids=tuple(assets),
            sector_code_by_asset_id=sectors,
            previous_target_weights=previous_weights,
            signal_cutoff=dates[-1],
            signal_source_max_date=dates[-1],
        )
    )

    assert decision.is_complete
    assert decision.original_long_asset_ids == (
        "A00", "A01", "A02", "A03", "A04",
        "A05", "A06", "A07", "A08", "A10",
    )
    assert decision.original_short_asset_ids == (
        "A29", "A28", "A27", "A26", "A25",
        "A24", "A23", "A22", "A21", "A19",
    )
    assert decision.raw_target_weights[decision.raw_target_weights > 0].sum() == pytest.approx(0.6)
    assert decision.raw_target_weights[decision.raw_target_weights < 0].sum() == pytest.approx(-0.4)
    pd.testing.assert_series_equal(
        decision.signal_audit.Momentum, expected_momentum,
        check_names=False, atol=1e-15, rtol=0,
    )
    for actual, expected in zip(
        (close, volume, previous_weights),
        original_inputs,
    ):
        if isinstance(actual, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected)
        else:
            pd.testing.assert_series_equal(actual, expected)


def test_backtest_initial_holdings_track_targets_with_whole_share_rounding():
    backtest_data, dates = make_synthetic_backtest_data()
    _, _, holdings = run_backtest(
        backtest_data,
        dates,
        _DEFAULT_STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
    )
    first_decision_date = pd.Timestamp(holdings["Date"].min())
    first_next_date = pd.Timestamp(
        holdings.loc[
            holdings["Date"].eq(first_decision_date), "Next_Date"
        ].iloc[0]
    )

    active = backtest_data.pit_matrix.loc[first_decision_date]
    candidate_frame = pd.DataFrame({
        "Current_Close": backtest_data.data_close.loc[first_decision_date],
        "Next_Close": backtest_data.data_close.loc[first_next_date],
    })
    eligible = candidate_frame.loc[
        sorted(set(active[active].index) & set(candidate_frame.index))
    ]
    eligible = eligible[
        np.isfinite(eligible["Current_Close"])
        & np.isfinite(eligible["Next_Close"])
        & eligible["Current_Close"].gt(0.0)
        & eligible["Next_Close"].gt(0.0)
    ]

    decision = _DEFAULT_STRATEGY.decide(
        StrategyDecisionContext(
            close_history=backtest_data.data_close.loc[:first_decision_date],
            volume_history=backtest_data.data_volume.loc[:first_decision_date],
            candidate_asset_ids=tuple(sorted(eligible.index.astype(str))),
            sector_code_by_asset_id=sector_rows_asof(
                backtest_data.sector_assignments,
                first_decision_date,
            )["GICS_Sector_Code"].astype(str).to_dict(),
            previous_target_weights=pd.Series(dtype=float),
            signal_cutoff=first_decision_date,
            signal_source_max_date=first_decision_date,
        )
    )
    implemented = holdings.loc[
        holdings["Date"].eq(first_decision_date)
    ].set_index("Asset_ID")["Weight"]

    expected = decision.raw_target_weights.sort_index()
    implemented = implemented.sort_index()
    assert implemented.index.equals(expected.index)
    assert np.sign(implemented).equals(np.sign(expected))
    assert float((implemented - expected).abs().max()) < 1e-4


@pytest.mark.parametrize(
    (
        "apply_turnover_threshold",
        "apply_fees",
        "apply_spread",
        "buffer_enabled",
    ),
    (
        pytest.param(True, True, True, False, id="TTTF"),
        pytest.param(True, False, False, True, id="TFFT"),
        pytest.param(False, True, False, False, id="FTFF"),
        pytest.param(False, False, True, True, id="FFTT"),
        pytest.param(False, False, False, False, id="FFFF"),
    ),
)
def test_backtest_supports_representative_flag_combinations(
    apply_turnover_threshold,
    apply_fees,
    apply_spread,
    buffer_enabled,
):
    backtest_data, dates = make_synthetic_backtest_data()
    strategy = momentum_test_strategy(
        selection=StockSelection(20, 20), exposure=ExposureParameters(1.5, 0.7),
        turnover_threshold=0.015,
        buffer=BufferParameters(buffer_enabled, 1.6 if buffer_enabled else None),
    )
    nav, diagnostics = run_backtest(
        backtest_data,
        dates,
        strategy,
        apply_turnover_threshold=apply_turnover_threshold,
        apply_fees=apply_fees,
        apply_spread=apply_spread,
        return_diagnostics=True,
    )

    assert not nav.empty
    assert not diagnostics.empty
    assert np.isfinite(nav.to_numpy(dtype=float)).all()
    if not apply_fees:
        assert diagnostics["Fees"].eq(0.0).all()
    if not apply_spread:
        assert diagnostics["Spread_Cost"].eq(0.0).all()
    if not apply_turnover_threshold:
        _, thresholded_diagnostics = run_backtest(
            backtest_data,
            dates,
            strategy,
            apply_turnover_threshold=True,
            apply_fees=apply_fees,
            apply_spread=apply_spread,
            return_diagnostics=True,
        )
        assert diagnostics["Turnover_Threshold"].eq(
            strategy.turnover_threshold
        ).all()
        assert diagnostics["Num_Trades"].sum() > thresholded_diagnostics[
            "Num_Trades"
        ].sum()


def test_backtest_empty_investment_preserves_requested_return_shapes():
    backtest_data, dates = make_synthetic_backtest_data()
    assets = list(backtest_data.data_close.columns[:5])
    sparse = BacktestDataset(
        data_close=backtest_data.data_close.loc[:, assets],
        data_volume=backtest_data.data_volume.loc[:, assets],
        pit_matrix=backtest_data.pit_matrix.loc[:, assets],
        sector_assignments=backtest_data.sector_assignments.loc[
            backtest_data.sector_assignments["Asset_ID"].isin(assets)
        ].copy(),
        valid_trading_days=backtest_data.valid_trading_days,
        rolling_dollar_vol=backtest_data.rolling_dollar_vol.loc[:, assets],
        asset_to_ticker={asset: asset for asset in assets},
    )

    nav, diagnostics, holdings = run_backtest(
        sparse,
        dates,
        _DEFAULT_STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
    )

    assert nav.empty
    assert diagnostics.empty
    assert holdings.empty
    assert list(holdings.columns) == [
        "Date",
        "Next_Date",
        "Asset_ID",
        "Ticker",
        "GICS_Sector_Code",
        "Sector",
        "Sector_As_Of_Date",
        "Sector_Source_Type",
        "Sector_Source_Reference",
        "Weight",
        "Stock_Return",
    ]


def test_explicit_momentum_runs_through_the_same_engine_with_zero_threshold():
    backtest_data, dates = make_synthetic_backtest_data()
    strategy = momentum_test_strategy()
    audit = _BacktestAuditCollector(strategy)

    nav, diagnostics, holdings = _run_backtest_impl(
        backtest_data,
        dates,
        strategy,
        return_diagnostics=True,
        return_holdings=True,
        strategy_audit=audit,
    )

    assert not nav.empty
    assert diagnostics["Turnover_Threshold"].eq(0.0).all()
    assert diagnostics["Num_Trades"].gt(0).all()
    assert holdings.groupby("Date")["Asset_ID"].nunique().eq(20).all()
    decisions = audit.decisions_frame()
    trades = audit.trades_frame()
    assert list(decisions.columns) == decision_audit_columns(strategy)
    assert list(trades.columns) == trade_audit_columns(strategy)
    selected = decisions.loc[decisions["Final_Selected_Side"].ne("")]
    assert selected.groupby(["Signal_Cutoff", "Final_Selected_Side"]).size().eq(10).all()
    assert selected.loc[
        selected["Final_Selected_Side"].eq("Long"), "Final_Target_Weight"
    ].eq(0.05).all()
    assert selected.loc[
        selected["Final_Selected_Side"].eq("Short"), "Final_Target_Weight"
    ].eq(-0.05).all()


@pytest.mark.parametrize(
    "display_mapping",
    [
        {},
        {"extra": "EXTRA"},
        {"asset": ""},
    ],
)
def test_data_acquisition_requires_complete_nonfabricated_display_metadata(
    display_mapping,
):
    date = pd.DatetimeIndex([pd.Timestamp("2024-01-31")])
    close = pd.DataFrame({"asset": [10.0]}, index=date)
    with pytest.raises(ValueError, match="asset_to_ticker"):
        BacktestDataset(
            data_close=close,
            data_volume=close,
            pit_matrix=pd.DataFrame({"asset": [True]}, index=date),
            sector_assignments=_sector_assignments(date, ["asset"]),
            valid_trading_days=date,
            rolling_dollar_vol=close,
            asset_to_ticker=display_mapping,
        )
