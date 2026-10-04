"""Focused contracts for selected-run backtest strategy audit exports."""

from types import SimpleNamespace

import pandas as pd
import pytest

from backtest.engine import decision_audit_columns, trade_audit_columns
from backtest.evaluation import save_strategy_audit_tables
from backtest.paths import BacktestPaths
from portfolio_core.strategies import build_registered_strategy


_STRATEGY = build_registered_strategy("momentum")
_DECISION_COLUMNS = decision_audit_columns(_STRATEGY)
_TRADE_COLUMNS = trade_audit_columns(_STRATEGY)


def _audit_metadata() -> dict:
    return {
        "Signal_Cutoff": pd.Timestamp("2022-01-31"),
        "Execution_Date": pd.Timestamp("2022-01-31"),
        "Valuation_End": pd.Timestamp("2022-02-28"),
        "Asset_ID": "AAA.N",
        "Ticker": "AAA",
        "GICS_Sector_Code": "20",
        "Sector": "Industrials",
        "Sector_As_Of_Date": pd.Timestamp("2022-01-31"),
        "Sector_Source_Type": "Wikipedia",
        "Sector_Source_Reference": "revision-1",
        "Strategy_ID": _STRATEGY.strategy_id,
        "Strategy_Version": _STRATEGY.strategy_version,
        "Strategy_Score": 0.1,
        "Strategy_Rank": 1,
        "Strategy_Eligible": True,
        "Strategy_Exclusion_Reason": "",
        "Momentum": 0.2,
        "Sizing_Volatility": 0.1,
        "Construction_Sector": "20",
        "Signal_Selected_Side": "Long",
        "Execution_Eligible": True,
        "Final_Selected_Side": "Long",
        "Execution_Adjustment": "",
    }


def _decision_audit() -> pd.DataFrame:
    return pd.DataFrame([{
        **_audit_metadata(),
        "Signal_Raw_Target_Weight": 0.1,
        "Final_Target_Weight": 0.1,
        "Applied_Target_Weight": 0.1,
        "Decision_Complete": True,
        "Used_Hold_Logic": False,
        "Applied_Rule": "selected_long",
    }], columns=_DECISION_COLUMNS)


def _trade_audit() -> pd.DataFrame:
    return pd.DataFrame([{
        **_audit_metadata(),
        "Signal_Raw_Target_Weight": 0.2,
        "Final_Target_Weight": 0.2,
        "Applied_Rule": "increase_long",
        "Current_Capital": 1_000.0,
        "Current_Weight": 0.1,
        "Target_Weight": 0.2,
        "Weight_Drift": 0.1,
        "Turnover_Threshold": 0.01,
        "Current_Position_Value": 100.0,
        "Requested_Target_Position_Value": 200.0,
        "Target_Position_Value": 200.0,
        "Trade_Notional": 100.0,
        "Current_Shares": 2.0,
        "Requested_Target_Shares": 4.0,
        "Applied_Target_Shares": 4.0,
        "Trade_Shares": 2.0,
        "Execution_Price": 50.0,
        "Effective_Execution_Price": 50.05,
        "Order_Count": 1,
        "Fixed_Fee": 2.0,
        "Spread_Rate": 0.001,
        "Spread_Cost": 0.1,
        "Cash_Effect": -102.1,
        "Restricted_Proceeds_Change": 0.0,
        "Turnover_Contribution": 0.1,
    }], columns=_TRADE_COLUMNS)


def test_strategy_audit_exports_fixed_schema_and_dates(tmp_path):
    paths = BacktestPaths(tmp_path / "backtest").for_strategy(
        _STRATEGY.strategy_id
    )
    results = SimpleNamespace(
        strategy=_STRATEGY,
        research_diagnostics=pd.DataFrame(),
        sector_residuals=pd.DataFrame(),
        decisions=_decision_audit(),
        trades=_trade_audit(),
    )

    save_strategy_audit_tables(results, paths)

    assert paths.strategy_decisions_csv.is_file()
    assert paths.strategy_trades_csv.is_file()
    decisions = pd.read_csv(paths.strategy_decisions_csv, dtype=str)
    trades = pd.read_csv(paths.strategy_trades_csv, dtype=str)
    assert list(decisions.columns) == _DECISION_COLUMNS
    assert list(trades.columns) == _TRADE_COLUMNS
    assert decisions.loc[0, "Signal_Cutoff"] == "2022-01-31"
    assert decisions.loc[0, "Sector_As_Of_Date"] == "2022-01-31"
    assert trades.loc[0, "Execution_Date"] == "2022-01-31"
    assert trades.loc[0, "Sector_As_Of_Date"] == "2022-01-31"
    assert trades.loc[0, "Trade_Notional"] == "100"


def test_strategy_audit_rejects_misaligned_sector_provenance(tmp_path):
    paths = BacktestPaths(tmp_path / "backtest").for_strategy(
        _STRATEGY.strategy_id
    )
    decisions = _decision_audit()
    decisions.loc[0, "Sector_As_Of_Date"] = pd.Timestamp("2021-12-31")

    with pytest.raises(ValueError, match="Signal_Cutoff"):
        save_strategy_audit_tables(
            SimpleNamespace(
                strategy=_STRATEGY,
                research_diagnostics=pd.DataFrame(),
                sector_residuals=pd.DataFrame(),
                decisions=decisions,
                trades=_trade_audit(),
            ),
            paths,
        )

    assert not paths.strategy_tables_dir.exists()


@pytest.mark.parametrize(
    ("exit_rule", "side"),
    (
        ("exit_long", 1),
        ("exit_short", -1),
        ("mandatory_off_universe_liquidation", 1),
        ("mandatory_off_universe_liquidation", -1),
    ),
)
def test_strategy_audit_allows_prior_sector_only_for_complete_exit(
    tmp_path, exit_rule, side,
):
    paths = BacktestPaths(tmp_path / "backtest").for_strategy(
        _STRATEGY.strategy_id
    )
    trade = _trade_audit()
    trade.loc[0, "Sector_As_Of_Date"] = pd.Timestamp("2021-12-31")
    trade.loc[0, "Applied_Rule"] = exit_rule
    trade.loc[0, "Current_Position_Value"] = side * 100.0
    trade.loc[0, "Current_Weight"] = side * 0.1
    trade.loc[0, "Current_Shares"] = side * 2.0
    trade.loc[0, "Target_Weight"] = 0.0
    trade.loc[0, "Requested_Target_Position_Value"] = 0.0
    trade.loc[0, "Target_Position_Value"] = 0.0
    trade.loc[0, "Trade_Notional"] = -side * 100.0
    trade.loc[0, "Requested_Target_Shares"] = 0.0
    trade.loc[0, "Applied_Target_Shares"] = 0.0
    trade.loc[0, "Trade_Shares"] = -side * 2.0
    execution_price = 50.0 - side * 0.05
    trade.loc[0, "Effective_Execution_Price"] = execution_price
    trade.loc[0, "Cash_Effect"] = side * 2.0 * execution_price - 2.0

    save_strategy_audit_tables(
        SimpleNamespace(
            strategy=_STRATEGY,
            research_diagnostics=pd.DataFrame(),
            sector_residuals=pd.DataFrame(),
            decisions=_decision_audit(),
            trades=trade,
        ),
        paths,
    )

    saved = pd.read_csv(paths.strategy_trades_csv, dtype=str)
    assert saved.loc[0, "Sector_As_Of_Date"] == "2021-12-31"
    assert saved.loc[0, "Applied_Rule"] == exit_rule


def test_strategy_audit_rejects_prior_non_exit_provenance(tmp_path):
    prior_non_exit = _trade_audit()
    prior_non_exit.loc[0, "Sector_As_Of_Date"] = pd.Timestamp("2021-12-31")
    with pytest.raises(ValueError, match="only for complete exits"):
        save_strategy_audit_tables(
            SimpleNamespace(
                strategy=_STRATEGY,
                research_diagnostics=pd.DataFrame(),
                sector_residuals=pd.DataFrame(),
                decisions=_decision_audit(),
                trades=prior_non_exit,
            ),
            BacktestPaths(tmp_path / "prior"),
        )


@pytest.mark.parametrize("exit_rule", ("exit_long", "mandatory_off_universe_liquidation"))
def test_strategy_audit_rejects_prior_partial_exit_provenance(tmp_path, exit_rule):
    partial_exit = _trade_audit()
    partial_exit.loc[0, "Sector_As_Of_Date"] = pd.Timestamp("2021-12-31")
    partial_exit.loc[0, "Applied_Rule"] = exit_rule
    partial_exit.loc[0, "Target_Position_Value"] = 50.0
    partial_exit.loc[0, "Trade_Notional"] = -50.0
    with pytest.raises(ValueError, match="only for complete exits"):
        save_strategy_audit_tables(
            SimpleNamespace(
                strategy=_STRATEGY,
                research_diagnostics=pd.DataFrame(),
                sector_residuals=pd.DataFrame(),
                decisions=_decision_audit(),
                trades=partial_exit,
            ),
            BacktestPaths(tmp_path / "partial"),
        )


def test_strategy_audit_rejects_future_provenance(tmp_path):
    future = _trade_audit()
    future.loc[0, "Sector_As_Of_Date"] = pd.Timestamp("2022-02-01")
    future.loc[0, "Applied_Rule"] = "exit_long"
    with pytest.raises(ValueError, match="future-dated"):
        save_strategy_audit_tables(
            SimpleNamespace(
                strategy=_STRATEGY,
                research_diagnostics=pd.DataFrame(),
                sector_residuals=pd.DataFrame(),
                decisions=_decision_audit(),
                trades=future,
            ),
            BacktestPaths(tmp_path / "future"),
        )
