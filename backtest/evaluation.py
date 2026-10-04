"""Explicit-strategy development/test evaluation and audit reporting."""
import dataclasses

import numpy as np
import pandas as pd
from portfolio_core.performance_metrics import annualized_ratio
from portfolio_core.corporate_actions import AUDIT_COLUMNS as ACTION_AUDIT_COLUMNS
from portfolio_core.accounting_config import (
    PortfolioAccountingConfig,
    spread_sensitivity_configs,
)
from portfolio_core.accounting_ledger import InfeasibleRebalanceError
from portfolio_core.io import atomic_write_dataframe
from portfolio_core.price_basis import PriceBasisSpec
from portfolio_core.strategies import Strategy
from portfolio_core.simulation_assumptions import build_simulation_assumptions

from .config import DEFAULT_CONFIG, BacktestEvaluationConfig
from .data_loading import BacktestDataset
from .engine import (
    _BacktestAuditCollector,
    _run_backtest_impl,
    decision_audit_columns,
    trade_audit_columns,
)
from .paths import BacktestPaths


def calculate_return_metrics(
    returns: pd.Series,
    *,
    cash_interest_rate: float,
) -> dict[str, float]:
    """Calculate monthly validation return, excess Sharpe and maximum drawdown."""
    values = pd.to_numeric(returns, errors="raise").astype(float)
    if len(values) < 2:
        raise ValueError("Validation metrics require at least two monthly returns")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Validation returns must be finite and nonmissing")
    pct_return = 100.0 * (float(np.prod(1.0 + values)) - 1.0)
    rf_monthly = (1.0 + cash_interest_rate) ** (1.0 / 12.0) - 1.0
    excess = values - rf_monthly
    sharpe = annualized_ratio(excess, 12.0, zero_variance_value=0.0)
    wealth = pd.Series(
        np.concatenate(([1.0], (1.0 + values).cumprod().to_numpy())),
        dtype=float,
    )
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "PctReturn": pct_return,
        "Sharpe": sharpe,
        "Maximum_Drawdown_Pct": 100.0 * float(drawdown.min()),
    }


CORPORATE_ACTION_OUTPUT_COLUMNS = [
    "Record_Type",
    "Period_Start",
    "Period_End",
    "Interest_Effect",
    *ACTION_AUDIT_COLUMNS,
    "Interest_Amount",
    "Cash_Interest_Credit",
    "Loan_Interest_Charge",
    "Counterfactual_Interest_Without_Actions",
]

@dataclasses.dataclass(frozen=True)
class BacktestAnalysisResult:
    development_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    strategy: Strategy
    development_results: pd.Series
    test_results: pd.Series
    full_results: pd.Series
    full_diag_net: pd.DataFrame
    development_results_gross: pd.Series
    test_results_gross: pd.Series
    test_holdings_gross: pd.DataFrame
    full_results_gross: pd.Series
    corporate_actions: pd.DataFrame
    transaction_cost_summary: pd.DataFrame
    decisions: pd.DataFrame
    trades: pd.DataFrame
    price_basis: PriceBasisSpec
    simulation_fingerprint: str
    accounting_config: PortfolioAccountingConfig
    spread_sensitivity: pd.DataFrame = dataclasses.field(
        default_factory=pd.DataFrame
    )
    research_diagnostics: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    sector_residuals: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)

def evaluate_strategy(
    backtest_data: BacktestDataset,
    dates,
    strategy: Strategy,
    include_costs=True,
    return_diagnostics=False,
    accounting_config: PortfolioAccountingConfig = DEFAULT_CONFIG.accounting,
    corporate_action_audit_records: list[dict[str, object]] | None = None,
    strategy_audit: _BacktestAuditCollector | None = None,
    return_holdings=False,
    *,
    require_complete_window=True,
):
    """Evaluate a window; only chronological development may defer activation."""
    dates = pd.DatetimeIndex(dates)
    if require_complete_window and (
        len(dates) < 2 or dates.has_duplicates or not dates.is_monotonic_increasing
    ):
        raise ValueError("A scored window needs at least two unique ordered NAV endpoints")
    outputs = _run_backtest_impl(
        backtest_data,
        dates,
        strategy,
        apply_fees=include_costs,
        apply_spread=include_costs,
        return_diagnostics=return_diagnostics,
        return_holdings=return_holdings,
        accounting_config=accounting_config,
        corporate_action_audit_records=corporate_action_audit_records,
        strategy_audit=strategy_audit,
    )
    if require_complete_window:
        nav = outputs[0] if return_diagnostics or return_holdings else outputs
        if nav.empty or nav.index[0] > dates[0]:
            raise InfeasibleRebalanceError(
                f"{dates[0]}: research account cannot start at the required NAV endpoint"
            )
        if not nav.index.equals(dates):
            raise ValueError("Evaluation NAV index does not match the complete requested window")
        returns = nav.pct_change(fill_method=None).iloc[1:]
        if not np.isfinite(nav).all() or not np.isfinite(returns).all():
            raise ValueError("Evaluation requires finite NAV and every scored return")
    return outputs


def summarize_transaction_costs(diag_df: pd.DataFrame, period_name: str) -> dict:
    """Summarize realized transaction costs and turnover for appendix reporting."""
    if diag_df is None or diag_df.empty:
        return {
            "Period": period_name,
            "Total_Trades": 0,
            "Total_Fees": 0.0,
            "Total_Spread_Cost": 0.0,
            "Total_Transaction_Costs": 0.0,
            "Average_Monthly_Turnover": np.nan,
            "Median_Monthly_Turnover": np.nan,
            "Maximum_Monthly_Turnover": np.nan,
        }

    total_fees = float(diag_df["Fees"].sum())
    total_spread = float(diag_df["Spread_Cost"].sum())

    return {
        "Period": period_name,
        "Total_Trades": int(diag_df["Num_Trades"].sum()),
        "Total_Fees": total_fees,
        "Total_Spread_Cost": total_spread,
        "Total_Transaction_Costs": total_fees + total_spread,
        "Average_Monthly_Turnover": float(diag_df["Turnover"].mean()),
        "Median_Monthly_Turnover": float(diag_df["Turnover"].median()),
        "Maximum_Monthly_Turnover": float(diag_df["Turnover"].max()),
    }


def _cumulative_pct_return(series: pd.Series) -> float:
    if series.empty or len(series) < 2:
        return float("nan")
    return float(100.0 * (series.iloc[-1] / series.iloc[0] - 1.0))


def evaluation_period_label(prefix: str, dates) -> str:
    index = pd.DatetimeIndex(dates)
    if index.empty:
        return prefix
    start_year = int(index.min().year)
    end_year = int(index.max().year)
    years = str(start_year) if start_year == end_year else f"{start_year}-{end_year}"
    return f"{prefix} ({years})"


def calculate_spread_sensitivity(
    backtest_data: BacktestDataset,
    base_result: BacktestAnalysisResult,
) -> pd.DataFrame:
    """Run low/base/high spread cases while keeping the base ranking official."""

    rows: list[dict[str, object]] = []
    periods = (
        ("Development_PctReturn", base_result.development_dates),
        ("Test_PctReturn", base_result.test_dates),
        ("Full_PctReturn", backtest_data.valid_trading_days),
    )
    base_series = {
        "Development_PctReturn": base_result.development_results,
        "Test_PctReturn": base_result.test_results,
        "Full_PctReturn": base_result.full_results,
    }
    for case, case_accounting in spread_sensitivity_configs(
        base_result.accounting_config
    ).items():
        coefficient = case_accounting.transaction_costs.liquidity_bps
        series_by_metric = (
            base_series
            if case == "base"
            else {
                metric: evaluate_strategy(
                    backtest_data,
                    dates,
                    base_result.strategy,
                    include_costs=True,
                    return_diagnostics=False,
                    accounting_config=case_accounting,
                    require_complete_window=metric == "Test_PctReturn",
                )
                for metric, dates in periods
            }
        )
        simulation_fingerprint = (
            base_result.simulation_fingerprint
            if case == "base"
            else str(
                build_simulation_assumptions(
                    case_accounting,
                    base_result.price_basis,
                )["simulation_fingerprint"]
            )
        )
        rows.append({
            "Sensitivity_Case": case,
            "Liquidity_Coefficient_Bps": float(coefficient),
            **{
                metric: _cumulative_pct_return(series_by_metric[metric])
                for metric, _dates in periods
            },
            "Price_Basis": base_result.price_basis.price_basis_id,
            "Simulation_Fingerprint": simulation_fingerprint,
            "Drives_Official_Ranking": case == "base",
        })
    return pd.DataFrame(rows)


def run_strategy_evaluation(
    backtest_data: BacktestDataset,
    strategy: Strategy,
    *,
    evaluation_config: BacktestEvaluationConfig = DEFAULT_CONFIG.evaluation,
    accounting_config: PortfolioAccountingConfig = DEFAULT_CONFIG.accounting,
) -> BacktestAnalysisResult:
    """Calculate development, test, and full results without writing outputs."""
    print("\n--- Starting Explicit Strategy Development/Test Evaluation ---")

    valid_trading_days = backtest_data.valid_trading_days
    development_dates = valid_trading_days[
        valid_trading_days <= pd.Timestamp(evaluation_config.validation_end_date)
    ]
    test_dates = valid_trading_days[
        valid_trading_days >= pd.Timestamp(evaluation_config.test_start_date)
    ]

    development_results, development_diag_net = evaluate_strategy(
        backtest_data,
        development_dates,
        strategy,
        include_costs=True,
        return_diagnostics=True,
        accounting_config=accounting_config,
        require_complete_window=False,
    )

    test_results, test_diag_net = evaluate_strategy(
        backtest_data,
        test_dates,
        strategy,
        include_costs=True,
        return_diagnostics=True,
        accounting_config=accounting_config,
        require_complete_window=True,
    )

    corporate_action_records: list[dict[str, object]] = []
    strategy_audit = _BacktestAuditCollector(strategy)
    full_results, full_diag_net = evaluate_strategy(
        backtest_data,
        valid_trading_days,
        strategy,
        include_costs=True,
        return_diagnostics=True,
        accounting_config=accounting_config,
        corporate_action_audit_records=corporate_action_records,
        strategy_audit=strategy_audit,
        require_complete_window=False,
    )

    corporate_actions = pd.DataFrame(corporate_action_records).reindex(
        columns=CORPORATE_ACTION_OUTPUT_COLUMNS
    )
    if not corporate_actions.empty:
        corporate_actions = corporate_actions.sort_values(
            ["Period_Start", "Period_End", "Record_Type", "Event_ID"],
            kind="stable",
        ).reset_index(drop=True)
    development_results_gross = evaluate_strategy(
        backtest_data,
        development_dates,
        strategy,
        include_costs=False,
        return_diagnostics=False,
        accounting_config=accounting_config,
        require_complete_window=False,
    )

    test_results_gross, test_holdings_gross = evaluate_strategy(
        backtest_data,
        test_dates,
        strategy,
        include_costs=False,
        return_diagnostics=False,
        return_holdings=True,
        accounting_config=accounting_config,
        require_complete_window=True,
    )

    full_results_gross = evaluate_strategy(
        backtest_data,
        valid_trading_days,
        strategy,
        include_costs=False,
        return_diagnostics=False,
        accounting_config=accounting_config,
        require_complete_window=False,
    )

    transaction_cost_summary = pd.DataFrame([
        summarize_transaction_costs(
            development_diag_net,
            evaluation_period_label("Development Period", development_dates),
        ),
        summarize_transaction_costs(
            test_diag_net,
            evaluation_period_label("Test Window", test_dates),
        ),
        summarize_transaction_costs(
            full_diag_net,
            evaluation_period_label("Full Backtest Window", valid_trading_days),
        ),
    ])
    decisions = strategy_audit.decisions_frame()
    trades = strategy_audit.trades_frame()
    price_basis = backtest_data.price_basis
    assumptions = build_simulation_assumptions(
        accounting_config,
        price_basis,
    )
    print(
        f"\n>>> Strategy Parameters Used: "
        f"{dataclasses.asdict(strategy.parameters)} <<<"
    )

    return BacktestAnalysisResult(
        development_dates=development_dates,
        test_dates=test_dates,
        strategy=strategy,
        development_results=development_results,
        test_results=test_results,
        full_results=full_results,
        full_diag_net=full_diag_net,
        development_results_gross=development_results_gross,
        test_results_gross=test_results_gross,
        test_holdings_gross=test_holdings_gross,
        full_results_gross=full_results_gross,
        corporate_actions=corporate_actions,
        transaction_cost_summary=transaction_cost_summary,
        decisions=decisions,
        trades=trades,
        price_basis=price_basis,
        simulation_fingerprint=str(assumptions["simulation_fingerprint"]),
        accounting_config=accounting_config,
        research_diagnostics=pd.DataFrame(strategy_audit.research_records),
        sector_residuals=pd.DataFrame(strategy_audit.sector_records),
    )


def save_evaluation_summary_tables(
    results: BacktestAnalysisResult,
    paths: BacktestPaths,
) -> None:
    """Persist completed accounting and transaction-cost summaries."""
    atomic_write_dataframe(
        results.corporate_actions,
        paths.security_event_accounting_audit_csv,
        index=False,
        date_format="%Y-%m-%d",
        float_format=None,
        lineterminator="\n",
    )
    atomic_write_dataframe(
        results.transaction_cost_summary,
        paths.transaction_cost_summary_csv,
        index=False,
        date_format=None,
        float_format=None,
    )
    if not results.spread_sensitivity.empty:
        atomic_write_dataframe(
            results.spread_sensitivity,
            paths.spread_sensitivity_csv,
            index=False,
            date_format=None,
            float_format=None,
        )
    nav_audit = results.full_diag_net.rename_axis("Date").reset_index()
    if not nav_audit.empty:
        if not np.allclose(
            nav_audit["NAV"],
            nav_audit["Cash"] + nav_audit["Signed_Market_Value"],
            atol=1e-7,
            rtol=0.0,
        ):
            raise ValueError("Backtest NAV audit does not reconcile to cash and positions")
        if not np.allclose(
            nav_audit["Interest"],
            nav_audit["Cash_Interest_Credit"]
            - nav_audit["Loan_Interest_Charge"],
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError("Backtest NAV financing split does not reconcile")
    atomic_write_dataframe(
        nav_audit,
        paths.strategy_nav_csv,
        index=False,
        date_format="%Y-%m-%d",
        float_format=None,
        lineterminator="\n",
    )
    print(
        f"\nSaved security-event accounting audit to "
        f"{paths.security_event_accounting_audit_csv}"
    )
    print(
        f"Saved transaction cost summary to "
        f"{paths.transaction_cost_summary_csv}"
    )
    print(f"Saved portfolio-accounting NAV audit to {paths.strategy_nav_csv}")


def _validated_strategy_audit(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    sector_date_column: str,
    artifact_name: str,
) -> pd.DataFrame:
    """Validate one deterministic analysis-output table before writing it."""
    if list(frame.columns) != columns:
        raise ValueError(
            f"{artifact_name} must have columns {columns}; "
            f"found {list(frame.columns)}"
        )
    result = frame.copy()
    for column in (
        "Signal_Cutoff",
        "Execution_Date",
        "Valuation_End",
        "Sector_As_Of_Date",
    ):
        result[column] = pd.to_datetime(result[column], errors="raise")
    sector_dates = result["Sector_As_Of_Date"]
    expected_dates = result[sector_date_column]
    if artifact_name == "Backtest trade audit":
        if sector_dates.gt(expected_dates).any():
            raise ValueError(
                "Backtest trade audit contains future-dated sector provenance"
            )
        prior = sector_dates.lt(expected_dates)
        exit_rule = result["Applied_Rule"].isin(
            ["exit_long", "exit_short", "mandatory_off_universe_liquidation"]
        )
        current_values = pd.to_numeric(
            result["Current_Position_Value"],
            errors="raise",
        )
        target_values = pd.to_numeric(
            result["Target_Position_Value"],
            errors="raise",
        )
        trade_values = pd.to_numeric(result["Trade_Notional"], errors="raise")
        complete_exit = (
            exit_rule
            & ~np.isclose(current_values, 0.0, atol=1e-12, rtol=0.0)
            & np.isclose(target_values, 0.0, atol=1e-12, rtol=0.0)
            & np.isclose(
                trade_values,
                -current_values,
                atol=1e-10,
                rtol=0.0,
            )
        )
        if (prior & ~complete_exit).any():
            raise ValueError(
                "Backtest trade audit permits prior sector provenance only "
                "for complete exits"
            )
    elif not sector_dates.equals(expected_dates):
        raise ValueError(
            f"{artifact_name} sector provenance is not aligned to "
            f"{sector_date_column}"
        )
    provenance = [
        "Asset_ID",
        "Ticker",
        "GICS_Sector_Code",
        "Sector",
        "Sector_Source_Type",
        "Sector_Source_Reference",
        "Applied_Rule",
    ]
    for column in provenance:
        result[column] = result[column].astype(str).str.strip()
    if result[provenance].isin(["", "Unknown", "nan"]).any().any():
        raise ValueError(f"{artifact_name} contains incomplete audit provenance")
    key = [sector_date_column, "Asset_ID"]
    if result.duplicated(key).any():
        raise ValueError(f"{artifact_name} contains duplicate asset/date rows")
    if artifact_name == "Backtest trade audit" and not result.empty:
        if result["Trade_Notional"].eq(0.0).any():
            raise ValueError("Backtest trade audit contains a zero-value trade")
        if not result["Order_Count"].isin([1, 2]).all():
            raise ValueError("Backtest trade audit contains an invalid order count")
        if (result[["Fixed_Fee", "Spread_Rate", "Spread_Cost"]] < 0.0).any().any():
            raise ValueError("Backtest trade audit contains a negative cost")
        if not np.allclose(
            result["Target_Position_Value"] - result["Current_Position_Value"],
            result["Trade_Notional"],
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError("Backtest trade audit position values do not reconcile")
        if not np.allclose(
            result["Applied_Target_Shares"] - result["Current_Shares"],
            result["Trade_Shares"],
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError("Backtest trade-audit shares do not reconcile")
        if not np.allclose(
            result["Trade_Shares"] * result["Execution_Price"],
            result["Trade_Notional"],
            atol=1e-7,
            rtol=0.0,
        ):
            raise ValueError("Backtest trade notionals do not reconcile")
        if not np.allclose(
            -result["Trade_Shares"] * result["Effective_Execution_Price"]
            - result["Fixed_Fee"],
            result["Cash_Effect"],
            atol=1e-7,
            rtol=0.0,
        ):
            raise ValueError("Backtest trade cash effects do not reconcile")
    return result.sort_values(key, kind="stable").reset_index(drop=True)


def save_strategy_audit_tables(
    results: BacktestAnalysisResult,
    paths: BacktestPaths,
) -> None:
    """Write selected full-period decisions and executed trades as CSV audits."""
    decisions = _validated_strategy_audit(
        results.decisions,
        columns=decision_audit_columns(results.strategy),
        sector_date_column="Signal_Cutoff",
        artifact_name="Backtest decision audit",
    )
    trades = _validated_strategy_audit(
        results.trades,
        columns=trade_audit_columns(results.strategy),
        sector_date_column="Execution_Date",
        artifact_name="Backtest trade audit",
    )
    for frame, path in (
        (decisions, paths.strategy_decisions_csv),
        (trades, paths.strategy_trades_csv),
    ):
        atomic_write_dataframe(
            frame,
            path,
            index=False,
            date_format="%Y-%m-%d",
            float_format="%.17g",
            lineterminator="\n",
        )
        print(f"Saved backtest strategy audit to {path}")
    for frame, path in (
        (results.research_diagnostics, paths.strategy_research_diagnostics_csv),
        (results.sector_residuals, paths.strategy_sector_residuals_csv),
    ):
        atomic_write_dataframe(
            frame, path, index=False, date_format="%Y-%m-%d",
            float_format="%.17g", lineterminator="\n",
        )
