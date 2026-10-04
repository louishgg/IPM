"""Prepared-only Brinson analysis for the historical live strategy."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
from portfolio_core.io import atomic_write_dataframe

from portfolio_core.brinson_attribution import (
    ATTRIBUTION_METHOD,
    STOCK_EFFECT_COLUMNS,
    BENCHMARK_CONSTITUENT_COLUMNS,
    BrinsonResult,
    build_brinson_benchmark,
    run_brinson_pipeline,
)
from portfolio_core.interval_valuation import (
    has_interval_event,
    value_one_share_through_events,
)
from portfolio_core.brinson_plotting import (
    prepare_active_decomposition_plot_frame,
    prepare_benchmark_consistency_plot_frame,
    render_benchmark_consistency,
    render_signed_decomposition,
)
from portfolio_core.sector_assignments import (
    SECTOR_AUDIT_COLUMNS,
    sector_rows_asof,
    validate_holding_sector_provenance,
)

from .analysis_data import (
    LiveAnalysisInputs,
    LiveAnalysisResult,
    load_analysis_inputs,
    membership_as_of,
    price_series,
    read_prepared_csv,
    validate_analysis_manifest,
    validate_account_nav,
)
from .config import LiveConfig
from .paths import LivePaths
from .preparation_artifacts import PREPARE_COMMAND
from .strategy_universe import (
    DOWNLOAD_PRICES_COMMAND,
    DOWNLOAD_SHARES_COMMAND,
    live_non_extinguished_assets,
)


SHARES_COLUMNS = {"Date", "Asset_ID", "Shares_Outstanding"}


@dataclass(frozen=True, slots=True)
class LiveBrinsonResult(BrinsonResult):
    """Shared stock attribution plus the full live account reconciliation."""

    account_reconciliation: pd.DataFrame


def load_prepared_shares(paths: LivePaths) -> pd.DataFrame:
    """Load the point-in-time shares table and reject future/backfilled state."""
    path = Path(paths.shares.prepared_shares_csv)
    validate_analysis_manifest(paths, {"shares_prepared"})
    shares = read_prepared_csv(path, "shares outstanding", DOWNLOAD_SHARES_COMMAND)
    missing = sorted(SHARES_COLUMNS - set(shares.columns))
    if missing:
        raise ValueError(f"Prepared shares outstanding is missing columns: {missing}")
    try:
        shares["Date"] = pd.to_datetime(
            shares["Date"], format="%Y-%m-%d", errors="raise"
        ).dt.tz_localize(None)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prepared shares contains an invalid Date") from exc
    shares["Asset_ID"] = shares["Asset_ID"].astype(str).str.strip()
    shares["Shares_Outstanding"] = pd.to_numeric(
        shares["Shares_Outstanding"], errors="coerce"
    )
    if shares.duplicated(["Date", "Asset_ID"]).any():
        raise ValueError("Prepared shares must be unique at (Date, Asset_ID)")
    invalid = (
        shares["Asset_ID"].eq("")
        | shares["Shares_Outstanding"].isna()
        | ~np.isfinite(shares["Shares_Outstanding"])
        | (shares["Shares_Outstanding"] <= 0.0)
    )
    if invalid.any():
        raise ValueError("Prepared shares requires nonblank assets and positive finite values")
    return shares.sort_values(["Date", "Asset_ID"], kind="stable").reset_index(drop=True)


def _shares_as_of(
    shares: pd.DataFrame,
    date: pd.Timestamp,
    asset_ids: list[str],
) -> pd.Series:
    eligible = shares.loc[
        shares["Asset_ID"].isin(asset_ids) & (shares["Date"] <= pd.Timestamp(date))
    ].sort_values(["Asset_ID", "Date"], kind="stable")
    latest = eligible.groupby("Asset_ID", sort=True).tail(1).set_index("Asset_ID")
    values = latest["Shares_Outstanding"].reindex(asset_ids)
    if values.isna().any():
        missing = values.index[values.isna()].tolist()
        raise RuntimeError(
            f"Prepared shares have no observation on or before {date.date()} for "
            f"active constituents {missing}. Run `{DOWNLOAD_SHARES_COMMAND}` then "
            f"`{PREPARE_COMMAND}`."
        )
    return values.astype(float)


def _optional_prepared_price(
    market: pd.DataFrame,
    date: pd.Timestamp,
    field: str,
    asset_id: str,
) -> float | None:
    """Return one finite positive prepared price, if available."""
    values = market.loc[
        market["Date"].eq(pd.Timestamp(date))
        & market["Asset_ID"].eq(str(asset_id)),
        field,
    ]
    if len(values) != 1:
        return None
    try:
        price = float(values.iloc[0])
    except (TypeError, ValueError):
        return None
    return price if np.isfinite(price) and price > 0.0 else None


def _benchmark_event_values(
    inputs: LiveAnalysisInputs,
    active: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    field: str,
    *,
    context: str,
) -> dict[str, float]:
    """Value complete one-share economics for assets with interval events."""
    actions = inputs.corporate_actions
    event_assets = [
        asset_id
        for asset_id in active
        if has_interval_event(
            actions.events,
            actions.legs,
            asset_id,
            start,
            end,
        )
    ]
    values: dict[str, float] = {}
    for asset_id in event_assets:
        valuation = value_one_share_through_events(
            actions.events,
            actions.legs,
            actions.sources,
            asset_id,
            start,
            end,
            lambda successor: _optional_prepared_price(
                inputs.market_daily,
                end,
                field,
                successor,
            ),
        )
        if valuation is None:
            raise RuntimeError(
                f"Missing prepared event-adjusted {field.lower()} value for "
                f"{context} on {pd.Timestamp(end).date()}: [{asset_id!r}]. "
                f"Run `{DOWNLOAD_PRICES_COMMAND}` then "
                f"`{PREPARE_COMMAND}`."
            )
        event_result, end_value = valuation
        monetary_currencies = set(
            event_result.cash_flows.get(
                "Currency", pd.Series(dtype="string")
            ).astype(str)
        ) | set(
            event_result.nontradable_rights.get(
                "Currency", pd.Series(dtype="string")
            ).astype(str)
        )
        unsupported_currencies = monetary_currencies - {"USD"}
        if unsupported_currencies:
            raise RuntimeError(
                f"Unsupported benchmark corporate-action currencies for "
                f"{asset_id}: {sorted(unsupported_currencies)}"
            )
        value = float(end_value)
        if not np.isfinite(value) or value < 0.0:
            raise RuntimeError(
                f"Invalid benchmark corporate-action value for {asset_id}"
            )
        values[asset_id] = value
    return values


def build_live_benchmark_sector_series(
    inputs: LiveAnalysisInputs,
    shares: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build live benchmark inputs, then delegate pure weighting to shared code."""
    metadata = inputs.metadata.set_index("Asset_ID", drop=False)
    constituent_rows: list[dict[str, Any]] = []
    audit_context: list[dict[str, Any]] = []

    for schedule_row in inputs.evaluation_periods.itertuples(index=False):
        # Shared benchmark input requires a nonblank period key, including
        # cash-only account intervals. It is not an execution identifier.
        period_id = str(schedule_row.Rebalance_ID) or "initial_cash"
        date = pd.Timestamp(schedule_row.Period_Start)
        next_date = pd.Timestamp(schedule_row.Period_End)
        nominal_active = membership_as_of(inputs.membership, date)
        active = sorted(
            live_non_extinguished_assets(
                nominal_active,
                as_of_date=date,
                corporate_action_events=inputs.corporate_actions.events,
                corporate_action_legs=inputs.corporate_actions.legs,
            )
        )
        missing_metadata = sorted(set(nominal_active) - set(metadata.index))
        if missing_metadata:
            raise RuntimeError(
                f"Execution-date membership lacks metadata on {date.date()}: "
                f"{missing_metadata}"
            )
        sector_rows = sector_rows_asof(
            inputs.sector_assignments,
            date,
        )
        missing_sectors = sorted(set(active) - set(sector_rows.index.astype(str)))
        if missing_sectors:
            raise RuntimeError(
                f"Execution-date benchmark lacks sector assignments on "
                f"{date.date()}: {missing_sectors}"
            )
        unknown = sorted(
            asset_id for asset_id in active
            if str(sector_rows.at[asset_id, "GICS_Sector_Code"])
            in {"", "Unknown"}
            or str(sector_rows.at[asset_id, "Sector"]) in {"", "Unknown"}
        )
        if unknown:
            raise RuntimeError(
                f"Execution-date benchmark has unknown sectors on {date.date()}: {unknown}"
            )
        start_prices = price_series(
            inputs.market_daily,
            date,
            str(schedule_row.Start_Field),
            active,
            context=f"{period_id} Brinson benchmark start",
        )
        event_values = _benchmark_event_values(
            inputs,
            active,
            date,
            next_date,
            str(schedule_row.End_Field),
            context=f"{period_id} Brinson benchmark end",
        )
        quoted_assets = [
            asset_id for asset_id in active if asset_id not in event_values
        ]
        end_values = price_series(
            inputs.market_daily,
            next_date,
            str(schedule_row.End_Field),
            quoted_assets,
            context=f"{period_id} Brinson benchmark end",
        )
        if event_values:
            end_values = pd.concat(
                [end_values, pd.Series(event_values, dtype="float64")]
            )
        end_values = end_values.reindex(active)
        shares_t0 = _shares_as_of(shares, date, active)
        constituent_rows.extend(
            {
                "Period_ID": period_id,
                "Start_Date": date,
                "End_Date": next_date,
                "Asset_ID": asset_id,
                "GICS_Sector_Code": str(
                    sector_rows.at[asset_id, "GICS_Sector_Code"]
                ),
                "Sector": str(sector_rows.at[asset_id, "Sector"]),
                "Start_Price": float(start_prices.at[asset_id]),
                "End_Value_Per_Start_Share": float(end_values.at[asset_id]),
                "Shares_Outstanding": float(shares_t0.at[asset_id]),
            }
            for asset_id in active
        )

        benchmark_start = inputs.benchmark_daily.loc[
            inputs.benchmark_daily["Date"].eq(date),
            str(schedule_row.Start_Field),
        ]
        benchmark_end = inputs.benchmark_daily.loc[
            inputs.benchmark_daily["Date"].eq(next_date),
            str(schedule_row.End_Field),
        ]
        if len(benchmark_start) != 1 or len(benchmark_end) != 1:
            raise RuntimeError(
                "Prepared total-return benchmark lacks period endpoints for "
                f"{date.date()} -> {next_date.date()}"
            )
        total_return = float(
            benchmark_end.iloc[0] / benchmark_start.iloc[0] - 1.0
        )
        audit_context.append({
            "Period_ID": period_id,
            "PiT_Active_Count": int(len(nominal_active)),
            "Dropped_Constituent_Count": int(
                len(nominal_active) - len(active)
            ),
            "Missing_Price_Count": 0,
            "Missing_Shares_Count": 0,
            "Unknown_Sector_Count": 0,
            "SP500TR_Return": total_return,
        })

    constituents = pd.DataFrame(
        constituent_rows,
        columns=BENCHMARK_CONSTITUENT_COLUMNS,
    )
    sector, shared_audit = build_brinson_benchmark(constituents)
    sector = sector.rename(columns={"Period_ID": "Rebalance_ID"})
    audit = shared_audit.merge(
        pd.DataFrame(audit_context),
        on="Period_ID",
        how="left",
        validate="one_to_one",
    )
    audit["Return_Diff"] = (
        audit["Reconstructed_Benchmark_Return"] - audit["SP500TR_Return"]
    )
    audit = audit.rename(columns={"Period_ID": "Rebalance_ID"})
    for frame in (sector, audit):
        frame["Rebalance_ID"] = frame["Rebalance_ID"].replace("initial_cash", "")
    sector = sector[[
        "Rebalance_ID", "Date", "Next_Date", "GICS_Sector_Code", "Sector",
        "Benchmark_Weight", "Benchmark_Return", "Benchmark_Contribution",
        "Constituent_Count", "Sector_Market_Cap",
    ]]
    audit = audit[[
        "Rebalance_ID", "Date", "Next_Date", "PiT_Active_Count",
        "Valid_Constituent_Count", "Dropped_Constituent_Count",
        "Missing_Price_Count", "Missing_Shares_Count", "Unknown_Sector_Count",
        "Benchmark_Market_Cap", "Sector_Weight_Sum",
        "Reconstructed_Benchmark_Return", "SP500TR_Return", "Return_Diff",
    ]]
    return sector, audit



def validate_executed_holdings(
    holdings: pd.DataFrame,
    inputs: LiveAnalysisInputs,
) -> pd.DataFrame:
    """Validate that attribution consumes actual simulated share holdings."""
    required = {
        "Rebalance_ID",
        "Date",
        "Next_Date",
        "Asset_ID",
        *SECTOR_AUDIT_COLUMNS,
        "Shares",
        "Weight",
        "Stock_Return",
    }
    missing = sorted(required - set(holdings.columns))
    if missing:
        raise ValueError(f"Live strategy holdings is missing columns: {missing}")
    result = validate_holding_sector_provenance(
        holdings,
        inputs.sector_assignments,
        context="Live strategy",
    )
    result["Next_Date"] = pd.to_datetime(
        result["Next_Date"], errors="raise"
    ).dt.tz_localize(None)
    for column in ("Shares", "Weight", "Stock_Return"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if (
        result[["Shares", "Weight", "Stock_Return"]].isna().any().any()
        or not np.isfinite(result[["Shares", "Weight", "Stock_Return"]].to_numpy(dtype=float)).all()
        or result["Shares"].eq(0).any()
    ):
        raise ValueError("Live strategy holdings requires finite, nonzero executed positions")
    if not np.allclose(result["Shares"], np.round(result["Shares"]), atol=0.0):
        raise ValueError("Live strategy holdings Shares must be whole numbers")
    if result.duplicated(["Rebalance_ID", "Asset_ID"]).any():
        raise ValueError("Live strategy holdings contains duplicate assets within a period")

    scheduled = set(
        zip(
            inputs.schedule["Rebalance_ID"].astype(str),
            inputs.schedule["Execution_Date"],
            inputs.schedule["Valuation_End"],
        )
    )
    held_periods = set(
        zip(result["Rebalance_ID"].astype(str), result["Date"], result["Next_Date"])
    )
    if held_periods != scheduled:
        raise ValueError("Live strategy holdings periods do not match the prepared schedule")
    return result.sort_values(["Date", "Asset_ID"], kind="stable").reset_index(drop=True)


def run_live_brinson(
    inputs: LiveAnalysisInputs,
    shares: pd.DataFrame,
    holdings: pd.DataFrame,
    nav: pd.DataFrame,
) -> LiveBrinsonResult:
    """Compute Brinson effects from simulated executed holdings."""
    executed = validate_executed_holdings(holdings, inputs)
    account = validate_account_nav(nav, periods=inputs.evaluation_periods)
    benchmark_sector, benchmark_audit = build_live_benchmark_sector_series(
        inputs, shares
    )
    stock = run_brinson_pipeline(
        benchmark_sector,
        benchmark_audit,
        executed,
    )
    reconciliation = _account_reconciliation(account, executed, stock)
    return LiveBrinsonResult(
        **{item.name: getattr(stock, item.name) for item in fields(BrinsonResult)},
        account_reconciliation=reconciliation,
    )


def _account_reconciliation(
    nav: pd.DataFrame, holdings: pd.DataFrame, stock: BrinsonResult,
) -> pd.DataFrame:
    """Bridge unchanged signed stock effects to net account excess returns.

    Stock weights use post-trade equity. Multiplying their effects by
    post-trade / start equity puts all contributions on the same denominator.
    The net-exposure bridge includes the uninvested cash interval.
    """
    combined = stock.period_attribution.loc[
        stock.period_attribution["Side"].eq("Combined")
    ].set_index("Date")
    benchmark = stock.benchmark_audit.set_index("Date")
    rows = []
    effect_columns = (
        *STOCK_EFFECT_COLUMNS, "Net_Exposure_Effect",
        "Benchmark_Reconstruction_Effect", "Cash_Interest_Effect",
        "Borrowing_Interest_Effect", "Fixed_Fee_Effect", "Spread_Effect",
    )
    for period in nav.itertuples(index=False):
        held = holdings.loc[holdings["Date"].eq(period.Period_Start)]
        if len(held) != period.Position_Count:
            raise ValueError("Live NAV position count does not match executed holdings")
        ratio = period.Post_Trade_NAV / period.Start_NAV
        exposure = float(held["Weight"].sum())
        stock_return = float((held["Weight"] * held["Stock_Return"]).sum())
        if not np.isclose(
            stock_return * period.Post_Trade_NAV,
            period.End_NAV - period.Post_Trade_NAV - period.Interest,
            atol=1e-7, rtol=0.0,
        ):
            raise ValueError("Live stock returns do not reconcile account P&L")
        b = float(benchmark.at[period.Period_Start, "Reconstructed_Benchmark_Return"])
        if not np.isclose(
            benchmark.at[period.Period_Start, "SP500TR_Return"],
            period.Benchmark_Return, atol=1e-12, rtol=0.0,
        ):
            raise ValueError("Live NAV benchmark return does not match attribution")
        stock_effects = dict.fromkeys(STOCK_EFFECT_COLUMNS, 0.0)
        if period.Period_Type == "invested":
            effects = combined.loc[period.Period_Start]
            stock_effects = {
                column: ratio * float(effects[f"Scaled_{column}"])
                for column in STOCK_EFFECT_COLUMNS
            }
        row = {
            "Attribution_Method": ATTRIBUTION_METHOD,
            "Rebalance_ID": period.Rebalance_ID,
            "Period_Type": period.Period_Type,
            "Period_Start": period.Period_Start,
            "Start_Field": period.Start_Field,
            "Period_End": period.Period_End,
            "End_Field": period.End_Field,
            "Start_NAV": period.Start_NAV,
            "Post_Trade_NAV": period.Post_Trade_NAV,
            "Stock_Return_On_Start_NAV": ratio * stock_return,
            "Net_Exposure_On_Start_NAV": ratio * exposure,
            "Account_Return": period.Period_Return,
            "Benchmark_Return": period.Benchmark_Return,
            "Reconstructed_Benchmark_Return": b,
            "Account_Excess_Return": period.Period_Return - period.Benchmark_Return,
            **stock_effects,
            "Net_Exposure_Effect": (ratio * exposure - 1.0) * b,
            "Benchmark_Reconstruction_Effect": b - period.Benchmark_Return,
            "Cash_Interest_Effect": period.Cash_Interest_Credit / period.Start_NAV,
            "Borrowing_Interest_Effect": -period.Loan_Interest_Charge / period.Start_NAV,
            "Fixed_Fee_Effect": -period.Fixed_Fees / period.Start_NAV,
            "Spread_Effect": -period.Spread_Cost / period.Start_NAV,
        }
        row["Total_Effect"] = sum(row[name] for name in effect_columns)
        row["Residual"] = row["Account_Excess_Return"] - row["Total_Effect"]
        rows.append(row)
    result = pd.DataFrame(rows)
    if not np.allclose(result["Residual"], 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("Live attribution does not reconcile interval account excess returns")

    # Exact telescoping link: prior account growth times subsequent benchmark
    # growth. Summing linked interval effects gives compounded excess return.
    portfolio_growth = 1.0 + result["Account_Return"].to_numpy()
    benchmark_growth = 1.0 + result["Benchmark_Return"].to_numpy()
    link = np.array([
        np.prod(portfolio_growth[:i]) * np.prod(benchmark_growth[i + 1:])
        for i in range(len(result))
    ])
    result["Link_Factor"] = link
    for column in (*effect_columns, "Account_Excess_Return", "Total_Effect"):
        result[f"Linked_{column}"] = result[column] * link
    result["Cumulative_Account_Return"] = np.cumprod(portfolio_growth) - 1.0
    result["Cumulative_Benchmark_Return"] = np.cumprod(benchmark_growth) - 1.0
    compounded_excess = np.prod(portfolio_growth) - np.prod(benchmark_growth)
    if not np.isclose(
        result["Linked_Total_Effect"].sum(), compounded_excess,
        atol=1e-10, rtol=0.0,
    ):
        raise ValueError("Live attribution does not reconcile compounded account excess return")
    return result


def _plot_window(
    result: LiveBrinsonResult, benchmark_plot: pd.DataFrame, decomposition_plot: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Verify full account coverage before extending stock curves over initial cash."""
    account = result.account_reconciliation
    required = ["Period_Start", "Period_End", "Period_Type"]
    if missing := set(required) - set(account.columns):
        raise ValueError(f"Live chart account periods are missing columns: {sorted(missing)}")
    periods = account[required].rename(columns={
        "Period_Start": "Date", "Period_End": "Next_Date",
    }).copy()
    for column in ("Date", "Next_Date"):
        periods[column] = pd.to_datetime(periods[column], errors="raise").dt.tz_localize(None)
    periods = periods.sort_values("Next_Date", kind="stable").reset_index(drop=True)
    dates = ["Date", "Next_Date"]
    if not periods[dates].equals(benchmark_plot[dates]):
        raise ValueError("Live benchmark must cover every recorded account period")
    invested = periods.loc[periods["Period_Type"].eq("invested"), dates].reset_index(drop=True)
    if not invested.equals(decomposition_plot[dates]):
        raise ValueError("Live attribution must cover every invested account period")
    cash = periods["Period_Type"].eq("cash")
    if (
        not periods["Period_Type"].isin(["cash", "invested"]).all()
        or (periods.loc[cash, "Next_Date"] > invested["Date"].iloc[0]).any()
    ):
        raise ValueError("Only recorded initial cash periods may extend the stock chart")
    return periods["Date"].iloc[0], periods["Next_Date"].iloc[-1]


def save_brinson_result(result: LiveBrinsonResult, paths: LivePaths) -> None:
    """Persist all live Brinson tables and figures."""
    benchmark_plot = prepare_benchmark_consistency_plot_frame(result.benchmark_audit)
    decomposition_plot = prepare_active_decomposition_plot_frame(
        result.period_attribution
    )
    window_start, window_end = _plot_window(result, benchmark_plot, decomposition_plot)
    attribution = paths.attribution
    outputs = (
        (result.benchmark_sector, attribution.benchmark_sector_csv),
        (result.benchmark_audit, attribution.benchmark_audit_csv),
        (result.sector_attribution, attribution.sector_attribution_csv),
        (result.period_attribution, attribution.monthly_attribution_csv),
        (result.total_attribution, attribution.period_attribution_csv),
        (result.account_reconciliation, attribution.account_reconciliation_csv),
    )
    for frame, path in outputs:
        atomic_write_dataframe(
            frame,
            path,
            index=False,
            date_format="%Y-%m-%d",
            float_format=None,
            lineterminator="\n",
        )

    render_benchmark_consistency(
        benchmark_plot,
        attribution.benchmark_consistency_png,
        reconstructed_label="Reconstructed live S&P 500 sector benchmark",
        title="Live Benchmark Consistency Check",
        xlabel="Date",
        figsize=(9, 5.2),
        atomic=True,
        window_start=window_start, window_end=window_end, date_frequency="daily",
    )
    render_signed_decomposition(
        decomposition_plot,
        attribution.active_decomposition_period_png,
        attribution.active_decomposition_cumulative_png,
        period_title="Live Brinson-Fachler Active Return Decomposition",
        cumulative_title="Live Cumulative Brinson-Fachler Active Return Decomposition",
        xlabel="Date",
        period_figsize=(10, 5.5),
        cumulative_figsize=(9, 5.2),
        label_rotation=0,
        marker_size=4,
        atomic=True,
        window_start=window_start, window_end=window_end, date_frequency="daily",
    )


def remove_saved_brinson_result(paths: LivePaths) -> None:
    """Remove generated attribution for one successfully replaced strategy."""
    for path in paths.attribution.result_files:
        path.unlink(missing_ok=True)


def load_saved_strategy_holdings(paths: LivePaths) -> pd.DataFrame:
    path = paths.strategy_holdings_csv
    strategy_option = (
        f" --strategy {paths.strategy_id}"
        if paths.strategy_id is not None
        else " --strategy ID"
    )
    return read_prepared_csv(
        path,
        "live strategy holdings",
        f"python -m live.analyze strategy{strategy_option}",
    )


def run_brinson(
    config: LiveConfig,
    *,
    strategy: LiveAnalysisResult | None = None,
) -> LiveBrinsonResult:
    inputs = load_analysis_inputs(config.paths, config=config)
    shares = load_prepared_shares(config.paths)
    holdings = (
        strategy.holdings
        if strategy is not None
        else load_saved_strategy_holdings(config.paths)
    )
    nav = strategy.nav if strategy is not None else read_prepared_csv(
        config.paths.strategy_nav_csv, "live account NAV",
        f"python -m live.analyze strategy --strategy {config.paths.strategy_id}",
    )
    return run_live_brinson(inputs, shares, holdings, nav)
