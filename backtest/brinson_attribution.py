"""Brinson attribution data preparation and holdings export."""

import numpy as np
import pandas as pd
from portfolio_core.io import atomic_write_dataframe
from portfolio_core.brinson_attribution import (
    BENCHMARK_CONSTITUENT_COLUMNS,
    BrinsonResult,
    add_benchmark_consistency_columns as _derive_benchmark_consistency_columns,
    build_brinson_benchmark,
    run_brinson_pipeline,
)
from portfolio_core.portfolio_lifecycle import (
    asset_has_interval_event,
    asset_is_extinguished_from_events,
    event_valuation_result,
)
from portfolio_core.sector_assignments import (
    SECTOR_AUDIT_COLUMNS,
    sector_rows_asof,
    validate_holding_sector_provenance,
)
from portfolio_core.brinson_plotting import (
    prepare_active_decomposition_plot_frame,
    prepare_benchmark_consistency_plot_frame,
    render_benchmark_consistency,
    render_signed_decomposition,
)
from portfolio_core.plot_style import LINE_FIGSIZE, PERIOD_FIGSIZE
from .data_loading import BacktestDataset, active_pit_asset_ids
from .paths import BacktestPaths


def save_gross_brinson_holdings(
    holdings: pd.DataFrame,
    paths: BacktestPaths,
) -> None:
    """Persist already-computed gross holdings for later attribution."""
    atomic_write_dataframe(
        holdings,
        paths.brinson_holdings_gross_csv,
        index=False,
        date_format=None,
        float_format=None,
    )
    print(
        f"\nSaved gross Brinson holdings to "
        f"{paths.brinson_holdings_gross_csv}"
    )


def load_gross_brinson_holdings(paths: BacktestPaths) -> pd.DataFrame:
    """Load gross-of-cost Brinson holdings exported by the selected strategy."""
    holdings_path = paths.brinson_holdings_gross_csv
    if not holdings_path.exists():
        strategy_option = (
            f" --strategy {paths.strategy_id}"
            if paths.strategy_id is not None
            else " --strategy ID"
        )
        raise FileNotFoundError(
            f"Missing Brinson holdings file: "
            f"{holdings_path}. "
            "Run `python -m backtest.analyze strategy"
            f"{strategy_option}` first."
        )

    holdings = pd.read_csv(
        holdings_path,
        parse_dates=["Date", "Next_Date"],
    )
    required = {
        "Date",
        "Next_Date",
        "Asset_ID",
        "Ticker",
        *SECTOR_AUDIT_COLUMNS,
        "Weight",
        "Stock_Return",
    }
    missing = sorted(required - set(holdings.columns))
    if missing:
        raise ValueError(
            f"Gross Brinson holdings are missing canonical columns: {missing}"
        )
    holdings["Asset_ID"] = holdings["Asset_ID"].astype(str)
    holdings["Ticker"] = holdings["Ticker"].astype(str)
    holdings["Weight"] = holdings["Weight"].astype(float)
    holdings["Stock_Return"] = holdings["Stock_Return"].astype(float)
    return holdings.sort_values(["Date", "Asset_ID"]).reset_index(drop=True)


def get_brinson_periods_from_holdings(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """Return unique Date -> Next_Date periods covered by the holdings export."""
    periods = (
        holdings_df[["Date", "Next_Date"]]
        .drop_duplicates()
        .sort_values(["Date", "Next_Date"])
        .reset_index(drop=True)
    )
    if periods.empty:
        raise RuntimeError("No Brinson periods found in holdings data.")
    return periods


def _brinson_end_value_per_start_share(
    backtest_data: BacktestDataset,
    asset_id: str,
    date: pd.Timestamp,
    next_date: pd.Timestamp,
    price_t0: object,
    price_t1: object,
) -> float:
    """Return complete end value of one start-date constituent share."""
    try:
        start_price = float(price_t0)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(start_price) or start_price <= 0.0:
        return np.nan

    if asset_has_interval_event(backtest_data, asset_id, date, next_date):
        valuation = event_valuation_result(
            backtest_data,
            asset_id,
            date,
            next_date,
        )
        if valuation is None:
            return np.nan
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
        if monetary_currencies - {"USD"}:
            return np.nan
        end_value_float = float(end_value)
    else:
        try:
            end_value_float = float(price_t1)
        except (TypeError, ValueError):
            return np.nan

    if not np.isfinite(end_value_float) or end_value_float < 0.0:
        return np.nan
    return end_value_float


def build_brinson_benchmark_sector_series(
    backtest_data: BacktestDataset,
    shares_monthly: pd.DataFrame,
    periods: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build domain inputs and delegate benchmark weighting to shared code."""
    data_close = backtest_data.data_close
    price_asset_ids = set(data_close.columns.astype(str))
    from .share_resolution import validate_prepared_shares
    shares_table = validate_prepared_shares(shares_monthly)
    factors = shares_table.set_index(["Date", "Asset_ID"])["Capitalization_Factor"]
    shares_monthly = shares_table.pivot(index="Date", columns="Asset_ID", values="Shares_Outstanding")
    share_asset_ids = set(shares_monthly.columns.astype(str))
    constituent_rows: list[dict[str, object]] = []
    audit_context: list[dict[str, object]] = []
    events = backtest_data.security_events
    event_legs = backtest_data.security_event_legs

    for _, period in periods.iterrows():
        date = pd.Timestamp(period["Date"])
        next_date = pd.Timestamp(period["Next_Date"])
        period_id = date.strftime("%Y-%m-%d")
        if date not in data_close.index or next_date not in data_close.index:
            raise RuntimeError(
                f"Missing price data for Brinson period {date.date()} -> "
                f"{next_date.date()}."
            )
        if date not in shares_monthly.index:
            raise RuntimeError(
                f"Missing shares outstanding data for Brinson date {date.date()}."
            )

        active_asset_ids = active_pit_asset_ids(backtest_data, date)
        extinguished_predecessors = sorted(
            asset_id
            for asset_id in active_asset_ids
            if asset_is_extinguished_from_events(
                events,
                event_legs,
                asset_id,
                date,
            )
        )
        eligible = sorted(
            set(active_asset_ids) - set(extinguished_predecessors)
        )
        unavailable_asset_ids = sorted(set(eligible) - price_asset_ids)
        undeclared_unavailable = sorted(
            set(unavailable_asset_ids)
            - set(map(str, backtest_data.unavailable_members))
        )
        if undeclared_unavailable:
            raise RuntimeError(
                "Active membership constituents lack prices without reviewed "
                f"unavailable identities on {date.date()}: "
                f"{undeclared_unavailable}."
            )
        candidate_asset_ids = sorted(set(eligible) & price_asset_ids)
        missing_share_assets = sorted(
            set(candidate_asset_ids) - share_asset_ids
        )
        if missing_share_assets:
            raise RuntimeError(
                "Missing prepared shares for priced membership constituents on "
                f"{date.date()}: {missing_share_assets}. "
                "Run `python -m backtest.prepare brinson`; inspect any unresolved "
                "reviewed-shares diagnostics and correct the required inputs."
            )
        if not candidate_asset_ids:
            raise RuntimeError(
                f"No priced membership constituents exist on {date.date()}."
            )
        sector_rows = sector_rows_asof(
            backtest_data.sector_assignments,
            date,
        )
        missing_sector_assets = sorted(
            set(candidate_asset_ids) - set(sector_rows.index.astype(str))
        )
        if missing_sector_assets:
            raise RuntimeError(
                "Missing period-start sector assignments for membership "
                f"constituents on {date.date()}: {missing_sector_assets}"
            )

        price_t0 = data_close.reindex(
            index=[date], columns=candidate_asset_ids
        ).iloc[0]
        price_t1 = data_close.reindex(
            index=[next_date], columns=candidate_asset_ids
        ).iloc[0]
        shares_t0 = pd.to_numeric(
            shares_monthly.loc[date].reindex(candidate_asset_ids),
            errors="coerce",
        ).astype("float64")
        missing_boundary_shares = sorted(
            shares_t0.index[
                shares_t0.isna()
                | ~np.isfinite(shares_t0)
                | shares_t0.le(0.0)
            ].astype(str)
        )
        if missing_boundary_shares:
            raise RuntimeError(
                "Missing or invalid prepared shares at Brinson "
                f"boundary {date.date()}: {missing_boundary_shares}. "
                "Run `python -m backtest.prepare brinson`; inspect any unresolved "
                "reviewed-shares diagnostics and correct the required inputs."
            )

        end_values = pd.Series(
            {
                asset_id: _brinson_end_value_per_start_share(
                    backtest_data,
                    asset_id,
                    date,
                    next_date,
                    price_t0.get(asset_id),
                    price_t1.get(asset_id),
                )
                for asset_id in candidate_asset_ids
            },
            dtype="float64",
        )
        invalid_price_assets = sorted(
            asset_id
            for asset_id in candidate_asset_ids
            if pd.isna(price_t0.get(asset_id))
            or pd.isna(end_values.get(asset_id))
            or not np.isfinite(float(price_t0.get(asset_id)))
            or float(price_t0.get(asset_id)) <= 0.0
        )
        if invalid_price_assets:
            raise RuntimeError(
                "Missing price for active priced membership constituents during "
                f"{date.date()} -> {next_date.date()}: {invalid_price_assets}"
            )
        unknown_sector_assets = sorted(
            asset_id for asset_id in candidate_asset_ids
            if str(sector_rows.at[asset_id, "GICS_Sector_Code"]) in {"", "Unknown"}
            or str(sector_rows.at[asset_id, "Sector"]) in {"", "Unknown"}
        )
        if unknown_sector_assets:
            raise RuntimeError(
                f"Unknown sector for membership constituents: {unknown_sector_assets}"
            )
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
                "Start_Price": float(price_t0.at[asset_id]),
                "End_Value_Per_Start_Share": float(end_values.at[asset_id]),
                "Shares_Outstanding": float(shares_t0.at[asset_id]),
            }
            for asset_id in candidate_asset_ids
        )
        audit_context.append({
            "Period_ID": period_id,
            "PiT_Active_Count": int(len(active_asset_ids)),
            "Priced_Constituent_Count": int(len(candidate_asset_ids)),
            "Extinguished_Predecessor_Count": int(
                len(extinguished_predecessors)
            ),
            "Extinguished_Predecessors": ";".join(extinguished_predecessors),
            "Unavailable_Member_Count": int(len(unavailable_asset_ids)),
            "Unavailable_Members": ";".join(unavailable_asset_ids),
            "Dropped_Constituent_Count": 0,
            "Missing_Price_Count": 0,
            "Missing_Shares_Count": 0,
            "Unknown_Sector_Count": 0,
        })

    constituents = pd.DataFrame(
        constituent_rows,
        columns=BENCHMARK_CONSTITUENT_COLUMNS,
    )
    capitalization_factors = pd.Series([
        factors.loc[(row.Start_Date, row.Asset_ID)] for row in constituents.itertuples()
    ], index=constituents.index, dtype=float)
    sector, shared_audit = build_brinson_benchmark(
        constituents, capitalization_factors=capitalization_factors,
    )
    audit = shared_audit.merge(
        pd.DataFrame(audit_context),
        on="Period_ID",
        how="left",
        validate="one_to_one",
    )
    sector = sector.drop(columns="Period_ID")
    audit = audit.drop(columns="Period_ID")
    audit = audit[[
        "Date", "Next_Date", "PiT_Active_Count", "Priced_Constituent_Count",
        "Extinguished_Predecessor_Count", "Extinguished_Predecessors",
        "Unavailable_Member_Count", "Unavailable_Members",
        "Valid_Constituent_Count", "Dropped_Constituent_Count",
        "Missing_Price_Count", "Missing_Shares_Count", "Unknown_Sector_Count",
        "Benchmark_Market_Cap", "Sector_Weight_Sum",
        "Reconstructed_Benchmark_Return",
    ]]
    return sector, audit



def load_sp500tr_monthly_returns(paths: BacktestPaths) -> pd.Series:
    """Load monthly S&P 500 total returns for benchmark consistency checks."""
    from .data_loading import load_prepared_benchmark

    try:
        benchmark = load_prepared_benchmark(paths.benchmark_csv)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing S&P 500 TR file: {paths.benchmark_csv}"
        ) from exc
    returns = benchmark.pct_change(fill_method=None)
    returns.name = "SP500TR_Return"
    return returns


def add_benchmark_consistency_columns(
    benchmark_audit_df: pd.DataFrame,
    paths: BacktestPaths,
) -> pd.DataFrame:
    """Add comparison against the S&P 500 total-return series."""
    sp500tr_returns = load_sp500tr_monthly_returns(paths)
    audit_df = benchmark_audit_df.copy()
    audit_df["SP500TR_Return"] = audit_df["Next_Date"].map(sp500tr_returns)
    return _derive_benchmark_consistency_columns(audit_df)


def run_brinson_attribution(
    backtest_data: BacktestDataset,
    holdings_df: pd.DataFrame | None = None,
    *,
    shares_monthly: pd.DataFrame,
    paths: BacktestPaths,
) -> BrinsonResult:
    """Calculate benchmark reconstruction and attribution without writing."""
    if holdings_df is None:
        holdings_df = load_gross_brinson_holdings(paths)
    else:
        holdings_df = holdings_df.copy()
        holdings_df["Date"] = pd.to_datetime(holdings_df["Date"])
        holdings_df["Next_Date"] = pd.to_datetime(holdings_df["Next_Date"])
    holdings_df = validate_holding_sector_provenance(
        holdings_df,
        backtest_data.sector_assignments,
        context="Gross Brinson",
    )

    periods = get_brinson_periods_from_holdings(holdings_df)
    benchmark_sector_df, benchmark_audit_df = build_brinson_benchmark_sector_series(
        backtest_data,
        shares_monthly,
        periods,
    )
    benchmark_audit_df = add_benchmark_consistency_columns(
        benchmark_audit_df,
        paths,
    )
    return run_brinson_pipeline(
        benchmark_sector_df,
        benchmark_audit_df,
        holdings_df,
    )


def save_brinson_result(
    result: BrinsonResult,
    paths: BacktestPaths,
) -> None:
    """Persist a completed backtest Brinson result and its plots."""
    if not isinstance(result, BrinsonResult):
        raise TypeError("Backtest Brinson result must be a BrinsonResult")
    benchmark_sector_df = result.benchmark_sector
    benchmark_audit_df = result.benchmark_audit
    sector_attribution_df = result.sector_attribution
    monthly_attribution_df = result.period_attribution
    period_attribution_df = result.total_attribution
    benchmark_plot = prepare_benchmark_consistency_plot_frame(benchmark_audit_df)
    combined = prepare_active_decomposition_plot_frame(monthly_attribution_df)
    if not combined[["Date", "Next_Date"]].equals(benchmark_plot[["Date", "Next_Date"]]):
        raise ValueError("Backtest benchmark and attribution must cover the same periods")
    window_start = benchmark_plot["Date"].iloc[0]
    window_end = benchmark_plot["Next_Date"].iloc[-1]
    attribution = paths.attribution

    for frame, path in (
        (benchmark_sector_df, attribution.benchmark_sector_csv),
        (benchmark_audit_df, attribution.benchmark_audit_csv),
        (sector_attribution_df, attribution.sector_attribution_csv),
        (monthly_attribution_df, attribution.monthly_attribution_csv),
        (period_attribution_df, attribution.period_attribution_csv),
    ):
        atomic_write_dataframe(
            frame,
            path,
            index=False,
            date_format=None,
            float_format=None,
        )
        print(f"Saved Brinson table to {path}")

    render_benchmark_consistency(
        benchmark_plot,
        attribution.benchmark_consistency_png,
        reconstructed_label="Reconstructed S&P 500 sector benchmark",
        title="Benchmark Consistency Check",
        xlabel="Month",
        figsize=LINE_FIGSIZE,
        atomic=False,
        window_start=window_start, window_end=window_end, date_frequency="monthly",
    )
    render_signed_decomposition(
        combined,
        attribution.active_decomposition_monthly_png,
        attribution.active_decomposition_cumulative_png,
        period_title="Monthly Brinson-Fachler Active Return Decomposition",
        cumulative_title="Cumulative Brinson-Fachler Active Return Decomposition",
        xlabel="Month",
        period_figsize=PERIOD_FIGSIZE,
        cumulative_figsize=LINE_FIGSIZE,
        label_rotation=45,
        marker_size=3,
        atomic=False,
        window_start=window_start, window_end=window_end, date_frequency="monthly",
    )
    for path in (
        attribution.benchmark_consistency_png,
        attribution.active_decomposition_monthly_png,
        attribution.active_decomposition_cumulative_png,
    ):
        print(f"Saved Brinson plot to {path}")

    corr = benchmark_plot["Benchmark_Return_Correlation"].iloc[0]
    mean_abs_diff = benchmark_plot["Mean_Absolute_Return_Diff"].iloc[0]
    print(
        "Benchmark consistency: "
        f"corr={corr:.4f}, mean_abs_diff={mean_abs_diff:.4%}. "
        "Differences are expected because reconstructed constituent returns "
        "are price-only while SP500TR includes dividends."
    )


def remove_saved_brinson_result(paths: BacktestPaths) -> None:
    """Remove generated attribution for one successfully replaced strategy."""
    for path in paths.attribution.result_files:
        path.unlink(missing_ok=True)
