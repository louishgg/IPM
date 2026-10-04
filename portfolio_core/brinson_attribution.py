"""Pure Brinson attribution calculations shared by backtest and live."""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum

import numpy as np
import pandas as pd


ATTRIBUTION_METHOD = "Brinson-Fachler"
STOCK_EFFECT_COLUMNS = ("Allocation_Effect", "Selection_Effect", "Interaction_Effect")
_RETURN_COLUMNS = (
    "Portfolio_Return", "Benchmark_Return", "Active_Return",
    *STOCK_EFFECT_COLUMNS, "Total_Effect", "Residual",
)


BENCHMARK_CONSTITUENT_COLUMNS = (
    "Period_ID",
    "Start_Date",
    "End_Date",
    "Asset_ID",
    "GICS_Sector_Code",
    "Sector",
    "Start_Price",
    "End_Value_Per_Start_Share",
    "Shares_Outstanding",
)


@dataclass(frozen=True, slots=True)
class BrinsonResult:
    """All common CSV-ready benchmark and attribution tables."""

    benchmark_sector: pd.DataFrame
    benchmark_audit: pd.DataFrame
    portfolio_sector: pd.DataFrame
    sector_attribution: pd.DataFrame
    period_attribution: pd.DataFrame
    total_attribution: pd.DataFrame


def build_brinson_benchmark(
    constituents: pd.DataFrame,
    *,
    capitalization_factors: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sector returns/weights and reconciliation audits for each period.

    Weights use start price * shares * optional capitalization factor, normalized
    across constituents. Factors affect weights only, never stock returns.
    """
    if tuple(constituents.columns) != BENCHMARK_CONSTITUENT_COLUMNS:
        raise ValueError(
            "Benchmark constituents must contain exactly "
            f"{list(BENCHMARK_CONSTITUENT_COLUMNS)}"
        )
    frame = constituents.copy()
    if frame.empty:
        raise RuntimeError("No benchmark constituents were supplied")
    for column in ("Period_ID", "Asset_ID", "GICS_Sector_Code", "Sector"):
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Benchmark constituents contain blank {column}")
    unknown_mask = frame["GICS_Sector_Code"].eq("Unknown") | frame[
        "Sector"
    ].eq("Unknown")
    if unknown_mask.any():
        unknown = sorted(frame.loc[unknown_mask, "Asset_ID"])
        raise ValueError(f"Benchmark constituents have unknown sectors: {unknown}")
    frame["Start_Date"] = pd.to_datetime(frame["Start_Date"], errors="raise")
    frame["End_Date"] = pd.to_datetime(frame["End_Date"], errors="raise")
    if frame["Start_Date"].dt.tz is not None or frame["End_Date"].dt.tz is not None:
        raise ValueError("Benchmark period dates must be timezone-naive")
    if frame["End_Date"].le(frame["Start_Date"]).any():
        raise ValueError("Benchmark end dates must follow start dates")
    for column in (
        "Start_Price",
        "End_Value_Per_Start_Share",
        "Shares_Outstanding",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    invalid = (
        ~np.isfinite(frame["Start_Price"])
        | frame["Start_Price"].le(0.0)
        | ~np.isfinite(frame["End_Value_Per_Start_Share"])
        | frame["End_Value_Per_Start_Share"].lt(0.0)
        | ~np.isfinite(frame["Shares_Outstanding"])
        | frame["Shares_Outstanding"].le(0.0)
    )
    if invalid.any():
        raise ValueError("Benchmark constituents contain invalid prices or shares")
    identity = ["Period_ID", "Asset_ID"]
    if frame.duplicated(identity).any():
        raise ValueError("Benchmark constituents duplicate a period Asset_ID")
    inconsistent = frame.groupby("Period_ID", sort=False)[
        ["Start_Date", "End_Date"]
    ].nunique()
    if inconsistent.gt(1).any().any():
        raise ValueError("Benchmark period IDs map to inconsistent dates")

    factors = pd.Series(1.0, index=frame.index)
    if capitalization_factors is not None:
        if not capitalization_factors.index.equals(frame.index):
            raise ValueError("Capitalization factors must exactly align with constituents")
        factors = pd.to_numeric(capitalization_factors, errors="raise")
        if (~np.isfinite(factors) | factors.le(0)).any():
            raise ValueError("Capitalization factors must be finite and positive")
    frame["Market_Cap"] = frame["Start_Price"] * frame["Shares_Outstanding"] * factors
    frame["Stock_Return"] = (
        frame["End_Value_Per_Start_Share"] / frame["Start_Price"] - 1.0
    )
    sector_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for period_id, group in frame.groupby("Period_ID", sort=False):
        group = group.copy()
        total_cap = float(group["Market_Cap"].sum())
        if not np.isfinite(total_cap) or total_cap <= 0.0:
            raise RuntimeError(f"Invalid benchmark market cap for {period_id}")
        group["Benchmark_Stock_Weight"] = group["Market_Cap"] / total_cap
        reconstructed = 0.0
        sector_weights: list[float] = []
        start = pd.Timestamp(group["Start_Date"].iloc[0])
        end = pd.Timestamp(group["End_Date"].iloc[0])
        for sector_code, sector_group in group.groupby(
            "GICS_Sector_Code", sort=True
        ):
            labels = sorted(set(sector_group["Sector"].astype(str)))
            if len(labels) != 1:
                raise ValueError(
                    "Benchmark constituents map one GICS sector code to "
                    f"multiple labels in {period_id}: {sector_code}={labels}"
                )
            sector = labels[0]
            weight = float(sector_group["Benchmark_Stock_Weight"].sum())
            sector_weights.append(weight)
            sector_return = safe_weighted_average(
                sector_group["Stock_Return"],
                sector_group["Benchmark_Stock_Weight"],
            )
            contribution = weight * sector_return
            reconstructed += contribution
            sector_rows.append({
                "Period_ID": str(period_id),
                "Date": start,
                "Next_Date": end,
                "GICS_Sector_Code": str(sector_code),
                "Sector": str(sector),
                "Benchmark_Weight": weight,
                "Benchmark_Return": sector_return,
                "Benchmark_Contribution": contribution,
                "Constituent_Count": int(len(sector_group)),
                "Sector_Market_Cap": float(sector_group["Market_Cap"].sum()),
            })
        audit_rows.append({
            "Period_ID": str(period_id),
            "Date": start,
            "Next_Date": end,
            "Valid_Constituent_Count": int(len(group)),
            "Benchmark_Market_Cap": total_cap,
            "Sector_Weight_Sum": float(fsum(sector_weights)),
            "Reconstructed_Benchmark_Return": float(reconstructed),
        })
    return (
        pd.DataFrame(sector_rows).sort_values(
            ["Date", "GICS_Sector_Code"], kind="stable"
        ).reset_index(drop=True),
        pd.DataFrame(audit_rows).sort_values("Date", kind="stable").reset_index(
            drop=True
        ),
    )


def add_benchmark_consistency_columns(
    benchmark_audit_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return benchmark audit data with derived consistency diagnostics."""
    audit_df = benchmark_audit_df.copy()
    required = {"Reconstructed_Benchmark_Return", "SP500TR_Return"}
    missing = sorted(required - set(audit_df.columns))
    if missing:
        raise ValueError(
            f"Benchmark consistency data is missing columns: {missing}"
        )
    audit_df["Return_Diff"] = (
        audit_df["Reconstructed_Benchmark_Return"] - audit_df["SP500TR_Return"]
    )
    audit_df["Abs_Return_Diff"] = audit_df["Return_Diff"].abs()

    valid = audit_df[
        ["Reconstructed_Benchmark_Return", "SP500TR_Return"]
    ].dropna()
    correlation = float(valid.corr().iloc[0, 1]) if len(valid) > 1 else np.nan
    audit_df["Benchmark_Return_Correlation"] = correlation
    audit_df["Mean_Absolute_Return_Diff"] = float(
        audit_df["Abs_Return_Diff"].mean()
    )
    audit_df["Max_Absolute_Return_Diff"] = float(
        audit_df["Abs_Return_Diff"].max()
    )
    audit_df["Cumulative_Reconstructed_Benchmark"] = (
        (1.0 + audit_df["Reconstructed_Benchmark_Return"]).cumprod() - 1.0
    )
    audit_df["Cumulative_SP500TR"] = (
        (1.0 + audit_df["SP500TR_Return"].fillna(0.0)).cumprod() - 1.0
    )
    return audit_df


def safe_weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Return a finite weighted average, or NaN when inputs are unusable."""
    values = pd.Series(values, dtype="float64")
    weights = pd.Series(weights, dtype="float64")
    mask = (
        values.notna()
        & weights.notna()
        & np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not mask.any():
        return np.nan

    clean_weights = weights[mask]
    weight_sum = float(clean_weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return np.nan

    return float(np.average(values[mask], weights=clean_weights))


def build_brinson_portfolio_sector_series(
    holdings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize absolute NAV weights separately within the long and short books.

    Sector returns retain the stock-return sign; Effective_Portfolio_Return
    negates it for shorts. Side_Gross_Exposure retains each book's positive gross.
    """
    required = {
        "Date",
        "Next_Date",
        "GICS_Sector_Code",
        "Sector",
        "Weight",
        "Stock_Return",
    }
    missing = sorted(required - set(holdings_df.columns))
    if missing:
        raise ValueError(f"Portfolio holdings are missing columns: {missing}")
    holdings_df = holdings_df.copy()
    holdings_df["Weight"] = pd.to_numeric(holdings_df["Weight"], errors="raise")
    if not np.isfinite(holdings_df["Weight"].to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError("Portfolio holdings require finite weights")
    # Zero positions do not create a sector or require a realized stock return.
    holdings_df = holdings_df.loc[holdings_df["Weight"].ne(0.0)].copy()
    holdings_df["Stock_Return"] = pd.to_numeric(
        holdings_df["Stock_Return"], errors="raise"
    )
    if not np.isfinite(holdings_df["Stock_Return"].to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError("Held positions require finite stock returns")
    for column in ("Date", "Next_Date"):
        holdings_df[column] = pd.to_datetime(holdings_df[column], errors="raise")
        if holdings_df[column].isna().any():
            raise ValueError(f"Portfolio holdings contain missing {column}")
    for column in ("GICS_Sector_Code", "Sector"):
        if holdings_df[column].isna().any():
            raise ValueError("Portfolio holdings contain unknown sectors")
        holdings_df[column] = holdings_df[column].astype(str).str.strip()
    unknown = holdings_df["GICS_Sector_Code"].isin(["", "Unknown"]) | (
        holdings_df["Sector"].isin(["", "Unknown"])
    )
    if unknown.any():
        raise ValueError("Portfolio holdings contain unknown sectors")
    rows = []

    for (date, next_date), period_df in holdings_df.groupby(["Date", "Next_Date"]):
        side_specs = [
            ("Long", period_df[period_df["Weight"] > 0.0]),
            ("Short", period_df[period_df["Weight"] < 0.0]),
        ]

        for side, side_df in side_specs:
            if side_df.empty:
                continue

            side_df = side_df.copy()
            side_df["Abs_Weight"] = side_df["Weight"].abs()
            side_gross = float(side_df["Abs_Weight"].sum())
            if not np.isfinite(side_gross) or side_gross <= 0.0:
                raise ValueError("Portfolio side gross exposure must be finite and positive")

            side_df["Portfolio_Stock_Weight"] = (
                side_df["Abs_Weight"] / side_gross
            )

            for sector_code, sector_df in side_df.groupby(
                "GICS_Sector_Code", sort=True
            ):
                labels = sorted(set(sector_df["Sector"].astype(str)))
                if len(labels) != 1:
                    raise ValueError(
                        "Portfolio holdings map one GICS sector code to "
                        f"multiple labels on {pd.Timestamp(date).date()}: "
                        f"{sector_code}={labels}"
                    )
                sector = labels[0]
                sector_weight = float(sector_df["Portfolio_Stock_Weight"].sum())
                sector_return = safe_weighted_average(
                    sector_df["Stock_Return"],
                    sector_df["Portfolio_Stock_Weight"],
                )
                effective_return = sector_return if side == "Long" else -sector_return

                rows.append({
                    "Date": pd.Timestamp(date),
                    "Next_Date": pd.Timestamp(next_date),
                    "Side": side,
                    "GICS_Sector_Code": str(sector_code),
                    "Sector": sector,
                    "Portfolio_Weight": sector_weight,
                    "Portfolio_Return": sector_return,
                    "Effective_Portfolio_Return": effective_return,
                    "Holding_Count": int(len(sector_df)),
                    "Side_Gross_Exposure": side_gross,
                    "Portfolio_Weight_Sum": float(
                        side_df["Portfolio_Stock_Weight"].sum()
                    ),
                })

    if not rows:
        raise RuntimeError("No long or short sector books could be built.")

    return pd.DataFrame(rows).sort_values(
        ["Date", "Side", "GICS_Sector_Code"]
    ).reset_index(drop=True)


def build_brinson_sector_attribution(
    benchmark_sector_df: pd.DataFrame,
    portfolio_sector_df: pd.DataFrame,
) -> pd.DataFrame:
    """Separate BF allocation, pure selection and interaction for each book.

    Allocation centers side-effective sector returns on the reconstructed
    benchmark total. Raw active contribution equals the three effects plus
    Benchmark_Centering_Adjustment, which cancels across normalized sectors.
    Unheld sectors keep NaN observed returns and zero selection/interaction.
    Short returns are negated; Scaled_* applies positive side gross once.
    """
    for frame, context, numeric in (
        (benchmark_sector_df, "Benchmark", ("Benchmark_Weight", "Benchmark_Return")),
        (portfolio_sector_df, "Portfolio", ("Portfolio_Weight", "Portfolio_Return", "Side_Gross_Exposure")),
    ):
        if frame[["Date", "Next_Date", "GICS_Sector_Code", "Sector"]].isna().any().any():
            raise ValueError(f"{context} sector series contains missing period or sector data")
        if not np.isfinite(frame[list(numeric)].to_numpy(dtype=float)).all():
            raise ValueError(f"{context} sector series requires finite weights and returns")
    if (benchmark_sector_df["Benchmark_Weight"] < 0.0).any():
        raise ValueError("Benchmark sector weights must be nonnegative")
    if (portfolio_sector_df["Portfolio_Weight"] <= 0.0).any():
        raise ValueError("Portfolio sector series must contain only held sectors with positive weights")
    rows = []
    benchmark_groups = {
        (pd.Timestamp(date), pd.Timestamp(next_date)): group.copy()
        for (date, next_date), group in benchmark_sector_df.groupby(
            ["Date", "Next_Date"]
        )
    }

    for (date, next_date, side), port_group in portfolio_sector_df.groupby(
        ["Date", "Next_Date", "Side"]
    ):
        period_key = (pd.Timestamp(date), pd.Timestamp(next_date))
        if period_key not in benchmark_groups:
            raise RuntimeError(
                f"Missing benchmark sector series for {date.date()} -> "
                f"{next_date.date()}."
            )

        bench_group = benchmark_groups[period_key]
        port_by_sector = port_group.set_index("GICS_Sector_Code")
        bench_by_sector = bench_group.set_index("GICS_Sector_Code")
        if port_by_sector.index.has_duplicates:
            raise ValueError(
                "Portfolio sector series duplicates a GICS code for "
                f"{pd.Timestamp(date).date()} {side}"
            )
        if bench_by_sector.index.has_duplicates:
            raise ValueError(
                "Benchmark sector series duplicates a GICS code for "
                f"{pd.Timestamp(date).date()}"
            )
        missing_benchmark = sorted(set(port_by_sector.index) - set(bench_by_sector.index))
        if missing_benchmark:
            raise ValueError(f"Held sectors are absent from the benchmark: {missing_benchmark}")
        if side not in {"Long", "Short"}:
            raise ValueError(f"Unsupported Brinson side: {side}")
        sign = 1.0 if side == "Long" else -1.0
        benchmark_total = sign * float(fsum(
            bench_group["Benchmark_Weight"] * bench_group["Benchmark_Return"]
        ))
        side_gross = float(port_group["Side_Gross_Exposure"].iloc[0])
        if side_gross <= 0.0 or not port_group["Side_Gross_Exposure"].eq(side_gross).all():
            raise ValueError("Portfolio side gross exposure must be positive and constant within a book")

        for sector_code in sorted(bench_by_sector.index):
            missing_portfolio = sector_code not in port_by_sector.index
            portfolio_label = (
                str(port_by_sector.loc[sector_code, "Sector"])
                if not missing_portfolio
                else ""
            )
            benchmark_label = str(bench_by_sector.loc[sector_code, "Sector"])
            if (
                portfolio_label
                and benchmark_label
                and portfolio_label != benchmark_label
            ):
                raise ValueError(
                    "Portfolio and benchmark sector labels disagree for code "
                    f"{sector_code} on {pd.Timestamp(date).date()}: "
                    f"{portfolio_label!r} != {benchmark_label!r}"
                )
            sector = benchmark_label or portfolio_label
            portfolio_weight = (
                float(port_by_sector.loc[sector_code, "Portfolio_Weight"])
                if not missing_portfolio
                else 0.0
            )
            benchmark_weight = (
                float(bench_by_sector.loc[sector_code, "Benchmark_Weight"])
            )
            portfolio_return = (
                float(port_by_sector.loc[sector_code, "Portfolio_Return"])
                if not missing_portfolio
                else np.nan
            )
            benchmark_return = (
                float(bench_by_sector.loc[sector_code, "Benchmark_Return"])
            )

            effective_portfolio_return = sign * portfolio_return
            effective_benchmark_return = sign * benchmark_return
            weight_difference = portfolio_weight - benchmark_weight
            allocation_effect = (
                weight_difference * (effective_benchmark_return - benchmark_total)
            )
            # The benchmark return is the attribution-only surrogate for an
            # unheld sector; it is never recorded as a realized portfolio return.
            return_difference = (
                0.0 if missing_portfolio
                else effective_portfolio_return - effective_benchmark_return
            )
            selection_effect = benchmark_weight * return_difference
            interaction_effect = weight_difference * return_difference
            effects = dict(zip(STOCK_EFFECT_COLUMNS, (
                allocation_effect, selection_effect, interaction_effect,
            )))
            centering_adjustment = weight_difference * benchmark_total
            total_effect = allocation_effect + selection_effect + interaction_effect
            portfolio_contribution = (
                0.0 if missing_portfolio else portfolio_weight * effective_portfolio_return
            )
            active_contribution = (
                portfolio_contribution - benchmark_weight * effective_benchmark_return
            )
            residual = active_contribution - centering_adjustment - total_effect

            rows.append({
                "Attribution_Method": ATTRIBUTION_METHOD,
                "Date": pd.Timestamp(date),
                "Next_Date": pd.Timestamp(next_date),
                "Side": side,
                "GICS_Sector_Code": str(sector_code),
                "Sector": sector,
                "Portfolio_Weight": portfolio_weight,
                "Benchmark_Weight": benchmark_weight,
                "Portfolio_Return": portfolio_return,
                "Benchmark_Return": benchmark_return,
                "Effective_Portfolio_Return": effective_portfolio_return,
                "Effective_Benchmark_Return": effective_benchmark_return,
                "Effective_Benchmark_Total_Return": benchmark_total,
                **effects,
                "Benchmark_Centering_Adjustment": centering_adjustment,
                "Total_Effect": total_effect,
                "Active_Contribution": active_contribution,
                "Residual": residual,
                "Side_Gross_Exposure": side_gross,
                **{f"Scaled_{column}": side_gross * value for column, value in effects.items()},
                "Scaled_Benchmark_Centering_Adjustment": side_gross * centering_adjustment,
                "Scaled_Total_Effect": side_gross * total_effect,
                "Scaled_Active_Contribution": side_gross * active_contribution,
                "Scaled_Residual": side_gross * residual,
                "Missing_Portfolio_Sector": missing_portfolio,
                "Missing_Benchmark_Sector": False,
            })

    if not rows:
        raise RuntimeError("No Brinson attribution rows were produced.")

    return pd.DataFrame(rows).sort_values(
        ["Date", "Side", "GICS_Sector_Code"]
    ).reset_index(drop=True)


def summarize_brinson_monthly_attribution(
    sector_attribution_df: pd.DataFrame,
) -> pd.DataFrame:
    """Sum sector effects by month and side, then combine gross-scaled books.

    Combined rows use Scaled_* fields only; unscaled fields are undefined (NaN).
    """
    side_rows = []

    for (date, next_date, side), group in sector_attribution_df.groupby(
        ["Date", "Next_Date", "Side"]
    ):
        portfolio_return = float(
            (group.loc[~group["Missing_Portfolio_Sector"], "Portfolio_Weight"]
             * group.loc[~group["Missing_Portfolio_Sector"], "Effective_Portfolio_Return"]).sum()
        )
        benchmark_return = float(
            (group["Benchmark_Weight"]
             * group["Effective_Benchmark_Return"]).sum()
        )
        active_return = portfolio_return - benchmark_return
        effects = {column: float(group[column].sum()) for column in STOCK_EFFECT_COLUMNS}
        total_effect = sum(effects.values())
        residual = active_return - total_effect
        side_gross = float(group["Side_Gross_Exposure"].iloc[0])

        side_rows.append({
            "Attribution_Method": ATTRIBUTION_METHOD,
            "Date": pd.Timestamp(date),
            "Next_Date": pd.Timestamp(next_date),
            "Side": side,
            "Side_Gross_Exposure": side_gross,
            "Portfolio_Return": portfolio_return,
            "Benchmark_Return": benchmark_return,
            "Active_Return": active_return,
            **effects,
            "Total_Effect": total_effect,
            "Residual": residual,
            "Scaled_Portfolio_Return": side_gross * portfolio_return,
            "Scaled_Benchmark_Return": side_gross * benchmark_return,
            "Scaled_Active_Return": side_gross * active_return,
            **{f"Scaled_{column}": side_gross * value for column, value in effects.items()},
            "Scaled_Total_Effect": side_gross * total_effect,
            "Scaled_Residual": side_gross * residual,
        })

    monthly_df = pd.DataFrame(side_rows).sort_values(
        ["Date", "Side"]
    ).reset_index(drop=True)

    combined_rows = []
    for (date, next_date), group in monthly_df.groupby(["Date", "Next_Date"]):
        scaled_active = float(group["Scaled_Active_Return"].sum())
        scaled_effects = {
            f"Scaled_{column}": float(group[f"Scaled_{column}"].sum())
            for column in STOCK_EFFECT_COLUMNS
        }
        scaled_total = sum(scaled_effects.values())
        scaled_residual = scaled_active - scaled_total

        combined_rows.append({
            "Attribution_Method": ATTRIBUTION_METHOD,
            "Date": pd.Timestamp(date),
            "Next_Date": pd.Timestamp(next_date),
            "Side": "Combined",
            "Side_Gross_Exposure": float(group["Side_Gross_Exposure"].sum()),
            **dict.fromkeys(_RETURN_COLUMNS, np.nan),
            "Scaled_Portfolio_Return": float(
                group["Scaled_Portfolio_Return"].sum()
            ),
            "Scaled_Benchmark_Return": float(
                group["Scaled_Benchmark_Return"].sum()
            ),
            "Scaled_Active_Return": scaled_active,
            **scaled_effects,
            "Scaled_Total_Effect": scaled_total,
            "Scaled_Residual": scaled_residual,
        })

    return pd.concat(
        [monthly_df, pd.DataFrame(combined_rows)],
        ignore_index=True,
    ).sort_values(["Date", "Side"]).reset_index(drop=True)


def summarize_brinson_period_attribution(
    monthly_attribution_df: pd.DataFrame,
) -> pd.DataFrame:
    """Sum monthly effects arithmetically, without compounding or linking.

    Total_* uses book-normalized effects for each side and gross-scaled effects
    for Combined; Total_Scaled_* always uses gross-scaled effects.
    """
    rows = []

    for side, group in monthly_attribution_df.groupby("Side", sort=True):
        prefix = "Scaled_" if side == "Combined" else ""
        total_active = float(group[f"{prefix}Active_Return"].sum())
        effects = {
            column: float(group[f"{prefix}{column}"].sum())
            for column in STOCK_EFFECT_COLUMNS
        }

        rows.append({
            "Attribution_Method": ATTRIBUTION_METHOD,
            "Side": side,
            "Months": int(len(group)),
            "Mean_Side_Gross_Exposure": float(
                group["Side_Gross_Exposure"].mean()
            ),
            "Total_Active_Return": total_active,
            **{f"Total_{column}": value for column, value in effects.items()},
            "Total_Effect": float(group[f"{prefix}Total_Effect"].sum()),
            "Total_Residual": float(group[f"{prefix}Residual"].sum()),
            **{
                column.replace("_Effect", "_Share_Of_Active"): (
                    value / total_active if abs(total_active) > 1e-12 else np.nan
                )
                for column, value in effects.items()
            },
            "Total_Scaled_Active_Return": float(
                group["Scaled_Active_Return"].sum()
            ),
            **{
                f"Total_Scaled_{column}": float(group[f"Scaled_{column}"].sum())
                for column in STOCK_EFFECT_COLUMNS
            },
            "Total_Scaled_Effect": float(
                group["Scaled_Total_Effect"].sum()
            ),
            "Total_Scaled_Residual": float(
                group["Scaled_Residual"].sum()
            ),
        })

    side_order = {"Long": 0, "Short": 1, "Combined": 2}
    return pd.DataFrame(rows).sort_values(
        by="Side",
        key=lambda col: col.map(side_order).fillna(99),
    ).reset_index(drop=True)


def validate_brinson_outputs(
    benchmark_audit_df: pd.DataFrame,
    portfolio_sector_df: pd.DataFrame,
    monthly_attribution_df: pd.DataFrame,
    *,
    sector_attribution_df: pd.DataFrame,
) -> None:
    """Reconcile sector, side and combined tables against their source identities."""
    def indexed(frame, keys, context):
        if frame.empty:
            raise RuntimeError(f"Brinson {context} requires nonempty accounting tables")
        invalid = frame[keys].isna().any(axis=1) | frame.duplicated(keys, keep=False)
        if invalid.any():
            identity = frame.loc[invalid, keys].iloc[0].to_dict()
            raise RuntimeError(f"Brinson {context} has a missing or duplicate identity: {identity}")
        return frame.set_index(keys).sort_index()

    def fail_at(mask, values, context):
        if np.any(mask):
            position = np.argwhere(mask)[0]
            identity = values.index[position[0]]
            field = values.columns[position[1]] if isinstance(values, pd.DataFrame) else values.name
            raise RuntimeError(f"Brinson {context}: {field} at {identity}")

    def require_finite(values, context):
        fail_at(~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)), values, context)

    def check_close(actual, expected, context, tolerance=1e-10):
        mismatch = ~np.isclose(actual, expected, atol=tolerance, rtol=0.0)
        if isinstance(actual, (pd.Series, pd.DataFrame)):
            fail_at(mismatch, actual, f"{context} does not reconcile")
        elif np.any(mismatch):
            raise RuntimeError(f"Brinson {context} does not reconcile")

    def same_keys(actual, expected, context):
        missing = expected.difference(actual)
        unexpected = actual.difference(expected)
        if len(missing) or len(unexpected):
            raise RuntimeError(
                f"Brinson {context} identities disagree: "
                f"missing={missing.tolist()[:3]}, unexpected={unexpected.tolist()[:3]}"
            )

    period_keys = ["Date", "Next_Date"]
    book_keys = [*period_keys, "Side"]
    sector_keys = [*book_keys, "GICS_Sector_Code"]
    audit = indexed(benchmark_audit_df, period_keys, "benchmark audit")
    portfolio = indexed(portfolio_sector_df, sector_keys, "portfolio sectors")
    sector = indexed(sector_attribution_df, sector_keys, "attribution sectors")
    monthly = indexed(monthly_attribution_df, book_keys, "interval attribution")
    require_finite(audit[["Sector_Weight_Sum", "Reconstructed_Benchmark_Return"]], "finite benchmark data required")
    check_close(audit["Sector_Weight_Sum"], 1.0, "benchmark sector weights", tolerance=1e-8)
    for frame in (sector, monthly):
        fail_at(~frame["Attribution_Method"].eq(ATTRIBUTION_METHOD), frame["Attribution_Method"],
                "attribution method is inconsistent")
    absent = sector["Missing_Portfolio_Sector"]
    observed_returns = ["Portfolio_Return", "Effective_Portfolio_Return"]
    fail_at(sector.loc[absent, observed_returns].notna(), sector.loc[absent, observed_returns],
            "absent portfolio sectors must retain undefined observed returns")
    require_finite(sector.loc[~absent, observed_returns], "held sectors require finite returns")
    numeric = sector.select_dtypes(include="number").drop(columns=observed_returns)
    require_finite(numeric, "sector effects require finite values")
    fail_at(sector["Missing_Benchmark_Sector"], sector["Missing_Benchmark_Sector"],
            "attribution requires benchmark sector coverage")
    check_close(sector.loc[absent, ["Portfolio_Weight", *STOCK_EFFECT_COLUMNS[1:]]],
                0.0, "absent-sector convention", tolerance=0.0)

    held = sector.loc[~absent]
    same_keys(held.index, portfolio.index, "held sectors")
    for column in ("Portfolio_Weight", "Portfolio_Return", "Side_Gross_Exposure"):
        check_close(held[column], portfolio[column].reindex(held.index), f"held {column}")
    side_sign = pd.Series(sector.index.get_level_values("Side"), index=sector.index).map(
        {"Long": 1.0, "Short": -1.0}
    )
    check_close(sector["Effective_Benchmark_Return"],
                side_sign * sector["Benchmark_Return"], "benchmark side sign")
    check_close(sector.loc[~absent, "Effective_Portfolio_Return"],
                side_sign[~absent] * sector.loc[~absent, "Portfolio_Return"], "portfolio side sign")
    check_close(sector["Total_Effect"], sum(sector[column] for column in STOCK_EFFECT_COLUMNS),
                "three-effect total")
    check_close(sector["Benchmark_Centering_Adjustment"],
                (sector["Portfolio_Weight"] - sector["Benchmark_Weight"])
                * sector["Effective_Benchmark_Total_Return"], "benchmark centering")
    active = (sector["Portfolio_Weight"] * sector["Effective_Portfolio_Return"].where(~absent, 0.0)
              - sector["Benchmark_Weight"] * sector["Effective_Benchmark_Return"])
    check_close(sector["Active_Contribution"], active, "raw active contribution")
    centered_residual = (
        active - sector["Benchmark_Centering_Adjustment"] - sector["Total_Effect"]
    ).rename("Residual")
    check_close(centered_residual, 0.0, "centered sector identity")
    check_close(sector["Residual"], centered_residual, "sector residual")
    for column in (
        *STOCK_EFFECT_COLUMNS,
        "Benchmark_Centering_Adjustment", "Total_Effect", "Active_Contribution", "Residual",
    ):
        check_close(sector[f"Scaled_{column}"], sector["Side_Gross_Exposure"] * sector[column],
                    f"gross scaling of {column}")
    books = sector.index.droplevel("GICS_Sector_Code").unique()
    periods = books.droplevel("Side").unique()
    expected_combined = pd.MultiIndex.from_tuples(
        [(*period, "Combined") for period in periods], names=book_keys,
    )
    same_keys(monthly.index, books.append(expected_combined), "interval attribution")
    missing_audit = periods.difference(audit.index)
    if len(missing_audit):
        raise RuntimeError(f"Brinson benchmark audit is missing intervals: {missing_audit.tolist()[:3]}")

    scaled_columns = [f"Scaled_{column}" for column in _RETURN_COLUMNS]
    for key, group in sector.groupby(level=book_keys):
        summary = monthly.loc[key]
        require_finite(monthly.loc[[key]].select_dtypes(include="number"), "interval effects require finite values")
        gross = float(group["Side_Gross_Exposure"].iloc[0])
        if gross <= 0.0:
            raise RuntimeError(f"Brinson Side_Gross_Exposure must be positive at {key}")
        check_close(group["Side_Gross_Exposure"], gross, "constant side gross exposure")
        check_close(summary["Side_Gross_Exposure"], gross, f"Side_Gross_Exposure at {key}")
        for column in ("Portfolio_Weight", "Benchmark_Weight"):
            check_close(group[column].sum(), 1.0, f"{column} at {key}", tolerance=1e-8)
        benchmark_total = float(fsum(group["Benchmark_Weight"] * group["Effective_Benchmark_Return"]))
        sign = 1.0 if key[2] == "Long" else -1.0
        check_close(group["Effective_Benchmark_Total_Return"], benchmark_total, "benchmark total")
        check_close(benchmark_total, sign * audit.at[key[:2], "Reconstructed_Benchmark_Return"],
                    f"Reconstructed_Benchmark_Return at {key}")
        check_close(group[["Benchmark_Centering_Adjustment", "Scaled_Benchmark_Centering_Adjustment"]].sum(),
                    0.0, f"aggregate centering cancellation at {key}")
        portfolio_return = float((group["Portfolio_Weight"] * group["Effective_Portfolio_Return"].fillna(0.0)).sum())
        effects = {column: float(group[column].sum()) for column in STOCK_EFFECT_COLUMNS}
        active_return = portfolio_return - benchmark_total
        expected = {
            "Portfolio_Return": portfolio_return, "Benchmark_Return": benchmark_total,
            "Active_Return": active_return, **effects, "Total_Effect": sum(effects.values()),
            "Residual": active_return - sum(effects.values()),
        }
        for column, value in expected.items():
            check_close(summary[column], value, f"{column} sector-to-side sum at {key}")
            check_close(summary[f"Scaled_{column}"], gross * summary[column],
                        f"Scaled_{column} side scaling at {key}")
        check_close(summary["Residual"], 0.0, f"Residual at {key}")
        check_close(summary["Scaled_Residual"], 0.0, f"Scaled_Residual at {key}")

    for period in periods:
        combined_key = (*period, "Combined")
        combined = monthly.loc[combined_key]
        unscaled = monthly.loc[[combined_key], list(_RETURN_COLUMNS)]
        fail_at(unscaled.notna(), unscaled, "Combined attribution must leave unscaled fields undefined")
        columns = ["Side_Gross_Exposure", *scaled_columns]
        require_finite(monthly.loc[[combined_key], columns], "combined effects require finite values")
        sides = monthly.loc[period].drop(index="Combined")
        check_close(combined[columns].astype(float), sides[columns].sum(), f"side-to-Combined sum at {period}")
        check_close(combined["Scaled_Residual"], 0.0, f"Scaled_Residual at {combined_key}")


def run_brinson_pipeline(
    benchmark_sector: pd.DataFrame,
    benchmark_audit: pd.DataFrame,
    holdings: pd.DataFrame,
) -> BrinsonResult:
    """Run the common portfolio, attribution, summary, and reconciliation steps."""
    portfolio_sector = build_brinson_portfolio_sector_series(holdings)
    sector_attribution = build_brinson_sector_attribution(
        benchmark_sector,
        portfolio_sector,
    )
    period_attribution = summarize_brinson_monthly_attribution(
        sector_attribution
    )
    total_attribution = summarize_brinson_period_attribution(
        period_attribution
    )
    validate_brinson_outputs(
        benchmark_audit,
        portfolio_sector,
        period_attribution,
        sector_attribution_df=sector_attribution,
    )
    return BrinsonResult(
        benchmark_sector=benchmark_sector,
        benchmark_audit=benchmark_audit,
        portfolio_sector=portfolio_sector,
        sector_attribution=sector_attribution,
        period_attribution=period_attribution,
        total_attribution=total_attribution,
    )


__all__ = [
    "ATTRIBUTION_METHOD",
    "STOCK_EFFECT_COLUMNS",
    "BENCHMARK_CONSTITUENT_COLUMNS",
    "BrinsonResult",
    "add_benchmark_consistency_columns",
    "build_brinson_portfolio_sector_series",
    "build_brinson_sector_attribution",
    "build_brinson_benchmark",
    "run_brinson_pipeline",
    "safe_weighted_average",
    "summarize_brinson_monthly_attribution",
    "summarize_brinson_period_attribution",
    "validate_brinson_outputs",
]
