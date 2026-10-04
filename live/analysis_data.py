"""Validated prepared strategy-data contracts and persistence for live analysis."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_core.corporate_actions import (
    EVENT_COLUMNS as CORPORATE_ACTION_EVENT_COLUMNS,
    LEG_COLUMNS as CORPORATE_ACTION_LEG_COLUMNS,
    SOURCE_COLUMNS as CORPORATE_ACTION_SOURCE_COLUMNS,
    apply_corporate_actions,
)
from portfolio_core.io import atomic_write_dataframe
from portfolio_core.price_basis import (
    PriceBasisSpec,
    price_basis_spec,
    validate_price_basis,
)
from portfolio_core.sector_assignments import load_sector_assignments, validate_sector_assignments

from .corporate_action_policy import (
    CORPORATE_ACTION_POLICY_COLUMNS,
    LiveCorporateActionBundle,
    _validate_valuation_dates,
)
from .monthly_history import (
    aggregate_monthly_market, empty_dividends, validate_dividends,
    validate_monthly_history,
)
from .preparation_artifacts import (
    PREPARE_COMMAND,
    validate_preparation_manifest,
)
from .paths import LivePaths
from .config import DEFAULT_CONFIG, DecisionPeriod, LiveConfig
from .price_coverage import YAHOO_SUPPLEMENT_SOURCE
from .strategy_universe import (
    DOWNLOAD_BENCHMARK_COMMAND,
    DOWNLOAD_PRICES_COMMAND,
    EVALUATION_PERIOD_COLUMNS,
    METADATA_COLUMNS,
    SCHEDULE_COLUMNS,
    SCHEDULE_DATE_COLUMNS,
    evaluation_periods,
    historical_monthly_dates,
    prepared_membership_asof,
    sector_assignment_requirements,
    validate_membership_provider_identities,
)


MARKET_COLUMNS = {
    "Price_Source",
    "Date",
    "Asset_ID",
    "Source_Ticker",
    "Yahoo_Ticker",
    "Open",
    "Close",
    "Volume",
}
MEMBERSHIP_COLUMNS = {"Effective_Date", "Asset_ID"}
BENCHMARK_COLUMNS = {"Date", "Open", "Close"}


@dataclass(frozen=True)
class LiveAnalysisInputs:
    """Validated prepared tables needed by the causal strategy."""

    market_daily: pd.DataFrame
    membership: pd.DataFrame
    metadata: pd.DataFrame
    sector_assignments: pd.DataFrame
    schedule: pd.DataFrame
    benchmark_daily: pd.DataFrame
    corporate_actions: LiveCorporateActionBundle
    evaluation_periods: pd.DataFrame
    price_basis: PriceBasisSpec = field(
        default_factory=lambda: price_basis_spec("live")
    )
    market_monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_dividends: pd.DataFrame = field(default_factory=empty_dividends)


@dataclass(frozen=True)
class LiveAnalysisResult:
    """In-memory historical live strategy artifacts."""

    nav: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    decisions: pd.DataFrame
    performance: pd.DataFrame
    corporate_actions: pd.DataFrame
    spread_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    research_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_residuals: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_nav: pd.DataFrame = field(default_factory=pd.DataFrame)


def read_prepared_csv(
    path: Path,
    label: str,
    command: str = PREPARE_COMMAND,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prepared {label}: {path}. Run `{command}` first."
        )
    return pd.read_csv(path, keep_default_na=False)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prepared {label} is missing columns: {missing}")


def _require_exact_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...],
    label: str,
) -> None:
    if tuple(frame.columns) != required:
        raise ValueError(
            f"Prepared {label} must have columns {list(required)}; "
            f"found {list(frame.columns)}"
        )


def validate_analysis_manifest(
    paths: LivePaths,
    required_artifacts: Collection[str],
) -> None:
    """Validate analysis inputs through the shared schema-v7 contract."""
    validate_preparation_manifest(
        paths,
        include_brinson=False,
        check_raw=False,
        required_artifacts=required_artifacts,
    )


def _parse_dates(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        try:
            frame[column] = pd.to_datetime(
                frame[column], format="%Y-%m-%d", errors="raise"
            ).dt.tz_localize(None)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Prepared {label} contains an invalid {column}; expected YYYY-MM-DD"
            ) from exc


def _empty_corporate_action_bundle() -> LiveCorporateActionBundle:
    return LiveCorporateActionBundle(
        events=pd.DataFrame(columns=CORPORATE_ACTION_EVENT_COLUMNS),
        legs=pd.DataFrame(columns=CORPORATE_ACTION_LEG_COLUMNS),
        sources=pd.DataFrame(columns=CORPORATE_ACTION_SOURCE_COLUMNS),
        policy=pd.DataFrame(columns=CORPORATE_ACTION_POLICY_COLUMNS),
    )


def _validate_market_table(market: pd.DataFrame) -> list[str]:
    _require_columns(market, MARKET_COLUMNS, "market data")
    _parse_dates(market, ["Date"], "market data")
    for column in ("Price_Source", "Asset_ID", "Source_Ticker", "Yahoo_Ticker"):
        market[column] = market[column].astype(str).str.strip()
    if market[["Price_Source", "Asset_ID"]].eq("").any().any():
        raise ValueError("Prepared market data contains a blank key")
    invalid_sources = sorted(
        set(market["Price_Source"]) - {"yahoo", YAHOO_SUPPLEMENT_SOURCE}
    )
    if invalid_sources:
        raise ValueError(f"Prepared market data has invalid Price_Source values: {invalid_sources}")
    market_key = ["Date", "Asset_ID"]
    if market.duplicated(market_key).any():
        raise ValueError(
            f"Prepared market data must be unique at {tuple(market_key)}"
        )
    raw_numeric = market[["Open", "Close", "Volume"]].replace("", np.nan)
    market_numeric = raw_numeric.apply(pd.to_numeric, errors="coerce")
    if (raw_numeric.notna() & market_numeric.isna()).any().any():
        raise ValueError("Prepared market data contains nonnumeric OHLCV values")
    finite_mask = (
        np.isfinite(market_numeric.to_numpy(dtype=float))
        | market_numeric.isna().to_numpy()
    )
    if not finite_mask.all():
        raise ValueError("Prepared market data contains nonfinite OHLCV values")
    if (
        (market_numeric[["Open", "Close"]] <= 0.0)
        & market_numeric[["Open", "Close"]].notna()
    ).any().any():
        raise ValueError("Prepared market prices must be positive")
    if (
        (market_numeric["Volume"] < 0.0)
        & market_numeric["Volume"].notna()
    ).any():
        raise ValueError("Prepared market volume cannot be negative")
    market[["Open", "Close", "Volume"]] = market_numeric.astype(float)
    return market_key


def _validate_membership_table(pit: pd.DataFrame) -> None:
    _require_columns(pit, MEMBERSHIP_COLUMNS, "membership")
    _parse_dates(pit, ["Effective_Date"], "membership")
    pit["Asset_ID"] = pit["Asset_ID"].astype(str).str.strip()
    if pit.duplicated(["Effective_Date", "Asset_ID"]).any():
        raise ValueError(
            "Prepared membership must be unique at (Effective_Date, Asset_ID)"
        )


def _validate_metadata_table(assets: pd.DataFrame) -> None:
    _require_exact_columns(assets, METADATA_COLUMNS, "asset metadata")
    for column in METADATA_COLUMNS:
        assets[column] = assets[column].astype(str).str.strip()
    if assets["Asset_ID"].eq("").any() or assets["Asset_ID"].duplicated().any():
        raise ValueError("Prepared asset metadata requires unique nonblank Asset_IDs")
    if assets[["Source_Ticker", "Yahoo_Ticker"]].eq("").any().any():
        raise ValueError("Prepared asset metadata contains a blank provider identity")


def _validate_schedule_table(periods: pd.DataFrame) -> None:
    _require_columns(periods, set(SCHEDULE_COLUMNS), "decision schedule")
    _parse_dates(
        periods,
        SCHEDULE_DATE_COLUMNS,
        "decision schedule",
    )
    if periods[list(SCHEDULE_DATE_COLUMNS)].isna().any().any():
        raise ValueError("Prepared schedule contains missing dates")
    periods["Rebalance_ID"] = periods["Rebalance_ID"].astype(str).str.strip()
    for column in ("Sizing_Field", "Execution_Field", "Valuation_Field"):
        periods[column] = periods[column].astype(str).str.title().str.strip()
    if periods["Rebalance_ID"].eq("").any() or periods["Rebalance_ID"].duplicated().any():
        raise ValueError("Prepared schedule requires unique nonblank Rebalance_IDs")
    ordered = periods.sort_values("Execution_Date", kind="stable").reset_index(drop=True)
    if not ordered.equals(periods.reset_index(drop=True)):
        raise ValueError("Prepared schedule must be sorted by Execution_Date")
    for row in periods.itertuples(index=False):
        DecisionPeriod(
            rebalance_id=str(row.Rebalance_ID),
            membership_effective_date=row.Membership_Effective_Date.date(),
            signal_cutoff=row.Signal_Cutoff.date(),
            sizing_date=row.Sizing_Date.date(),
            execution_date=row.Execution_Date.date(),
            valuation_end=row.Valuation_End.date(),
            sizing_field=str(row.Sizing_Field),
            execution_field=str(row.Execution_Field),
            valuation_field=str(row.Valuation_Field),
        )
    for previous, current in zip(
        periods.iloc[:-1].itertuples(),
        periods.iloc[1:].itertuples(),
    ):
        if pd.Timestamp(previous.Valuation_End) != pd.Timestamp(current.Execution_Date):
            raise ValueError(
                "Strategy periods must be contiguous: each Valuation_End must equal "
                "the next Execution_Date"
            )
        if str(previous.Valuation_Field) != str(current.Execution_Field):
            raise ValueError(
                "Adjacent live periods must use the same valuation and execution "
                "price field at their shared boundary"
            )
        field_order = {"Open": 0, "Close": 1}
        if (pd.Timestamp(current.Sizing_Date), field_order[current.Sizing_Field]) < (
            pd.Timestamp(previous.Execution_Date), field_order[previous.Execution_Field]
        ):
            raise ValueError(
                "Each next sizing checkpoint must lie between the adjacent "
                "execution dates"
            )


def _validate_sector_assignment_table(
    assignments: pd.DataFrame,
    membership: pd.DataFrame,
    schedule: pd.DataFrame,
    actions: LiveCorporateActionBundle,
    *,
    evaluation_start: object,
    history_dates=(),
) -> pd.DataFrame:
    normalized = validate_sector_assignments(assignments)
    requirements = sector_assignment_requirements(
        schedule,
        membership,
        evaluation_start=evaluation_start,
        corporate_action_events=actions.events,
        corporate_action_legs=actions.legs,
        history_dates=history_dates,
    )
    expected_pairs = set(
        requirements[["As_Of_Date", "Asset_ID"]].itertuples(index=False, name=None)
    )
    actual_pairs = set(
        normalized[["As_Of_Date", "Asset_ID"]].itertuples(index=False, name=None)
    )
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        unexpected = sorted(actual_pairs - expected_pairs)
        raise ValueError(
            "Prepared sector assignments do not exactly cover consumed live "
            f"asset/date pairs; missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    return normalized


def _validate_benchmark_table(benchmark: pd.DataFrame) -> None:
    _require_columns(benchmark, BENCHMARK_COLUMNS, "benchmark")
    _parse_dates(benchmark, ["Date"], "benchmark")
    benchmark_numeric = benchmark[["Open", "Close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        benchmark_numeric.isna().any().any()
        or not np.isfinite(benchmark_numeric.to_numpy(dtype=float)).all()
        or (benchmark_numeric <= 0.0).any().any()
    ):
        raise ValueError(
            "Prepared benchmark Open/Close values must be finite and positive"
        )
    benchmark[["Open", "Close"]] = benchmark_numeric.astype(float)
    if benchmark["Date"].duplicated().any():
        raise ValueError("Prepared benchmark must have one row per Date")


def _validate_corporate_action_tables(actions: LiveCorporateActionBundle) -> None:
    _require_columns(
        actions.events,
        set(CORPORATE_ACTION_EVENT_COLUMNS),
        "corporate-action events",
    )
    _require_columns(
        actions.legs,
        set(CORPORATE_ACTION_LEG_COLUMNS),
        "corporate-action legs",
    )
    _require_columns(
        actions.sources,
        set(CORPORATE_ACTION_SOURCE_COLUMNS),
        "corporate-action sources",
    )
    _require_columns(
        actions.policy,
        set(CORPORATE_ACTION_POLICY_COLUMNS),
        "corporate-action policy",
    )
    _parse_dates(actions.events, ["Effective_Date"], "corporate-action events")
    policy_ids = set(actions.policy["Event_ID"].astype(str))
    event_ids = set(actions.events["Event_ID"].astype(str))
    if policy_ids != event_ids:
        raise ValueError(
            "Prepared corporate-action policy and event tables contain different Event_IDs"
        )
    if not actions.policy.empty:
        if not actions.policy["Review_Status"].eq("approved").all():
            raise ValueError("Prepared corporate-action policy must be approved")
        applied = actions.policy["Apply_Accounting"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
        if not applied.all():
            raise ValueError(
                "Prepared corporate-action bundle contains a disabled policy row"
            )
        base = pd.to_numeric(
            actions.policy["CVR_Base_Value_Per_Unit"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.isfinite(base).all() or (base < 0.0).any():
            raise ValueError("Prepared corporate-action CVR base values are invalid")
        _validate_valuation_dates(actions.policy)
    if actions.events.empty:
        validation_start = validation_end = pd.Timestamp("1970-01-01")
    else:
        validation_start = (
            actions.events["Effective_Date"].min() - pd.Timedelta(days=1)
        )
        validation_end = actions.events["Effective_Date"].max()
    apply_corporate_actions(
        {},
        {},
        actions.events,
        actions.legs,
        actions.sources,
        start_exclusive=validation_start,
        end_inclusive=validation_end,
    )


def validate_analysis_inputs(
    market_daily: pd.DataFrame,
    membership: pd.DataFrame,
    metadata: pd.DataFrame,
    sector_assignments: pd.DataFrame,
    schedule: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    corporate_actions: LiveCorporateActionBundle | None = None,
    *,
    evaluation_start: object,
    evaluation_end: object,
    price_basis: PriceBasisSpec | None = None,
    history_start: object | None = None,
    market_monthly: pd.DataFrame | None = None,
    monthly_dividends: pd.DataFrame | None = None,
) -> LiveAnalysisInputs:
    """Validate and normalize all prepared strategy inputs without writing files."""
    market = market_daily.copy()
    pit = membership.copy()
    assets = metadata.copy()
    sectors = sector_assignments.copy()
    periods = schedule.copy()
    benchmark = benchmark_daily.copy()
    supplied_actions = (
        corporate_actions
        if corporate_actions is not None
        else _empty_corporate_action_bundle()
    )
    actions = LiveCorporateActionBundle(
        events=supplied_actions.events.copy(),
        legs=supplied_actions.legs.copy(),
        sources=supplied_actions.sources.copy(),
        policy=supplied_actions.policy.copy(),
    )
    supplied_price_basis = price_basis or price_basis_spec("live")
    if supplied_price_basis != price_basis_spec("live"):
        raise ValueError(
            f"Unsupported live price basis: {supplied_price_basis!r}"
        )

    market_key = _validate_market_table(market)
    _validate_membership_table(pit)
    _validate_metadata_table(assets)
    validate_membership_provider_identities(pit, assets)
    _validate_schedule_table(periods)
    # In-memory callers may supply daily-only fixtures. The production loader
    # below always requires both prepared monthly artifacts and their hashes.
    monthly = (aggregate_monthly_market(market, through=periods.Signal_Cutoff.max())
               if market_monthly is None else market_monthly.copy())
    dividends = (empty_dividends() if monthly_dividends is None else monthly_dividends.copy())
    monthly = validate_monthly_history(monthly)
    dividends = validate_dividends(dividends)
    account_periods = evaluation_periods(
        periods, evaluation_start=evaluation_start, evaluation_end=evaluation_end
    )
    _validate_benchmark_table(benchmark)
    _validate_corporate_action_tables(actions)
    sectors = _validate_sector_assignment_table(
        sectors,
        pit,
        periods,
        actions,
        evaluation_start=evaluation_start,
        history_dates=(historical_monthly_dates(
            market.loc[market.Date.ge(pd.Timestamp(history_start)), "Date"],
            through=periods.Signal_Cutoff.max(),
        ) if history_start is not None else ()),
    )

    unknown_members = sorted(set(pit["Asset_ID"]) - set(assets["Asset_ID"]))
    if unknown_members:
        raise ValueError(
            "Prepared membership has Asset_IDs missing from metadata: "
            f"{unknown_members[:20]}"
        )

    return LiveAnalysisInputs(
        market_daily=market.sort_values(market_key, kind="stable").reset_index(drop=True),
        membership=pit.sort_values(
            ["Effective_Date", "Asset_ID"], kind="stable"
        ).reset_index(drop=True),
        metadata=assets.sort_values("Asset_ID", kind="stable").reset_index(drop=True),
        sector_assignments=sectors,
        schedule=periods.reset_index(drop=True),
        evaluation_periods=account_periods,
        benchmark_daily=benchmark.sort_values(
            "Date", kind="stable"
        ).reset_index(drop=True),
        corporate_actions=LiveCorporateActionBundle(
            events=actions.events.sort_values(
                ["Effective_Date", "Event_ID"], kind="stable"
            ).reset_index(drop=True),
            legs=actions.legs.sort_values(
                ["Event_ID", "Leg_Order"], kind="stable"
            ).reset_index(drop=True),
            sources=actions.sources.sort_values(
                ["Event_ID", "Source_URL"], kind="stable"
            ).reset_index(drop=True),
            policy=actions.policy.sort_values(
                "Event_ID", kind="stable"
            ).reset_index(drop=True),
        ),
        price_basis=supplied_price_basis,
        market_monthly=monthly,
        monthly_dividends=dividends,
    )


def load_analysis_inputs(
    paths: LivePaths, *, config: LiveConfig = DEFAULT_CONFIG
) -> LiveAnalysisInputs:
    """Load prepared strategy CSVs; never fall back to raw data or downloads."""
    market_path = paths.market_daily_csv
    membership_path = paths.pit_membership_csv
    metadata_path = paths.asset_metadata_csv
    sector_assignments_path = paths.sector_assignments_csv
    schedule_path = paths.decision_schedule_csv
    benchmark_path = paths.prepared_benchmark_csv
    corporate_events_path = paths.prepared_corporate_action_events_csv
    corporate_legs_path = paths.prepared_corporate_action_legs_csv
    corporate_sources_path = paths.prepared_corporate_action_sources_csv
    corporate_policy_path = paths.prepared_corporate_action_policy_csv
    price_basis_path = paths.price_basis_csv
    validate_analysis_manifest(
        paths,
        {
            "market_daily",
            "market_monthly",
            "monthly_dividends_prepared",
            "pit_membership",
            "asset_metadata",
            "sector_assignments",
            "decision_schedule",
            "benchmark_prepared",
            "corporate_action_events_prepared",
            "corporate_action_legs_prepared",
            "corporate_action_sources_prepared",
            "corporate_action_policy_prepared",
            "price_basis",
        },
    )
    price_basis = validate_price_basis(
        read_prepared_csv(price_basis_path, "price basis"),
        "live",
    )
    return validate_analysis_inputs(
        read_prepared_csv(market_path, "market data"),
        read_prepared_csv(membership_path, "membership"),
        read_prepared_csv(metadata_path, "asset metadata"),
        load_sector_assignments(sector_assignments_path),
        read_prepared_csv(schedule_path, "decision schedule"),
        read_prepared_csv(
            benchmark_path,
            "benchmark",
            DOWNLOAD_BENCHMARK_COMMAND,
        ),
        LiveCorporateActionBundle(
            events=read_prepared_csv(
                corporate_events_path, "corporate-action events"
            ),
            legs=read_prepared_csv(corporate_legs_path, "corporate-action legs"),
            sources=read_prepared_csv(
                corporate_sources_path, "corporate-action sources"
            ),
            policy=read_prepared_csv(
                corporate_policy_path, "corporate-action policy"
            ),
        ),
        evaluation_start=config.market.competition_start,
        evaluation_end=config.market.competition_end,
        price_basis=price_basis,
        history_start=config.market.download_start,
        market_monthly=read_prepared_csv(paths.market_monthly_csv, "monthly market history"),
        monthly_dividends=read_prepared_csv(paths.monthly_dividends_csv, "monthly dividend evidence"),
    )


def validate_prepared_sector_contract(
    paths: LivePaths, *, config: LiveConfig = DEFAULT_CONFIG
) -> None:
    """Semantically validate dated sector coverage for ``live.prepare --check``."""
    market = read_prepared_csv(
        paths.market_daily_csv,
        "market data",
    )
    membership = read_prepared_csv(
        paths.pit_membership_csv,
        "membership",
    )
    metadata = read_prepared_csv(
        paths.asset_metadata_csv,
        "asset metadata",
    )
    schedule = read_prepared_csv(
        paths.decision_schedule_csv,
        "decision schedule",
    )
    assignments = load_sector_assignments(
        paths.sector_assignments_csv
    )
    actions = LiveCorporateActionBundle(
        events=read_prepared_csv(
            paths.prepared_corporate_action_events_csv,
            "corporate-action events",
        ),
        legs=read_prepared_csv(
            paths.prepared_corporate_action_legs_csv,
            "corporate-action legs",
        ),
        sources=read_prepared_csv(
            paths.prepared_corporate_action_sources_csv,
            "corporate-action sources",
        ),
        policy=read_prepared_csv(
            paths.prepared_corporate_action_policy_csv,
            "corporate-action policy",
        ),
    )
    validate_price_basis(
        read_prepared_csv(paths.price_basis_csv, "price basis"),
        "live",
    )
    _validate_market_table(market)
    validate_monthly_history(read_prepared_csv(paths.market_monthly_csv, "monthly market history"))
    validate_dividends(read_prepared_csv(paths.monthly_dividends_csv, "monthly dividend evidence"))
    _validate_membership_table(membership)
    _validate_metadata_table(metadata)
    validate_membership_provider_identities(membership, metadata)
    _validate_schedule_table(schedule)
    _validate_corporate_action_tables(actions)
    unknown_members = sorted(set(membership["Asset_ID"]) - set(metadata["Asset_ID"]))
    if unknown_members:
        raise ValueError(
            "Prepared membership has Asset_IDs missing from metadata: "
            f"{unknown_members[:20]}"
        )
    _validate_sector_assignment_table(
        assignments,
        membership,
        schedule,
        actions,
        evaluation_start=config.market.competition_start,
        history_dates=historical_monthly_dates(
            market.loc[market.Date.ge(pd.Timestamp(config.market.download_start)), "Date"],
            through=schedule.Signal_Cutoff.max(),
        ),
    )


def membership_as_of(
    membership: pd.DataFrame,
    date: pd.Timestamp,
) -> list[str]:
    try:
        return sorted(prepared_membership_asof(membership, date))
    except ValueError as exc:
        if "No prepared constituent membership" not in str(exc):
            raise
        raise RuntimeError(
            "No prepared constituent membership on or before "
            f"{pd.Timestamp(date).date()}"
        ) from exc


def price_series(
    market: pd.DataFrame,
    date: pd.Timestamp,
    field: str,
    asset_ids: Iterable[str],
    *,
    context: str,
) -> pd.Series:
    asset_list = sorted(set(str(asset_id) for asset_id in asset_ids))
    if not asset_list:
        return pd.Series(dtype=float)
    rows = market.loc[
        market["Date"].eq(pd.Timestamp(date))
        & market["Asset_ID"].isin(asset_list),
        ["Asset_ID", field],
    ]
    values = rows.set_index("Asset_ID")[field].reindex(asset_list).astype(float)
    invalid = values.isna() | ~np.isfinite(values) | (values <= 0.0)
    if invalid.any():
        missing = values.index[invalid].tolist()
        raise RuntimeError(
            f"Missing prepared {field.lower()} prices for {context} on "
            f"{pd.Timestamp(date).date()}: {missing}. Run "
            f"`{DOWNLOAD_PRICES_COMMAND}` then `{PREPARE_COMMAND}`."
        )
    return values


def validate_account_nav(
    nav: pd.DataFrame, *, periods: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate account continuity, cash intervals and saved period boundaries."""
    nav = nav.copy()
    if not nav.empty:
        required_nav = {
            *EVALUATION_PERIOD_COLUMNS,
            "Start_NAV", "Period_Return", "Benchmark_Return",
            "Fixed_Fees", "Spread_Cost", "Position_Count", "Order_Count",
            "End_Position_Count", "Post_Trade_Gross_Market_Value",
            "Post_Trade_NAV",
            "End_NAV",
            "Interest",
            "Cash_Interest_Credit",
            "Loan_Interest_Charge",
            "Post_Trade_Cash",
            "Post_Trade_Signed_Market_Value",
            "Restricted_Short_Proceeds",
            "Free_Cash",
            "Loan",
            "End_Cash",
            "End_Signed_Market_Value",
            "End_Restricted_Short_Proceeds",
            "End_Free_Cash",
            "End_Loan",
        }
        missing_nav = sorted(required_nav - set(nav.columns))
        if missing_nav:
            raise ValueError(
                f"Live NAV audit is missing accounting columns: {missing_nav}"
            )
        for column in ("Period_Start", "Period_End"):
            nav[column] = pd.to_datetime(nav[column], errors="raise")
        nav["Rebalance_ID"] = nav["Rebalance_ID"].fillna("").astype(str)
        numeric = sorted(required_nav - set(EVALUATION_PERIOD_COLUMNS))
        nav[numeric] = nav[numeric].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(nav[numeric].to_numpy(dtype=float)).all():
            raise ValueError("Live NAV audit requires finite accounting values")
        if (nav[["Start_NAV", "Post_Trade_NAV", "End_NAV"]] <= 0.0).any().any():
            raise ValueError("Live NAV audit requires positive equity")
        if periods is not None:
            actual = nav[list(EVALUATION_PERIOD_COLUMNS)].reset_index(drop=True)
            actual_rows = list(actual.itertuples(index=False, name=None))
            expected_rows = list(periods.itertuples(index=False, name=None))
            if actual_rows != expected_rows:
                raise ValueError("Live NAV boundaries do not match the evaluation periods")
        if len(nav) > 1:
            for end_column, start_column in (
                ("Period_End", "Period_Start"), ("End_Field", "Start_Field"),
            ):
                if not np.array_equal(
                    nav[end_column].iloc[:-1].to_numpy(),
                    nav[start_column].iloc[1:].to_numpy(),
                ):
                    raise ValueError("Live NAV interval boundaries are not contiguous")
        cash = nav.loc[nav["Period_Type"].eq("cash")]
        cash_zero = [
            "Position_Count", "End_Position_Count", "Order_Count", "Fixed_Fees",
            "Spread_Cost", "Post_Trade_Gross_Market_Value",
            "Post_Trade_Signed_Market_Value", "End_Signed_Market_Value",
            "Restricted_Short_Proceeds", "End_Restricted_Short_Proceeds",
            "Loan", "End_Loan",
        ]
        if not cash.empty and (
            cash[cash_zero].ne(0.0).any().any() or cash["Rebalance_ID"].ne("").any()
        ):
            raise ValueError("Live cash intervals must have no holdings, orders or costs")
        if nav.loc[nav["Period_Type"].eq("invested"), "Position_Count"].le(0).any():
            raise ValueError("Live invested intervals require holdings")
        reconciliations = (
            (
                nav["Start_NAV"].iloc[1:].to_numpy(),
                nav["End_NAV"].iloc[:-1].to_numpy(),
                "adjacent NAV",
            ),
            (
                nav["Period_Return"],
                nav["End_NAV"] / nav["Start_NAV"] - 1.0,
                "period returns",
            ),
            (
                nav["Start_NAV"] - nav["Post_Trade_NAV"],
                nav["Fixed_Fees"] + nav["Spread_Cost"],
                "transaction costs",
            ),
            (
                nav["Post_Trade_NAV"],
                nav["Post_Trade_Cash"]
                + nav["Post_Trade_Signed_Market_Value"],
                "post-trade NAV",
            ),
            (
                nav["End_NAV"],
                nav["End_Cash"] + nav["End_Signed_Market_Value"],
                "end NAV",
            ),
            (
                nav["Free_Cash"],
                nav["Post_Trade_Cash"]
                - nav["Restricted_Short_Proceeds"],
                "post-trade free cash",
            ),
            (
                nav["End_Free_Cash"],
                nav["End_Cash"]
                - nav["End_Restricted_Short_Proceeds"],
                "end free cash",
            ),
            (
                nav["Loan"],
                (-nav["Free_Cash"]).clip(lower=0.0),
                "post-trade loan",
            ),
            (
                nav["End_Loan"],
                (-nav["End_Free_Cash"]).clip(lower=0.0),
                "end loan",
            ),
            (
                nav["Interest"],
                nav["Cash_Interest_Credit"]
                - nav["Loan_Interest_Charge"],
                "period financing",
            ),
        )
        for actual, expected, label in reconciliations:
            if not np.allclose(actual, expected, atol=1e-7, rtol=0.0):
                raise ValueError(f"Live NAV audit does not reconcile {label}")
    elif periods is not None:
        raise ValueError("Live NAV audit cannot be empty")
    return nav


def save_analysis_result(result: LiveAnalysisResult, paths: LivePaths) -> None:
    """Persist a completed strategy only after all periods pass validation."""
    validate_account_nav(result.nav)
    outputs = (
        (result.nav, paths.strategy_nav_csv),
        (result.daily_nav, paths.strategy_daily_nav_csv),
        (result.holdings, paths.strategy_holdings_csv),
        (result.trades, paths.strategy_trades_csv),
        (result.decisions, paths.strategy_decisions_csv),
        (result.performance, paths.strategy_performance_csv),
        (result.corporate_actions, paths.strategy_corporate_actions_csv),
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
    if not result.spread_sensitivity.empty:
        atomic_write_dataframe(
            result.spread_sensitivity,
            paths.spread_sensitivity_csv,
            index=False,
            date_format=None,
            float_format=None,
            lineterminator="\n",
        )


__all__ = [
    "LiveAnalysisInputs",
    "LiveAnalysisResult",
    "load_analysis_inputs",
    "membership_as_of",
    "price_series",
    "read_prepared_csv",
    "save_analysis_result",
    "validate_analysis_manifest",
    "validate_analysis_inputs",
    "validate_prepared_sector_contract",
]
